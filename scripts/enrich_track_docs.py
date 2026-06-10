"""Offline LLM track-document enrichment (Track B — plan: drastically improve nDCG@20).

The dense recall channel is the weak link (recall@100 0.179 < BM25 0.346): a 0.6B
embedder over thin metadata can't connect a new-artist gold to an intent query like
"high-energy hip-hop for driving". This is the DOC-SIDE dual of propose-ground:
for each catalog track, an LLM writes ONE rich description (genre, mood, era,
instrumentation, similar artists, typical use) from its world knowledge. Those
docs are then re-embedded with a STRONGER model (scripts/embed_catalog.py
--doc-source ...) and wired as a new dense channel (use_doc_enriched). Attacks the
new-artist wall the session/lexical channels structurally can't reach.

`build_enrich_prompt` + `parse_enriched_doc` are pure and unit-tested; the LLM call
is integration (Colab). Routes to Gemini (gemini-*) or a local HF instruct model,
mirroring scripts/doc2query_generate.py.

Usage:
    python scripts/enrich_track_docs.py --model gemini-2.5-flash-lite --batch-size 16
    python scripts/enrich_track_docs.py --model Qwen/Qwen2.5-7B-Instruct --max-tracks 20

Output: parquet at experiments/cache/enriched_docs/<safe_model>/docs.parquet
  Columns: track_id (str), enriched_doc (str)
Then: python scripts/embed_catalog.py --model <strong-embedder> --label doc-enriched-v1 \
        --doc-source experiments/cache/enriched_docs/<safe_model>/docs.parquet
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_PROMPT_TEMPLATE = """You are a music expert writing a catalog description for a recommendation engine. Using your real-world knowledge of this track and artist, write ONE rich, factual paragraph (40-80 words) describing it: genre and subgenre, mood and energy, era, instrumentation and vocal style, lyrical themes, similar artists, and the occasions or activities it suits. Do not invent chart facts. Write only the paragraph, no preamble, no quotes.

Track metadata:
{metadata_lines}

Description:"""

_LABELS = {
    "track_name": "Title",
    "artist_name": "Artist",
    "album_name": "Album",
    "tag_list": "Tags",
    "release_date": "Release date",
}


def build_enrich_prompt(metadata: dict) -> str:
    """Prompt asking for one rich descriptive paragraph for a track. Absent/empty
    fields are skipped so the format degrades cleanly."""
    lines: list[str] = []
    for field, label in _LABELS.items():
        val = metadata.get(field)
        if val is None or val == "":
            continue
        rendered = ", ".join(str(v) for v in val) if isinstance(val, list) else str(val)
        if rendered:
            lines.append(f"- {label}: {rendered}")
    return _PROMPT_TEMPLATE.format(metadata_lines="\n".join(lines))


_PREAMBLE = re.compile(r"^\s*(here (is|'s)[^:]*:|description:|sure[,!.][^\n]*)\s*",
                       re.IGNORECASE)


def parse_enriched_doc(completion: str, max_chars: int = 1200) -> str:
    """Clean an LLM completion to a single-paragraph description: drop common
    preamble, strip surrounding quotes, collapse whitespace/newlines, cap length."""
    if not completion or not completion.strip():
        return ""
    text = _PREAMBLE.sub("", completion.strip())
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip('"“”‘’').strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]
    return text


def _gemini_generate(model_name: str, prompts: list[str], batch_size: int,
                     max_retries: int = 3) -> list[str]:
    from concurrent.futures import ThreadPoolExecutor
    sys.path.insert(0, str(REPO_ROOT / "music-crs-baselines"))
    from mcrs.query_rewriters.gemini_propose import GeminiClient
    client = GeminiClient(model=model_name, max_output_tokens=256)

    def _one(p: str) -> str:
        for _ in range(max_retries):
            try:
                return client.generate("You are a music cataloging expert.", p)
            except Exception:
                continue
        return ""
    workers = max(1, min(batch_size, len(prompts)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_one, prompts))


def _hf_generate(model_name: str, prompts: list[str], batch_size: int,
                 max_new_tokens: int) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    out: list[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        chat = [tok.apply_chat_template([{"role": "user", "content": p}],
                                        tokenize=False, add_generation_prompt=True)
                for p in batch]
        enc = tok(chat, padding=True, truncation=True, max_length=1024,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return out


def main() -> int:
    import argparse
    import os

    import pandas as pd
    from datasets import load_dataset
    from tqdm import tqdm

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gemini-2.5-flash-lite",
                   help="gemini-* (API) or a HF instruct model name.")
    p.add_argument("--catalog-dataset", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    p.add_argument("--catalog-split", default="all_tracks")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--max-tracks", type=int, default=None, help="smoke cap")
    p.add_argument("--cache-root", default=str(REPO_ROOT / "experiments" / "cache" / "enriched_docs"))
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    safe = args.model.replace("/", "_")
    out_dir = Path(args.cache_root) / safe
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "docs.parquet"

    done, existing = set(), []
    if args.resume and out_path.exists():
        df = pd.read_parquet(out_path)
        done = set(df["track_id"]); existing = df.to_dict("records")
        print(f"[enrich] resume: {len(done)} done", file=sys.stderr)

    ds = load_dataset(args.catalog_dataset, split=args.catalog_split)
    if args.max_tracks is not None:
        ds = ds.select(range(min(args.max_tracks, len(ds))))
    rows = [r for r in ds if r["track_id"] not in done]
    if not rows:
        print(f"[enrich] nothing to do; {out_path}", file=sys.stderr)
        return 0
    print(f"[enrich] generating {len(rows)} docs with {args.model}", file=sys.stderr)

    prompts = [build_enrich_prompt(r) for r in rows]
    is_gemini = args.model.lower().startswith("gemini")
    new_rows: list[dict] = []
    flush_every = 500
    CH = max(args.batch_size, 1) * 8  # generate in chunks so we can flush/resume

    def _flush():
        pd.DataFrame(existing + new_rows).to_parquet(str(out_path) + ".tmp", index=False)
        os.replace(str(out_path) + ".tmp", out_path)

    for s in tqdm(range(0, len(rows), CH), desc="enrich"):
        chunk, cprompts = rows[s:s + CH], prompts[s:s + CH]
        if is_gemini:
            raws = _gemini_generate(args.model, cprompts, args.batch_size)
        else:
            raws = _hf_generate(args.model, cprompts, args.batch_size, args.max_new_tokens)
        for r, raw in zip(chunk, raws):
            new_rows.append({"track_id": r["track_id"], "enriched_doc": parse_enriched_doc(raw)})
        if len(new_rows) % flush_every < CH:
            _flush()
    _flush()
    print(f"[enrich] wrote {len(existing) + len(new_rows)} rows -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
