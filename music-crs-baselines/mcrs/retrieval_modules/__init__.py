from pathlib import Path

from .bm25 import BM25_MODEL
from .bert import BERT_MODEL
from .dense_precomputed import DENSE_PRECOMPUTED
from .dense_local import DENSE_LOCAL
from .dense_multimodal_local import DENSE_MULTIMODAL_LOCAL
from .rrf import RRF_MODEL
from .sequential_rerank import SEQUENTIAL_RERANK


# Qwen3-Embedding standard instruction prefix for asymmetric retrieval.
# Queries are wrapped with this prefix before encoding; docs were embedded
# as-is offline by the challenge dataset provider.
QWEN3_MUSIC_INSTRUCT = (
    "Instruct: Given a conversational music request, "
    "retrieve tracks most relevant to the user's intent.\nQuery: "
)


def _wrrf_union_v1_specs(extra_config: dict, corpus_types: list[str] | None = None) -> list[dict]:
    """Sub-retriever specs for wrrf_union_v1. HyDE is opt-in via use_hyde.

    Kept as a standalone helper so the channel-gating logic is unit-testable
    without constructing any retriever (the hyde_qwen3 build loads Qwen2.5-7B).
    """
    ec = extra_config or {}
    corpus_types = corpus_types or ["track_name", "artist_name", "album_name"]
    specs = [
        {"type": "bm25",
         "corpus_types": ["track_name", "artist_name", "album_name",
                          "release_date", "tag_list"],
         "topk_internal": 100, "weight": float(ec.get("w_bm25", 1.0))},
        {"type": "dense_metadata_qwen3", "corpus_types": corpus_types,
         "topk_internal": 100, "weight": float(ec.get("w_qwen", 0.7))},
        {"type": "same_artist", "topk_internal": 100,
         "weight": float(ec.get("w_artist", 1.0))},
    ]
    if ec.get("use_hyde"):
        specs.append({
            "type": "hyde_qwen3", "topk_internal": 100,
            "weight": float(ec.get("w_hyde", 1.0)),
            "extra_config": {
                "hyde_model": ec.get("hyde_model", "Qwen/Qwen2.5-7B-Instruct"),
                "n_docs": int(ec.get("hyde_n_docs", 3)),
                "topk_per_doc": int(ec.get("hyde_topk_per_doc", 100)),
                "batch_size": int(ec.get("hyde_batch_size", 16)),
            },
        })
    return specs


def load_retrieval_module(
        retrieval_type: str,
        dataset_name: str,
        track_split_types: list[str],
        corpus_types: list[str] = ["track_name", "artist_name", "album_name"],
        cache_dir: str = "./cache",
        extra_config: dict | None = None,
    ):
    extra_config = extra_config or {}
    if retrieval_type == "bm25":
        return BM25_MODEL(dataset_name, track_split_types, corpus_types, cache_dir)
    elif retrieval_type == "bert":
        return BERT_MODEL(dataset_name, track_split_types, corpus_types, cache_dir)
    # Legacy key preserved so v3's config still loads; defaults = attributes col, no instruct.
    elif retrieval_type == "dense_precomputed":
        return DENSE_PRECOMPUTED(dataset_name, track_split_types, corpus_types, cache_dir)
    # Parameterized dense variants — each factory key pins (embed_col, instruct).
    elif retrieval_type == "dense_attributes_qwen3":
        return DENSE_PRECOMPUTED(
            dataset_name, track_split_types, corpus_types, cache_dir,
            embed_col="attributes-qwen3_embedding_0.6b",
            instruct=None, instruct_label="raw",
        )
    elif retrieval_type == "dense_metadata_qwen3":
        return DENSE_PRECOMPUTED(
            dataset_name, track_split_types, corpus_types, cache_dir,
            embed_col="metadata-qwen3_embedding_0.6b",
            instruct=None, instruct_label="raw",
        )
    elif retrieval_type == "dense_metadata_qwen3_instruct":
        return DENSE_PRECOMPUTED(
            dataset_name, track_split_types, corpus_types, cache_dir,
            embed_col="metadata-qwen3_embedding_0.6b",
            instruct=QWEN3_MUSIC_INSTRUCT, instruct_label="instruct-music-v1",
        )
    elif retrieval_type == "dense_attributes_qwen3_instruct":
        return DENSE_PRECOMPUTED(
            dataset_name, track_split_types, corpus_types, cache_dir,
            embed_col="attributes-qwen3_embedding_0.6b",
            instruct=QWEN3_MUSIC_INSTRUCT, instruct_label="instruct-music-v1",
        )
    elif retrieval_type == "dense_lyrics_qwen3_instruct":
        return DENSE_PRECOMPUTED(
            dataset_name, track_split_types, corpus_types, cache_dir,
            embed_col="lyrics-qwen3_embedding_0.6b",
            instruct=QWEN3_MUSIC_INSTRUCT, instruct_label="instruct-music-v1",
        )
    # Phase 1 Bundle A — BGE-M3 hybrid embedder over track metadata.
    # Embeddings precomputed via `scripts/embed_catalog.py` and cached locally.
    # BGE-M3 is symmetric (no instruct prefix) and uses model-default pooling
    # (sentence-transformers handles it).
    elif retrieval_type == "dense_metadata_bge_m3_local":
        return DENSE_LOCAL(
            dataset_name, track_split_types, corpus_types, cache_dir,
            model_name="BAAI/bge-m3",
            embed_label="bge-m3-metadata",
        )
    # nDCG-stretch Stage A — fine-tuned BGE-M3 (merged Hub repo). Reuses
    # DENSE_LOCAL; the embed_label distinguishes its precomputed catalog
    # pickle from the zero-shot BGE-M3 cache.
    elif retrieval_type == "dense_metadata_bge_m3_ft_local":
        hub_repo = extra_config.get("hub_repo", "OrRim123/recsys2026-bge-m3-music-v1-merged")
        return DENSE_LOCAL(
            dataset_name, track_split_types, corpus_types, cache_dir,
            model_name=hub_repo,
            embed_label="bge-m3-music-v1-merged",
        )
    # Phase 1 Bundle B — Qwen3-Embedding-4B over track metadata.
    # Same family as the current 0.6B; tests "is the dense just under-powered?"
    # Qwen3-Embedding asymmetric — apply the music instruct prefix.
    elif retrieval_type == "dense_metadata_qwen3_4b_local":
        return DENSE_LOCAL(
            dataset_name, track_split_types, corpus_types, cache_dir,
            model_name="Qwen/Qwen3-Embedding-4B",
            embed_label="qwen3-4b-metadata",
            instruct=QWEN3_MUSIC_INSTRUCT, instruct_label="instruct-music-v1",
        )
    # RRF hybrids. Each key pins a specific sub-retriever combination so the
    # config file needs only one field, per the existing factory signature.
    elif retrieval_type == "rrf_bm25_dense_metadata_v1":
        # BM25 (v2a champion corpus) + dense_metadata_qwen3_instruct (v3.1).
        # Each sub retrieves top-60; we fuse via RRF k=60 and trim to caller's topk.
        return RRF_MODEL(
            dataset_name, track_split_types, corpus_types, cache_dir,
            sub_specs=[
                {
                    "type": "bm25",
                    "corpus_types": [
                        "track_name", "artist_name", "album_name",
                        "release_date", "tag_list",
                    ],
                    "topk_internal": 60,
                },
                {
                    "type": "dense_metadata_qwen3_instruct",
                    "corpus_types": corpus_types,
                    "topk_internal": 60,
                },
            ],
            k=60,
        )
    # v5 multi-modal RRF: add a lyrics-qwen3 stream alongside v4.2's BM25 + dense-metadata.
    # Same weight pattern: BM25 heavy, each dense stream light + small topk to limit tail noise.
    # Query embedding cache is shared across metadata/lyrics streams (same encoder + instruct).
    elif retrieval_type == "wrrf_bm25_dense_lyrics_v1":
        return RRF_MODEL(
            dataset_name, track_split_types, corpus_types, cache_dir,
            sub_specs=[
                {
                    "type": "bm25",
                    "corpus_types": [
                        "track_name", "artist_name", "album_name",
                        "release_date", "tag_list",
                    ],
                    "topk_internal": 60,
                    "weight": 1.0,
                },
                {
                    "type": "dense_metadata_qwen3_instruct",
                    "corpus_types": corpus_types,
                    "topk_internal": 20,
                    "weight": 0.4,
                },
                {
                    "type": "dense_lyrics_qwen3_instruct",
                    "corpus_types": corpus_types,
                    "topk_internal": 20,
                    "weight": 0.4,
                },
            ],
            k=60,
        )
    # Phase 1 Bundle A — same shape as wrrf_bm25_dense_lyrics_v1 but the metadata-dense
    # sub is BGE-M3 (locally embedded) instead of Qwen3-Embedding-0.6B. Single-axis
    # swap: only the metadata-dense embedder changes, BM25 + lyrics-dense identical.
    elif retrieval_type == "wrrf_bm25_dense_lyrics_bge_m3_v1":
        return RRF_MODEL(
            dataset_name, track_split_types, corpus_types, cache_dir,
            sub_specs=[
                {
                    "type": "bm25",
                    "corpus_types": [
                        "track_name", "artist_name", "album_name",
                        "release_date", "tag_list",
                    ],
                    "topk_internal": 60,
                    "weight": 1.0,
                },
                {
                    "type": "dense_metadata_bge_m3_local",
                    "corpus_types": corpus_types,
                    "topk_internal": 20,
                    "weight": 0.4,
                },
                {
                    "type": "dense_lyrics_qwen3_instruct",
                    "corpus_types": corpus_types,
                    "topk_internal": 20,
                    "weight": 0.4,
                },
            ],
            k=60,
        )
    # nDCG-stretch Stage A factory. Same shape as wrrf_bm25_dense_lyrics_bge_m3_v1
    # but the metadata-dense sub uses our FINE-TUNED merged BGE-M3 model loaded
    # by DENSE_LOCAL (model_name=<hub_repo>, embed_label='bge-m3-music-v1-merged').
    #
    # `extra_config["bge_m3_hub_repo"]` overrides the default Hub path. The key
    # MUST be at YAML top level (run_inference_blindset.py forwards the entire
    # YAML dict as extra_config); do NOT nest under `extra_config:` in the YAML.
    elif retrieval_type == "wrrf_bm25_dense_lyrics_bge_m3_ft_v1":
        bge_m3_hub = extra_config.get(
            "bge_m3_hub_repo",
            "OrRim123/recsys2026-bge-m3-music-v1-merged",
        )
        return RRF_MODEL(
            dataset_name, track_split_types, corpus_types, cache_dir,
            sub_specs=[
                {
                    "type": "bm25",
                    "corpus_types": [
                        "track_name", "artist_name", "album_name",
                        "release_date", "tag_list",
                    ],
                    "topk_internal": 60,
                    "weight": 1.0,
                },
                {
                    "type": "dense_metadata_bge_m3_ft_local",
                    "corpus_types": corpus_types,
                    "topk_internal": 20,
                    "weight": 0.6,
                    "extra_config": {"hub_repo": bge_m3_hub},
                },
                {
                    "type": "dense_lyrics_qwen3_instruct",
                    "corpus_types": corpus_types,
                    "topk_internal": 20,
                    "weight": 0.4,
                },
            ],
            k=60,
        )
    # Phase 1 Bundle B — same shape but metadata-dense is Qwen3-Embedding-4B.
    elif retrieval_type == "wrrf_bm25_dense_lyrics_qwen3_4b_v1":
        return RRF_MODEL(
            dataset_name, track_split_types, corpus_types, cache_dir,
            sub_specs=[
                {
                    "type": "bm25",
                    "corpus_types": [
                        "track_name", "artist_name", "album_name",
                        "release_date", "tag_list",
                    ],
                    "topk_internal": 60,
                    "weight": 1.0,
                },
                {
                    "type": "dense_metadata_qwen3_4b_local",
                    "corpus_types": corpus_types,
                    "topk_internal": 20,
                    "weight": 0.4,
                },
                {
                    "type": "dense_lyrics_qwen3_instruct",
                    "corpus_types": corpus_types,
                    "topk_internal": 20,
                    "weight": 0.4,
                },
            ],
            k=60,
        )
    # Weighted RRF: BM25 heavy (weight 1.0), dense light (0.4). Addresses v4's
    # symmetric RRF near-miss by giving BM25's stronger rankings more mass and
    # cutting dense's topk_internal from 60 → 20 to reduce tail pollution.
    elif retrieval_type == "wrrf_bm25_dense_metadata_v1":
        return RRF_MODEL(
            dataset_name, track_split_types, corpus_types, cache_dir,
            sub_specs=[
                {
                    "type": "bm25",
                    "corpus_types": [
                        "track_name", "artist_name", "album_name",
                        "release_date", "tag_list",
                    ],
                    "topk_internal": 60,
                    "weight": 1.0,
                },
                {
                    "type": "dense_metadata_qwen3_instruct",
                    "corpus_types": corpus_types,
                    "topk_internal": 20,
                    "weight": 0.4,
                },
            ],
            k=60,
        )
    # SID (Semantic Item ID) generator (W4). Beam-searches a fine-tuned causal
    # LM to emit SID triplets; maps them back to track_ids via a trie lookup.
    elif retrieval_type == "sid_generator":
        from mcrs.retrieval_modules.sid_generator import SID_GENERATOR
        sid_hub_repo = extra_config.get(
            "sid_hub_repo",
            # W1 v2 default (2026-05-17): commit bea989b's cosine-Sinkhorn-at-all-levels
            # codebook produced this -v2-merged. v1-merged still exists on Hub for
            # comparison runs; override via `extra_config.sid_hub_repo` in YAML.
            "OrRim123/recsys2026-sid-generator-qwen15b-v2-merged",
        )
        return SID_GENERATOR(
            hub_repo=sid_hub_repo,
            sid_lookup_path=Path(cache_dir) / "sid" / "track_to_sid.parquet",
            device="cuda",
            num_beams=20,
            max_prompt_len=1024,
            cap_per_bucket=1,
        )
    # cf-bpr user x item affinity retriever (exp 025 A2).
    # Query-independent; scores per-user against all tracks via cosine on
    # precomputed cf-bpr embeddings. Warm users only; cold users get empty list
    # (RRF fusion falls back to other branches).
    elif retrieval_type == "cf_bpr":
        from .cf_bpr import CF_BPR
        return CF_BPR(dataset_name, track_split_types, corpus_types, cache_dir)
    # wRRF with cf-bpr as a 4th branch (on top of wrrf_bm25_dense_lyrics_v1).
    # cf-bpr weight chosen lower (0.25) than dense (0.4) because only ~43% of
    # Blind-A users are warm — we don't want this sub to dominate for the 57%
    # cold users who will see an empty ranking from it.
    elif retrieval_type == "wrrf_bm25_dense_lyrics_cfbpr_v1":
        return RRF_MODEL(
            dataset_name, track_split_types, corpus_types, cache_dir,
            sub_specs=[
                {
                    "type": "bm25",
                    "corpus_types": [
                        "track_name", "artist_name", "album_name",
                        "release_date", "tag_list",
                    ],
                    "topk_internal": 60,
                    "weight": 1.0,
                },
                {
                    "type": "dense_metadata_qwen3_instruct",
                    "corpus_types": corpus_types,
                    "topk_internal": 20,
                    "weight": 0.4,
                },
                {
                    "type": "dense_lyrics_qwen3_instruct",
                    "corpus_types": corpus_types,
                    "topk_internal": 20,
                    "weight": 0.4,
                },
                {
                    "type": "cf_bpr",
                    "corpus_types": corpus_types,
                    "topk_internal": 20,
                    "weight": 0.25,
                },
            ],
            k=60,
        )
    # W4 ensemble: existing 3-stream wRRF (BM25 + dense_metadata + dense_lyrics)
    # plus the new SID generator as a 4th stream. Initial SID weight = 0.5
    # (between BM25=1.0 and dense=0.4); tuned in W5 via extra_config.sid_stream_weight.
    elif retrieval_type == "wrrf_bm25_dense_sid_v1":
        sid_weight = float(extra_config.get("sid_stream_weight", 0.5))
        return RRF_MODEL(
            dataset_name, track_split_types, corpus_types, cache_dir,
            sub_specs=[
                {
                    "type": "bm25",
                    "corpus_types": [
                        "track_name", "artist_name", "album_name",
                        "release_date", "tag_list",
                    ],
                    "topk_internal": 60,
                    "weight": 1.0,
                },
                {
                    "type": "dense_metadata_qwen3_instruct",
                    "corpus_types": corpus_types,
                    "topk_internal": 20,
                    "weight": 0.4,
                },
                {
                    "type": "dense_lyrics_qwen3_instruct",
                    "corpus_types": corpus_types,
                    "topk_internal": 20,
                    "weight": 0.4,
                },
                {
                    "type": "sid_generator",
                    "corpus_types": corpus_types,
                    "topk_internal": 20,
                    "weight": sid_weight,
                    # Forward parent's sid_hub_repo (if set) to the SID factory.
                    # Without this the sub-retriever would always use the default,
                    # making the YAML override ineffective for wRRF ensembles.
                    "extra_config": {
                        "sid_hub_repo": extra_config.get("sid_hub_repo")
                    } if extra_config.get("sid_hub_repo") else {},
                },
            ],
            k=60,
        )
    # ------------------------------------------------------------------
    # Multi-modal Stage A (fresh-model branch). Plugs the trained
    # MultiModalBiEncoder into production via DENSE_MULTIMODAL_LOCAL.
    # `extra_config` keys:
    #   - model_dir            : filesystem path OR Hub repo of the saved
    #                            MultiModalBiEncoder (required).
    #   - embed_label          : subdir name under {cache_dir}/dense_local/
    #                            <safe>/ where the catalog pickle lives.
    #                            Default 'mm-v1'.
    #   - backbone_override    : optional override for the base model passed
    #                            to MultiModalBiEncoder.from_pretrained.
    #   - multimodal_artifacts : path to the precompute cache (tag_vocab,
    #                            user_cf*, etc.) — REQUIRED at inference time
    #                            for user_cf lookup + cold-user fallback.
    #   - query_max_len        : tokenizer max_length for query side
    #                            (default 384, matches training PASSAGE config).
    # All keys live at the YAML TOP LEVEL when the outer config invokes
    # this factory directly; the wrrf_bm25_multimodal_v1 composite below
    # forwards them via sub_spec extra_config.
    # ------------------------------------------------------------------
    elif retrieval_type == "dense_multimodal_local":
        model_dir = extra_config.get("model_dir") or extra_config.get("hub_repo")
        if not model_dir:
            raise ValueError(
                "dense_multimodal_local requires extra_config['model_dir'] "
                "(filesystem path OR Hub repo of the saved MultiModalBiEncoder)."
            )
        mm_artifacts = extra_config.get("multimodal_artifacts")
        if not mm_artifacts:
            raise ValueError(
                "dense_multimodal_local requires extra_config['multimodal_artifacts'] "
                "(path to scripts/precompute_multimodal_artifacts.py output)."
            )
        return DENSE_MULTIMODAL_LOCAL(
            dataset_name, track_split_types, corpus_types, cache_dir,
            model_dir=model_dir,
            embed_label=extra_config.get("embed_label", "mm-v1"),
            backbone_override=extra_config.get("backbone_override"),
            multimodal_artifacts=mm_artifacts,
            query_max_len=int(extra_config.get("query_max_len", 384)),
        )
    # nDCG-stretch Stage A multi-modal composite. BM25 (proven lexical
    # complement) + DENSE_MULTIMODAL_LOCAL. The multi-modal tower
    # SUBSUMES dense_metadata + dense_lyrics + cf_bpr — those streams
    # become redundant because text, lyrics, CF, audio, and tag/year are
    # all fused inside the encoder.
    #
    # topk_internal bumped to 100 (vs the 60 used by legacy wRRFs) so the
    # candidate pool is rich enough to feed Stage B (multi-modal cross-
    # encoder reranker) when that lands in Phase 6+.
    elif retrieval_type == "wrrf_bm25_multimodal_v1":
        mm_model_dir = extra_config.get("mm_model_dir") or extra_config.get("hub_repo")
        if not mm_model_dir:
            raise ValueError(
                "wrrf_bm25_multimodal_v1 requires extra_config['mm_model_dir'] "
                "(or legacy alias 'hub_repo') — the multi-modal model path."
            )
        mm_artifacts = extra_config.get("multimodal_artifacts")
        if not mm_artifacts:
            raise ValueError(
                "wrrf_bm25_multimodal_v1 requires extra_config['multimodal_artifacts']."
            )
        return RRF_MODEL(
            dataset_name, track_split_types, corpus_types, cache_dir,
            sub_specs=[
                {
                    "type": "bm25",
                    "corpus_types": [
                        "track_name", "artist_name", "album_name",
                        "release_date", "tag_list",
                    ],
                    "topk_internal": 100,
                    "weight": 1.0,
                },
                {
                    "type": "dense_multimodal_local",
                    "corpus_types": corpus_types,
                    "topk_internal": 100,
                    "weight": 0.7,
                    "extra_config": {
                        "model_dir": mm_model_dir,
                        "embed_label": extra_config.get("embed_label", "mm-v1"),
                        "backbone_override": extra_config.get("backbone_override"),
                        "multimodal_artifacts": mm_artifacts,
                        "query_max_len": int(extra_config.get("query_max_len", 384)),
                    },
                },
            ],
            k=60,
        )
    elif retrieval_type == "same_artist":
        from .same_artist import SameArtistRetriever
        return SameArtistRetriever(dataset_name, track_split_types, corpus_types, cache_dir)
    elif retrieval_type == "session_cf":
        from .session_cf import SessionCFRetriever
        return SessionCFRetriever(dataset_name, track_split_types, corpus_types, cache_dir)
    elif retrieval_type == "hyde_qwen3":
        # HyDE recall channel: an LLM writes pseudo-track descriptions, the qwen3
        # dense retriever matches them in the catalog embedding space, and RRF
        # fuses the per-doc lists. NOTE: LLAMA_MODEL is the generic HF causal-LM
        # wrapper (name is legacy) — it loads whatever `hyde_model` names, i.e.
        # Qwen2.5-7B-Instruct here, NOT Llama.
        import os
        from ..lm_modules.llama import LLAMA_MODEL
        from ..query_rewriters.hyde import HydeGenerator
        from .hyde_qwen3 import HydeQwen3Retriever
        # hyde_model accepts an HF repo id OR a local path (e.g. a Drive-cached
        # checkpoint) — transformers' from_pretrained handles both. sdpa attention
        # speeds up batched decode and needs no build (unlike flash-attn).
        lm = LLAMA_MODEL(
            model_name=extra_config.get("hyde_model", "Qwen/Qwen2.5-7B-Instruct"),
            attn_implementation="sdpa")
        prompt_path = os.path.join(
            os.path.dirname(__file__), "..", "system_prompts", "hyde_pseudo_track.txt")
        generator = HydeGenerator(
            lm, prompt_path, cache_dir=cache_dir,
            n_docs=int(extra_config.get("n_docs", 3)),
            batch_size=int(extra_config.get("batch_size", 16)))
        inner = load_retrieval_module(
            "dense_metadata_qwen3", dataset_name, track_split_types,
            corpus_types, cache_dir, extra_config={})
        return HydeQwen3Retriever(
            generator, inner, topk_per_doc=int(extra_config.get("topk_per_doc", 100)))
    elif retrieval_type == "wrrf_union_v1":
        # 3-channel recall union: lexical + frozen-Qwen semantic + session
        # artist continuity. session_cf was DROPPED after the G1 ablation
        # (2026-05-25): its marginal recall was +0.002 @100 / +0.008 @20 over
        # the union-without-it — below the 0.01 keep rule (user-CF is near-inert
        # here). This removes only the recall channel; cfbpr_score stays as an
        # LGBM rerank feature pending G2 importances. Union recall after drop:
        # ~0.478 @100, ~0.346 @20 (vs 4ch 0.480 / 0.354).
        return RRF_MODEL(
            dataset_name, track_split_types, corpus_types, cache_dir,
            sub_specs=_wrrf_union_v1_specs(extra_config, corpus_types),
            k=60,
        )
    # Sequential retrieve-then-rerank.
    elif retrieval_type == "bm25_then_dense_rerank_v1":
        # BM25 (v2a 5-field corpus) first-stage top-100 → dense metadata+instruct rerank.
        return SEQUENTIAL_RERANK(
            dataset_name, track_split_types, corpus_types, cache_dir,
            first_stage_spec={
                "type": "bm25",
                "corpus_types": [
                    "track_name", "artist_name", "album_name",
                    "release_date", "tag_list",
                ],
                "topk_internal": 100,
            },
            reranker_spec={
                "embed_col": "metadata-qwen3_embedding_0.6b",
                "instruct": QWEN3_MUSIC_INSTRUCT,
                "instruct_label": "instruct-music-v1",
            },
        )
    else:
        raise ValueError(f"Unsupported retrieval type: {retrieval_type}")
