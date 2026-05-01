"""StateTracker — RA-Rec-style structured user-state extraction.

Per RecSys_Challenge_Plan §A1. For every (session, turn), extract a JSON-like
user_state block with 6 keys (mood, intent, energy, sonic_pref, era_pref,
familiarity) from the user query + prior dialog. The extracted state then:
  - feeds A2 CMQR (rewrites injected with state values)
  - is cached and re-used at later turns of the same session
  - is included in the response-generation prompt envelope

Parse-failure policy (plan §A1):
  1. retry once with a more directive nudge prompt
  2. fall back to cached prior-turn state from the same session
  3. if no cached state, return None (caller's policy: at training-time drop
     the rollout from the batch; at inference-time fall back to last_user
     query without state injection — never zero-reward the responder for a
     retrieval-stack failure)

Model: Qwen-2.5-1.5B-Instruct via the existing LLAMA_MODEL wrapper. Already
loaded in the exp 021 pipeline, so no extra weight load.

Caching: states cached at `{cache_dir}/state/{session_id}__{turn_number}.json`
with `ensure_ascii=False`. Idempotent — call site can call `.extract()` freely
and it'll hit cache after the first pass.

Cache-path warning: `run_inference_devset.py:71` runs `os.system("rm -rf cache")`
at startup — so a default `cache_dir="./cache"` would be wiped on every run.
For prod use, point `cache_dir` at `experiments/cache/` (the convention
established by cf-bpr at `music-crs-baselines/experiments/cache/cf_bpr/`).
The smoke harness uses a separate `cache/smoke_state/` path that's safe
because the smoke runs standalone (not via `run_inference_devset.py`).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Optional

# Keys must match music-crs-baselines/mcrs/system_prompts/state_extraction.txt
# AND scripts/reward_fns.py ALLOWED_STATE_KEYS.
ALLOWED_KEYS = ("mood", "intent", "energy", "sonic_pref", "era_pref", "familiarity")

ENVELOPE = re.compile(r"<user_state>(.*?)</user_state>", re.DOTALL | re.IGNORECASE)
# Permissive fallback: Qwen-1.5B at greedy frequently drops the OPENING tag and
# emits e.g. "user state:\nmood: ...\n...\n</user_state>". Match on the closing
# tag alone and take everything before it as the body. Also accept "user state:"
# (or "user_state:") as a soft opening.
PERMISSIVE_CLOSE = re.compile(r"</user_state>", re.IGNORECASE)
PERMISSIVE_OPEN = re.compile(r"user[_ ]state\s*:?", re.IGNORECASE)
# Output-prefix priming: forcing the model to start its generation at this
# string guarantees the opening tag.
PREFIX_PRIME = "<user_state>\n"
RETRY_PREFIX = (
    "Your previous output did not contain a valid <user_state> block. "
    "Output ONLY a single <user_state>...</user_state> block, nothing else.\n\n"
)


def _parse_body(body: str) -> dict[str, str]:
    """Extract {key: value} pairs for allowed keys from a body of text."""
    state: dict[str, str] = {}
    for line in body.splitlines():
        line = line.strip().rstrip(",")
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        if key in ALLOWED_KEYS:
            state[key] = value.strip().rstrip(",")
    return state


def parse_user_state(text: str) -> Optional[dict[str, str]]:
    """Parse a <user_state> body out of model output, tolerantly.

    Strategy (in order of strictness):
      1. Strict: full `<user_state>...</user_state>` envelope.
      2. Closing-tag-only: take everything BEFORE `</user_state>` as the body
         (Qwen-1.5B greedy frequently drops the opening tag).
      3. Bare key:value lines starting with `user state:` (no tags at all).

    Returns a dict with at least one allowed key, or None if parsing fails.
    """
    if not text:
        return None

    # 1. Strict envelope
    m = ENVELOPE.search(text)
    if m:
        state = _parse_body(m.group(1))
        if state:
            return state

    # 2. Closing tag only — common Qwen-1.5B failure mode
    close_match = PERMISSIVE_CLOSE.search(text)
    if close_match:
        body = text[:close_match.start()]
        # Trim a leading "user state:" or similar header if present.
        opener = PERMISSIVE_OPEN.search(body)
        if opener and opener.end() < len(body):
            body = body[opener.end():]
        state = _parse_body(body)
        if state:
            return state

    # 3. No tags at all but starts with "user state:" header
    opener = PERMISSIVE_OPEN.search(text)
    if opener:
        body = text[opener.end():]
        # Stop at obvious noise markers (model started hallucinating a dialog)
        for stop in ("\nuser:", "\nassistant:", "\n\n"):
            idx = body.find(stop)
            if idx > 0:
                body = body[:idx]
                break
        state = _parse_body(body)
        if state:
            return state

    return None


def build_extraction_prompt(user_query: str, history_text: str) -> list[dict]:
    """Build the chat-history payload that LLAMA_MODEL.response_generation expects.

    The system prompt comes from state_extraction.txt; this function provides
    the user-side context.
    """
    history_text = (history_text or "").strip() or "(none)"
    user_query = (user_query or "").strip()
    user_msg = f"user query: \"{user_query}\"\nprior dialog:\n{history_text}"
    return [{"role": "user", "content": user_msg}]


class StateTracker:
    """Per-turn user-state extractor with parse-failure fallback + cache.

    Args:
        lm: an LM wrapper exposing `response_generation(sys_prompt, chat_history,
            recommend_item, max_new_tokens)`. Either LLAMA_MODEL or VLLM_MODEL.
        prompt_path: path to state_extraction.txt (or any system prompt that asks
            the model to emit a <user_state> block).
        cache_dir: parent directory; states are saved to `{cache_dir}/state/`.
        max_new_tokens: cap for the state-extraction generation. 96 is plenty
            (6 keys × ~10 tokens average + envelope tags ≈ 80 tokens).
        previous_states: optional pre-loaded mapping `{session_id: last_state}`
            used as the second-step fallback if a session's first turn fails.
    """

    def __init__(
        self,
        lm,
        prompt_path: str | Path,
        cache_dir: str | Path = "./cache",
        max_new_tokens: int = 96,
        debug_failures: bool = False,
    ):
        self.lm = lm
        self.prompt = Path(prompt_path).read_text(encoding="utf-8")
        self.cache_root = Path(cache_dir) / "state"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.max_new_tokens = max_new_tokens
        self.debug_failures = debug_failures
        # Detect backend once: HF (LLAMA_MODEL exposes .lm + .tokenizer)
        # vs vLLM (VLLM_MODEL exposes .lm = vllm.LLM, .tokenizer, no .device).
        # The HF path uses model.generate(); the vLLM path uses lm.lm.generate()
        # with vllm.SamplingParams. Detected by walking attributes — no
        # imports needed (so the module stays MPS-importable without vllm).
        self._backend = self._detect_backend()
        # In-memory cache to avoid repeated disk hits within one process.
        self._memcache: dict[tuple[str, int], dict[str, str]] = {}
        # Per-session "last successful state" for the parse-failure fallback.
        self._session_last_state: dict[str, dict[str, str]] = {}
        # Captured raw failure outputs (for diagnosis when debug_failures=True).
        self.failures: list[dict] = []
        # Counters for instrumentation.
        self.stats = {
            "calls": 0,
            "cache_hits": 0,
            "ok_first_try": 0,
            "ok_after_retry": 0,
            "fallback_to_prior": 0,
            "drop": 0,
        }

    def _detect_backend(self) -> str:
        """Return 'hf' (LLAMA_MODEL) or 'vllm' (VLLM_MODEL).

        Detection by attribute presence:
          - HF wrapper has .device (str) AND .lm exposes .generate (HF causal LM).
          - vLLM wrapper has .lm exposing .chat / .generate via vllm.LLM (no .device).
        """
        cls_name = type(self.lm).__name__.lower()
        if "vllm" in cls_name:
            return "vllm"
        if hasattr(self.lm, "device") and hasattr(self.lm, "lm") and hasattr(self.lm.lm, "generate"):
            return "hf"
        # Fallback: prefer hf if it walks like one, else vllm.
        if hasattr(self.lm, "device"):
            return "hf"
        return "vllm"

    # -------- Caching -------------------------------------------------------

    def _cache_path(self, session_id: str, turn_number: int) -> Path:
        # Sanitise session_id for filesystem (UUIDs are safe; defensive anyway).
        safe = re.sub(r"[^A-Za-z0-9_\-]", "_", str(session_id))
        return self.cache_root / f"{safe}__{int(turn_number):d}.json"

    def _load_cached(self, session_id: str, turn_number: int) -> Optional[dict]:
        key = (session_id, int(turn_number))
        if key in self._memcache:
            return self._memcache[key]
        cp = self._cache_path(session_id, turn_number)
        if cp.exists():
            try:
                with cp.open("r", encoding="utf-8") as f:
                    state = json.load(f)
                if isinstance(state, dict) and state:
                    self._memcache[key] = state
                    self._session_last_state[session_id] = state
                    return state
            except (OSError, json.JSONDecodeError):
                pass
        return None

    def _save_cache(self, session_id: str, turn_number: int, state: dict) -> None:
        cp = self._cache_path(session_id, turn_number)
        with cp.open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        self._memcache[(session_id, int(turn_number))] = state
        self._session_last_state[session_id] = state

    # -------- Generation ----------------------------------------------------

    def _generate(
        self,
        system_prompt: str,
        user_query: str,
        history_text: str,
        prefix_prime: bool = True,
    ) -> str:
        """Direct chat completion: system → user → assistant(<prefix>).

        Bypasses LLAMA_MODEL.response_generation because that wrapper appends
        a fake empty `assistant` turn before the generation prompt — fine for
        response generation (where the assistant content is the recommended
        item), but malforms the prompt for state extraction (no track exists).

        prefix_prime=True forces the assistant's output to start with
        `<user_state>\\n`. This guarantees the opening tag (Qwen-1.5B greedy
        was dropping it and emitting `user state:\\n...` instead). Returned
        text re-prepends the prefix so parse_user_state sees the full envelope.

        Backends:
          - HF (LLAMA_MODEL): uses model.generate() on self.lm.device.
          - vLLM (VLLM_MODEL): uses vllm.LLM.generate() with SamplingParams.
        """
        chat_history = build_extraction_prompt(user_query, history_text)
        messages = [{"role": "system", "content": system_prompt}, *chat_history]
        tokenizer = self.lm.tokenizer

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        if prefix_prime:
            prompt_text = prompt_text + PREFIX_PRIME

        if self._backend == "vllm":
            decoded = self._generate_vllm(prompt_text)
        else:
            decoded = self._generate_hf(prompt_text)

        if prefix_prime:
            # Re-prepend so parse_user_state sees the full envelope.
            return PREFIX_PRIME + decoded
        return decoded

    def _generate_hf(self, prompt_text: str) -> str:
        """HF causal LM path: tokenize → model.generate → decode new tokens."""
        tokenizer = self.lm.tokenizer
        model = self.lm.lm  # the underlying HF causal LM
        device = self.lm.device

        token_inputs = tokenizer(prompt_text, return_tensors="pt")
        input_ids = token_inputs.input_ids.to(device)
        attention_mask = token_inputs.attention_mask.to(device)

        # Greedy by default (deterministic). The retry-prompt nudge handles
        # parse failures more reliably than temperature variation on small models.
        import torch
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen = outputs[:, input_ids.shape[1]:]
        return tokenizer.batch_decode(gen, skip_special_tokens=True)[0]

    def _generate_vllm(self, prompt_text: str) -> str:
        """vLLM path: pass the rendered prompt string + SamplingParams to vllm.LLM.generate.

        Why .generate (not .chat): we already rendered the chat template above
        AND prefix-primed the assistant's output with `<user_state>\\n`. vLLM's
        .chat() would re-render the template, dropping our prime. .generate()
        accepts the literal prompt verbatim — preserves both the chat scaffold
        and the prefix prime.
        """
        from vllm import SamplingParams  # vLLM-only import — kept lazy

        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=self.max_new_tokens,
        )
        outputs = self.lm.lm.generate([prompt_text], sampling, use_tqdm=False)
        # vLLM returns list[RequestOutput]; one prompt → one entry, one greedy completion.
        return outputs[0].outputs[0].text

    @staticmethod
    def was_fallback(state: Optional[dict]) -> bool:
        """True iff `state` was produced by the prior-turn fallback (Gap 10).

        Callers should use this rather than peeking at `_fallback` directly.
        Returns False for None (drop) and for fresh extractions.
        """
        return bool(state and state.get("_fallback") == "prior_turn_state")

    def extract(
        self,
        session_id: str,
        turn_number: int,
        user_query: str,
        history_text: str = "",
    ) -> Optional[dict[str, str]]:
        """Return state dict or None. Side-effects: cache + stats update.

        Return-shape contract (Gap 10):
          - `dict` with the 6 allowed keys → fresh extraction (or cached fresh).
          - `dict` containing `_fallback: "prior_turn_state"` → parse failed for
            this turn; we returned the session's last successful state. Callers
            can detect this via `StateTracker.was_fallback(state)` and downgrade
            confidence (e.g. CMQR weight, responder envelope flag).
          - `None` → parse failed AND no prior state available; caller's policy
            (training: drop; inference: fall back to no state injection).
        """
        self.stats["calls"] += 1

        # 1. Cache hit?
        cached = self._load_cached(session_id, turn_number)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return cached

        # 2. First try at greedy temp.
        text = self._generate(self.prompt, user_query, history_text)
        state = parse_user_state(text)
        if state:
            self.stats["ok_first_try"] += 1
            self._save_cache(session_id, turn_number, state)
            return state

        # 3. Retry with a directive nudge (LM doesn't expose temp; we use a
        #    stronger prompt instead — equivalent effect for small models).
        text2 = self._generate(RETRY_PREFIX + self.prompt, user_query, history_text)
        state = parse_user_state(text2)
        if state:
            self.stats["ok_after_retry"] += 1
            self._save_cache(session_id, turn_number, state)
            return state

        # 4. Diagnosis: capture raw outputs when both attempts fail.
        if self.debug_failures:
            self.failures.append({
                "session_id": session_id,
                "turn_number": int(turn_number),
                "user_query": (user_query or "")[:120],
                "raw_output_attempt1": (text or "")[:300],
                "raw_output_attempt2": (text2 or "")[:300],
            })

        # 5. Fallback to the session's last successful state.
        prior = self._session_last_state.get(session_id)
        if prior:
            self.stats["fallback_to_prior"] += 1
            # Cache the prior state under this turn so subsequent calls hit cache,
            # but DON'T promote it to "last" (we don't want stale state to chain).
            cp = self._cache_path(session_id, turn_number)
            payload = {**prior, "_fallback": "prior_turn_state"}
            with cp.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self._memcache[(session_id, int(turn_number))] = payload
            return payload

        # 6. Drop — caller decides what to do.
        self.stats["drop"] += 1
        return None

    # -------- Stats helpers -------------------------------------------------

    def parse_validity(self) -> float:
        n = self.stats["calls"] - self.stats["cache_hits"]
        if n == 0:
            return 1.0
        ok = self.stats["ok_first_try"] + self.stats["ok_after_retry"]
        return ok / n

    def report(self) -> dict:
        return {
            **self.stats,
            "parse_validity_excl_cache": round(self.parse_validity(), 4),
        }
