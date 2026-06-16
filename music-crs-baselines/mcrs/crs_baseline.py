import os
import re
import torch
from typing import Optional, Any, List, Dict
from mcrs.db_item import MusicCatalogDB
from mcrs.db_user import UserProfileDB
from mcrs.lm_modules import load_lm_module
from mcrs.retrieval_modules import load_retrieval_module
from mcrs.retrieval_modules.sasrec_model import build_user_dialog, build_sasrec_context
from mcrs.rerankers import load_reranker_module
from mcrs.response_rerankers import load_response_reranker_module


_RESPONSE_TAG_RE = re.compile(
    r"<response>\s*(.*?)\s*</response>", re.DOTALL | re.IGNORECASE
)
_USER_STATE_BLOCK_RE = re.compile(
    r"<user_state>.*?</user_state>", re.DOTALL | re.IGNORECASE
)


def extract_cot_response(raw: str) -> str:
    """Strip the CoT envelope from a CoT-prompt LM output.

    The CoT prompts (response_generation_cot_*) instruct the model to emit
    a structured <user_state>...</user_state> block followed by the
    user-facing <response>...</response>. Only the latter goes to Gemini.

    Strategy:
      1. Prefer the contents of <response>...</response> if both tags exist.
      2. Fallback: drop any <user_state>...</user_state> block and return
         the rest, stripped. Handles partial generations where the model
         produced the user_state but ran out of tokens before closing
         <response>, OR forgot the response tags entirely.
      3. Final fallback: the original string. Never returns empty unless
         the model itself produced empty output.
    """
    if not raw:
        return raw
    m = _RESPONSE_TAG_RE.search(raw)
    if m:
        return m.group(1).strip()
    cleaned = _USER_STATE_BLOCK_RE.sub("", raw).strip()
    # Drop a stray opening <response> tag if the model never closed it.
    cleaned = re.sub(r"</?response>", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned or raw.strip()


def build_retrieval_query(
    session_memory: list[dict],
    mode: str = "raw",
    goal_text: Optional[str] = None,
    user_profile: Optional[dict] = None,
    max_history_turns: int = 6,
    state: Optional[dict] = None,
) -> str:
    """Format the conversation history for the retriever.

    Modes:
      'raw'              : 021-champion behaviour. Newline-joined "role: content"
                           across ALL turns including assistant text + expanded
                           music-turn metadata. Heavy noise on multi-turn data.
      'last_user'        : Just the last user turn's content (no role prefix).
                           Strips assistant text, prior music metadata, and the
                           role labels. Best for dense encoder (avoids 512-token
                           truncation hiding the actual query) and for BM25
                           (no spurious 'user'/'music'/'assistant' tokens).
      'last_user_with_goal': last_user + ' || goal: <listener_goal>' when
                           goal_text is provided. Adds light context useful
                           when the user query is short.
      'compact_colbert'  : The 512-ceiling concession for ColBERT. High-signal
                           slice only: 'goal: <g>' + 'culture: <c>' (labelled,
                           FIRST for truncation-safety) + the bare most-recent
                           user turn (no role prefix). goal/culture omitted when
                           absent -> degrades to bare last_user. No demographics.
                           The SHARED builder for serve + the ColBERT train-data
                           builder so the query is byte-identical between training
                           and inference. (nb74 DEV harness NOT yet wired to this —
                           DEV ColBERT recall is not a valid gate for this mode until
                           it is.)
      'raw_with_goal'    : 'raw' (full newline-joined history) plus a trailing
                           '\\ngoal: <listener_goal>' line when goal_text is
                           provided. Matches the nb74 cell-4 dev recall harness
                           so the offline query can be aligned with serve. With
                           no goal it is byte-identical to 'raw'.
      'bge_m3_structured': Structured 4-block format (plan §Task 4 step 3) used
                           for the BGE-M3-FT fine-tune + matching inference. Layout:
                             [USER]: age=<X> country=<Y> gender=<Z>
                             [GOAL]: <listener_goal>
                             [HISTORY]: U: <content> | A: <content> | ...
                             [QUERY]: <last user content>
                           History is the last `max_history_turns` BEFORE the
                           final user turn (which becomes [QUERY]). Missing
                           fields render as 'unknown' or empty rather than
                           dropping the block — keeps the format positionally
                           stable so the encoder can learn the sections.

    Returns the formatted query string.
    """
    if mode == "raw":
        return "\n".join(
            f"{t.get('role','')}: {t.get('content','')}" for t in session_memory
        )
    if mode == "raw_with_goal":
        base = "\n".join(
            f"{t.get('role','')}: {t.get('content','')}" for t in session_memory
        )
        gt = (goal_text or "").strip()
        return f"{base}\ngoal: {gt}" if gt else base
    if mode == "raw_enriched":
        # Tier-1 #3.2: raw_with_goal plus cold-firable taste signals
        # (preferred_musical_culture + age/country/gender). Each line is appended
        # only when its field is present, so the format degrades to raw_with_goal
        # (and to raw with no goal). These fields are legal at inference.
        base = "\n".join(
            f"{t.get('role','')}: {t.get('content','')}" for t in session_memory
        )
        lines = [base]
        gt = (goal_text or "").strip()
        if gt:
            lines.append(f"goal: {gt}")
        up = user_profile if isinstance(user_profile, dict) else {}
        culture = str(up.get("preferred_musical_culture") or "").strip()
        if culture:
            lines.append(f"culture: {culture}")
        age = up.get("age_group") or up.get("age")
        country = up.get("country_name") or up.get("country_code")
        gender = up.get("gender")
        if age or country or gender:
            lines.append(f"user: age={age or 'unknown'} "
                         f"country={country or 'unknown'} gender={gender or 'unknown'}")
        return "\n".join(lines)
    # Last user turn — find it from the END of session_memory.
    last_user = ""
    last_user_idx: Optional[int] = None
    for i in range(len(session_memory) - 1, -1, -1):
        if session_memory[i].get("role") == "user":
            last_user = str(session_memory[i].get("content", "")).strip()
            last_user_idx = i
            break
    if mode == "compact_colbert":
        # "Humble" ColBERT query (the 512-ceiling concession): feed only the
        # high-signal slice — goal + culture + the most recent user turn — instead
        # of the full dialog, which truncates to noise (75-88% of real queries
        # exceed any feasible q_len). goal/culture are labelled and placed FIRST so
        # right-truncation on a rare long user turn drops only the query tail, never
        # the durable intent/taste. The user turn carries NO role prefix (role labels
        # are retrieval noise, cf. mode="last_user"). Lines are omitted when absent,
        # so the format degrades to bare last_user. No demographics (intentional).
        # SHARED by serve + the ColBERT train-data builder so the query is byte-identical
        # between training and inference (nb74 DEV harness not yet wired — see docstring).
        lines = []
        gt = (goal_text or "").strip()
        if gt:
            lines.append(f"goal: {gt}")
        # Self-defending profile parse (I1): a caller may hand a JSON/repr STRING
        # instead of a dict; parse it so 'culture' survives regardless of caller —
        # else serve (string) drops culture while train (dict) keeps it -> a silent
        # train/serve skew on the ~75% warm sessions this query exists to help.
        up = user_profile
        if isinstance(up, str):
            import json as _json
            try:
                up = _json.loads(up)
            except Exception:
                try:
                    import ast as _ast
                    up = _ast.literal_eval(up)
                except Exception:
                    up = {}
        up = up if isinstance(up, dict) else {}
        culture = str(up.get("preferred_musical_culture") or "").strip()
        if culture:
            lines.append(f"culture: {culture}")
        lines.append(last_user)
        return "\n".join(lines)
    if mode == "bge_m3_structured":
        # User block (age/country/gender from user_profile if present).
        up = user_profile if isinstance(user_profile, dict) else {}
        age = up.get("age_group") or up.get("age") or "unknown"
        country = up.get("country_code") or up.get("country") or "unknown"
        gender = up.get("gender") or "unknown"
        # History block: turns BEFORE the final user, last N kept.
        if last_user_idx is None:
            history_turns = session_memory[-max_history_turns:]
        else:
            history_turns = session_memory[:last_user_idx][-max_history_turns:]
        history_parts = []
        for t in history_turns:
            role = t.get("role", "")
            content = str(t.get("content", "")).strip().replace("|", "/")
            if role == "user":
                history_parts.append(f"U: {content}")
            elif role == "assistant":
                history_parts.append(f"A: {content}")
            elif role == "music":
                # Music-turn content is a track_id; downstream callers may have
                # already expanded it to metadata, but keep the role marker.
                history_parts.append(f"A: {content}")
        history_block = " | ".join(history_parts)
        goal_block = (goal_text or "").strip()
        blocks = [f"[USER]: age={age} country={country} gender={gender}"]
        # [STATE] block (serve-safe StateTracker intent). Emitted ONLY when a
        # state dict is passed (bge_m3_ft v2 path) so the v1 model's format is
        # unchanged. Positionally stable: all 6 keys always rendered, 'unknown'
        # when absent (same discipline as [USER]). The raw `thought` field is
        # NEVER used (leakage: empty in Blind-A); user_state is LM-extracted at
        # both train and serve, so no schema mismatch.
        if state is not None:
            st = state if isinstance(state, dict) else {}
            _keys = ("mood", "intent", "energy", "sonic_pref", "era_pref", "familiarity")
            state_block = " ".join(
                f"{k}={(str(st.get(k)).strip() or 'unknown').replace('|', '/')}"
                if st.get(k) not in (None, "") else f"{k}=unknown"
                for k in _keys
            )
            blocks.append(f"[STATE]: {state_block}")
        blocks.append(f"[GOAL]: {goal_block}")
        blocks.append(f"[HISTORY]: {history_block}")
        blocks.append(f"[QUERY]: {last_user}")
        return "\n".join(blocks)
    if not last_user:
        # Fallback to raw if there's no user turn (shouldn't happen on inference).
        return "\n".join(
            f"{t.get('role','')}: {t.get('content','')}" for t in session_memory
        )
    if mode == "last_user_with_goal" and goal_text:
        return f"{last_user} || goal: {goal_text}"
    return last_user

def build_sasrec_extra_features(
    per_sub: list,
    labels: list,
    batch_retrieval_items: list,
    sentinel: int = 10000,
) -> list | None:
    """Per-candidate side features aligned to batch_retrieval_items, computed
    from the union's per-sub rankings:
      'sasrec_rank'    : 1-indexed position in the SASRec sub (sentinel if absent),
                         only when a 'sasrec_seq' sub is present.
      'n_channels_hit' : how many union channels surfaced this candidate (>=1),
                         always emitted (Lever 2 cross-channel agreement).
    Returns None only if there are no per-sub rankings at all (nothing to add).
    MUST mirror WRRFRunner.run in scripts/build_lgbm_features.py (train/serve
    parity)."""
    if not labels or not per_sub:
        return None
    has_sasrec = "sasrec_seq" in labels
    sidx = labels.index("sasrec_seq") if has_sasrec else None
    out = []
    for qi, cand_list in enumerate(batch_retrieval_items):
        rankmap = ({tid: r + 1 for r, tid in enumerate(per_sub[sidx][qi])}
                   if has_sasrec else {})
        # Cross-channel agreement: count channels that surfaced each tid.
        hit_count: dict = {}
        for s in range(len(per_sub)):
            for t in per_sub[s][qi]:
                hit_count[t] = hit_count.get(t, 0) + 1
        row = []
        for tid in cand_list:
            d = {"n_channels_hit": hit_count.get(tid, 1)}
            if has_sasrec:
                d["sasrec_rank"] = rankmap.get(tid, sentinel)
            row.append(d)
        out.append(row)
    return out


class CRS_BASELINE:
    """
    Conversational Recommender System (CRS) baseline that wires together an LLM module and an item retrieval module over a music catalog and user profiles.
    Attributes:
        cache_dir: Local path for caching artifacts and indices.
        lm_type: Identifier/name for the LLM backend to load.
        retrieval_type: Retrieval backend to use (e.g., "bm25").
        item_db_name: Hugging Face dataset or DB name for item metadata.
        user_db_name: Hugging Face dataset or DB name for user metadata.
        split_types: Dataset split names to load (e.g., ["test_warm", "test_cold"]).
        corpus_types: Item fields used for retrieval (e.g., title, artist, album).
        device: Compute device for the LLM (e.g., "cuda", "cpu").
        dtype: Torch dtype used by the LLM.
        lm: Loaded LLM module used for response generation.
        retrieval: Retrieval module used to fetch candidate items.
        item_db: Item metadata database accessor.
        user_db: User profile database accessor.
        prompts_dir: Directory containing prompt templates.
        role_prompt: Loaded prompt templates keyed by role.
        session_memory: In-memory list of message dicts for the current session.
    """
    def __init__(self,
        lm_type="meta-llama/Llama-3.2-1B-Instruct",
        retrieval_type="bm25",
        item_db_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        user_db_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Metadata",
        track_split_types: list[str] = ["all_tracks"], # for test
        user_split_types: list[str] = ["all_users"],
        corpus_types: list[str] = ["track_name", "artist_name", "album_name"],
        cache_dir="./cache",
        device="cuda",
        attn_implementation="eager",
        dtype=torch.bfloat16,
        response_prompt_name: str = "response_generation",
        reranker_type: Optional[str] = None,
        reranker_model_path: Optional[str] = None,
        reranker_multimodal_artifacts: Optional[str] = None,
        reranker_max_output_tokens: int = 512,
        # EXP-010: the LLM listwise reranker WINDOW (how many of the retrieval pool
        # the reranker reads + reorders). Default 50 = bit-identical to every shipped
        # config; reranker_k: 100 lets it see wall golds at union rank 51-100.
        reranker_k: int = 50,
        reranker_k2: int = 24,   # llm_listwise_2stage: stage-2 shortlist size (k1=reranker_k)
        reranker_stage1_model: str = "gemini-2.5-flash-lite",  # 2stage stage-1 (cheap coarse filter)
        # EXP-014: richer per-candidate context for the LLM listwise reranker
        # (release year + wider tag list + goal_category). Off = config 205 prompt.
        reranker_rich_candidates: bool = False,
        # EXP: disable the LLM-listwise reranker's "thinking" tokens (gemini-2.5
        # thinking_config). None = default (thinking on); 0 = no thinking (direct
        # ranking). Hypothesis: thinking hurts the terse listwise task (pro@2048
        # regressed). Plumbed to GeminiClient via load_reranker_module.
        reranker_thinking_budget: Optional[int] = None,
        retrieval_topk: int = 20,
        response_max_new_tokens: int = 64,
        top_n_for_prompt: int = 1,
        query_preprocessing_mode: str = "raw",
        # ---- Responder goal injection (opt-in, default OFF) --------------
        # When True, the listener_goal text is appended to the responder's
        # system prompt (see _get_system_prompt). Gives the local LLM the
        # user's stated intent, which is otherwise dropped before generation.
        # Default OFF so the shipped config 194 prompt is bit-identical.
        responder_use_goal: bool = False,
        # ---- SASRec goal-ful context (opt-in, default OFF) ---------------
        # When True, the listener_goal is appended to the SASRec channel's
        # context (build_sasrec_context) at serve — the REQUIRED serve half of
        # the in-pool goal-ful SASRec 3-way parity (SASRec_Improved_Plan.md).
        # Default OFF -> goal=None -> build_sasrec_context == build_user_dialog,
        # bit-identical to the goal-less sasrec_v1 path.
        sasrec_context_use_goal: bool = False,
        response_reranker_type: Optional[str] = None,
        response_reranker_model_path: Optional[str] = None,
        response_n_candidates: int = 3,
        response_temperatures: Optional[List[float]] = None,
        use_vllm: bool = False,
        # ---- W4 P0 #3: LoRA adapter on top of lm_type ---------------------
        # When set (HF repo id or local dir from W4 KTO / W5 S-DPO / W6 RGRPO),
        # the LM is wrapped with PEFT and the trained adapter is loaded.
        # Works on both LLAMA_MODEL and VLLM_MODEL paths.
        lora_path: Optional[str] = None,
        lora_max_rank: int = 32,
        # ---- W1 StateTracker integration (opt-in, default OFF) -----------
        # When use_state_tracker=True, a per-turn user-state extractor runs
        # before retrieval. Extracted state is threaded into batch results
        # under the `extracted_state` key so downstream consumers (W2 CMQR,
        # W4 responder prompt envelope) can read it. Default OFF — exp 021
        # path stays bit-exact when the flag is absent.
        use_state_tracker: bool = False,
        state_tracker_prompt_name: str = "state_extraction",
        state_tracker_max_new_tokens: int = 96,
        # ---- W2 CMQR multi-query rewriter (opt-in, default OFF) ----------
        # When use_cmqr=True, the inner retriever is wrapped by CMQR_REWRITER:
        # for each query, emit N rewrites (injecting StateTracker fields),
        # call the inner retriever per rewrite, RRF-fuse the N ranked lists,
        # return top-`retrieval_topk`. Reuses self.lm (no extra weight load).
        # Cache stores rewrites by (session_id, turn_number) at
        # `{cache_dir}/cmqr/`. Implies use_state_tracker=True so rewrites can
        # actually inject state fields; raises if the flag is missing.
        use_cmqr: bool = False,
        cmqr_prompt_name: str = "cmqr_rewrites",
        cmqr_n_rewrites: int = 4,
        cmqr_topk_per_rewrite: int = 50,
        cmqr_rrf_k: int = 60,
        cmqr_max_new_tokens: int = 96,
        # ---- W5 extra_config: YAML overrides forwarded to the factory -------
        # Supports sid_hub_repo (str) and sid_stream_weight (float) overrides
        # for the SID weight sweep without forking YAML config files.
        extra_config: dict | None = None,
    ):
        """Initialize the CRS baseline components.

        Args:
            lm_type: LLM model identifier to load for response generation.
            retrieval_type: Retrieval backend name (e.g., "bm25").
            item_db_name: Dataset/DB name for item metadata.
            user_db_name: Dataset/DB name for user metadata.
            split_types: Dataset split names to load.
            corpus_types: Item metadata fields used for retrieval.
            cache_dir: Local directory for caching artifacts/indices.
            device: Compute device for the LLM (e.g., "cuda", "cpu").
            dtype: Torch dtype for the LLM weights/tensors.
        """
        self.cache_dir = cache_dir
        self.extra_config = extra_config or {}
        self.lm_type = lm_type
        self.retrieval_type = retrieval_type
        self.item_db_name = item_db_name
        self.user_db_name = user_db_name
        self.track_split_types = track_split_types
        self.user_split_types = user_split_types
        self.corpus_types = corpus_types
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.use_vllm = use_vllm
        self.lm = load_lm_module(
            self.lm_type, self.device, self.attn_implementation, self.dtype,
            use_vllm=self.use_vllm,
            lora_path=lora_path,
            lora_max_rank=int(lora_max_rank),
        )
        self.retrieval = load_retrieval_module(self.retrieval_type, self.item_db_name, self.track_split_types, self.corpus_types, self.cache_dir, extra_config=self.extra_config)
        self.reranker_type = reranker_type
        self.reranker_model_path = reranker_model_path
        self.reranker_max_output_tokens = reranker_max_output_tokens
        self.reranker_k = reranker_k
        self.reranker = load_reranker_module(
            reranker_type, self.item_db_name, self.track_split_types, self.corpus_types, self.cache_dir,
            model_path=reranker_model_path,
            multimodal_artifacts=reranker_multimodal_artifacts,
            max_output_tokens=reranker_max_output_tokens,
            k=reranker_k,
            k2=reranker_k2,
            stage1_model=reranker_stage1_model,
            rich_candidates=reranker_rich_candidates,
            thinking_budget=reranker_thinking_budget,
        )
        # Tier-2 #4.1: lazily built when the reranker lists qwen_meta_cos/bm25_score.
        self._relevance_scorer = None
        # When reranker is present, pull a larger candidate pool from retrieval
        # (retrieval_topk) and shrink to 20 via rerank. Otherwise retrieval
        # returns exactly 20. Submission format fixed at 20 tids.
        self.retrieval_topk = retrieval_topk
        self.response_max_new_tokens = response_max_new_tokens
        # Top-N tracks passed to the LM prompt as "recommended tracks". Default
        # 1 (same as 021 champion). Setting >1 decouples retrieval's top-1 from
        # the LM's cited track — the LM can pick the one that best matches the
        # query rather than being forced to describe whatever the reranker
        # bumped to rank 1. See exp 028 post-mortem on the coupling hypothesis.
        self.top_n_for_prompt = max(1, int(top_n_for_prompt))
        # query_preprocessing_mode controls how session_memory is formatted
        # before being sent to the retriever. 'raw' (default) preserves 021
        # champion behaviour. 'last_user' / 'last_user_with_goal' clean the
        # query — strip role prefixes + drop multi-turn assistant/music noise.
        # See build_retrieval_query() above for the modes.
        if query_preprocessing_mode not in (
            "raw", "raw_with_goal", "raw_enriched", "last_user",
            "last_user_with_goal", "bge_m3_structured",
        ):
            raise ValueError(f"unknown query_preprocessing_mode: {query_preprocessing_mode!r}")
        self.query_preprocessing_mode = query_preprocessing_mode
        # Responder goal injection (see _get_system_prompt). Default off.
        self.responder_use_goal = bool(responder_use_goal)
        self.sasrec_context_use_goal = bool(sasrec_context_use_goal)
        # Response reranker (exp 026+): sample K responses and pick the best via
        # a reward model trained on train goal_progress_assessments.
        self.response_reranker_type = response_reranker_type
        self.response_reranker = load_response_reranker_module(
            response_reranker_type, model_path=response_reranker_model_path,
        )
        self.response_n_candidates = response_n_candidates
        self.response_temperatures = response_temperatures or [0.3, 0.7, 1.0]
        self.item_db = MusicCatalogDB(self.item_db_name, self.track_split_types, self.corpus_types)
        self.user_db = UserProfileDB(self.user_db_name, self.user_split_types)
        self.prompts_dir = os.path.join(os.path.dirname(__file__), "system_prompts")
        # response_prompt_name is config-overrideable so we can A/B test prompt
        # variants (stock vs persona vs few-shot) without editing code. All
        # variants must live under system_prompts/{name}.txt.
        self.response_prompt_name = response_prompt_name
        response_prompt_path = f"{self.prompts_dir}/{self.response_prompt_name}.txt"
        if not os.path.isfile(response_prompt_path):
            raise FileNotFoundError(
                f"response prompt '{self.response_prompt_name}' not found at {response_prompt_path}"
            )
        self.role_prompt = {
            "role_play": open(f"{self.prompts_dir}/roleplay.txt", "r", encoding="utf-8").read(),
            "personalization": open(f"{self.prompts_dir}/personalization.txt", "r", encoding="utf-8").read(),
            "response_generation": open(response_prompt_path, "r", encoding="utf-8").read(),
        }
        self.session_memory = []

        # StateTracker — opt-in via config (Gap 1). The tracker reuses self.lm
        # so no extra weights are loaded. Cache lives at experiments/cache/state/
        # by default to survive the `rm -rf cache` in run_inference_devset.py:71.
        self.use_state_tracker = bool(use_state_tracker)
        self.state_tracker = None
        if self.use_state_tracker:
            from mcrs.query_rewriters.state_tracker import StateTracker
            state_prompt_path = f"{self.prompts_dir}/{state_tracker_prompt_name}.txt"
            if not os.path.isfile(state_prompt_path):
                raise FileNotFoundError(
                    f"state_tracker prompt '{state_tracker_prompt_name}' not found at {state_prompt_path}"
                )
            # cache_dir at the BASELINES level is `./cache` (transient) but the
            # 021 yaml ships `cache_dir: "../experiments/cache"` already; we
            # honour whatever the caller passed.
            self.state_tracker = StateTracker(
                lm=self.lm,
                prompt_path=state_prompt_path,
                cache_dir=self.cache_dir,
                max_new_tokens=int(state_tracker_max_new_tokens),
            )

        # intent_state Q* rewriter — opt-in via extra_config use_intent_state (plan §3/§5,
        # W1.a). Rewrites each turn's raw retrieval query into a clean, self-contained
        # query_star (goal+profile-seeded) via Gemini, feeding the content channels. Cached
        # by (session, turn); falls back to the raw query on any failure (never breaks serve).
        self.use_intent_state = bool(self.extra_config.get("use_intent_state"))
        self.intent_state_rewriter = None
        if self.use_intent_state:
            from mcrs.query_rewriters.intent_state import IntentStateRewriter
            self.intent_state_rewriter = IntentStateRewriter(
                cache_dir=self.cache_dir,
                model=self.extra_config.get("intent_state_model", "gemini-2.5-flash-lite"),
            )

        # CMQR — opt-in via config (W2). Wraps self.retrieval transparently:
        # downstream code calls self.retrieval.batch_text_to_item_retrieval(...)
        # and gets RRF-fused rewrites without knowing CMQR is in the path.
        # Requires StateTracker to be on (otherwise rewrites can't inject
        # state fields and degrade to vanilla query-expansion).
        self.use_cmqr = bool(use_cmqr)
        self.cmqr = None
        if self.use_cmqr:
            if not self.use_state_tracker:
                raise ValueError(
                    "use_cmqr=True requires use_state_tracker=True — CMQR injects "
                    "user_state fields into rewrites and needs the tracker to fill them."
                )
            from mcrs.query_rewriters.cmqr import CMQR_REWRITER
            cmqr_prompt_path = f"{self.prompts_dir}/{cmqr_prompt_name}.txt"
            if not os.path.isfile(cmqr_prompt_path):
                raise FileNotFoundError(
                    f"cmqr prompt '{cmqr_prompt_name}' not found at {cmqr_prompt_path}"
                )
            # Wrap the existing retriever; CMQR keeps it as `self.inner` and
            # exposes the same batch_text_to_item_retrieval interface.
            self.cmqr = CMQR_REWRITER(
                lm=self.lm,
                inner_retriever=self.retrieval,
                prompt_path=cmqr_prompt_path,
                cache_dir=self.cache_dir,
                n_rewrites=int(cmqr_n_rewrites),
                topk_per_rewrite=int(cmqr_topk_per_rewrite),
                rrf_k=int(cmqr_rrf_k),
                max_new_tokens=int(cmqr_max_new_tokens),
            )
            # Replace self.retrieval so the rest of the pipeline is unchanged.
            self._inner_retrieval = self.retrieval  # keep a handle for diagnostics
            self.retrieval = self.cmqr

        # A6 catalog-membership guard — always-on safety net for any future
        # path that emits track_ids (W4-W6 trained responders may hallucinate).
        # No-op for W2 (retrieval-only outputs are catalog-bounded by construction).
        # Build the valid set once from item_db; ~50k strings, negligible memory.
        try:
            self._valid_catalog: Optional[set] = set(self.item_db.metadata_dict.keys())
        except (AttributeError, KeyError):
            self._valid_catalog = None  # graceful: skip filter if db lacks the dict

    def _reset_session_memory(self):
        """Clear all messages stored in the current session memory.
        """
        self.session_memory = []

    def _upload_session_memory(self, chat_history: List[Dict[str, Any]]):
        """Upload the session memory to the database.
        """
        self.session_memory = chat_history

    def _get_system_prompt(self, user_id: Optional[str] = None,
                           goal_text: Optional[str] = None) -> str:
        """Build the system prompt, optionally personalized with a user profile.
        Args:
            user_id: Optional user identifier. When provided, includes a personalization segment derived from the user's profile.
            goal_text: Optional listener_goal text. Appended as a session-goal
                segment ONLY when responder_use_goal=True (config-gated, default
                off). Gives the responder the user's stated intent — otherwise
                dropped before the LLM. No-ops when the flag is off or goal_text
                is empty, so the shipped prompt is bit-identical by default.
        Returns:
            The final system prompt string used for the LLM.
        """
        system_prompt = self.role_prompt["role_play"] + self.role_prompt["response_generation"]
        if user_id:
            user_profile_str = self.user_db.id_to_profile_str(user_id)
            system_prompt += self.role_prompt["personalization"] + '\n' + user_profile_str
        if getattr(self, "responder_use_goal", False) and goal_text:
            system_prompt += (
                "\n\n[SESSION GOAL] The listener's stated goal for this session: "
                f"{goal_text}"
            )
        return system_prompt

    def _played_tids_for(self, prior_history: list) -> list[str]:
        """Raw played track_ids for the session-aware channels (same_artist,
        session_cf) + the LGBM session-continuity features.

        Bug #1 fix: this reuses the shared played_tids_from_context helper
        (role in ("music","assistant") + a track_id fallback) so history_tids
        cannot silently diverge from what those channels compute. The previous
        inline filter checked role=="music", but both inference parsers rewrite
        music turns to role=="assistant" (run_inference_devset.py /
        run_inference_blindset.py) — so it always returned [] at serve, running
        SASRec sequence + the LGBM session features history-blind in production.
        """
        from .retrieval_modules.session_history import played_tids_from_context
        if self._valid_catalog is not None:
            return played_tids_from_context(
                {"chat_history": prior_history}, self._valid_catalog)
        # No catalog to validate against (db lacks metadata_dict) — accept raw
        # track_ids carried on music/assistant turns directly.
        return [str(t["track_id"]) for t in prior_history
                if t.get("role") in ("music", "assistant") and t.get("track_id")]

    @staticmethod
    def _sasrec_dialog_turns(prior_history: list, user_query) -> list:
        """Turn list handed to build_user_dialog for the weight-1.0 SASRec channel.

        SASRec was trained (train_sasrec._walk_split) on user-turns-only dialog
        built from the turns before the music turn PLUS the current-turn user
        request. prior_history here is the PRE-append history (it excludes the
        current request), so we append it back to match the training
        distribution and the build_lgbm_features feature builder. Dropping the
        current request (the old `build_user_dialog(prior_history)` call) was a
        train/serve skew — build_user_dialog keeps only role=="user" turns, so
        music/assistant turns are excluded regardless.
        """
        return list(prior_history) + [{"role": "user", "content": user_query}]

    def _finalize_topk(self, items: list, pool: list, played_set: set,
                       k: int = 20) -> list:
        """Assemble the final top-k: dedupe (keep first), drop ids not in the
        catalog (when known), EXCLUDE already-played tracks (bug #2), and
        backfill from the pre-rerank pool.

        Played-exclusion is safe by construction: the gold is ALWAYS a new
        track, so a played track in the top-k is a guaranteed miss; dropping it
        only promotes real candidates (non-decreasing for nDCG@k). Depends on
        bug #1 being fixed so played_set (from history_tids) is non-empty.
        """
        valid = self._valid_catalog
        played = played_set or set()
        seen: set = set()
        kept: list = []
        for src in (items, pool):
            for tid in src:
                if len(kept) >= k:
                    break
                if tid in seen or tid in played:
                    continue
                if valid is not None and tid not in valid:
                    continue
                kept.append(tid)
                seen.add(tid)
            if len(kept) >= k:
                break
        return kept[:k]

    def chat(self, user_query: str, user_id: Optional[str] = None) -> dict[str, Any]:
        """Run a single CRS turn: retrieve items and generate a response.
        Args:
            user_query: The user's latest message or request.
            user_id: Optional user identifier for personalization.
        Returns:
            A dictionary with keys:
                - user_id: The user identifier (may be None).
                - user_query: Echo of the input query.
                - retrieval_items: List of retrieved item IDs (top candidates).
                - recommend_item: Metadata for the top recommended item.
                - response: The generated assistant response string.
        """
        self.session_memory.append({"role": "user", "content": user_query})
        # stage0. system prompt
        system_prompt = self._get_system_prompt(user_id)
        # stage1. retrieval
        retrieval_input = "\n".join([f"{conversation['role']}: {conversation['content']}" for conversation in self.session_memory])
        retrieval_items = self.retrieval.text_to_item_retrieval(retrieval_input, topk=20)
        recommend_item = self.item_db.id_to_metadata(retrieval_items[0])
        # stage2. response generation
        response = self.lm.response_generation(system_prompt, self.session_memory, recommend_item)
        return {
            "user_id": user_id,
            "user_query": user_query,
            "retrieval_items": retrieval_items,
            "recommend_item": recommend_item,
            "response": response,
        }

    def batch_chat(self, batch_data: List[Dict[str, Any]],
                   generate_response: bool = True) -> List[Dict[str, Any]]:
        """Run multiple CRS turns in batch: retrieve items and generate responses.
        Args:
            batch_data: List of dictionaries, each containing:
                - user_query: The user's latest message or request.
                - user_id: Optional user identifier for personalization.
                - session_memory: List of chat history messages.
            generate_response: When False, skip Stage 2 (the responder LM) and
                return a schema-valid stub response ("ok") instead. retrieval_items
                (the nDCG axis) is still fully computed via retrieval + rerank.
                Used by run_inference_blindset.py --retrieval_only to get fresh
                track_ids cheaply when reusing frozen responses (EXP-016).
        Returns:
            A list of dictionaries, each with keys:
                - user_id: The user identifier (may be None).
                - user_query: Echo of the input query.
                - retrieval_items: List of retrieved item IDs (top candidates).
                - recommend_item: Metadata for the top recommended item.
                - response: The generated assistant response string.
        """
        # Prepare batch inputs
        sys_prompts = []
        retrieval_inputs = []
        # Option B: per-row compact ColBERT query (goal+culture+last user turn),
        # built via the SAME build_retrieval_query(mode="compact_colbert") as the
        # train builder. Threaded into batch_context['colbert_query']; the union
        # routes it to ColBERT only when colbert_compact_query is set (else ignored).
        colbert_queries: list[str] = []
        session_memories = []
        user_ids: list[Optional[str]] = []
        goal_categories: list[Optional[str]] = []
        goal_specificities: list[Optional[str]] = []
        user_profiles_raw: list[Any] = []
        extracted_states: list[Optional[dict]] = []

        for data in batch_data:
            user_query = data['user_query']
            user_id = data.get('user_id')
            session_memory = data['session_memory'].copy()
            session_memory.append({"role": "user", "content": user_query})

            cg = data.get('conversation_goal') or {}
            goal_text = (cg.get('listener_goal') or "").strip() or None
            # Pass goal_text so the responder system prompt can carry the
            # listener's stated intent (only when responder_use_goal=True; the
            # method no-ops on the goal otherwise, so default behaviour is
            # bit-identical).
            sys_prompts.append(self._get_system_prompt(user_id, goal_text=goal_text))
            # bge_m3_structured needs user_profile too; pass it through so the
            # structured [USER] block renders age/country/gender. Other modes
            # ignore the kwarg.
            user_profile_for_query = data.get('user_profile_raw')
            if isinstance(user_profile_for_query, str):
                try:
                    import json as _json
                    user_profile_for_query = _json.loads(user_profile_for_query)
                except Exception:
                    try:
                        import ast as _ast
                        user_profile_for_query = _ast.literal_eval(user_profile_for_query)
                    except Exception:
                        user_profile_for_query = None
            retrieval_input = build_retrieval_query(
                session_memory,
                mode=self.query_preprocessing_mode,
                goal_text=goal_text,
                user_profile=user_profile_for_query,
            )
            # intent_state Q* rewrite (opt-in): replace the raw query with a clean,
            # self-contained query_star. fallback_query=retrieval_input means any failure
            # degrades to the exact raw query (bit-identical to use_intent_state=False).
            if self.use_intent_state and self.intent_state_rewriter is not None:
                sid = data.get('session_id') or ""
                tn = int(data.get('turn_number') or 0)
                history_text = "\n".join(
                    f"{t.get('role','')}: {t.get('content','')}" for t in session_memory[:-1])
                try:
                    qs = self.intent_state_rewriter.rewrite(
                        sid, tn, history_text, user_query, goal_text or "",
                        user_profile_for_query, fallback_query=retrieval_input)
                    retrieval_input = qs.get("query_star") or retrieval_input
                except Exception as e:
                    print(f"[CRS_BASELINE] intent_state failed on session={sid[:8]} turn={tn}: {e!r}")
            retrieval_inputs.append(retrieval_input)
            # Compact ColBERT query from the SAME session_memory (includes the current
            # user turn -> "last user turn") + parsed profile. Independent of the Q*
            # rewrite above (that only affects the shared raw_enriched query).
            colbert_queries.append(build_retrieval_query(
                session_memory, mode="compact_colbert",
                goal_text=goal_text, user_profile=user_profile_for_query))
            session_memories.append(session_memory)
            user_ids.append(user_id)
            # Session-level side channels for task-aware rerankers (A1 LGBM).
            # Back-compat: absent in batch_data -> None, rerankers handle gracefully.
            cg = data.get('conversation_goal') or {}
            goal_categories.append(cg.get('category'))
            goal_specificities.append(cg.get('specificity'))
            user_profiles_raw.append(data.get('user_profile_raw'))

            # StateTracker (opt-in, Gap 1). Extract user_state BEFORE retrieval
            # so W2 CMQR can use it. Per-turn — caches inside StateTracker.
            # Falls back to None on any exception so this can never break the
            # production path; loud-warns so failures are visible in logs.
            state: Optional[dict] = None
            if self.use_state_tracker and self.state_tracker is not None:
                sid = data.get('session_id') or ""
                tn = int(data.get('turn_number') or 0)
                # If session_id/turn_number aren't set on the batch row, skip
                # extraction silently — the inference scripts always set them.
                if sid and tn > 0:
                    history_text = "\n".join(
                        f"{t.get('role','')}: {t.get('content','')}"
                        for t in session_memory[:-1]  # exclude the appended user query
                    )
                    try:
                        state = self.state_tracker.extract(sid, tn, user_query, history_text)
                    except Exception as e:
                        # Never let state extraction kill the pipeline.
                        print(f"[CRS_BASELINE] state_tracker failed on "
                              f"session={sid[:8]} turn={tn}: {e!r}")
                        state = None
            extracted_states.append(state)

        # If CMQR is wrapping the retriever, hand it the per-query context
        # (session_id, turn_number, extracted_state) so it can cache by
        # (session, turn) and inject state fields into rewrites. The set is
        # ephemeral — only valid for the next batch_text_to_item_retrieval call.
        if self.use_cmqr and self.cmqr is not None:
            self.cmqr.set_batch_context(
                session_ids=[d.get("session_id") for d in batch_data],
                turn_numbers=[int(d.get("turn_number") or 0) for d in batch_data],
                extracted_states=extracted_states,
            )

        # Build per-query structured context for SID-style retrievers.
        # Other retrievers accept-and-ignore via try/except TypeError back-compat.
        # session_memory at this point has the prior history WITHOUT the current user
        # query (it was appended above into session_memories but batch_data holds
        # the original pre-append list — use data.get("session_memory") for prior).
        batch_context = []
        # colbert_queries is built positionally in the first loop (one append per
        # batch_data row, no skips); pin that so a future early-continue there can't
        # silently misalign the compact query to the wrong session.
        assert len(colbert_queries) == len(batch_data), (
            f"colbert_queries ({len(colbert_queries)}) misaligned with batch_data "
            f"({len(batch_data)}) — first-loop appends must be 1:1 with batch_data")
        for _i, data in enumerate(batch_data):
            prior_history = data.get("session_memory", [])  # {role, content} dicts
            # Recover RAW played track_ids via the shared helper (bug #1 fix):
            # the old inline filter checked role=="music", but the inference
            # parsers rewrite music turns to role=="assistant" -> it returned []
            # at serve, running SASRec + the LGBM session features history-blind.
            _played = self._played_tids_for(prior_history)
            batch_context.append({
                "chat_history": prior_history,
                "current_user_query": data["user_query"],
                "user_profile": data.get("user_profile_raw"),
                "conversation_goal": data.get("conversation_goal"),
                "history_tids": _played,
                # Option B per-channel query (built in the first loop, same index).
                "colbert_query": colbert_queries[_i],
                # SASRec was trained on user-turns-only dialog that INCLUDES the
                # current user request (train_sasrec._walk_split). prior_history
                # is the pre-append history, so rebuild the training turn list via
                # _sasrec_dialog_turns; else the channel sees a dialog missing the
                # current request (train/serve skew on a weight-1.0 channel) or
                # falls back to the noisy full raw query.
                # Goal-ful context only when sasrec_context_use_goal (the in-pool
                # model); else goal_text=None -> build_sasrec_context degrades to
                # build_user_dialog, bit-identical to the goal-less sasrec_v1 path.
                "user_dialog": build_sasrec_context(
                    self._sasrec_dialog_turns(prior_history, data["user_query"]),
                    goal_text=((data.get("conversation_goal") or {}).get("listener_goal")
                               if self.sasrec_context_use_goal else None)),
            })

        # Stage 1: Batch retrieval. Pull retrieval_topk (default 20; 40 when
        # a reranker is configured) candidates per query. user_ids and
        # batch_context thread through so cf-bpr/SID-style retrievers can use them.
        stage1_topk = self.retrieval_topk
        if hasattr(self.retrieval, 'batch_text_to_item_retrieval'):
            try:
                batch_retrieval_items = self.retrieval.batch_text_to_item_retrieval(
                    retrieval_inputs, topk=stage1_topk,
                    user_ids=user_ids, batch_context=batch_context,
                )
            except TypeError:
                # Back-compat: retriever predates batch_context kwarg.
                try:
                    batch_retrieval_items = self.retrieval.batch_text_to_item_retrieval(
                        retrieval_inputs, topk=stage1_topk, user_ids=user_ids,
                    )
                except TypeError:
                    batch_retrieval_items = self.retrieval.batch_text_to_item_retrieval(
                        retrieval_inputs, topk=stage1_topk,
                    )
        else:
            batch_retrieval_items = [self.retrieval.text_to_item_retrieval(inp, topk=stage1_topk) for inp in retrieval_inputs]
        # Capture the pre-rerank pool — A6 backfill source if catalog filter
        # has to drop hallucinated UUIDs after rerank.
        batch_retrieval_pool = [list(items) for items in batch_retrieval_items]

        # Stage 1b: Rerank (optional). Post-retrieval reranker scores each
        # candidate and keeps the top-20 for submission. The rerank() call
        # forwards user + goal side-channels; rerankers that don't use them
        # (e.g. BGE cross-encoder) accept-and-ignore, while task-aware ones
        # (LGBM LambdaMART) use them as categorical features.
        if self.reranker is not None:
            # Per-query session info for rerankers that compute session-continuity
            # features (LGBM LambdaMART). played_tids = the RAW prior track_ids
            # already threaded into batch_context['history_tids'] above.
            extra_session_info = [
                {"played_tids": bc.get("history_tids", [])} for bc in batch_context
            ]
            # Build per-candidate SASRec rank features for LGBM models that
            # list "sasrec_rank_inv" in their feature set. This issues a second
            # call to batch_per_sub_rankings (only available on RRF union
            # retrievers) — negligible for Blind-A (80 queries) and gated so
            # non-LGBM / non-union paths are completely unaffected.
            extra_features_per_candidate = None
            _reranker_feats = getattr(self.reranker, "features", [])
            if (("sasrec_rank_inv" in _reranker_feats
                 or "n_channels_hit" in _reranker_feats)
                    and hasattr(self.retrieval, "batch_per_sub_rankings")):
                try:
                    _per_sub, _labels = self.retrieval.batch_per_sub_rankings(
                        retrieval_inputs, user_ids=user_ids, batch_context=batch_context)
                    extra_features_per_candidate = build_sasrec_extra_features(
                        _per_sub, _labels, batch_retrieval_items)
                except Exception as e:
                    print(f"[crs_baseline] sasrec extra-features skipped: {e!r}")
                    extra_features_per_candidate = None
            # Tier-2 #4.1: inject qwen_meta_cos + bm25_score for models that list
            # them, via the shared RelevanceScorer (same computation as the train
            # feature builder -> parity). Merged into the per-candidate dicts.
            if any(f in _reranker_feats for f in ("qwen_meta_cos", "bm25_score")):
                try:
                    if self._relevance_scorer is None:
                        from mcrs.rerankers.relevance_scorer import RelevanceScorer
                        self._relevance_scorer = RelevanceScorer(
                            self.item_db_name, self.track_split_types,
                            self.corpus_types, self.cache_dir)
                    _rel = self._relevance_scorer.feats_for_batch(
                        retrieval_inputs, batch_retrieval_items)
                    if extra_features_per_candidate is None:
                        extra_features_per_candidate = [
                            [{} for _ in items] for items in batch_retrieval_items]
                    for _qi, _per_cand in enumerate(_rel):
                        for _ci, _f in enumerate(_per_cand):
                            extra_features_per_candidate[_qi][_ci].update(_f)
                except Exception as e:
                    print(f"[crs_baseline] relevance extra-features skipped: {e!r}")
            try:
                batch_retrieval_items = self.reranker.rerank(
                    retrieval_inputs, batch_retrieval_items, topk=20,
                    user_ids=user_ids,
                    goal_categories=goal_categories,
                    goal_specificities=goal_specificities,
                    user_profiles_raw=user_profiles_raw,
                    extra_session_info=extra_session_info,
                    extra_features_per_candidate=extra_features_per_candidate,
                )
            except TypeError:
                # Back-compat: reranker predates the side-channel kwargs
                # (e.g. BGE cross-encoder doesn't accept extra_session_info).
                batch_retrieval_items = self.reranker.rerank(
                    retrieval_inputs, batch_retrieval_items, topk=20,
                )

        # Stage 1c: dedupe + catalog-membership guard + PLAYED-TRACK EXCLUSION
        # (bug #2) + backfill. Played tracks are dropped because the gold is
        # always a NEW track, so a played track in the top-20 is a guaranteed
        # miss; excluding it only promotes real candidates (non-decreasing for
        # nDCG). The catalog filter still guards hallucinated ids from W4-W6
        # responders (retrievers are catalog-bounded). Applied unconditionally
        # (the old code skipped the whole block when _valid_catalog was None);
        # _finalize_topk handles a None catalog gracefully. played_set comes from
        # batch_context["history_tids"] — non-empty only after the bug #1 fix.
        filtered = []
        for items, pool, bc in zip(
                batch_retrieval_items, batch_retrieval_pool, batch_context):
            played_set = set(bc.get("history_tids") or [])
            filtered.append(self._finalize_topk(items, pool, played_set))
        batch_retrieval_items = filtered

        # Build the "recommend_item(s)" string passed to the LM. When
        # top_n_for_prompt=1 this is identical to 021 champion (just
        # id_to_metadata of the top-1). When >1, concatenate the top-N
        # tracks into a multi-line "Candidate tracks" block so the LM can
        # pick the best match for its response.
        def _format_recommend_items(tids: list[str], n: int) -> str:
            if n <= 1:
                return self.item_db.id_to_metadata(tids[0])
            lines = []
            for rank, tid in enumerate(tids[:n], start=1):
                try:
                    meta = self.item_db.id_to_metadata(tid)
                except Exception:
                    meta = f"track_id: {tid}"
                lines.append(f"Candidate {rank}: {meta}")
            return "\n".join(lines)

        recommend_items = [
            _format_recommend_items(items, self.top_n_for_prompt)
            for items in batch_retrieval_items
        ]

        # Stage 2: Batch response generation. Skipped in retrieval-only mode —
        # the responder is the costly stage (~35-65 min on Blind-A) and EXP-016
        # reuses frozen responses, so we emit a schema-valid stub and graft real
        # responses downstream. retrieval_items (the nDCG axis) is fully computed
        # above regardless. The stub mirrors run_inference_devset_retrieval_only.
        if not generate_response:
            responses = ["ok"] * len(batch_data)
        elif self.response_reranker is not None and hasattr(self.lm, 'batch_response_generation_multi'):
            # Multi-candidate + reward-model rerank path (exp 026+).
            # Sample K responses per query, score each, ship the best one.
            candidates_per_query = self.lm.batch_response_generation_multi(
                sys_prompts, session_memories, recommend_items,
                max_new_tokens=self.response_max_new_tokens,
                temperatures=self.response_temperatures[: self.response_n_candidates],
            )
            # Build per-query context strings matching the reward model's
            # training format (see scripts/build_reward_dataset.py).
            contexts: list[str] = []
            for i, data in enumerate(batch_data):
                rec_meta = recommend_items[i] if i < len(recommend_items) else ""
                user_query = data.get('user_query', '')
                # Minimal context — no goal metadata available at Blind-A
                # inference time (we don't know listener_goal); rely on
                # query + recommended_track + prior history to drive scoring.
                history_text = "\n".join(
                    f"{t['role']}: {t['content']}" for t in data.get('session_memory', [])
                )[:1000]
                ctx = (
                    f"User query: {user_query}\n"
                    f"Recommended track: {rec_meta}\n"
                    f"Prior dialog: {history_text}"
                )
                contexts.append(ctx)
            best_idx = self.response_reranker.rerank(contexts, candidates_per_query)
            responses = [cands[bi] for cands, bi in zip(candidates_per_query, best_idx)]
        elif hasattr(self.lm, 'batch_response_generation'):
            # Standard greedy path (exp 021/024 etc.).
            responses = self.lm.batch_response_generation(
                sys_prompts, session_memories, recommend_items,
                max_new_tokens=self.response_max_new_tokens,
            )
        else:
            responses = [self.lm.response_generation(sys_prompts[i], session_memories[i], recommend_items[i])
                        for i in range(len(batch_data))]

        # CoT prompts ask the LM to emit <user_state>...</user_state> then
        # <response>...</response>. Gemini scores predicted_response, so
        # strip the user_state envelope before returning. Triggered by
        # naming convention: response_prompt_name starts with
        # 'response_generation_cot_'. Skipped under the retrieval-only stub.
        if generate_response and self.response_prompt_name.startswith("response_generation_cot_"):
            responses = [extract_cot_response(r) for r in responses]

        # Prepare results
        results = []
        for i, data in enumerate(batch_data):
            row = {
                "user_id": data.get('user_id'),
                "user_query": data['user_query'],
                "retrieval_items": batch_retrieval_items[i],
                "recommend_item": recommend_items[i],
                "response": responses[i],
            }
            # Side-channel: extracted user_state. Only present when
            # use_state_tracker=True; absent on the default exp 021 path.
            if self.use_state_tracker:
                state = extracted_states[i]
                row["extracted_state"] = state
                # Surface the fallback flag explicitly so downstream consumers
                # (CMQR, responder envelope) can downweight on stale state.
                from mcrs.query_rewriters.state_tracker import StateTracker
                row["state_was_fallback"] = StateTracker.was_fallback(state)
            results.append(row)

        return results

    def save_caches(self) -> None:
        """Persist any in-memory caches that components have built up.

        Currently saves:
          - ProRank score cache (~32 MB at full devset; ~800k entries) —
            critical because batch_chat builds it up purely in memory
            and the inference driver process exits before any save would
            happen otherwise. Per W3 review P0 #1.
          - StateTracker / CMQR caches are already file-backed (each
            extraction writes its own JSON), so no explicit save needed.

        Safe to call from a try/finally block; never raises.
        """
        try:
            if self.reranker is not None and hasattr(self.reranker, "save_cache"):
                self.reranker.save_cache()
                print(f"[CRS_BASELINE] saved reranker cache")
        except Exception as e:  # noqa: BLE001
            # Never let cache-save failure crash the run; the predictions
            # are already on disk by the time this is called.
            print(f"[CRS_BASELINE] save_caches: reranker save failed: {e!r}")
