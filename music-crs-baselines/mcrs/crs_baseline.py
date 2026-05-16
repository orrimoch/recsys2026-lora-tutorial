import os
import re
import torch
from typing import Optional, Any, List, Dict
from mcrs.db_item import MusicCatalogDB
from mcrs.db_user import UserProfileDB
from mcrs.lm_modules import load_lm_module
from mcrs.retrieval_modules import load_retrieval_module
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

    Returns the formatted query string.
    """
    if mode == "raw":
        return "\n".join(
            f"{t.get('role','')}: {t.get('content','')}" for t in session_memory
        )
    # Last user turn — find it from the END of session_memory.
    last_user = ""
    for t in reversed(session_memory):
        if t.get("role") == "user":
            last_user = str(t.get("content", "")).strip()
            break
    if not last_user:
        # Fallback to raw if there's no user turn (shouldn't happen on inference).
        return "\n".join(
            f"{t.get('role','')}: {t.get('content','')}" for t in session_memory
        )
    if mode == "last_user_with_goal" and goal_text:
        return f"{last_user} || goal: {goal_text}"
    return last_user

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
        retrieval_topk: int = 20,
        response_max_new_tokens: int = 64,
        top_n_for_prompt: int = 1,
        query_preprocessing_mode: str = "raw",
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
        self.retrieval = load_retrieval_module(self.retrieval_type, self.item_db_name, self.track_split_types, self.corpus_types, self.cache_dir)
        self.reranker_type = reranker_type
        self.reranker_model_path = reranker_model_path
        self.reranker = load_reranker_module(
            reranker_type, self.item_db_name, self.track_split_types, self.corpus_types, self.cache_dir,
            model_path=reranker_model_path,
        )
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
        if query_preprocessing_mode not in ("raw", "last_user", "last_user_with_goal"):
            raise ValueError(f"unknown query_preprocessing_mode: {query_preprocessing_mode!r}")
        self.query_preprocessing_mode = query_preprocessing_mode
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

    def _get_system_prompt(self, user_id: Optional[str] = None) -> str:
        """Build the system prompt, optionally personalized with a user profile.
        Args:
            user_id: Optional user identifier. When provided, includes a personalization segment derived from the user's profile.
        Returns:
            The final system prompt string used for the LLM.
        """
        system_prompt = self.role_prompt["role_play"] + self.role_prompt["response_generation"]
        if user_id:
            user_profile_str = self.user_db.id_to_profile_str(user_id)
            system_prompt += self.role_prompt["personalization"] + '\n' + user_profile_str
        return system_prompt

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

    def batch_chat(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run multiple CRS turns in batch: retrieve items and generate responses.
        Args:
            batch_data: List of dictionaries, each containing:
                - user_query: The user's latest message or request.
                - user_id: Optional user identifier for personalization.
                - session_memory: List of chat history messages.
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

            sys_prompts.append(self._get_system_prompt(user_id))
            cg = data.get('conversation_goal') or {}
            goal_text = (cg.get('listener_goal') or "").strip() or None
            retrieval_input = build_retrieval_query(
                session_memory,
                mode=self.query_preprocessing_mode,
                goal_text=goal_text,
            )
            retrieval_inputs.append(retrieval_input)
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
        for data in batch_data:
            prior_history = data.get("session_memory", [])  # {role, content} dicts
            batch_context.append({
                "chat_history": prior_history,
                "current_user_query": data["user_query"],
                "user_profile": data.get("user_profile_raw"),
                "conversation_goal": data.get("conversation_goal"),
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
            try:
                batch_retrieval_items = self.reranker.rerank(
                    retrieval_inputs, batch_retrieval_items, topk=20,
                    user_ids=user_ids,
                    goal_categories=goal_categories,
                    goal_specificities=goal_specificities,
                    user_profiles_raw=user_profiles_raw,
                )
            except TypeError:
                # Back-compat: reranker predates the side-channel kwargs.
                batch_retrieval_items = self.reranker.rerank(
                    retrieval_inputs, batch_retrieval_items, topk=20,
                )

        # Stage 1c: A6 catalog-membership filter + A7 dedupe. Hard guard for
        # any track_id that snuck in from a hallucinating model (W4-W6 trained
        # responders are the threat surface; W2 retrievers are catalog-bounded
        # by construction so this is a no-op here). Drops invalid IDs and
        # backfills from the pre-rerank pool to keep len(items) ≥ 20.
        if self._valid_catalog is not None:
            filtered = []
            for items, pool in zip(batch_retrieval_items, batch_retrieval_pool):
                # Dedupe-keep-first → catalog-filter → backfill from pool.
                seen: set = set()
                kept: list = []
                for tid in items:
                    if tid in seen or tid not in self._valid_catalog:
                        continue
                    kept.append(tid)
                    seen.add(tid)
                # Backfill from the pre-rerank pool until len == 20 (or pool
                # is exhausted). Pool ids are also filtered+deduped on insert.
                if len(kept) < 20:
                    for tid in pool:
                        if len(kept) >= 20:
                            break
                        if tid in seen or tid not in self._valid_catalog:
                            continue
                        kept.append(tid)
                        seen.add(tid)
                filtered.append(kept[:20])
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

        # Stage 2: Batch response generation.
        if self.response_reranker is not None and hasattr(self.lm, 'batch_response_generation_multi'):
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
        # 'response_generation_cot_'.
        if self.response_prompt_name.startswith("response_generation_cot_"):
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
