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
        # dense content channel. Qwen3-Embedding-0.6B is ASYMMETRIC: the query
        # side needs the instruct prefix or recall collapses (nb74 Stage 7: raw
        # dev recall@100 0.0894 -> instruct 0.1789, 2x). Default to the instruct
        # variant; dense_instruct=False restores the raw variant for ablation.
        {"type": ("dense_metadata_qwen3" if ec.get("dense_instruct") is False
                  else "dense_metadata_qwen3_instruct"),
         "corpus_types": corpus_types,
         "topk_internal": 100, "weight": float(ec.get("w_qwen", 0.7))},
        {"type": "same_artist", "topk_internal": 100,
         "weight": float(ec.get("w_artist", 1.0))},
    ]
    # Segment-aware routing (2026-06-08): cold queries (no played history) lean on
    # CONTENT channels; warm queries lean on the SESSION channel. The RRF layer
    # picks each query's weights by history presence. Opt-in via
    # use_segment_routing; default off -> shipped fixed-weight fusion unchanged.
    if ec.get("use_segment_routing"):
        for s in specs:
            t = s["type"]
            if t == "bm25":  # lexical content — on for both
                s["cold_weight"] = float(ec.get("w_bm25_cold", ec.get("w_bm25", 1.0)))
                s["warm_weight"] = float(ec.get("w_bm25_warm", ec.get("w_bm25", 1.0)))
            elif "dense_metadata_qwen3" in t:  # dense content — SYMMETRIC.
                # Validated 2026-06-08: upweighting dense for cold backfired
                # (turn-1 nDCG -0.0021); keeping it symmetric removed the cold
                # regression while warm kept the same_artist gain (overall +0.0042).
                s["cold_weight"] = float(ec.get("w_qwen_cold", ec.get("w_qwen", 0.7)))
                s["warm_weight"] = float(ec.get("w_qwen_warm", ec.get("w_qwen", 0.7)))
            elif t == "same_artist":  # session continuity — only when warm
                s["cold_weight"] = float(ec.get("w_artist_cold", 0.0))
                s["warm_weight"] = float(ec.get("w_artist_warm", 1.5))
    # Attributes content channel (Tier-1 #3.1a, 2026-06-07): a 2nd dense content
    # view via the precomputed attributes-qwen3 embeddings (instruct query side,
    # Qwen3 is asymmetric). Verified ORTHOGONAL to metadata-dense (0.62 same-track
    # cosine, 3.6% top-10 neighbor overlap), so it can surface new-artist golds the
    # metadata/lexical/session channels miss (the new-artist wall). Opt-in via
    # use_attributes so the shipped config 194 is unchanged. Default weight 0.4
    # mirrors the metadata-dense mass; sweep only after turn-1 recall lifts.
    if ec.get("use_attributes"):
        specs.append({
            "type": "dense_attributes_qwen3_instruct",
            "corpus_types": corpus_types,
            "topk_internal": 100,
            "weight": float(ec.get("w_attributes", 0.4)),
        })
    # Lyrics content channel (A1, roadmap 2026-05-30): a 2nd content view via the
    # precomputed lyrics-qwen3 embeddings. Different signal than metadata-dense,
    # so it can surface new-artist golds the metadata/lexical/session channels
    # miss (the 98.8%-new-artist wall). Opt-in via use_lyrics so the shipped
    # config 194 is unchanged. Default weight 0.4 mirrors the metadata-dense mass.
    if ec.get("use_lyrics"):
        specs.append({
            "type": "dense_lyrics_qwen3_instruct",
            "corpus_types": corpus_types,
            "topk_internal": 100,
            "weight": float(ec.get("w_lyrics", 0.4)),
        })
    # Doc-side enriched content channel (Track B): LLM-written rich per-track docs
    # re-embedded with a stronger model. The doc side has never been enriched (all
    # 5 LLM levers enrich the QUERY); this directly attacks the weak dense channel
    # + the new-artist wall. Opt-in via use_doc_enriched. Default weight 0.7 (it is
    # meant to be a PRIMARY content channel, not a 0.4 side view). Requires the
    # embed pickle (enrich_track_docs.py -> embed_catalog.py --doc-source).
    if ec.get("use_doc_enriched"):
        specs.append({
            "type": "dense_doc_enriched_local",
            "corpus_types": corpus_types,
            "topk_internal": 100,
            "weight": float(ec.get("w_doc_enriched", 0.7)),
            "extra_config": {
                "embed_model": ec.get("doc_enriched_model", "BAAI/bge-m3"),
                "embed_label": ec.get("doc_enriched_label", "doc-enriched-v1"),
                "instruct": ec.get("doc_enriched_instruct", False),
            },
        })
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
    # Structured-query channel (Tier-1 #3.1b, 2026-06-07): an LLM distils each turn
    # into a content-only synthetic query (genres/moods/era/culture/intent), which
    # the qwen3 dense retriever matches in the catalog space. De-noises the dialogue
    # and de-emphasizes the seed artist -> reaches new-artist golds on pivot turns.
    # Opt-in via use_structured_query (loads Qwen2.5-7B like HyDE). Default weight 0.5.
    if ec.get("use_structured_query"):
        specs.append({
            "type": "structured_query", "topk_internal": 100,
            "weight": float(ec.get("w_structured_query", 0.5)),
            "extra_config": {
                "sq_model": ec.get("sq_model", "Qwen/Qwen2.5-7B-Instruct"),
                "batch_size": int(ec.get("sq_batch_size", 16)),
                "inner_dense": ec.get("sq_inner_dense", "dense_metadata_qwen3_instruct"),
            },
        })
    # Two-tower content channel (Tier-1 #3.3, 2026-06-07): a learned query head +
    # ItemFusion over the 5 frozen catalog modalities, trained contrastively on
    # (intent->gold) pairs. Pure content+intent (no session input) -> reaches the
    # new-artist wall session/lexical channels can't. Opt-in via use_two_tower.
    if ec.get("use_two_tower"):
        specs.append({
            "type": "two_tower", "topk_internal": 100,
            "weight": float(ec.get("w_two_tower", 0.7)),
            "extra_config": {"model_dir": ec.get("two_tower_model_dir", "two_tower_v1")},
        })
    # RAG propose-then-ground channel (Tier-1 #3.5, 2026-06-07): an LLM proposes
    # real "Artist - Title" tracks (incl. new artists in the same style), each
    # grounded to a catalog track by dense NN. Sources candidates from the LLM's
    # world knowledge — outside the collaborative/content graph — to reach the
    # new-artist wall. Opt-in via use_propose_ground (loads Qwen2.5-7B like HyDE).
    if ec.get("use_propose_ground"):
        specs.append({
            "type": "propose_ground", "topk_internal": 100,
            "weight": float(ec.get("w_propose_ground", 0.5)),
            "extra_config": {
                "pg_model": ec.get("pg_model", "Qwen/Qwen2.5-7B-Instruct"),
                "n_proposals": int(ec.get("pg_n_proposals", 20)),
                "batch_size": int(ec.get("pg_batch_size", 16)),
                "inner_dense": ec.get("pg_inner_dense", "dense_metadata_qwen3_instruct"),
            },
        })
    if ec.get("use_sasrec"):
        specs.append({
            "type": "sasrec_seq", "topk_internal": 100,
            "weight": float(ec.get("w_sasrec", 1.0)),
            "extra_config": {
                "model_dir": ec.get("sasrec_model_dir", "sasrec_v1"),
                "max_len": int(ec.get("sasrec_max_len", 50)),
            },
        })
    # cf-bpr user x item affinity channel (Lever 4, roadmap 2026-05-30): query-
    # independent user-taste signal, ORTHOGONAL to the session (same_artist/sasrec)
    # and content (bm25/dense) channels — can surface popular tracks by NEW artists
    # the user's latent taste likes. Opt-in via use_cfbpr; low default weight (0.25)
    # since only ~43% of users are warm (cold users -> empty list, RRF falls back
    # to the other channels). cf_bpr ignores batch_context but uses user_ids, which
    # RRF.batch_per_sub_rankings supplies via its try/except signature fallback.
    if ec.get("use_cfbpr"):
        specs.append({
            "type": "cf_bpr", "topk_internal": 100,
            "weight": float(ec.get("w_cfbpr", 0.25)),
        })
    # Related-artist channel (Lever 3, roadmap 2026-05-30): cross-session artist
    # co-occurrence reaches NEW artists (the ~96%-new-artist wall) that
    # same_artist/sasrec structurally cannot. Stage 16 probe: ~29% of union-missed
    # golds reachable at top-100 co-occ artists. Opt-in via use_related_artist.
    if ec.get("use_related_artist"):
        specs.append({
            "type": "related_artist", "topk_internal": 100,
            "weight": float(ec.get("w_related_artist", 1.0)),
        })
    # CLAP audio->session RECALL channel (2026-06-06): mean-pool the played tracks'
    # CLAP audio -> nearest catalog tracks ("sounds like the session"). nb74 Stage 23
    # probe = best new-artist wall rescue (269 union-missed new-artist golds, +0.035
    # recall@100 ceiling). Opt-in via use_clap_recall; weight-swept like related_artist
    # (a weak channel can inject RRF noise — gate on union recall@100 then nDCG).
    if ec.get("use_clap_recall"):
        specs.append({
            "type": "clap_recall", "topk_internal": 100,
            "weight": float(ec.get("w_clap_recall", 1.0)),
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
    # Track B — doc-side LLM-enriched documents re-embedded with a stronger model.
    # Pipeline: scripts/enrich_track_docs.py writes a rich per-track description ->
    # scripts/embed_catalog.py --doc-source <docs.parquet> --model <embed_model>
    # --label <embed_label> caches the pickle -> DENSE_LOCAL loads it. Attacks the
    # weak dense channel (recall@100 0.179 < BM25 0.346) + the new-artist wall.
    # extra_config: embed_model (default bge-m3, symmetric), embed_label, instruct.
    elif retrieval_type == "dense_doc_enriched_local":
        ec = extra_config or {}
        _use_instruct = bool(ec.get("instruct", False))
        return DENSE_LOCAL(
            dataset_name, track_split_types, corpus_types, cache_dir,
            model_name=ec.get("embed_model", "BAAI/bge-m3"),
            embed_label=ec.get("embed_label", "doc-enriched-v1"),
            instruct=(QWEN3_MUSIC_INSTRUCT if _use_instruct else None),
            instruct_label=("instruct-music-v1" if _use_instruct else "raw"),
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
    elif retrieval_type == "related_artist":
        from .related_artist import RelatedArtistRetriever
        return RelatedArtistRetriever(dataset_name, track_split_types, corpus_types, cache_dir)
    elif retrieval_type == "clap_recall":
        from .clap_recall import ClapRecallRetriever
        return ClapRecallRetriever(dataset_name, track_split_types, corpus_types, cache_dir)
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
    elif retrieval_type == "structured_query":
        # Structured-query channel: LLM (Qwen2.5-7B via the generic HF wrapper)
        # extracts a content-only synthetic query; an inner qwen3 dense retriever
        # matches it in the catalog space. Mirrors the hyde_qwen3 build.
        import os
        from ..lm_modules.llama import LLAMA_MODEL
        from ..query_rewriters.structured_query import StructuredQueryExtractor
        from .structured_query_channel import StructuredQueryRetriever
        lm = LLAMA_MODEL(
            model_name=extra_config.get("sq_model", "Qwen/Qwen2.5-7B-Instruct"),
            attn_implementation="sdpa")
        prompt_path = os.path.join(
            os.path.dirname(__file__), "..", "system_prompts", "structured_query.txt")
        extractor = StructuredQueryExtractor(
            lm, prompt_path, cache_dir=cache_dir,
            batch_size=int(extra_config.get("batch_size", 16)))
        inner = load_retrieval_module(
            extra_config.get("inner_dense", "dense_metadata_qwen3_instruct"),
            dataset_name, track_split_types, corpus_types, cache_dir, extra_config={})
        return StructuredQueryRetriever(extractor, inner)
    elif retrieval_type == "propose_ground":
        # RAG propose-then-ground: LLM proposes real Artist - Title tracks; an
        # inner dense retriever grounds each to a catalog track by NN; RRF fuses.
        # pg_model="gemini-*" routes to the Gemini API backend (no local 7B load);
        # anything else uses the local HF model. Mirrors the hyde_qwen3 build.
        import os
        from ..query_rewriters.gemini_propose import build_propose_generator
        from .propose_ground_channel import ProposeGroundRetriever
        prompt_path = os.path.join(
            os.path.dirname(__file__), "..", "system_prompts", "propose_tracks.txt")
        generator = build_propose_generator(
            extra_config.get("pg_model", "Qwen/Qwen2.5-7B-Instruct"),
            prompt_path, cache_dir,
            n_proposals=int(extra_config.get("n_proposals", 20)),
            batch_size=int(extra_config.get("batch_size", 16)))
        inner = load_retrieval_module(
            extra_config.get("inner_dense", "dense_metadata_qwen3_instruct"),
            dataset_name, track_split_types, corpus_types, cache_dir, extra_config={})
        return ProposeGroundRetriever(generator, inner)
    elif retrieval_type == "sasrec_seq":
        # SASRec recall channel. Loads the trained model + precomputes the item-repr
        # matrix; the dialog context token is encoded by bge-base-en-v1.5 (English,
        # 768-dim), with oldest-first truncation so recent turns survive the 512-cap.
        import os
        import pickle
        import warnings
        import torch
        from .sasrec_model import SasrecModel
        from .sasrec_seq import SasrecRetriever
        ec = extra_config or {}
        model_dir = os.path.join(cache_dir, "retrieval_v2", "sasrec",
                                 ec.get("model_dir", "sasrec_v1"))
        # New checkpoints save item_feats as a torch tensor + track_ids as a
        # list[str], both safe under PyTorch 2.6's weights_only=True. Older
        # checkpoints (numpy item_feats) trip the safe loader's unpickler, so
        # we fall back to weights_only=False with a warning to nudge a retrain.
        ckpt_path = os.path.join(model_dir, "sasrec.pt")
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError:
            warnings.warn(
                "legacy SASRec checkpoint loaded weights_only=False; "
                "retrain to refresh", stacklevel=2)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = SasrecModel(**ckpt["model_kwargs"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        item_feats = torch.as_tensor(ckpt["item_feats"], dtype=torch.float32)
        track_ids = ckpt["track_ids"]
        with torch.no_grad():
            item_repr = model.item_fusion(item_feats)
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(ec.get("ctx_model", "BAAI/bge-base-en-v1.5"))
        st.max_seq_length = 512
        try:
            st.tokenizer.truncation_side = "left"  # keep most-recent turns
        except Exception:
            pass

        def _text_encode(texts):
            return st.encode(list(texts), convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)

        return SasrecRetriever(model, item_repr, track_ids, item_feats, _text_encode,
                               max_len=int(ec.get("max_len", 50)))
    elif retrieval_type == "two_tower":
        # Two-tower content channel: learned query head + ItemFusion over the 5
        # frozen catalog modalities. Loads the trained model + precomputes the
        # item-repr matrix; the query is encoded by the SAME Qwen3 encoder the
        # dense channel uses (the contrastive heads align the spaces).
        import os
        import torch
        from .two_tower_model import TwoTowerModel
        from .two_tower_channel import TwoTowerRetriever
        ec = extra_config or {}
        model_dir = os.path.join(cache_dir, "retrieval_v2", "two_tower",
                                 ec.get("model_dir", "two_tower_v1"))
        ckpt = torch.load(os.path.join(model_dir, "two_tower.pt"),
                          map_location="cpu", weights_only=False)
        model = TwoTowerModel(**ckpt["model_kwargs"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        item_feats = torch.as_tensor(ckpt["item_feats"], dtype=torch.float32)
        track_ids = ckpt["track_ids"]
        with torch.no_grad():
            item_repr = model.encode_item(item_feats)
        qenc = DENSE_PRECOMPUTED(
            dataset_name, track_split_types, corpus_types, cache_dir,
            embed_col="metadata-qwen3_embedding_0.6b",
            instruct=QWEN3_MUSIC_INSTRUCT, instruct_label="instruct-music-v1")

        def _query_encode(texts):
            return qenc._encode_queries(list(texts))

        return TwoTowerRetriever(model, item_repr, track_ids, _query_encode)
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
