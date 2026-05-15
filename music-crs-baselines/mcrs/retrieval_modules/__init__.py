from .bm25 import BM25_MODEL
from .bert import BERT_MODEL
from .dense_precomputed import DENSE_PRECOMPUTED
from .dense_local import DENSE_LOCAL
from .rrf import RRF_MODEL
from .sequential_rerank import SEQUENTIAL_RERANK


# Qwen3-Embedding standard instruction prefix for asymmetric retrieval.
# Queries are wrapped with this prefix before encoding; docs were embedded
# as-is offline by the challenge dataset provider.
QWEN3_MUSIC_INSTRUCT = (
    "Instruct: Given a conversational music request, "
    "retrieve tracks most relevant to the user's intent.\nQuery: "
)


def load_retrieval_module(
        retrieval_type: str,
        dataset_name: str,
        track_split_types: list[str],
        corpus_types: list[str] = ["track_name", "artist_name", "album_name"],
        cache_dir: str = "./cache"
    ):
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
