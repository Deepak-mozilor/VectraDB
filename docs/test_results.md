# VectraDB Test Results

**Date:** 2026-03-22
**Branch:** `feat/gpu-processing`
**Rust:** stable
**Platform:** macOS (Darwin 25.3.0)

## Summary

| Status | Count |
|--------|-------|
| **Passed** | 171 |
| **Failed** | 0 |
| **Ignored** | 2 |
| **Total** | 173 |

**Result: ALL TESTS PASSED**

---

## Per-Crate Breakdown

### vectradb-api (16 tests) — ALL PASSED

| Test | Status |
|------|--------|
| `auth::tests::test_write_detection` | PASS |
| `auth::tests::test_no_keys_means_disabled` | PASS |
| `auth::tests::test_auth_disabled` | PASS |
| `auth::tests::test_invalid_key_rejected` | PASS |
| `auth::tests::test_readonly_key_cannot_write` | PASS |
| `auth::tests::test_multiple_keys` | PASS |
| `auth::tests::test_admin_key_has_full_access` | PASS |
| `metrics::tests::test_path_normalization` | PASS |
| `rate_limit::tests::test_rate_limiter_disabled` | PASS |
| `rate_limit::tests::test_rate_limiter_per_ip_isolation` | PASS |
| `rate_limit::tests::test_rate_limiter_allows_within_limit` | PASS |
| `rate_limit::tests::test_rate_limiter_blocks_over_limit` | PASS |
| `rate_limit::tests::test_cleanup_stale_entries` | PASS |
| `rate_limit::tests::test_rate_limiter_refills_over_time` | PASS |
| `tests::test_search_request` | PASS |
| `tests::test_create_vector_request` | PASS |

### vectradb-chunkers (22 tests) — ALL PASSED

| Test | Status |
|------|--------|
| `document::tests::test_chunk_by_sentences` | PASS |
| `document::tests::test_metadata_creation` | PASS |
| `document::tests::test_document_chunker_creation` | PASS |
| `document::tests::test_chunk_by_paragraphs` | PASS |
| `markdown::tests::test_chunk_by_semantic_blocks` | PASS |
| `markdown::tests::test_chunk_by_headings` | PASS |
| `markdown::tests::test_extract_heading_text` | PASS |
| `markdown::tests::test_markdown_chunker_creation` | PASS |
| `markdown::tests::test_markdown_metadata` | PASS |
| `code::tests::test_language_detection` | PASS |
| `code::tests::test_code_metadata` | PASS |
| `code::tests::test_chunk_by_logical_blocks` | PASS |
| `code::tests::test_code_chunker_creation` | PASS |
| `production::tests::test_adaptive_chunking` | PASS |
| `production::tests::test_production_chunker_creation` | PASS |
| `production::tests::test_content_analysis` | PASS |
| `production::tests::test_quality_scoring` | PASS |
| `production::tests::test_syllable_counting` | PASS |
| `tests::test_chunking_config_default` | PASS |
| `tests::test_find_semantic_boundaries` | PASS |
| `tests::test_split_with_overlap` | PASS |
| `tests::test_chunker_factory` | PASS |

### vectradb-components (30 tests) — ALL PASSED

| Test | Status |
|------|--------|
| `filter::tests::test_from_clauses_empty_returns_none` | PASS |
| `filter::tests::test_from_clauses_must_and_must_not` | PASS |
| `filter::tests::test_not_equals` | PASS |
| `filter::tests::test_and_filter` | PASS |
| `filter::tests::test_equals` | PASS |
| `filter::tests::test_or_filter` | PASS |
| `filter::tests::test_in` | PASS |
| `filter::tests::test_exists_and_not_exists` | PASS |
| `filter::tests::test_nested_and_or` | PASS |
| `indexing::tests::test_hash_function` | PASS |
| `indexing::tests::test_linear_index` | PASS |
| `indexing::tests::test_hash_index` | PASS |
| `similarity::tests::test_cosine_similarity` | PASS |
| `similarity::tests::test_euclidean_distance` | PASS |
| `similarity::tests::test_find_similar_vectors` | PASS |
| `storage::tests::test_in_memory_db_creation` | PASS |
| `storage::tests::test_in_memory_db_dimension_check` | PASS |
| `storage::tests::test_in_memory_db_stats` | PASS |
| `storage::tests::test_in_memory_db_upsert` | PASS |
| `tensor::tests::test_cosine_sim` | PASS |
| `tensor::tests::test_cross_correlation` | PASS |
| `tensor::tests::test_dot_product` | PASS |
| `tensor::tests::test_tensor_creation` | PASS |
| `tensor::tests::test_tensor_indexing` | PASS |
| `tensor::tests::test_tensor_row_2d` | PASS |
| `tensor::tests::test_tensor_shape_mismatch` | PASS |
| `tensor::tests::test_tensor_subtensor_dim0` | PASS |
| `vector_operations::tests::test_create_vector_document` | PASS |
| `vector_operations::tests::test_normalize_vector` | PASS |
| `vector_operations::tests::test_validate_vector` | PASS |

### vectradb-eval (10 tests) — ALL PASSED

| Test | Status |
|------|--------|
| `tests::test_recall_at_k` | PASS |
| `tests::test_precision_at_k` | PASS |
| `tests::test_reciprocal_rank` | PASS |
| `tests::test_reciprocal_rank_first` | PASS |
| `tests::test_reciprocal_rank_not_found` | PASS |
| `tests::test_ndcg_perfect` | PASS |
| `tests::test_ndcg_reversed` | PASS |
| `tests::test_average_precision` | PASS |
| `tests::test_evaluate_aggregate` | PASS |
| `tests::test_empty_relevant` | PASS |

### vectradb-search (32 tests) — ALL PASSED

| Test | Status |
|------|--------|
| `hnsw::tests::test_hnsw_creation` | PASS |
| `hnsw::tests::test_hnsw_dimension_mismatch` | PASS |
| `hnsw::tests::test_hnsw_insert_and_search` | PASS |
| `es4d::tests::test_distance_det_no_termination` | PASS |
| `es4d::tests::test_distance_det_early_termination` | PASS |
| `es4d::tests::test_dimension_reordering` | PASS |
| `es4d::tests::test_reorder_vector` | PASS |
| `es4d::tests::test_es4d_dimension_mismatch` | PASS |
| `es4d::tests::test_es4d_insert_and_search` | PASS |
| `es4d::tests::test_es4d_remove` | PASS |
| `es4d::tests::test_kmeans_basic` | PASS |
| `es4d::tests::test_es4d_build_index_full` | PASS |
| `lsh::tests::test_hamming_distance` | PASS |
| `lsh::tests::test_lsh_creation` | PASS |
| `lsh::tests::test_lsh_hash_signature` | PASS |
| `lsh::tests::test_lsh_insert_and_search` | PASS |
| `pq::tests::test_pq_creation` | PASS |
| `pq::tests::test_pq_encode_decode` | PASS |
| `pq::tests::test_pq_insert_and_search` | PASS |
| `simd::tests::test_cosine_identical` | PASS |
| `simd::tests::test_cosine_orthogonal` | PASS |
| `simd::tests::test_dot_product_known` | PASS |
| `simd::tests::test_l2_distance_known` | PASS |
| `simd::tests::test_l2_squared_known` | PASS |
| `simd::tests::test_high_dimensional` | PASS |
| `simd::tests::test_non_aligned_lengths` | PASS |
| `simd::tests::test_zero_vectors` | PASS |
| `tensor::tests::test_basic_search` | PASS |
| `tensor::tests::test_crud_operations` | PASS |
| `tensor::tests::test_shifting_search` | PASS |
| `tensor::tests::test_shifting_search_cosine_exact_match` | PASS |
| `tensor::tests::test_weighted_similarity` | PASS |

### vectradb-storage (2 tests) — ALL PASSED

| Test | Status |
|------|--------|
| `tests::test_persistent_db_creation` | PASS |
| `tests::test_persistent_db_operations` | PASS |

### vectradb-tfidf (19 tests) — ALL PASSED

| Test | Status |
|------|--------|
| `scoring::tests::test_tf_raw` | PASS |
| `scoring::tests::test_tf_sublinear` | PASS |
| `scoring::tests::test_idf` | PASS |
| `scoring::tests::test_idf_smooth` | PASS |
| `tokenizer::tests::test_simple_tokenizer_basic` | PASS |
| `tokenizer::tests::test_simple_tokenizer_punctuation` | PASS |
| `tokenizer::tests::test_simple_tokenizer_stop_words` | PASS |
| `tokenizer::tests::test_simple_tokenizer_min_length` | PASS |
| `tokenizer::tests::test_empty_input` | PASS |
| `tests::test_add_and_count` | PASS |
| `tests::test_remove_document` | PASS |
| `tests::test_search_relevance` | PASS |
| `tests::test_search_machine_learning` | PASS |
| `tests::test_search_by_threshold` | PASS |
| `tests::test_search_no_match` | PASS |
| `tests::test_matched_terms` | PASS |
| `tests::test_tfidf_vector` | PASS |
| `tests::test_reindex_document` | PASS |
| `tests::test_empty_index` | PASS |

### vectradb-py (1 test) — ALL PASSED

| Test | Status |
|------|--------|
| `tests::test_py_vector_document_conversion` | PASS |

### stress_tests (39 tests) — ALL PASSED

| Test | Status |
|------|--------|
| `test_cosine_identical_vectors` | PASS |
| `test_cosine_orthogonal_vectors` | PASS |
| `test_cosine_opposite_vectors` | PASS |
| `test_euclidean_known_value` | PASS |
| `test_euclidean_triangle_inequality` | PASS |
| `test_manhattan_known_value` | PASS |
| `test_dot_product_known_value` | PASS |
| `test_normalize_produces_unit_vector` | PASS |
| `test_validate_vector_rejects_nan` | PASS |
| `test_validate_vector_rejects_infinity` | PASS |
| `test_validate_vector_rejects_empty` | PASS |
| `test_zero_vector_cosine` | PASS |
| `test_single_dimension_vectors` | PASS |
| `test_dimension_mismatch_returns_error` | PASS |
| `test_dimension_enforcement` | PASS |
| `test_inmemory_crud_1000_vectors` | PASS |
| `test_upsert_creates_and_updates` | PASS |
| `test_duplicate_vector_id_returns_error` | PASS |
| `test_delete_nonexistent_vector` | PASS |
| `test_get_nonexistent_vector` | PASS |
| `test_search_recall_comparison_all_algorithms` | PASS |
| `test_all_algorithms_return_sorted_results` | PASS |
| `test_top_1_is_nearest_neighbor` | PASS |
| `test_high_dimensional_512` | PASS |
| `test_search_performance_benchmark` | PASS |
| `test_algorithm_insert_delete_consistency` | PASS |
| `test_concurrent_reads_and_writes` | PASS |
| `test_hnsw_heavy_churn` | PASS |
| `test_persistent_storage_survives_restart` | PASS |
| `test_persistent_db_search_with_filter` | PASS |
| `test_filter_condition_all_variants` | PASS |
| `test_filter_complex_boolean_logic` | PASS |
| `test_es4d_build_index_then_incremental_insert` | PASS |
| `test_es4d_det_effectiveness_high_dim` | PASS |
| `test_tensor_basic_search_finds_exact_match` | PASS |
| `test_tensor_shifting_search_finds_embedded_signal` | PASS |
| `test_tensor_cross_correlation_detects_scaled_copy` | PASS |
| `test_tensor_weighted_similarity_emphasizes_first_dimension` | PASS |
| `test_tensor_subtensor_extraction_correctness` | PASS |

### Other Crates (0 unit tests, run only doc-tests)

| Crate | Unit Tests | Doc Tests |
|-------|-----------|-----------|
| `vectradb-embeddings` | 0 | 0 |
| `vectradb-rag` | 0 | 0 |
| `vectradb-server` | 0 | 0 |

---

## Test Categories

| Category | Tests | Description |
|----------|-------|-------------|
| **Similarity Metrics** | 15 | Cosine, Euclidean, Manhattan, dot product correctness |
| **Vector Operations** | 8 | Validation, normalization, document creation |
| **Search Algorithms** | 32 | HNSW, ES4D, LSH, PQ insert/search/remove, SIMD distance |
| **Metadata Filtering** | 11 | Boolean logic (AND/OR/NOT), equals, in, exists |
| **Storage & Persistence** | 4 | Sled KV store, restart survival, CRUD operations |
| **Authentication** | 7 | API keys, roles (admin/read-only), disabled mode |
| **Rate Limiting** | 6 | Token bucket, per-IP isolation, refill, cleanup |
| **TF-IDF** | 19 | Tokenizer, scoring (TF/IDF), inverted index, search |
| **Evaluation Metrics** | 10 | Recall@k, Precision@k, MRR, NDCG, MAP |
| **Chunking** | 22 | Document, markdown, code chunking, quality scoring |
| **Tensor Operations** | 12 | Multi-dimensional tensors, shifting search, cross-correlation |
| **Stress & Integration** | 39 | Scale (1000+ vectors), concurrency, recall comparison, heavy churn |
| **Python Bindings** | 1 | PyO3 vector document conversion |
