//! TF-IDF sparse text retrieval engine for VectraDB.
//!
//! Provides an inverted index with configurable tokenization and TF-IDF scoring.
//! Designed to complement dense vector search for hybrid retrieval.

pub mod scoring;
pub mod tokenizer;

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tokenizer::{SimpleTokenizer, Tokenizer};

/// TF-IDF index configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TfIdfConfig {
    /// Minimum document frequency — terms appearing in fewer docs are ignored.
    pub min_df: usize,
    /// Maximum document frequency ratio (0.0–1.0). Terms in more than this
    /// fraction of docs are ignored (e.g., 0.95 = skip terms in >95% of docs).
    pub max_df_ratio: f32,
    /// Use sublinear TF: `1 + log(tf)` instead of raw count.
    pub sublinear_tf: bool,
}

impl Default for TfIdfConfig {
    fn default() -> Self {
        Self {
            min_df: 1,
            max_df_ratio: 0.95,
            sublinear_tf: true,
        }
    }
}

/// A posting list entry: one document's data for a term.
#[derive(Debug, Clone)]
struct PostingEntry {
    doc_id: String,
    term_freq: usize,
}

/// Sparse search result from TF-IDF retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseSearchResult {
    pub id: String,
    pub score: f32,
    pub matched_terms: Vec<String>,
}

/// TF-IDF inverted index for sparse text retrieval.
pub struct TfIdfIndex {
    /// term -> list of (doc_id, tf)
    inverted_index: HashMap<String, Vec<PostingEntry>>,
    /// doc_id -> L2 norm of its TF-IDF vector
    doc_norms: HashMap<String, f32>,
    /// doc_id -> set of terms in that document
    doc_terms: HashMap<String, HashSet<String>>,
    /// Total number of indexed documents
    doc_count: usize,
    config: TfIdfConfig,
    tokenizer: Box<dyn Tokenizer>,
}

impl TfIdfIndex {
    /// Create a new TF-IDF index with the given config and default tokenizer.
    pub fn new(config: TfIdfConfig) -> Self {
        Self {
            inverted_index: HashMap::new(),
            doc_norms: HashMap::new(),
            doc_terms: HashMap::new(),
            doc_count: 0,
            config,
            tokenizer: Box::new(SimpleTokenizer::default()),
        }
    }

    /// Create a new TF-IDF index with a custom tokenizer.
    pub fn with_tokenizer(config: TfIdfConfig, tokenizer: Box<dyn Tokenizer>) -> Self {
        Self {
            inverted_index: HashMap::new(),
            doc_norms: HashMap::new(),
            doc_terms: HashMap::new(),
            doc_count: 0,
            config,
            tokenizer,
        }
    }

    /// Index a document. If the document ID already exists, it is re-indexed.
    pub fn add_document(&mut self, id: &str, text: &str) {
        // Remove existing document first
        if self.doc_terms.contains_key(id) {
            self.remove_document(id);
        }

        let tokens = self.tokenizer.tokenize(text);

        // Count term frequencies
        let mut term_counts: HashMap<String, usize> = HashMap::new();
        for token in &tokens {
            *term_counts.entry(token.clone()).or_insert(0) += 1;
        }

        // Store terms for this document
        let term_set: HashSet<String> = term_counts.keys().cloned().collect();
        self.doc_terms.insert(id.to_string(), term_set);

        // Add to inverted index
        for (term, count) in &term_counts {
            self.inverted_index
                .entry(term.clone())
                .or_default()
                .push(PostingEntry {
                    doc_id: id.to_string(),
                    term_freq: *count,
                });
        }

        self.doc_count += 1;

        // Recompute norm for this document
        self.recompute_doc_norm(id);
    }

    /// Remove a document from the index.
    pub fn remove_document(&mut self, id: &str) {
        if let Some(terms) = self.doc_terms.remove(id) {
            for term in &terms {
                if let Some(postings) = self.inverted_index.get_mut(term) {
                    postings.retain(|p| p.doc_id != id);
                    if postings.is_empty() {
                        self.inverted_index.remove(term);
                    }
                }
            }
            self.doc_norms.remove(id);
            self.doc_count -= 1;
        }
    }

    /// Search for documents matching the query text, returning top-k results.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<SparseSearchResult> {
        let query_tfidf = self.compute_tfidf_vector(query);
        if query_tfidf.is_empty() {
            return vec![];
        }

        let query_norm = l2_norm_sparse(&query_tfidf);
        if query_norm == 0.0 {
            return vec![];
        }

        let mut scores = self.compute_scores(&query_tfidf, query_norm);
        scores.sort_by(|a, b| b.score.total_cmp(&a.score));
        scores.truncate(top_k);
        scores
    }

    /// Search by score threshold: returns all results with score >= min_score.
    pub fn search_by_threshold(
        &self,
        query: &str,
        min_score: f32,
        max_results: usize,
    ) -> Vec<SparseSearchResult> {
        let query_tfidf = self.compute_tfidf_vector(query);
        if query_tfidf.is_empty() {
            return vec![];
        }

        let query_norm = l2_norm_sparse(&query_tfidf);
        if query_norm == 0.0 {
            return vec![];
        }

        let mut scores = self.compute_scores(&query_tfidf, query_norm);
        scores.retain(|r| r.score >= min_score);
        scores.sort_by(|a, b| b.score.total_cmp(&a.score));
        scores.truncate(max_results);
        scores
    }

    /// Get the TF-IDF sparse vector for a text string.
    pub fn compute_tfidf_vector(&self, text: &str) -> HashMap<String, f32> {
        let tokens = self.tokenizer.tokenize(text);
        let mut term_counts: HashMap<String, usize> = HashMap::new();
        for token in &tokens {
            *term_counts.entry(token.clone()).or_insert(0) += 1;
        }

        let mut tfidf = HashMap::new();
        for (term, count) in &term_counts {
            let df = self
                .inverted_index
                .get(term)
                .map_or(0, |postings| postings.len());

            // Apply min_df and max_df_ratio filters
            if df < self.config.min_df {
                continue;
            }
            if self.doc_count > 0 && (df as f32 / self.doc_count as f32) > self.config.max_df_ratio
            {
                continue;
            }

            let tf = if self.config.sublinear_tf {
                scoring::tf_sublinear(*count)
            } else {
                scoring::tf_raw(*count)
            };
            let idf = scoring::idf_smooth(self.doc_count, df);

            tfidf.insert(term.clone(), tf * idf);
        }

        tfidf
    }

    /// Number of indexed documents.
    pub fn document_count(&self) -> usize {
        self.doc_count
    }

    /// Number of unique terms in the index.
    pub fn vocabulary_size(&self) -> usize {
        self.inverted_index.len()
    }

    // --- internal helpers ---

    fn compute_scores(
        &self,
        query_tfidf: &HashMap<String, f32>,
        query_norm: f32,
    ) -> Vec<SparseSearchResult> {
        // Accumulate dot products per document
        let mut doc_dots: HashMap<String, f32> = HashMap::new();
        let mut doc_matched_terms: HashMap<String, Vec<String>> = HashMap::new();

        for (term, query_weight) in query_tfidf {
            if let Some(postings) = self.inverted_index.get(term) {
                let df = postings.len();
                let idf = scoring::idf_smooth(self.doc_count, df);

                for entry in postings {
                    let doc_tf = if self.config.sublinear_tf {
                        scoring::tf_sublinear(entry.term_freq)
                    } else {
                        scoring::tf_raw(entry.term_freq)
                    };
                    let doc_weight = doc_tf * idf;

                    *doc_dots.entry(entry.doc_id.clone()).or_insert(0.0) +=
                        query_weight * doc_weight;
                    doc_matched_terms
                        .entry(entry.doc_id.clone())
                        .or_default()
                        .push(term.clone());
                }
            }
        }

        // Compute cosine similarity
        doc_dots
            .into_iter()
            .map(|(doc_id, dot)| {
                let doc_norm = self.doc_norms.get(&doc_id).copied().unwrap_or(1.0);
                let score = if doc_norm > 0.0 && query_norm > 0.0 {
                    dot / (doc_norm * query_norm)
                } else {
                    0.0
                };
                let matched_terms = doc_matched_terms.remove(&doc_id).unwrap_or_default();
                SparseSearchResult {
                    id: doc_id,
                    score,
                    matched_terms,
                }
            })
            .collect()
    }

    fn recompute_doc_norm(&mut self, doc_id: &str) {
        let terms = match self.doc_terms.get(doc_id) {
            Some(t) => t,
            None => return,
        };

        let mut norm_sq = 0.0_f32;
        for term in terms {
            if let Some(postings) = self.inverted_index.get(term) {
                let df = postings.len();
                let idf = scoring::idf_smooth(self.doc_count, df);

                if let Some(entry) = postings.iter().find(|p| p.doc_id == doc_id) {
                    let tf = if self.config.sublinear_tf {
                        scoring::tf_sublinear(entry.term_freq)
                    } else {
                        scoring::tf_raw(entry.term_freq)
                    };
                    let weight = tf * idf;
                    norm_sq += weight * weight;
                }
            }
        }

        self.doc_norms.insert(doc_id.to_string(), norm_sq.sqrt());
    }
}

fn l2_norm_sparse(v: &HashMap<String, f32>) -> f32 {
    v.values().map(|w| w * w).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_index() -> TfIdfIndex {
        let mut idx = TfIdfIndex::new(TfIdfConfig::default());
        idx.add_document("d1", "the quick brown fox jumps over the lazy dog");
        idx.add_document("d2", "machine learning and deep learning for NLP");
        idx.add_document("d3", "the brown fox is quick and jumps high");
        idx.add_document("d4", "vector database for machine learning embeddings");
        idx.add_document("d5", "deep neural networks learn representations");
        idx
    }

    #[test]
    fn test_add_and_count() {
        let idx = test_index();
        assert_eq!(idx.document_count(), 5);
        assert!(idx.vocabulary_size() > 0);
    }

    #[test]
    fn test_remove_document() {
        let mut idx = test_index();
        assert_eq!(idx.document_count(), 5);
        idx.remove_document("d1");
        assert_eq!(idx.document_count(), 4);
        // Searching should not return d1
        let results = idx.search("quick brown fox", 10);
        assert!(results.iter().all(|r| r.id != "d1"));
    }

    #[test]
    fn test_search_relevance() {
        let idx = test_index();
        let results = idx.search("quick brown fox", 5);
        assert!(!results.is_empty());
        // d1 and d3 both contain "quick brown fox" terms
        let top_ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(
            top_ids.contains(&"d1") || top_ids.contains(&"d3"),
            "Expected d1 or d3 in results"
        );
    }

    #[test]
    fn test_search_machine_learning() {
        let idx = test_index();
        let results = idx.search("machine learning", 5);
        let top_ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(top_ids.contains(&"d2") || top_ids.contains(&"d4"));
    }

    #[test]
    fn test_search_by_threshold() {
        let idx = test_index();
        let results = idx.search_by_threshold("quick brown fox", 0.01, 100);
        assert!(!results.is_empty());
        for r in &results {
            assert!(r.score >= 0.01);
        }
    }

    #[test]
    fn test_search_no_match() {
        let idx = test_index();
        let results = idx.search("zzzzz qqqqq", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_matched_terms() {
        let idx = test_index();
        let results = idx.search("quick fox", 5);
        if let Some(r) = results.first() {
            assert!(!r.matched_terms.is_empty());
        }
    }

    #[test]
    fn test_tfidf_vector() {
        let idx = test_index();
        let vec = idx.compute_tfidf_vector("machine learning deep");
        assert!(!vec.is_empty());
        // "machine" and "learning" should have weights
        assert!(vec.contains_key("machine"));
        assert!(vec.contains_key("learning"));
    }

    #[test]
    fn test_reindex_document() {
        let mut idx = test_index();
        idx.add_document("d1", "completely different text about databases");
        assert_eq!(idx.document_count(), 5); // same count, re-indexed
        let results = idx.search("quick brown fox", 5);
        // d1 no longer matches "quick brown fox"
        let d1_result = results.iter().find(|r| r.id == "d1");
        assert!(d1_result.is_none() || d1_result.unwrap().score < 0.01);
    }

    #[test]
    fn test_empty_index() {
        let idx = TfIdfIndex::new(TfIdfConfig::default());
        let results = idx.search("anything", 5);
        assert!(results.is_empty());
    }
}
