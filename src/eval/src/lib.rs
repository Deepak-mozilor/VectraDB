use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Ground truth for a single query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryGroundTruth {
    pub query_id: String,
    /// Relevant document IDs (ordered by relevance, most relevant first).
    pub relevant_ids: Vec<String>,
    /// Optional graded relevance scores (for NDCG). Same order as `relevant_ids`.
    pub relevance_scores: Option<Vec<f32>>,
}

/// Evaluation result for a single query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryEvalResult {
    pub query_id: String,
    pub recall_at_k: f64,
    pub precision_at_k: f64,
    pub average_precision: f64,
    pub reciprocal_rank: f64,
    pub ndcg: f64,
    pub latency_ms: f64,
}

/// Aggregated evaluation report across all queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    pub num_queries: usize,
    pub k: usize,
    pub recall_at_k: f64,
    pub precision_at_k: f64,
    pub mrr: f64,
    pub ndcg: f64,
    pub map: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub latency_mean_ms: f64,
    pub per_query: Vec<QueryEvalResult>,
}

/// Evaluator provides IR evaluation metric computations.
pub struct Evaluator;

impl Evaluator {
    /// Compute all metrics for a set of queries.
    ///
    /// * `results` – for each query: `(retrieved_ids, ground_truth, latency_ms)`
    /// * `k` – the cutoff for @k metrics
    pub fn evaluate(results: &[(Vec<String>, QueryGroundTruth, f64)], k: usize) -> EvalReport {
        let mut per_query = Vec::with_capacity(results.len());
        let mut latencies = Vec::with_capacity(results.len());

        for (retrieved, gt, latency_ms) in results {
            let relevance_map = Self::build_relevance_map(gt);

            let recall = Self::recall_at_k(retrieved, &gt.relevant_ids, k);
            let precision = Self::precision_at_k(retrieved, &gt.relevant_ids, k);
            let ap = Self::average_precision(retrieved, &gt.relevant_ids);
            let rr = Self::reciprocal_rank(retrieved, &gt.relevant_ids);
            let ndcg = Self::ndcg(retrieved, &relevance_map, k);

            latencies.push(*latency_ms);

            per_query.push(QueryEvalResult {
                query_id: gt.query_id.clone(),
                recall_at_k: recall,
                precision_at_k: precision,
                average_precision: ap,
                reciprocal_rank: rr,
                ndcg,
                latency_ms: *latency_ms,
            });
        }

        let n = per_query.len() as f64;
        let recall = per_query.iter().map(|q| q.recall_at_k).sum::<f64>() / n;
        let precision = per_query.iter().map(|q| q.precision_at_k).sum::<f64>() / n;
        let mrr = per_query.iter().map(|q| q.reciprocal_rank).sum::<f64>() / n;
        let ndcg = per_query.iter().map(|q| q.ndcg).sum::<f64>() / n;
        let map = per_query.iter().map(|q| q.average_precision).sum::<f64>() / n;
        let latency_mean = latencies.iter().sum::<f64>() / n;

        latencies.sort_by(|a, b| a.total_cmp(b));
        let p50 = percentile(&latencies, 50.0);
        let p95 = percentile(&latencies, 95.0);
        let p99 = percentile(&latencies, 99.0);

        EvalReport {
            num_queries: results.len(),
            k,
            recall_at_k: recall,
            precision_at_k: precision,
            mrr,
            ndcg,
            map,
            latency_p50_ms: p50,
            latency_p95_ms: p95,
            latency_p99_ms: p99,
            latency_mean_ms: latency_mean,
            per_query,
        }
    }

    /// Recall@k: fraction of relevant documents found in the top-k results.
    pub fn recall_at_k(retrieved: &[String], relevant: &[String], k: usize) -> f64 {
        if relevant.is_empty() {
            return 0.0;
        }
        let relevant_set: HashSet<&String> = relevant.iter().collect();
        let found = retrieved
            .iter()
            .take(k)
            .filter(|id| relevant_set.contains(id))
            .count();
        found as f64 / relevant.len() as f64
    }

    /// Precision@k: fraction of top-k results that are relevant.
    pub fn precision_at_k(retrieved: &[String], relevant: &[String], k: usize) -> f64 {
        if k == 0 {
            return 0.0;
        }
        let relevant_set: HashSet<&String> = relevant.iter().collect();
        let top_k: Vec<_> = retrieved.iter().take(k).collect();
        let found = top_k.iter().filter(|id| relevant_set.contains(*id)).count();
        found as f64 / top_k.len() as f64
    }

    /// Reciprocal Rank: 1/position of the first relevant result.
    pub fn reciprocal_rank(retrieved: &[String], relevant: &[String]) -> f64 {
        let relevant_set: HashSet<&String> = relevant.iter().collect();
        for (i, id) in retrieved.iter().enumerate() {
            if relevant_set.contains(id) {
                return 1.0 / (i + 1) as f64;
            }
        }
        0.0
    }

    /// NDCG@k: Normalized Discounted Cumulative Gain.
    ///
    /// Uses graded relevance from `relevance_map`. Documents not in the map
    /// have relevance 0.
    pub fn ndcg(retrieved: &[String], relevance_map: &HashMap<String, f32>, k: usize) -> f64 {
        let dcg = Self::dcg(retrieved, relevance_map, k);
        let idcg = Self::ideal_dcg(relevance_map, k);
        if idcg == 0.0 {
            return 0.0;
        }
        dcg / idcg
    }

    /// Average Precision: mean of precision values at each relevant result position.
    pub fn average_precision(retrieved: &[String], relevant: &[String]) -> f64 {
        if relevant.is_empty() {
            return 0.0;
        }
        let relevant_set: HashSet<&String> = relevant.iter().collect();
        let mut hits = 0;
        let mut sum_precision = 0.0;

        for (i, id) in retrieved.iter().enumerate() {
            if relevant_set.contains(id) {
                hits += 1;
                sum_precision += hits as f64 / (i + 1) as f64;
            }
        }

        sum_precision / relevant.len() as f64
    }

    // -- internal helpers --

    fn dcg(retrieved: &[String], relevance_map: &HashMap<String, f32>, k: usize) -> f64 {
        retrieved
            .iter()
            .take(k)
            .enumerate()
            .map(|(i, id)| {
                let rel = *relevance_map.get(id).unwrap_or(&0.0) as f64;
                (2.0_f64.powf(rel) - 1.0) / (i as f64 + 2.0).log2()
            })
            .sum()
    }

    fn ideal_dcg(relevance_map: &HashMap<String, f32>, k: usize) -> f64 {
        let mut scores: Vec<f32> = relevance_map.values().copied().collect();
        scores.sort_by(|a, b| b.total_cmp(a));
        scores
            .iter()
            .take(k)
            .enumerate()
            .map(|(i, &rel)| {
                let rel = rel as f64;
                (2.0_f64.powf(rel) - 1.0) / (i as f64 + 2.0).log2()
            })
            .sum()
    }

    fn build_relevance_map(gt: &QueryGroundTruth) -> HashMap<String, f32> {
        let mut map = HashMap::new();
        if let Some(scores) = &gt.relevance_scores {
            for (id, &score) in gt.relevant_ids.iter().zip(scores.iter()) {
                map.insert(id.clone(), score);
            }
        } else {
            // Binary relevance: all relevant docs get score 1.0
            for id in &gt.relevant_ids {
                map.insert(id.clone(), 1.0);
            }
        }
        map
    }
}

/// Compute the p-th percentile from a sorted slice.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_at_k() {
        let retrieved = vec!["a", "b", "c", "d", "e"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let relevant = vec!["a", "c", "f", "g", "h"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();

        // 2 of 5 relevant found in top-5
        let r = Evaluator::recall_at_k(&retrieved, &relevant, 5);
        assert!((r - 0.4).abs() < 1e-9);

        // top-2: only "a" is relevant
        let r2 = Evaluator::recall_at_k(&retrieved, &relevant, 2);
        assert!((r2 - 0.2).abs() < 1e-9);
    }

    #[test]
    fn test_precision_at_k() {
        let retrieved = vec!["a", "b", "c", "d", "e"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let relevant = vec!["a", "c", "f"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();

        // 2 relevant in top-5 => 2/5
        let p = Evaluator::precision_at_k(&retrieved, &relevant, 5);
        assert!((p - 0.4).abs() < 1e-9);

        // 1 relevant in top-2 => 1/2
        let p2 = Evaluator::precision_at_k(&retrieved, &relevant, 2);
        assert!((p2 - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reciprocal_rank() {
        let retrieved: Vec<String> = vec!["x", "y", "a", "b"]
            .into_iter()
            .map(String::from)
            .collect();
        let relevant: Vec<String> = vec!["a", "b"].into_iter().map(String::from).collect();

        // First relevant at position 3 => 1/3
        let rr = Evaluator::reciprocal_rank(&retrieved, &relevant);
        assert!((rr - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_reciprocal_rank_first() {
        let retrieved: Vec<String> = vec!["a", "b", "c"].into_iter().map(String::from).collect();
        let relevant: Vec<String> = vec!["a"].into_iter().map(String::from).collect();
        let rr = Evaluator::reciprocal_rank(&retrieved, &relevant);
        assert!((rr - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_reciprocal_rank_not_found() {
        let retrieved: Vec<String> = vec!["x", "y"].into_iter().map(String::from).collect();
        let relevant: Vec<String> = vec!["a"].into_iter().map(String::from).collect();
        let rr = Evaluator::reciprocal_rank(&retrieved, &relevant);
        assert!((rr - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_ndcg_perfect() {
        // Retrieved in perfect order
        let retrieved: Vec<String> = vec!["a", "b", "c"].into_iter().map(String::from).collect();
        let mut relevance = HashMap::new();
        relevance.insert("a".to_string(), 3.0);
        relevance.insert("b".to_string(), 2.0);
        relevance.insert("c".to_string(), 1.0);

        let ndcg = Evaluator::ndcg(&retrieved, &relevance, 3);
        assert!(
            (ndcg - 1.0).abs() < 1e-9,
            "perfect order should give NDCG=1.0"
        );
    }

    #[test]
    fn test_ndcg_reversed() {
        // Retrieved in worst order
        let retrieved: Vec<String> = vec!["c", "b", "a"].into_iter().map(String::from).collect();
        let mut relevance = HashMap::new();
        relevance.insert("a".to_string(), 3.0);
        relevance.insert("b".to_string(), 2.0);
        relevance.insert("c".to_string(), 1.0);

        let ndcg = Evaluator::ndcg(&retrieved, &relevance, 3);
        assert!(ndcg < 1.0, "reversed order should give NDCG < 1.0");
        assert!(ndcg > 0.0, "all relevant docs present so NDCG > 0.0");
    }

    #[test]
    fn test_average_precision() {
        // Retrieved: [rel, non, rel, non, rel]
        let retrieved: Vec<String> = vec!["a", "x", "b", "y", "c"]
            .into_iter()
            .map(String::from)
            .collect();
        let relevant: Vec<String> = vec!["a", "b", "c"].into_iter().map(String::from).collect();

        // AP = (1/1 + 2/3 + 3/5) / 3 = (1 + 0.667 + 0.6) / 3 ≈ 0.7556
        let ap = Evaluator::average_precision(&retrieved, &relevant);
        assert!((ap - 0.7556).abs() < 0.001);
    }

    #[test]
    fn test_evaluate_aggregate() {
        let results = vec![
            (
                vec!["a".to_string(), "b".to_string()],
                QueryGroundTruth {
                    query_id: "q1".to_string(),
                    relevant_ids: vec!["a".to_string()],
                    relevance_scores: None,
                },
                1.0, // latency
            ),
            (
                vec!["x".to_string(), "a".to_string()],
                QueryGroundTruth {
                    query_id: "q2".to_string(),
                    relevant_ids: vec!["a".to_string()],
                    relevance_scores: None,
                },
                2.0,
            ),
        ];

        let report = Evaluator::evaluate(&results, 2);
        assert_eq!(report.num_queries, 2);
        assert_eq!(report.k, 2);
        // q1 recall=1.0, q2 recall=1.0 => mean 1.0
        assert!((report.recall_at_k - 1.0).abs() < 1e-9);
        // q1 precision=0.5, q2 precision=0.5 => mean 0.5
        assert!((report.precision_at_k - 0.5).abs() < 1e-9);
        // q1 RR=1.0, q2 RR=0.5 => MRR=0.75
        assert!((report.mrr - 0.75).abs() < 1e-9);
        assert!((report.latency_mean_ms - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_empty_relevant() {
        let retrieved: Vec<String> = vec!["a".to_string()];
        let relevant: Vec<String> = vec![];
        assert_eq!(Evaluator::recall_at_k(&retrieved, &relevant, 5), 0.0);
        assert_eq!(Evaluator::average_precision(&retrieved, &relevant), 0.0);
        assert_eq!(Evaluator::reciprocal_rank(&retrieved, &relevant), 0.0);
    }
}
