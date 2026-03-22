/// Raw term frequency: count of term in document.
pub fn tf_raw(count: usize) -> f32 {
    count as f32
}

/// Sublinear term frequency: 1 + log(count). Returns 0 for count=0.
pub fn tf_sublinear(count: usize) -> f32 {
    if count == 0 {
        0.0
    } else {
        1.0 + (count as f32).ln()
    }
}

/// Inverse document frequency: log(N / df).
/// Returns 0 if df == 0.
pub fn idf(total_docs: usize, doc_freq: usize) -> f32 {
    if doc_freq == 0 {
        return 0.0;
    }
    (total_docs as f32 / doc_freq as f32).ln()
}

/// Smooth IDF variant: log(1 + N / df).
pub fn idf_smooth(total_docs: usize, doc_freq: usize) -> f32 {
    if doc_freq == 0 {
        return 0.0;
    }
    (1.0 + total_docs as f32 / doc_freq as f32).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tf_raw() {
        assert_eq!(tf_raw(0), 0.0);
        assert_eq!(tf_raw(5), 5.0);
    }

    #[test]
    fn test_tf_sublinear() {
        assert_eq!(tf_sublinear(0), 0.0);
        assert!((tf_sublinear(1) - 1.0).abs() < 1e-6); // 1 + log(1) = 1
        let val = tf_sublinear(10);
        assert!((val - (1.0 + (10.0_f32).ln())).abs() < 1e-5);
    }

    #[test]
    fn test_idf() {
        assert_eq!(idf(100, 0), 0.0);
        // log(100/10) = log(10) ≈ 2.3026
        assert!((idf(100, 10) - (10.0_f32).ln()).abs() < 1e-5);
        // log(100/100) = 0
        assert!((idf(100, 100) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_idf_smooth() {
        assert_eq!(idf_smooth(100, 0), 0.0);
        // log(1 + 100/10) = log(11) ≈ 2.3979
        assert!((idf_smooth(100, 10) - (11.0_f32).ln()).abs() < 1e-5);
    }
}
