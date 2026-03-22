use std::collections::HashSet;

/// Trait for pluggable tokenization strategies.
pub trait Tokenizer: Send + Sync {
    fn tokenize(&self, text: &str) -> Vec<String>;
}

/// Simple whitespace tokenizer with lowercase, punctuation stripping, and stop words.
pub struct SimpleTokenizer {
    pub lowercase: bool,
    pub stop_words: HashSet<String>,
    pub min_token_length: usize,
}

impl Default for SimpleTokenizer {
    fn default() -> Self {
        let stop_words: HashSet<String> = [
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should", "may", "might",
            "shall", "can", "need", "dare", "ought", "used", "to", "of", "in", "for", "on", "with",
            "at", "by", "from", "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "out", "off", "over", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all", "each", "every", "both", "few",
            "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
            "so", "than", "too", "very", "just", "because", "but", "and", "or", "if", "while",
            "about", "up", "this", "that", "these", "those", "it", "its", "i", "me", "my", "we",
            "our", "you", "your", "he", "him", "his", "she", "her", "they", "them", "their",
            "what", "which", "who", "whom",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        Self {
            lowercase: true,
            stop_words,
            min_token_length: 2,
        }
    }
}

impl Tokenizer for SimpleTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split(|c: char| c.is_whitespace() || c == ',' || c == ';')
            .map(|word| {
                let trimmed = word.trim_matches(|c: char| !c.is_alphanumeric());
                if self.lowercase {
                    trimmed.to_lowercase()
                } else {
                    trimmed.to_string()
                }
            })
            .filter(|word| word.len() >= self.min_token_length && !self.stop_words.contains(word))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tokenizer_basic() {
        let tokenizer = SimpleTokenizer::default();
        let tokens = tokenizer.tokenize("Hello, World! This is a test.");
        assert_eq!(tokens, vec!["hello", "world", "test"]);
    }

    #[test]
    fn test_simple_tokenizer_punctuation() {
        let tokenizer = SimpleTokenizer::default();
        let tokens = tokenizer.tokenize("machine-learning (ML) is great!");
        assert!(tokens.contains(&"machine-learning".to_string()));
        assert!(tokens.contains(&"ml".to_string()));
        assert!(tokens.contains(&"great".to_string()));
    }

    #[test]
    fn test_simple_tokenizer_stop_words() {
        let tokenizer = SimpleTokenizer::default();
        let tokens = tokenizer.tokenize("the quick brown fox");
        assert!(!tokens.contains(&"the".to_string()));
        assert!(tokens.contains(&"quick".to_string()));
        assert!(tokens.contains(&"brown".to_string()));
        assert!(tokens.contains(&"fox".to_string()));
    }

    #[test]
    fn test_simple_tokenizer_min_length() {
        let tokenizer = SimpleTokenizer::default();
        let tokens = tokenizer.tokenize("I x a b c test");
        // "I", "a", "b", "c", "x" are all either stop words or < min_token_length
        assert_eq!(tokens, vec!["test"]);
    }

    #[test]
    fn test_empty_input() {
        let tokenizer = SimpleTokenizer::default();
        let tokens = tokenizer.tokenize("");
        assert!(tokens.is_empty());
    }
}
