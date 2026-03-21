//! Metadata filtering for vector search.
//!
//! Filters allow narrowing search results by tag conditions.
//! Supports `must` (AND), `must_not` (AND NOT), and `should` (OR) clauses,
//! composable into arbitrarily nested boolean logic.
//!
//! # Example (conceptual)
//! ```ignore
//! // Find similar vectors WHERE category="article" AND source IN ("news","blog")
//! let filter = MetadataFilter::And(vec![
//!     MetadataFilter::Condition(FilterCondition::Equals {
//!         key: "category".into(), value: "article".into(),
//!     }),
//!     MetadataFilter::Or(vec![
//!         MetadataFilter::Condition(FilterCondition::Equals {
//!             key: "source".into(), value: "news".into(),
//!         }),
//!         MetadataFilter::Condition(FilterCondition::Equals {
//!             key: "source".into(), value: "blog".into(),
//!         }),
//!     ]),
//! ]);
//! assert!(filter.matches(&tags));
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single condition on a metadata tag.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterCondition {
    /// Tag value must equal the given value.
    Equals { key: String, value: String },
    /// Tag value must not equal the given value.
    NotEquals { key: String, value: String },
    /// Tag value must be one of the given values.
    In { key: String, values: Vec<String> },
    /// Tag key must exist (any value).
    Exists { key: String },
    /// Tag key must not exist.
    NotExists { key: String },
}

impl FilterCondition {
    /// Evaluate this condition against a set of tags.
    pub fn matches(&self, tags: &HashMap<String, String>) -> bool {
        match self {
            Self::Equals { key, value } => tags.get(key).map_or(false, |v| v == value),
            Self::NotEquals { key, value } => tags.get(key).map_or(true, |v| v != value),
            Self::In { key, values } => tags.get(key).map_or(false, |v| values.contains(v)),
            Self::Exists { key } => tags.contains_key(key),
            Self::NotExists { key } => !tags.contains_key(key),
        }
    }
}

/// Compound metadata filter with boolean logic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetadataFilter {
    /// A single condition.
    Condition(FilterCondition),
    /// All sub-filters must match (AND).
    And(Vec<MetadataFilter>),
    /// At least one sub-filter must match (OR).
    Or(Vec<MetadataFilter>),
}

impl MetadataFilter {
    /// Evaluate the filter against a set of tags.
    pub fn matches(&self, tags: &HashMap<String, String>) -> bool {
        match self {
            Self::Condition(c) => c.matches(tags),
            Self::And(filters) => filters.iter().all(|f| f.matches(tags)),
            Self::Or(filters) => filters.iter().any(|f| f.matches(tags)),
        }
    }

    /// Build a filter from the REST API's must/must_not/should format.
    pub fn from_clauses(
        must: Option<&[(String, String)]>,
        must_not: Option<&[(String, String)]>,
        should: Option<&[(String, String)]>,
    ) -> Option<Self> {
        let mut parts = Vec::new();

        if let Some(must) = must {
            for (key, value) in must {
                parts.push(MetadataFilter::Condition(FilterCondition::Equals {
                    key: key.clone(),
                    value: value.clone(),
                }));
            }
        }

        if let Some(must_not) = must_not {
            for (key, value) in must_not {
                parts.push(MetadataFilter::Condition(FilterCondition::NotEquals {
                    key: key.clone(),
                    value: value.clone(),
                }));
            }
        }

        if let Some(should) = should {
            if !should.is_empty() {
                let or_conditions: Vec<MetadataFilter> = should
                    .iter()
                    .map(|(key, value)| {
                        MetadataFilter::Condition(FilterCondition::Equals {
                            key: key.clone(),
                            value: value.clone(),
                        })
                    })
                    .collect();
                parts.push(MetadataFilter::Or(or_conditions));
            }
        }

        match parts.len() {
            0 => None,
            1 => Some(parts.remove(0)),
            _ => Some(MetadataFilter::And(parts)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tags(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn test_equals() {
        let cond = FilterCondition::Equals {
            key: "color".into(),
            value: "red".into(),
        };
        assert!(cond.matches(&tags(&[("color", "red")])));
        assert!(!cond.matches(&tags(&[("color", "blue")])));
        assert!(!cond.matches(&tags(&[])));
    }

    #[test]
    fn test_not_equals() {
        let cond = FilterCondition::NotEquals {
            key: "status".into(),
            value: "deleted".into(),
        };
        assert!(cond.matches(&tags(&[("status", "active")])));
        assert!(cond.matches(&tags(&[]))); // missing key passes NotEquals
        assert!(!cond.matches(&tags(&[("status", "deleted")])));
    }

    #[test]
    fn test_in() {
        let cond = FilterCondition::In {
            key: "lang".into(),
            values: vec!["en".into(), "fr".into(), "de".into()],
        };
        assert!(cond.matches(&tags(&[("lang", "en")])));
        assert!(cond.matches(&tags(&[("lang", "fr")])));
        assert!(!cond.matches(&tags(&[("lang", "es")])));
        assert!(!cond.matches(&tags(&[])));
    }

    #[test]
    fn test_exists_and_not_exists() {
        let exists = FilterCondition::Exists {
            key: "color".into(),
        };
        let not_exists = FilterCondition::NotExists {
            key: "color".into(),
        };
        let t = tags(&[("color", "red")]);
        assert!(exists.matches(&t));
        assert!(!not_exists.matches(&t));
        assert!(!exists.matches(&tags(&[])));
        assert!(not_exists.matches(&tags(&[])));
    }

    #[test]
    fn test_and_filter() {
        let filter = MetadataFilter::And(vec![
            MetadataFilter::Condition(FilterCondition::Equals {
                key: "category".into(),
                value: "article".into(),
            }),
            MetadataFilter::Condition(FilterCondition::Equals {
                key: "lang".into(),
                value: "en".into(),
            }),
        ]);
        assert!(filter.matches(&tags(&[("category", "article"), ("lang", "en")])));
        assert!(!filter.matches(&tags(&[("category", "article"), ("lang", "fr")])));
        assert!(!filter.matches(&tags(&[("category", "video"), ("lang", "en")])));
    }

    #[test]
    fn test_or_filter() {
        let filter = MetadataFilter::Or(vec![
            MetadataFilter::Condition(FilterCondition::Equals {
                key: "source".into(),
                value: "news".into(),
            }),
            MetadataFilter::Condition(FilterCondition::Equals {
                key: "source".into(),
                value: "blog".into(),
            }),
        ]);
        assert!(filter.matches(&tags(&[("source", "news")])));
        assert!(filter.matches(&tags(&[("source", "blog")])));
        assert!(!filter.matches(&tags(&[("source", "wiki")])));
    }

    #[test]
    fn test_nested_and_or() {
        // category="article" AND (source="news" OR source="blog")
        let filter = MetadataFilter::And(vec![
            MetadataFilter::Condition(FilterCondition::Equals {
                key: "category".into(),
                value: "article".into(),
            }),
            MetadataFilter::Or(vec![
                MetadataFilter::Condition(FilterCondition::Equals {
                    key: "source".into(),
                    value: "news".into(),
                }),
                MetadataFilter::Condition(FilterCondition::Equals {
                    key: "source".into(),
                    value: "blog".into(),
                }),
            ]),
        ]);
        assert!(filter.matches(&tags(&[("category", "article"), ("source", "news")])));
        assert!(filter.matches(&tags(&[("category", "article"), ("source", "blog")])));
        assert!(!filter.matches(&tags(&[("category", "article"), ("source", "wiki")])));
        assert!(!filter.matches(&tags(&[("category", "video"), ("source", "news")])));
    }

    #[test]
    fn test_from_clauses_must_and_must_not() {
        let filter = MetadataFilter::from_clauses(
            Some(&[("category".into(), "article".into())]),
            Some(&[("status".into(), "deleted".into())]),
            None,
        )
        .unwrap();

        assert!(filter.matches(&tags(&[("category", "article"), ("status", "active")])));
        assert!(!filter.matches(&tags(&[("category", "article"), ("status", "deleted")])));
    }

    #[test]
    fn test_from_clauses_empty_returns_none() {
        assert!(MetadataFilter::from_clauses(None, None, None).is_none());
    }
}
