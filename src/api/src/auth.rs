//! API key authentication middleware.
//!
//! Supports two roles:
//! - **Admin** keys: full read + write access
//! - **Read-only** keys: search, get, list, stats only
//!
//! When no keys are configured, auth is disabled (backward compatible).
//! The `/health` endpoint is always accessible without auth.
//!
//! # Usage
//!
//! ```bash
//! # Admin key (read + write)
//! curl -H "Authorization: Bearer vdb-admin-secret" http://localhost:8080/vectors
//!
//! # Read-only key (search only)
//! curl -H "Authorization: Bearer vdb-ro-key" http://localhost:8080/search -d '...'
//! ```

use axum::{
    body::Body,
    extract::State,
    http::{Method, Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::collections::HashSet;
use std::sync::Arc;

/// Authentication configuration.
#[derive(Debug, Clone)]
pub struct AuthConfig {
    /// Admin API keys (full read + write access).
    admin_keys: HashSet<String>,
    /// Read-only API keys (search, get, list, stats).
    readonly_keys: HashSet<String>,
    /// Whether auth is enabled. False when no keys are configured.
    pub enabled: bool,
}

impl AuthConfig {
    /// Create auth config from key lists. Auth is disabled if both lists are empty.
    pub fn new(admin_keys: Vec<String>, readonly_keys: Vec<String>) -> Self {
        let enabled = !admin_keys.is_empty() || !readonly_keys.is_empty();
        Self {
            admin_keys: admin_keys.into_iter().collect(),
            readonly_keys: readonly_keys.into_iter().collect(),
            enabled,
        }
    }

    /// Disabled auth (no keys required).
    pub fn disabled() -> Self {
        Self {
            admin_keys: HashSet::new(),
            readonly_keys: HashSet::new(),
            enabled: false,
        }
    }

    /// Validate a token. Returns true if the token is authorized for the request.
    fn validate(&self, token: &str, requires_write: bool) -> Result<(), AuthError> {
        if self.admin_keys.contains(token) {
            return Ok(());
        }
        if self.readonly_keys.contains(token) {
            if requires_write {
                return Err(AuthError::Forbidden);
            }
            return Ok(());
        }
        Err(AuthError::InvalidKey)
    }
}

enum AuthError {
    MissingKey,
    InvalidKey,
    Forbidden,
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let (status, msg) = match self {
            AuthError::MissingKey => (
                StatusCode::UNAUTHORIZED,
                r#"{"error":"unauthorized","message":"Missing API key. Set Authorization: Bearer <key>"}"#,
            ),
            AuthError::InvalidKey => (
                StatusCode::UNAUTHORIZED,
                r#"{"error":"unauthorized","message":"Invalid API key"}"#,
            ),
            AuthError::Forbidden => (
                StatusCode::FORBIDDEN,
                r#"{"error":"forbidden","message":"Read-only API key cannot perform write operations"}"#,
            ),
        };
        (
            status,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            msg,
        )
            .into_response()
    }
}

/// Returns true if this request mutates data (write operation).
fn is_write_request(method: &Method, path: &str) -> bool {
    match *method {
        Method::POST => {
            // POST /search, /search/text, /embed are reads
            !path.starts_with("/search") && !path.starts_with("/embed")
        }
        Method::PUT | Method::DELETE => true,
        _ => false,
    }
}

/// Axum middleware that checks the `Authorization: Bearer <key>` header.
///
/// - `/health` is always accessible (for load balancer probes)
/// - Write operations require an admin key
/// - Read operations accept both admin and read-only keys
/// - When auth is disabled, all requests pass through
pub async fn auth_middleware(
    State(auth): State<Arc<AuthConfig>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    // Auth disabled → pass through
    if !auth.enabled {
        return next.run(request).await;
    }

    // /health is always public (load balancer probes)
    if request.uri().path() == "/health" {
        return next.run(request).await;
    }

    // Extract bearer token
    let token = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| {
            v.strip_prefix("Bearer ")
                .or_else(|| v.strip_prefix("bearer "))
        });

    let token = match token {
        Some(t) => t.trim(),
        None => return AuthError::MissingKey.into_response(),
    };

    let requires_write = is_write_request(request.method(), request.uri().path());

    match auth.validate(token, requires_write) {
        Ok(()) => next.run(request).await,
        Err(e) => e.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_disabled() {
        let auth = AuthConfig::disabled();
        assert!(!auth.enabled);
    }

    #[test]
    fn test_admin_key_has_full_access() {
        let auth = AuthConfig::new(vec!["admin-key".into()], vec![]);
        assert!(auth.validate("admin-key", false).is_ok());
        assert!(auth.validate("admin-key", true).is_ok());
    }

    #[test]
    fn test_readonly_key_cannot_write() {
        let auth = AuthConfig::new(vec![], vec!["ro-key".into()]);
        assert!(auth.validate("ro-key", false).is_ok());
        assert!(auth.validate("ro-key", true).is_err());
    }

    #[test]
    fn test_invalid_key_rejected() {
        let auth = AuthConfig::new(vec!["admin-key".into()], vec!["ro-key".into()]);
        assert!(auth.validate("wrong-key", false).is_err());
        assert!(auth.validate("wrong-key", true).is_err());
    }

    #[test]
    fn test_write_detection() {
        assert!(is_write_request(&Method::POST, "/vectors"));
        assert!(is_write_request(&Method::POST, "/vectors/text"));
        assert!(is_write_request(&Method::PUT, "/vectors/id1"));
        assert!(is_write_request(&Method::DELETE, "/vectors/id1"));

        // These POSTs are reads
        assert!(!is_write_request(&Method::POST, "/search"));
        assert!(!is_write_request(&Method::POST, "/search/text"));
        assert!(!is_write_request(&Method::POST, "/embed"));

        // GETs are always reads
        assert!(!is_write_request(&Method::GET, "/vectors"));
        assert!(!is_write_request(&Method::GET, "/stats"));
    }

    #[test]
    fn test_no_keys_means_disabled() {
        let auth = AuthConfig::new(vec![], vec![]);
        assert!(!auth.enabled);
    }

    #[test]
    fn test_multiple_keys() {
        let auth = AuthConfig::new(
            vec!["admin1".into(), "admin2".into()],
            vec!["ro1".into(), "ro2".into()],
        );
        assert!(auth.validate("admin1", true).is_ok());
        assert!(auth.validate("admin2", true).is_ok());
        assert!(auth.validate("ro1", false).is_ok());
        assert!(auth.validate("ro2", false).is_ok());
        assert!(auth.validate("ro1", true).is_err());
    }
}
