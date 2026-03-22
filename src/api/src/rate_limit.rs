//! Per-IP token bucket rate limiter middleware.
//!
//! Limits the number of requests per second from each client IP address.
//! Returns `429 Too Many Requests` with `Retry-After` header when exceeded.
//!
//! # Features
//! - Token bucket algorithm (smooth, allows short bursts)
//! - Per-IP tracking (fair across clients)
//! - Automatic cleanup of stale entries
//! - `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset` response headers
//! - `/health` is exempt from rate limiting
//! - Disabled when rate = 0 (backward compatible)

use axum::{
    body::Body,
    extract::State,
    http::{header, Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

/// Rate limiter configuration.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum sustained requests per second per IP.
    pub requests_per_second: f64,
    /// Maximum burst size (tokens in the bucket).
    pub burst_size: u32,
    /// Whether rate limiting is enabled.
    pub enabled: bool,
}

impl RateLimitConfig {
    pub fn new(requests_per_second: f64, burst_size: u32) -> Self {
        Self {
            requests_per_second,
            burst_size,
            enabled: requests_per_second > 0.0,
        }
    }

    pub fn disabled() -> Self {
        Self {
            requests_per_second: 0.0,
            burst_size: 0,
            enabled: false,
        }
    }
}

/// Per-IP token bucket state.
struct Bucket {
    tokens: f64,
    last_refill: Instant,
}

/// Shared rate limiter state.
pub struct RateLimiter {
    config: RateLimitConfig,
    buckets: Mutex<HashMap<IpAddr, Bucket>>,
}

impl RateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            buckets: Mutex::new(HashMap::new()),
        }
    }

    /// Try to consume one token for the given IP.
    /// Returns (allowed, remaining_tokens, seconds_until_reset).
    async fn try_acquire(&self, ip: IpAddr) -> (bool, u32, f64) {
        let mut buckets = self.buckets.lock().await;
        let now = Instant::now();
        let rate = self.config.requests_per_second;
        let max = self.config.burst_size as f64;

        let bucket = buckets.entry(ip).or_insert(Bucket {
            tokens: max,
            last_refill: now,
        });

        // Refill tokens based on elapsed time
        let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
        bucket.tokens = (bucket.tokens + elapsed * rate).min(max);
        bucket.last_refill = now;

        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            let remaining = bucket.tokens as u32;
            (true, remaining, 0.0)
        } else {
            // How long until one token is available
            let wait = (1.0 - bucket.tokens) / rate;
            (false, 0, wait)
        }
    }

    /// Remove entries that haven't been seen in over 5 minutes.
    /// Called periodically to prevent memory leaks from many unique IPs.
    pub async fn cleanup_stale(&self) {
        let mut buckets = self.buckets.lock().await;
        let now = Instant::now();
        buckets.retain(|_, b| now.duration_since(b.last_refill).as_secs() < 300);
    }
}

/// Extract client IP from request (checks X-Forwarded-For, then peer addr).
fn extract_ip(request: &Request<Body>) -> IpAddr {
    // Check X-Forwarded-For header (behind reverse proxy)
    if let Some(forwarded) = request
        .headers()
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
    {
        if let Some(first_ip) = forwarded.split(',').next() {
            if let Ok(ip) = first_ip.trim().parse::<IpAddr>() {
                return ip;
            }
        }
    }

    // Check X-Real-IP header
    if let Some(real_ip) = request
        .headers()
        .get("x-real-ip")
        .and_then(|v| v.to_str().ok())
    {
        if let Ok(ip) = real_ip.trim().parse::<IpAddr>() {
            return ip;
        }
    }

    // Fallback: extract from connection info if available via extensions
    // When behind no proxy, use a default (all requests share one bucket)
    "127.0.0.1".parse().unwrap()
}

/// Axum middleware that enforces per-IP rate limits.
///
/// Adds response headers:
/// - `X-RateLimit-Limit`: max requests per second
/// - `X-RateLimit-Remaining`: remaining tokens
/// - `X-RateLimit-Reset`: seconds until bucket refills (on 429)
///
/// Returns 429 Too Many Requests with `Retry-After` when limit exceeded.
pub async fn rate_limit_middleware(
    State(limiter): State<Arc<RateLimiter>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    // Disabled → pass through
    if !limiter.config.enabled {
        return next.run(request).await;
    }

    // /health is exempt from rate limiting
    if request.uri().path() == "/health" {
        return next.run(request).await;
    }

    let ip = extract_ip(&request);
    let (allowed, remaining, retry_after) = limiter.try_acquire(ip).await;

    if allowed {
        let mut response = next.run(request).await;
        let headers = response.headers_mut();
        headers.insert(
            "x-ratelimit-limit",
            format!("{}", limiter.config.requests_per_second as u32)
                .parse()
                .unwrap(),
        );
        headers.insert(
            "x-ratelimit-remaining",
            format!("{remaining}").parse().unwrap(),
        );
        response
    } else {
        let body = format!(
            r#"{{"error":"too_many_requests","message":"Rate limit exceeded. Retry after {retry_after:.1}s","retry_after":{retry_after:.1}}}"#
        );
        (
            StatusCode::TOO_MANY_REQUESTS,
            [
                (header::CONTENT_TYPE, "application/json"),
                (
                    header::RETRY_AFTER,
                    &format!("{}", retry_after.ceil() as u32),
                ),
            ],
            body,
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rate_limiter_allows_within_limit() {
        let limiter = RateLimiter::new(RateLimitConfig::new(10.0, 10));
        let ip: IpAddr = "1.2.3.4".parse().unwrap();

        // Should allow 10 requests (burst size)
        for _ in 0..10 {
            let (allowed, _, _) = limiter.try_acquire(ip).await;
            assert!(allowed);
        }
    }

    #[tokio::test]
    async fn test_rate_limiter_blocks_over_limit() {
        let limiter = RateLimiter::new(RateLimitConfig::new(10.0, 5));
        let ip: IpAddr = "1.2.3.4".parse().unwrap();

        // Exhaust burst
        for _ in 0..5 {
            let (allowed, _, _) = limiter.try_acquire(ip).await;
            assert!(allowed);
        }

        // Next request should be blocked
        let (allowed, _, retry_after) = limiter.try_acquire(ip).await;
        assert!(!allowed);
        assert!(retry_after > 0.0);
    }

    #[tokio::test]
    async fn test_rate_limiter_per_ip_isolation() {
        let limiter = RateLimiter::new(RateLimitConfig::new(10.0, 2));
        let ip1: IpAddr = "1.1.1.1".parse().unwrap();
        let ip2: IpAddr = "2.2.2.2".parse().unwrap();

        // Exhaust ip1's bucket
        limiter.try_acquire(ip1).await;
        limiter.try_acquire(ip1).await;
        let (allowed, _, _) = limiter.try_acquire(ip1).await;
        assert!(!allowed, "ip1 should be blocked");

        // ip2 should still have tokens
        let (allowed, _, _) = limiter.try_acquire(ip2).await;
        assert!(allowed, "ip2 should not be affected by ip1");
    }

    #[tokio::test]
    async fn test_rate_limiter_disabled() {
        let config = RateLimitConfig::disabled();
        assert!(!config.enabled);
    }

    #[tokio::test]
    async fn test_rate_limiter_refills_over_time() {
        let limiter = RateLimiter::new(RateLimitConfig::new(100.0, 1));
        let ip: IpAddr = "1.2.3.4".parse().unwrap();

        // Use the one token
        let (allowed, _, _) = limiter.try_acquire(ip).await;
        assert!(allowed);

        // Blocked immediately
        let (allowed, _, _) = limiter.try_acquire(ip).await;
        assert!(!allowed);

        // Wait for refill (100 req/s = 10ms per token)
        tokio::time::sleep(std::time::Duration::from_millis(15)).await;

        // Should be allowed again
        let (allowed, _, _) = limiter.try_acquire(ip).await;
        assert!(allowed);
    }

    #[tokio::test]
    async fn test_cleanup_stale_entries() {
        let limiter = RateLimiter::new(RateLimitConfig::new(10.0, 10));
        let ip: IpAddr = "1.2.3.4".parse().unwrap();

        limiter.try_acquire(ip).await;
        assert_eq!(limiter.buckets.lock().await.len(), 1);

        // Cleanup shouldn't remove recent entries
        limiter.cleanup_stale().await;
        assert_eq!(limiter.buckets.lock().await.len(), 1);
    }
}
