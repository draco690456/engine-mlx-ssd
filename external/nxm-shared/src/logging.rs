//! Logging utilities for all NXM projects.
//!
//! Provides a unified tracing setup with per-functionality filtering.
//!
//! # Usage
//!
//! ```rust,ignore
//! use nxm_shared::logging;
//!
//! fn main() {
//!     logging::init("info,nexum::kvcache=debug,nexum::ops=trace");
//!     // or for development:
//!     logging::init_dev();
//! }
//! ```
//!
//! In your modules, use the `target` parameter for per-feature filtering:
//!
//! ```rust,ignore
//! tracing::info!(target: "nexum::engine::prefill", tokens = 512, "prefill started");
//! tracing::debug!(target: "nexum::kvcache::turboquant", "compressing cold KV");
//! ```

use tracing_subscriber::{fmt, EnvFilter};

/// Initialize tracing with a custom filter string.
///
/// Filter syntax: `level` or `target=level` separated by commas.
///
/// Examples:
/// - `"info"` — everything at info level
/// - `"warn,nexum::ops=debug"` — warn globally, debug for ops
/// - `"info,nexum::kvcache=trace"` — info globally, trace for kvcache
pub fn init(filter: &str) {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(filter));

    fmt()
        .with_env_filter(env_filter)
        .with_target(true)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .compact()
        .init();
}

/// Initialize tracing for development: debug level, with file/line info.
pub fn init_dev() {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("debug"));

    fmt()
        .with_env_filter(env_filter)
        .with_target(true)
        .with_thread_ids(false)
        .with_file(true)
        .with_line_number(true)
        .init();
}

/// Initialize tracing for tests: only errors unless RUST_LOG is set.
pub fn init_test() {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("error"));

    let _ = fmt()
        .with_env_filter(env_filter)
        .with_test_writer()
        .try_init();
}
