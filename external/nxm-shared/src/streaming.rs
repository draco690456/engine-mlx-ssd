//! SSE streaming helpers — format StreamChunk as Server-Sent Events.

use crate::types::{Delta, StreamChoice, StreamChunk};

/// Create a streaming chunk with content.
pub fn content_chunk(id: &str, model: &str, created: u64, content: &str, is_first: bool) -> StreamChunk {
    StreamChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![StreamChoice {
            index: 0,
            delta: Delta {
                role: if is_first { Some("assistant".to_string()) } else { None },
                content: Some(content.to_string()),
            },
            finish_reason: None,
        }],
    }
}

/// Create a streaming chunk signaling end of generation.
pub fn stop_chunk(id: &str, model: &str, created: u64) -> StreamChunk {
    StreamChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![StreamChoice {
            index: 0,
            delta: Delta {
                role: None,
                content: None,
            },
            finish_reason: Some("stop".to_string()),
        }],
    }
}

/// Format a StreamChunk as an SSE `data:` line.
pub fn to_sse(chunk: &StreamChunk) -> String {
    format!("data: {}\n\n", serde_json::to_string(chunk).unwrap_or_default())
}

/// The final SSE event signaling stream end.
pub fn sse_done() -> &'static str {
    "data: [DONE]\n\n"
}
