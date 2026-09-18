//! API types — chat messages, roles, etc.
//!
//! OpenAI-compatible types for the server boundary.

/// Chat message role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    pub fn as_str(&self) -> &str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// Borrowed chat message — used in trait signatures to avoid allocation.
#[derive(Debug, Clone, Copy)]
pub struct ChatMsg<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

/// Owned chat message — used in API request/response types.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }

    /// Convert to borrowed ChatMsg.
    pub fn as_msg(&self) -> ChatMsg<'_> {
        ChatMsg {
            role: self.role.as_str(),
            content: &self.content,
        }
    }
}
