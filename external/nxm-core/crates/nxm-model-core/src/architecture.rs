//! Model architecture enum — all supported architectures.
//!
//! Used for dispatch: load correct weights, select correct forward path.

/// All supported model architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Architecture {
    // ── Dense Transformers ──
    Llama,
    Mistral,
    Qwen2,
    Qwen3,
    Qwen35,
    Qwen36,
    Phi3,
    Phi4,
    Gemma2,
    Gemma3,
    Gemma4,
    Gemma4E,
    SmolLM,
    SmolLM3,
    MiniCPM,
    MiniCPM5,

    // ── MoE (Mixture of Experts) ──
    Mixtral,
    DeepSeek2,
    DeepSeek3,
    LagunaXS2,
    NorthMiniCode,
    GptOss,

    // ── Hybrid (Attention + SSM/Linear) ──
    Ornith,
    Nemotron,
    GatedDeltaNet,
    Mamba2,
    Zamba2,

    // ── Specialized ──
    LFM,
    LFM2,
    VibeThinker,
    DiffusionLM,

    // ── Unknown / custom ──
    Unknown,
}

impl Architecture {
    /// Whether this architecture uses MoE routing.
    pub fn is_moe(&self) -> bool {
        matches!(
            self,
            Architecture::Mixtral
                | Architecture::DeepSeek2
                | Architecture::DeepSeek3
                | Architecture::LagunaXS2
                | Architecture::NorthMiniCode
                | Architecture::GptOss
                | Architecture::Gemma4E
        )
    }

    /// Whether this architecture uses linear attention / SSM.
    pub fn is_hybrid(&self) -> bool {
        matches!(
            self,
            Architecture::Ornith
                | Architecture::Nemotron
                | Architecture::GatedDeltaNet
                | Architecture::Mamba2
                | Architecture::Zamba2
        )
    }

    /// Whether this architecture supports MTP (Multi-Token Prediction).
    pub fn supports_mtp(&self) -> bool {
        matches!(
            self,
            Architecture::Qwen36
                | Architecture::Gemma4
                | Architecture::Gemma4E
                | Architecture::DeepSeek3
        )
    }

    /// Display name for the architecture.
    pub fn display_name(&self) -> &str {
        match self {
            Architecture::Llama => "LLaMA",
            Architecture::Mistral => "Mistral",
            Architecture::Qwen2 => "Qwen2",
            Architecture::Qwen3 => "Qwen3",
            Architecture::Qwen35 => "Qwen3.5",
            Architecture::Qwen36 => "Qwen3.6",
            Architecture::Phi3 => "Phi-3",
            Architecture::Phi4 => "Phi-4",
            Architecture::Gemma2 => "Gemma 2",
            Architecture::Gemma3 => "Gemma 3",
            Architecture::Gemma4 => "Gemma 4",
            Architecture::Gemma4E => "Gemma 4E",
            Architecture::SmolLM => "SmolLM",
            Architecture::SmolLM3 => "SmolLM3",
            Architecture::MiniCPM => "MiniCPM",
            Architecture::MiniCPM5 => "MiniCPM5",
            Architecture::Mixtral => "Mixtral",
            Architecture::DeepSeek2 => "DeepSeek-V2",
            Architecture::DeepSeek3 => "DeepSeek-V3",
            Architecture::LagunaXS2 => "Laguna XS.2",
            Architecture::NorthMiniCode => "North Mini Code",
            Architecture::GptOss => "GPT-OSS",
            Architecture::Ornith => "Ornith",
            Architecture::Nemotron => "Nemotron",
            Architecture::GatedDeltaNet => "GatedDeltaNet",
            Architecture::Mamba2 => "Mamba-2",
            Architecture::Zamba2 => "Zamba-2",
            Architecture::LFM => "LFM",
            Architecture::LFM2 => "LFM2",
            Architecture::VibeThinker => "VibeThinker",
            Architecture::DiffusionLM => "DiffusionLM",
            Architecture::Unknown => "Unknown",
        }
    }
}
