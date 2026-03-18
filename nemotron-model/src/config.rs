use serde::Deserialize;
use std::error::Error;
use std::fmt;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum LayerBlockType {
    Mamba,
    Moe,
    Attention,
}

impl LayerBlockType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mamba => "mamba",
            Self::Moe => "moe",
            Self::Attention => "attention",
        }
    }
}

impl fmt::Display for LayerBlockType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelConfigError {
    InvalidHybridOverrideSymbol { position: usize, symbol: char },
    LayerCountMismatch { expected: usize, actual: usize },
}

impl fmt::Display for ModelConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHybridOverrideSymbol { position, symbol } => {
                write!(
                    f,
                    "invalid hybrid override symbol {symbol:?} at character index {position}"
                )
            }
            Self::LayerCountMismatch { expected, actual } => {
                write!(
                    f,
                    "resolved layer count {actual} does not match num_hidden_layers {expected}"
                )
            }
        }
    }
}

impl Error for ModelConfigError {}

pub fn parse_hybrid_override_pattern(
    pattern: &str,
) -> Result<Vec<LayerBlockType>, ModelConfigError> {
    let mut block_types = Vec::with_capacity(pattern.len());

    for (position, symbol) in pattern.chars().enumerate() {
        match symbol {
            'M' | 'm' => block_types.push(LayerBlockType::Mamba),
            'E' | 'e' => block_types.push(LayerBlockType::Moe),
            '*' => block_types.push(LayerBlockType::Attention),
            '-' | '_' | ' ' | '\n' | '\r' | '\t' | ',' => {}
            _ => {
                return Err(ModelConfigError::InvalidHybridOverrideSymbol { position, symbol });
            }
        }
    }

    Ok(block_types)
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default)]
pub struct ModelConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub layers_block_type: Vec<LayerBlockType>,
    pub hybrid_override_pattern: String,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub num_query_groups: Option<usize>,
    pub head_dim: Option<usize>,
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub sliding_window: Option<usize>,
    pub use_mamba_kernels: bool,
    #[serde(alias = "mamba_state_dim")]
    pub ssm_state_size: Option<usize>,
    pub mamba_num_heads: Option<usize>,
    pub mamba_head_dim: Option<usize>,
    #[serde(alias = "mamba_num_groups", alias = "n_group")]
    pub n_groups: Option<usize>,
    pub conv_kernel: Option<usize>,
    pub chunk_size: Option<usize>,
    pub time_step_min: Option<f32>,
    pub time_step_max: Option<f32>,
    pub time_step_floor: Option<f32>,
    #[serde(alias = "num_local_experts", alias = "moe_num_experts")]
    pub n_routed_experts: Option<usize>,
    pub n_shared_experts: Option<usize>,
    #[serde(alias = "moe_top_k")]
    pub num_experts_per_tok: Option<usize>,
    pub routed_scaling_factor: Option<f32>,
    pub topk_group: Option<usize>,
    pub norm_topk_prob: Option<bool>,
    pub moe_intermediate_size: Option<usize>,
    pub moe_shared_expert_intermediate_size: Option<usize>,
    pub moe_shared_expert_overlap: Option<bool>,
    pub use_cache: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub num_logits_to_keep: Option<usize>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SpecialTokenIds {
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub pad: Option<u32>,
}

impl ModelConfig {
    pub fn from_json_str(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }

    pub fn display_name(&self) -> &str {
        self.architectures
            .first()
            .map(String::as_str)
            .unwrap_or(self.model_type.as_str())
    }

    pub fn attention_head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or_else(|| self.hidden_size / self.num_attention_heads)
    }

    pub fn key_value_head_count(&self) -> usize {
        self.num_key_value_heads
            .or(self.num_query_groups)
            .unwrap_or(self.num_attention_heads)
    }

    pub fn special_token_ids(&self) -> SpecialTokenIds {
        SpecialTokenIds {
            bos: self.bos_token_id,
            eos: self.eos_token_id,
            pad: self.pad_token_id,
        }
    }

    pub fn layer_block_types(&self) -> Result<Vec<LayerBlockType>, ModelConfigError> {
        let block_types = if !self.layers_block_type.is_empty() {
            self.layers_block_type.clone()
        } else {
            parse_hybrid_override_pattern(&self.hybrid_override_pattern)?
        };

        if self.num_hidden_layers != 0 && block_types.len() != self.num_hidden_layers {
            return Err(ModelConfigError::LayerCountMismatch {
                expected: self.num_hidden_layers,
                actual: block_types.len(),
            });
        }

        Ok(block_types)
    }

    pub fn layer_block_type(
        &self,
        layer_index: usize,
    ) -> Result<Option<LayerBlockType>, ModelConfigError> {
        Ok(self.layer_block_types()?.get(layer_index).copied())
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architectures: vec!["NemotronHForCausalLM".to_string()],
            model_type: "nemotron_h".to_string(),
            vocab_size: 131_072,
            hidden_size: 2_688,
            intermediate_size: 1_856,
            num_hidden_layers: 52,
            max_position_embeddings: 262_144,
            tie_word_embeddings: false,
            layers_block_type: Vec::new(),
            hybrid_override_pattern: "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
                .to_string(),
            num_attention_heads: 32,
            num_key_value_heads: Some(2),
            num_query_groups: Some(2),
            head_dim: Some(128),
            attention_bias: false,
            attention_dropout: 0.0,
            sliding_window: None,
            use_mamba_kernels: true,
            ssm_state_size: Some(128),
            mamba_num_heads: Some(64),
            mamba_head_dim: Some(64),
            n_groups: Some(8),
            conv_kernel: Some(4),
            chunk_size: Some(128),
            time_step_min: Some(0.001),
            time_step_max: Some(0.1),
            time_step_floor: Some(0.0001),
            n_routed_experts: Some(128),
            n_shared_experts: Some(1),
            num_experts_per_tok: Some(6),
            routed_scaling_factor: Some(2.5),
            topk_group: Some(1),
            norm_topk_prob: Some(true),
            moe_intermediate_size: Some(1_856),
            moe_shared_expert_intermediate_size: Some(3_712),
            moe_shared_expert_overlap: None,
            use_cache: true,
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            pad_token_id: Some(0),
            num_logits_to_keep: Some(1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{LayerBlockType, ModelConfig, SpecialTokenIds};

    /// Verifies that a full Nemotron-3-Nano-30B-A3B JSON config deserializes correctly
    /// and that `layer_block_types()` resolves the hybrid override pattern to 52 layers.
    ///
    /// This catches regressions in serde field mappings, alias resolution, and
    /// hybrid pattern parsing for the target model.
    #[test]
    fn deserializes_target_config_fields_and_resolves_hybrid_layer_types() {
        let config = ModelConfig::from_json_str(
            r#"{
                "architectures": ["NemotronHForCausalLM"],
                "model_type": "nemotron_h",
                "vocab_size": 131072,
                "hidden_size": 2688,
                "intermediate_size": 1856,
                "num_hidden_layers": 52,
                "max_position_embeddings": 262144,
                "tie_word_embeddings": false,
                "hybrid_override_pattern": "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
                "num_attention_heads": 32,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "attention_bias": false,
                "attention_dropout": 0.0,
                "use_mamba_kernels": true,
                "mamba_num_heads": 64,
                "mamba_head_dim": 64,
                "ssm_state_size": 128,
                "conv_kernel": 4,
                "chunk_size": 128,
                "n_groups": 8,
                "n_routed_experts": 128,
                "n_shared_experts": 1,
                "num_experts_per_tok": 6,
                "routed_scaling_factor": 2.5,
                "topk_group": 1,
                "norm_topk_prob": true,
                "moe_intermediate_size": 1856,
                "moe_shared_expert_intermediate_size": 3712,
                "time_step_min": 0.001,
                "time_step_max": 0.1,
                "time_step_floor": 0.0001,
                "use_cache": true,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": 0,
                "num_logits_to_keep": 1
            }"#,
        )
        .expect("config should deserialize");

        assert_eq!(config.ssm_state_size, Some(128));
        assert_eq!(config.hidden_size, 2_688);
        assert_eq!(config.intermediate_size, 1_856);
        assert_eq!(config.num_hidden_layers, 52);
        assert_eq!(config.n_routed_experts, Some(128));
        assert_eq!(config.n_shared_experts, Some(1));
        assert_eq!(config.num_experts_per_tok, Some(6));
        assert_eq!(config.routed_scaling_factor, Some(2.5));
        assert_eq!(config.topk_group, Some(1));
        assert_eq!(config.norm_topk_prob, Some(true));
        assert_eq!(config.moe_intermediate_size, Some(1_856));
        assert_eq!(config.moe_shared_expert_intermediate_size, Some(3_712));
        assert_eq!(config.time_step_min, Some(0.001));
        assert_eq!(config.time_step_max, Some(0.1));
        assert_eq!(config.time_step_floor, Some(0.0001));
        assert_eq!(
            config
                .layer_block_types()
                .expect("pattern should resolve")
                .len(),
            52
        );
        assert_eq!(
            config
                .layer_block_type(1)
                .expect("layer access should succeed"),
            Some(LayerBlockType::Moe)
        );
        assert_eq!(
            config
                .layer_block_type(5)
                .expect("layer access should succeed"),
            Some(LayerBlockType::Attention)
        );
        assert_eq!(config.attention_head_dim(), 128);
        assert_eq!(config.key_value_head_count(), 2);
        assert_eq!(
            config.special_token_ids(),
            SpecialTokenIds {
                bos: Some(1),
                eos: Some(2),
                pad: Some(0),
            }
        );
    }

    /// Verifies that `ModelConfig::default()` produces Nemotron-3-Nano-30B-A3B reference values.
    ///
    /// This catches accidental changes to default hyperparameters that would silently
    /// misconfigure the target model.
    #[test]
    fn default_config_matches_target_model_values() {
        let config = ModelConfig::default();

        assert_eq!(config.model_type, "nemotron_h");
        assert_eq!(config.hidden_size, 2_688);
        assert_eq!(config.intermediate_size, 1_856);
        assert_eq!(config.num_hidden_layers, 52);
        assert_eq!(config.vocab_size, 131_072);
        assert_eq!(config.max_position_embeddings, 262_144);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, Some(2));
        assert_eq!(config.num_query_groups, Some(2));
        assert_eq!(config.head_dim, Some(128));
        assert_eq!(
            config.hybrid_override_pattern,
            "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
        );
        assert_eq!(config.mamba_num_heads, Some(64));
        assert_eq!(config.mamba_head_dim, Some(64));
        assert_eq!(config.ssm_state_size, Some(128));
        assert_eq!(config.conv_kernel, Some(4));
        assert_eq!(config.chunk_size, Some(128));
        assert_eq!(config.n_groups, Some(8));
        assert_eq!(config.n_routed_experts, Some(128));
        assert_eq!(config.n_shared_experts, Some(1));
        assert_eq!(config.num_experts_per_tok, Some(6));
        assert_eq!(config.routed_scaling_factor, Some(2.5));
        assert_eq!(config.topk_group, Some(1));
        assert_eq!(config.norm_topk_prob, Some(true));
        assert_eq!(config.moe_intermediate_size, Some(1_856));
        assert_eq!(config.moe_shared_expert_intermediate_size, Some(3_712));
        assert_eq!(config.time_step_min, Some(0.001));
        assert_eq!(config.time_step_max, Some(0.1));
        assert_eq!(config.time_step_floor, Some(0.0001));
        assert_eq!(config.bos_token_id, Some(1));
        assert_eq!(config.eos_token_id, Some(2));
        assert_eq!(config.pad_token_id, Some(0));
        assert!(config.use_cache);
        assert_eq!(config.num_logits_to_keep, Some(1));
        assert_eq!(
            config.special_token_ids(),
            SpecialTokenIds {
                bos: Some(1),
                eos: Some(2),
                pad: Some(0),
            }
        );
        assert_eq!(
            config
                .layer_block_types()
                .expect("default pattern should resolve")
                .len(),
            config.num_hidden_layers
        );
    }
}
