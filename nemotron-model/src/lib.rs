pub mod config;
pub mod generate;
pub mod model;
pub mod tokenizer;
pub mod weights;

pub use config::{
    parse_hybrid_override_pattern, LayerBlockType, ModelConfig, ModelConfigError, SpecialTokenIds,
};
pub use generate::{generation_preview, GenerationError, GenerationRequest, GenerationResult};
pub use model::{
    EmbeddingTable, ModelForwardError, ModelForwardOutput, ModelRuntime, ModelTextError,
    NemotronModel,
};
pub use tokenizer::{ModelTokenizer, TokenizerError, TokenizerSpec, DEFAULT_TOKENIZER_FILE};
pub use weights::{
    LoadedTensor, WeightError, WeightFile, WeightManifest, WeightTensorDescriptor,
    DEFAULT_WEIGHT_ROOT, SAFETENSORS_FILE_EXTENSION, SAFETENSORS_INDEX_FILE,
};

pub fn workspace_summary() -> String {
    format!(
        "nemotron-rs scaffold: {} kernels, {} nn layers",
        nemotron_nn::kernel_count(),
        nemotron_nn::planned_layers().len()
    )
}
