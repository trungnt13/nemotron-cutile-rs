pub use attention::{AttentionForwardShape, AttentionLayer, AttentionLayerError};
pub use block::{BlockError, BlockMixer, NemotronBlock};
pub use cache::{CacheError, HybridCache, KvCache, KvCacheShape, LayerCache};
pub use mamba2::{Mamba2Cache, Mamba2Error, Mamba2ForwardShape, Mamba2Mixer};
pub use mlp::{
    supported_mlp_kernels, MlpBackend, MlpError, MlpKernel, MlpLayer, MlpShape, DENSE_RELU2_HOST,
};
pub use moe::{
    MlpShapeMismatch, MoeBackend, MoeError, MoeKernel, MoeLayer, MoeShape, MOE_DENSE_HOST,
};
pub mod attention;
pub mod block;
pub mod cache;
pub mod linear;
pub mod mamba2;
pub mod mlp;
pub mod moe;

pub use linear::{
    supported_linear_kernels, DenseLinearWeights, Int4LinearWeights, LinearBackend, LinearError,
    LinearKernel, LinearProjection, LinearShape, LinearWeightKind, LinearWeights, DENSE_F32_HOST,
};
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LayerStub {
    pub name: &'static str,
    pub summary: &'static str,
}

pub fn planned_layers() -> [LayerStub; 7] {
    [
        attention::SPEC,
        block::SPEC,
        cache::SPEC,
        linear::SPEC,
        mamba2::SPEC,
        mlp::SPEC,
        moe::SPEC,
    ]
}

pub fn kernel_count() -> usize {
    nemotron_kernels::planned_kernels().len()
}
