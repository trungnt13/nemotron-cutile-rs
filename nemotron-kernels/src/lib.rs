pub mod activations;
pub mod attention;
pub mod conv1d;
pub mod device;
pub mod embedding;
pub mod gemm;
pub mod moe_routing;
pub mod quantize;
pub mod rms_norm;
pub mod softmax;
pub mod ssm;
pub mod tensor;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelStub {
    pub name: &'static str,
    pub summary: &'static str,
}

pub fn planned_kernels() -> [KernelStub; 10] {
    [
        activations::SPEC,
        attention::SPEC,
        conv1d::SPEC,
        embedding::SPEC,
        gemm::SPEC,
        moe_routing::SPEC,
        quantize::SPEC,
        rms_norm::SPEC,
        softmax::SPEC,
        ssm::SPEC,
    ]
}
