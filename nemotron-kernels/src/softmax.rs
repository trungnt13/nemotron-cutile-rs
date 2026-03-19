use crate::tensor::{GpuTensor, TensorError};
use crate::KernelStub;

pub const SPEC: KernelStub = KernelStub {
    name: "softmax_host",
    summary: "Numerically stable softmax kernels.",
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SoftmaxBackend {
    HostFallback,
    Cutile,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SoftmaxKernel {
    pub name: &'static str,
    pub backend: SoftmaxBackend,
}

#[cfg(target_os = "linux")]
pub const SOFTMAX: SoftmaxKernel = SoftmaxKernel {
    name: "softmax_cutile",
    backend: SoftmaxBackend::Cutile,
};

#[cfg(not(target_os = "linux"))]
pub const SOFTMAX: SoftmaxKernel = SoftmaxKernel {
    name: "softmax_host",
    backend: SoftmaxBackend::HostFallback,
};

pub fn supported_softmax_kernels() -> [SoftmaxKernel; 1] {
    [SOFTMAX]
}

#[cfg(any(target_os = "linux", test))]
const CUTILE_SOFTMAX_MAX_WIDTH: usize = 4096;

pub fn softmax_host(values: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; values.len()];
    softmax_into_host(values, &mut output)
        .expect("softmax_host output buffer is allocated from input length and cannot mismatch");
    output
}

pub fn softmax_in_place_host(values: &mut [f32]) {
    let output = softmax_host(values);
    values.copy_from_slice(&output);
}

pub fn softmax_into_host(input: &[f32], output: &mut [f32]) -> Result<(), SoftmaxError> {
    if input.len() != output.len() {
        return Err(SoftmaxError::LengthMismatch {
            input: input.len(),
            output: output.len(),
        });
    }

    if input.is_empty() {
        return Ok(());
    }

    let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f64;

    for (value, slot) in input.iter().copied().zip(output.iter_mut()) {
        let weight = (value - max).exp();
        *slot = weight;
        sum += f64::from(weight);
    }

    if sum == 0.0 {
        return Err(SoftmaxError::ZeroPartition);
    }

    let inv_sum = (1.0 / sum) as f32;
    for value in output.iter_mut() {
        *value *= inv_sum;
    }

    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SoftmaxError {
    LengthMismatch { input: usize, output: usize },
    ZeroPartition,
    DeviceError(String),
}

impl From<TensorError> for SoftmaxError {
    fn from(e: TensorError) -> Self {
        SoftmaxError::DeviceError(e.to_string())
    }
}

#[cfg(any(target_os = "linux", test))]
fn supports_cutile_softmax(numel: usize) -> bool {
    numel > 0
        && numel <= CUTILE_SOFTMAX_MAX_WIDTH
        && numel.is_power_of_two()
        && i32::try_from(numel).is_ok()
}

#[cfg(target_os = "linux")]
#[cutile::module]
mod cutile_softmax {
    use cutile::core::*;

    #[cutile::entry()]
    fn softmax<const BM: i32, const BN: i32>(
        x: &Tensor<f32, { [-1, -1] }>,
        y: &mut Tensor<f32, { [BM, BN] }>,
    ) {
        let tile_x: Tile<f32, { [BM, BN] }> = load_tile_like_2d(x, y);
        let tile_x_max: Tile<f32, { [BM] }> = reduce_max(tile_x, 1i32);
        let tile_x_max: Tile<f32, { [BM, BN] }> =
            tile_x_max.reshape(const_shape![BM, 1]).broadcast(y.shape());
        let numerator: Tile<f32, { [BM, BN] }> = exp(tile_x - tile_x_max);
        let denominator: Tile<f32, { [BM] }> = reduce_sum(numerator, 1i32);
        let denominator = denominator
            .reshape(const_shape![BM, 1])
            .broadcast(y.shape());
        y.store(numerator / denominator);
    }
}

// ---------------------------------------------------------------------------
// Async GPU API
// ---------------------------------------------------------------------------

async fn softmax_host_bridge(input: &GpuTensor) -> Result<GpuTensor, SoftmaxError> {
    let data = input.to_host_async().await?;
    let result = softmax_host(&data);
    Ok(GpuTensor::from_host_async(&result, input.shape()).await?)
}

#[cfg(target_os = "linux")]
fn active_backend(input: &GpuTensor) -> SoftmaxBackend {
    if supports_cutile_softmax(input.numel()) {
        SoftmaxBackend::Cutile
    } else {
        SoftmaxBackend::HostFallback
    }
}

#[cfg(not(target_os = "linux"))]
fn active_backend(_input: &GpuTensor) -> SoftmaxBackend {
    SoftmaxBackend::HostFallback
}

#[cfg(target_os = "linux")]
async fn softmax_cutile(input: &GpuTensor) -> Result<GpuTensor, SoftmaxError> {
    use cutile::api;
    use cutile::tile_kernel::{IntoDeviceOperation, IntoDeviceOperationPartition, TileKernel};

    let numel = input.numel();
    let partition_width = i32::try_from(numel).map_err(|_| {
        SoftmaxError::DeviceError(format!(
            "cutile softmax width {numel} exceeds the supported i32 launch bound"
        ))
    })?;
    let flattened_input = input.cutile_tensor_for_shape(&[1, numel]).await?;
    let output = api::zeros::<2, f32>([1, numel]).partition([1, partition_width]);
    let generics = vec!["1".to_string(), numel.to_string()];
    let (_input, output) =
        cutile_softmax::softmax_async(flattened_input.device_operation(), output)
            .generics(generics)
            .await
            .map_err(|error| {
                SoftmaxError::DeviceError(format!("cutile softmax launch failed: {error:?}"))
            })?;
    GpuTensor::from_cutile_tensor(output.unpartition(), input.shape()).map_err(SoftmaxError::from)
}

/// Async GPU softmax normalization.
pub async fn softmax(input: &GpuTensor) -> Result<GpuTensor, SoftmaxError> {
    match active_backend(input) {
        #[cfg(target_os = "linux")]
        SoftmaxBackend::Cutile => softmax_cutile(input).await,
        SoftmaxBackend::HostFallback => softmax_host_bridge(input).await,
        #[cfg(not(target_os = "linux"))]
        SoftmaxBackend::Cutile => unreachable!("cutile backend is unavailable on this platform"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(lhs: f32, rhs: f32) {
        let diff = (lhs - rhs).abs();
        assert!(
            diff <= 1e-6,
            "values differ: left={lhs:?}, right={rhs:?}, diff={diff:?}"
        );
    }

    fn approx_slice_eq(lhs: &[f32], rhs: &[f32], tolerance: f32) {
        assert_eq!(lhs.len(), rhs.len(), "slice lengths differ");
        for (left, right) in lhs.iter().zip(rhs.iter()) {
            let diff = (left - right).abs();
            assert!(
                diff <= tolerance,
                "values differ: left={left:?}, right={right:?}, diff={diff:?}, tolerance={tolerance:?}"
            );
        }
    }

    /// Verifies that the softmax kernel registry reports the active primary backend for the current platform.
    ///
    /// This catches accidental backend tag changes as Linux adds cutile compute while other platforms keep host fallback.
    #[test]
    fn reports_platform_primary_backend() {
        #[cfg(target_os = "linux")]
        let expected = [SoftmaxKernel {
            name: "softmax_cutile",
            backend: SoftmaxBackend::Cutile,
        }];

        #[cfg(not(target_os = "linux"))]
        let expected = [SoftmaxKernel {
            name: "softmax_host",
            backend: SoftmaxBackend::HostFallback,
        }];

        assert_eq!(supported_softmax_kernels(), expected);
    }

    /// Verifies that the cutile dispatch heuristic only accepts flattened widths within the current supported bound.
    ///
    /// This catches regressions where Linux would try to launch the tile kernel for sizes outside the planned safety envelope.
    #[test]
    fn cutile_support_heuristic_respects_width_boundaries() {
        assert!(supports_cutile_softmax(1));
        assert!(supports_cutile_softmax(8));
        assert!(supports_cutile_softmax(CUTILE_SOFTMAX_MAX_WIDTH));
        assert!(!supports_cutile_softmax(0));
        assert!(!supports_cutile_softmax(7));
        assert!(!supports_cutile_softmax(CUTILE_SOFTMAX_MAX_WIDTH + 1));
    }

    /// Verifies softmax_host output against known reference values for [1, 2, 3].
    ///
    /// This catches errors in the exp/normalize formula.
    #[test]
    fn softmax_matches_reference_values() {
        let output = softmax_host(&[1.0, 2.0, 3.0]);

        approx_eq(output[0], 0.09003057);
        approx_eq(output[1], 0.24472848);
        approx_eq(output[2], 0.66524094);
    }

    /// Verifies that large inputs produce the same result as shifted inputs (numerical stability).
    ///
    /// This catches missing max-subtraction in the softmax_host, which would cause overflow.
    #[test]
    fn softmax_is_stable_for_large_inputs() {
        let output = softmax_host(&[1000.0, 1001.0, 1002.0]);

        approx_eq(output[0], 0.09003057);
        approx_eq(output[1], 0.24472848);
        approx_eq(output[2], 0.66524094);
    }

    /// Verifies that softmax_host outputs sum to 1.0 across a wide value range.
    ///
    /// This catches partition normalization errors.
    #[test]
    fn softmax_outputs_sum_to_one() {
        let output = softmax_host(&[-10.0, 0.0, 10.0, 20.0]);
        let sum: f32 = output.iter().sum();

        approx_eq(sum, 1.0);
    }

    /// Verifies that the in-place variant produces uniform probabilities for equal inputs.
    ///
    /// This catches issues where in-place writes are skipped or mis-ordered.
    #[test]
    fn softmax_in_place_updates_buffer() {
        let mut values = [0.0, 0.0, 0.0];
        softmax_in_place_host(&mut values);

        approx_eq(values[0], 1.0 / 3.0);
        approx_eq(values[1], 1.0 / 3.0);
        approx_eq(values[2], 1.0 / 3.0);
    }

    /// Verifies that mismatched input/output lengths are rejected.
    ///
    /// This catches missing length validation in the _into variant.
    #[test]
    fn softmax_into_rejects_length_mismatch() {
        let mut output = [0.0; 1];
        let error = softmax_into_host(&[1.0, 2.0], &mut output).unwrap_err();

        assert_eq!(
            error,
            SoftmaxError::LengthMismatch {
                input: 2,
                output: 1,
            }
        );
    }

    /// Verifies that empty input produces empty output without error.
    ///
    /// This catches panics on zero-length slices.
    #[test]
    fn softmax_handles_empty_input() {
        assert_eq!(softmax_host(&[]), Vec::<f32>::new());
    }

    /// Verifies that the async GPU softmax matches the host fallback.
    /// This catches regressions in the GPU data transfer path.
    #[tokio::test]
    async fn gpu_softmax_matches_host_fallback() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let expected = softmax_host(&data);
        let gpu_input = GpuTensor::from_host(&data, &[4]).unwrap();
        let result = super::softmax(&gpu_input).await.unwrap();
        approx_slice_eq(&result.to_host(), &expected, 5e-6);
    }

    /// Verifies that async GPU softmax preserves the input shape when the
    /// tensor has higher-rank metadata. This catches regressions where the
    /// wrapper flattens tensor metadata during host fallback.
    #[tokio::test]
    async fn gpu_softmax_preserves_input_shape() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let expected = softmax_host(&data);
        let gpu_input = GpuTensor::from_host(&data, &[2, 2]).unwrap();

        let result = super::softmax(&gpu_input).await.unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        approx_slice_eq(&result.to_host(), &expected, 5e-6);
    }

    /// Verifies that the async GPU softmax stays numerically stable for large shifted inputs.
    ///
    /// This catches regressions where the cutile path would skip max-subtraction and diverge from the host contract.
    #[tokio::test]
    async fn gpu_softmax_is_stable_for_large_inputs() {
        let data = vec![1000.0, 1001.0, 1002.0, 1003.0];
        let expected = softmax_host(&data);
        let gpu_input = GpuTensor::from_host(&data, &[4]).unwrap();

        let result = super::softmax(&gpu_input).await.unwrap();

        approx_slice_eq(&result.to_host(), &expected, 5e-5);
    }

    /// Verifies that Linux falls back to the host bridge when the flattened width exceeds the current cutile softmax limit.
    ///
    /// This catches regressions where oversized rows would try to launch an unsupported tile shape instead of preserving correctness.
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn gpu_softmax_falls_back_for_large_rows() {
        let width = CUTILE_SOFTMAX_MAX_WIDTH + 1;
        let data = vec![0.25; width];
        let gpu_input = GpuTensor::from_host(&data, &[width]).unwrap();

        assert_eq!(active_backend(&gpu_input), SoftmaxBackend::HostFallback);

        let result = super::softmax(&gpu_input).await.unwrap();

        approx_slice_eq(&result.to_host(), &softmax_host(&data), 5e-6);
    }
}
