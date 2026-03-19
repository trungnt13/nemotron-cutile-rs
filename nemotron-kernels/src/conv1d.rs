use crate::tensor::{GpuTensor, TensorError};
use crate::KernelStub;

#[cfg(target_os = "linux")]
use crate::gemm::{gemm, GemmShape};

pub const SPEC: KernelStub = KernelStub {
    name: "conv1d",
    summary: "Depthwise causal Conv1D kernels for Mamba-2.",
};

#[cfg(target_os = "linux")]
const CUTILE_GEMM_TILE_M: usize = 16;
#[cfg(target_os = "linux")]
const CUTILE_GEMM_TILE_N: usize = 16;
#[cfg(target_os = "linux")]
const CUTILE_GEMM_TILE_K: usize = 8;
#[cfg(any(target_os = "linux", test))]
const CUTILE_MAX_KERNEL_SIZE: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Conv1dBackend {
    HostFallback,
    Cutile,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Conv1dKernel {
    pub name: &'static str,
    pub backend: Conv1dBackend,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Conv1dShape {
    pub sequence_len: usize,
    pub channel_count: usize,
    pub kernel_size: usize,
}

impl Conv1dShape {
    pub const fn new(sequence_len: usize, channel_count: usize, kernel_size: usize) -> Self {
        Self {
            sequence_len,
            channel_count,
            kernel_size,
        }
    }

    pub const fn input_len(self) -> usize {
        self.sequence_len * self.channel_count
    }

    pub const fn weight_len(self) -> usize {
        self.channel_count * self.kernel_size
    }

    pub const fn output_len(self) -> usize {
        self.input_len()
    }
}

#[cfg(target_os = "linux")]
pub const DEPTHWISE_CAUSAL_CONV1D: Conv1dKernel = Conv1dKernel {
    name: "depthwise_causal_conv1d_cutile",
    backend: Conv1dBackend::Cutile,
};

#[cfg(not(target_os = "linux"))]
pub const DEPTHWISE_CAUSAL_CONV1D: Conv1dKernel = Conv1dKernel {
    name: "depthwise_causal_conv1d_host",
    backend: Conv1dBackend::HostFallback,
};

pub fn supported_conv1d_kernels() -> [Conv1dKernel; 1] {
    [DEPTHWISE_CAUSAL_CONV1D]
}

pub fn depthwise_causal_conv1d_host(
    input: &[f32],
    weights: &[f32],
    shape: Conv1dShape,
) -> Result<Vec<f32>, Conv1dError> {
    let mut output = vec![0.0; shape.output_len()];
    depthwise_causal_conv1d_into_host(input, weights, shape, &mut output)?;
    Ok(output)
}

pub fn depthwise_causal_conv1d_into_host(
    input: &[f32],
    weights: &[f32],
    shape: Conv1dShape,
    output: &mut [f32],
) -> Result<(), Conv1dError> {
    validate_shape(input, weights, shape, output)?;

    for timestep in 0..shape.sequence_len {
        for channel in 0..shape.channel_count {
            let mut acc = 0.0_f64;

            for tap in 0..shape.kernel_size {
                if timestep < tap {
                    break;
                }

                let input_timestep = timestep - tap;
                let input_index = input_timestep * shape.channel_count + channel;
                let weight_index = channel * shape.kernel_size + (shape.kernel_size - 1 - tap);

                acc += f64::from(input[input_index]) * f64::from(weights[weight_index]);
            }

            output[timestep * shape.channel_count + channel] = acc as f32;
        }
    }

    Ok(())
}

fn validate_shape(
    input: &[f32],
    weights: &[f32],
    shape: Conv1dShape,
    output: &mut [f32],
) -> Result<(), Conv1dError> {
    if shape.channel_count == 0 || shape.kernel_size == 0 {
        return Err(Conv1dError::InvalidShape(shape));
    }

    if input.len() != shape.input_len() {
        return Err(Conv1dError::LengthMismatch {
            argument: "input",
            expected: shape.input_len(),
            actual: input.len(),
        });
    }

    if weights.len() != shape.weight_len() {
        return Err(Conv1dError::LengthMismatch {
            argument: "weights",
            expected: shape.weight_len(),
            actual: weights.len(),
        });
    }

    if output.len() != shape.output_len() {
        return Err(Conv1dError::LengthMismatch {
            argument: "output",
            expected: shape.output_len(),
            actual: output.len(),
        });
    }

    Ok(())
}

fn validate_tensor_lengths(
    input: &GpuTensor,
    weights: &GpuTensor,
    shape: Conv1dShape,
) -> Result<(), Conv1dError> {
    if shape.channel_count == 0 || shape.kernel_size == 0 {
        return Err(Conv1dError::InvalidShape(shape));
    }

    if input.numel() != shape.input_len() {
        return Err(Conv1dError::LengthMismatch {
            argument: "input",
            expected: shape.input_len(),
            actual: input.numel(),
        });
    }

    if weights.numel() != shape.weight_len() {
        return Err(Conv1dError::LengthMismatch {
            argument: "weights",
            expected: shape.weight_len(),
            actual: weights.numel(),
        });
    }

    Ok(())
}

#[cfg(target_os = "linux")]
fn cutile_gemm_shape(shape: Conv1dShape) -> Option<GemmShape> {
    let k = shape.channel_count.checked_mul(shape.kernel_size)?;
    Some(GemmShape::new(shape.sequence_len, k, shape.channel_count))
}

#[cfg(target_os = "linux")]
fn supports_cutile_depthwise_causal_conv1d(shape: Conv1dShape) -> bool {
    let Some(gemm_shape) = cutile_gemm_shape(shape) else {
        return false;
    };

    shape.sequence_len > 0
        && shape.channel_count > 0
        && shape.kernel_size > 0
        && shape.kernel_size <= CUTILE_MAX_KERNEL_SIZE
        && i32::try_from(shape.sequence_len).is_ok()
        && i32::try_from(shape.channel_count).is_ok()
        && i32::try_from(shape.kernel_size).is_ok()
        && gemm_shape.m % CUTILE_GEMM_TILE_M == 0
        && gemm_shape.k % CUTILE_GEMM_TILE_K == 0
        && gemm_shape.n % CUTILE_GEMM_TILE_N == 0
}

#[cfg(target_os = "linux")]
fn build_cutile_gemm_operands(
    input: &[f32],
    weights: &[f32],
    shape: Conv1dShape,
) -> Result<(Vec<f32>, Vec<f32>, GemmShape), Conv1dError> {
    if shape.channel_count == 0 || shape.kernel_size == 0 {
        return Err(Conv1dError::InvalidShape(shape));
    }

    if input.len() != shape.input_len() {
        return Err(Conv1dError::LengthMismatch {
            argument: "input",
            expected: shape.input_len(),
            actual: input.len(),
        });
    }

    if weights.len() != shape.weight_len() {
        return Err(Conv1dError::LengthMismatch {
            argument: "weights",
            expected: shape.weight_len(),
            actual: weights.len(),
        });
    }

    let gemm_shape = cutile_gemm_shape(shape).ok_or(Conv1dError::InvalidShape(shape))?;
    let mut lhs = vec![0.0; gemm_shape.lhs_len()];
    let mut rhs = vec![0.0; gemm_shape.rhs_len()];

    for timestep in 0..shape.sequence_len {
        let row = &mut lhs[timestep * gemm_shape.k..(timestep + 1) * gemm_shape.k];
        for channel in 0..shape.channel_count {
            for tap in 0..shape.kernel_size {
                let column = channel * shape.kernel_size + tap;
                if timestep >= tap {
                    row[column] = input[(timestep - tap) * shape.channel_count + channel];
                }
            }
        }
    }

    for channel in 0..shape.channel_count {
        for tap in 0..shape.kernel_size {
            let row = channel * shape.kernel_size + tap;
            rhs[row * gemm_shape.n + channel] =
                weights[channel * shape.kernel_size + (shape.kernel_size - 1 - tap)];
        }
    }

    Ok((lhs, rhs, gemm_shape))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Conv1dError {
    InvalidShape(Conv1dShape),
    LengthMismatch {
        argument: &'static str,
        expected: usize,
        actual: usize,
    },
    DeviceError(String),
}

impl From<TensorError> for Conv1dError {
    fn from(e: TensorError) -> Self {
        Conv1dError::DeviceError(e.to_string())
    }
}

fn backend_for_shape(_shape: Conv1dShape) -> Conv1dBackend {
    #[cfg(target_os = "linux")]
    {
        if supports_cutile_depthwise_causal_conv1d(_shape) {
            return Conv1dBackend::Cutile;
        }
    }

    Conv1dBackend::HostFallback
}

#[cfg(target_os = "linux")]
async fn depthwise_causal_conv1d_via_cutile_gemm(
    input: &GpuTensor,
    weights: &GpuTensor,
    shape: Conv1dShape,
) -> Result<GpuTensor, Conv1dError> {
    let input_data = input.to_host_async().await?;
    let weight_data = weights.to_host_async().await?;
    let (lhs, rhs, gemm_shape) = build_cutile_gemm_operands(&input_data, &weight_data, shape)?;
    let gpu_lhs = GpuTensor::from_host_async(&lhs, &[gemm_shape.m, gemm_shape.k]).await?;
    let gpu_rhs = GpuTensor::from_host_async(&rhs, &[gemm_shape.k, gemm_shape.n]).await?;
    let mut result = gemm(&gpu_lhs, &gpu_rhs, gemm_shape)
        .await
        .map_err(|error| {
            Conv1dError::DeviceError(format!("cutile conv1d GEMM failed: {error:?}"))
        })?;
    result.reshape(input.shape())?;
    Ok(result)
}

async fn depthwise_causal_conv1d_host_bridge(
    input: &GpuTensor,
    weights: &GpuTensor,
    shape: Conv1dShape,
) -> Result<GpuTensor, Conv1dError> {
    let input_data = input.to_host_async().await?;
    let weight_data = weights.to_host_async().await?;
    let result = depthwise_causal_conv1d_host(&input_data, &weight_data, shape)?;
    Ok(GpuTensor::from_host_async(&result, input.shape()).await?)
}

// ---------------------------------------------------------------------------
// Async GPU API
// ---------------------------------------------------------------------------

/// Async GPU depthwise causal 1D convolution.
pub async fn depthwise_causal_conv1d(
    input: &GpuTensor,
    weights: &GpuTensor,
    shape: Conv1dShape,
) -> Result<GpuTensor, Conv1dError> {
    validate_tensor_lengths(input, weights, shape)?;

    match backend_for_shape(shape) {
        #[cfg(target_os = "linux")]
        Conv1dBackend::Cutile => {
            depthwise_causal_conv1d_via_cutile_gemm(input, weights, shape).await
        }
        Conv1dBackend::HostFallback => {
            depthwise_causal_conv1d_host_bridge(input, weights, shape).await
        }
        #[cfg(not(target_os = "linux"))]
        Conv1dBackend::Cutile => {
            unreachable!("cutile conv1d backend is unavailable on this platform")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_slice_with_tolerance(lhs: &[f32], rhs: &[f32], tolerance: f32) {
        assert_eq!(lhs.len(), rhs.len(), "slice lengths differ");
        for (index, (left, right)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let diff = (left - right).abs();
            assert!(
                diff <= tolerance,
                "index {index}: left={left:?}, right={right:?}, diff={diff:?}, tolerance={tolerance:?}"
            );
        }
    }

    fn approx_eq_slice(lhs: &[f32], rhs: &[f32]) {
        approx_eq_slice_with_tolerance(lhs, rhs, 1e-6);
    }

    /// Verifies that the Conv1D kernel registry reports the current platform's primary backend.
    ///
    /// This catches accidental backend metadata drift as Linux adds a real cutile path while other platforms keep host fallback.
    #[test]
    fn reports_platform_primary_backend() {
        #[cfg(target_os = "linux")]
        let expected = [Conv1dKernel {
            name: "depthwise_causal_conv1d_cutile",
            backend: Conv1dBackend::Cutile,
        }];

        #[cfg(not(target_os = "linux"))]
        let expected = [Conv1dKernel {
            name: "depthwise_causal_conv1d_host",
            backend: Conv1dBackend::HostFallback,
        }];

        assert_eq!(supported_conv1d_kernels(), expected);
    }

    /// Verifies that Linux only selects the cutile Conv1D backend when its internal GEMM transform is tile-aligned and bounded.
    ///
    /// This catches risky device dispatch for unsupported shapes while preserving host fallback everywhere else.
    #[test]
    fn selects_backend_from_shape_constraints() {
        let supported = Conv1dShape::new(16, 16, 4);
        let unaligned_sequence = Conv1dShape::new(12, 16, 4);
        let oversized_kernel = Conv1dShape::new(16, 16, CUTILE_MAX_KERNEL_SIZE + 1);

        #[cfg(target_os = "linux")]
        {
            assert_eq!(backend_for_shape(supported), Conv1dBackend::Cutile);
            assert_eq!(
                backend_for_shape(unaligned_sequence),
                Conv1dBackend::HostFallback
            );
            assert_eq!(
                backend_for_shape(oversized_kernel),
                Conv1dBackend::HostFallback
            );
        }

        #[cfg(not(target_os = "linux"))]
        {
            assert_eq!(backend_for_shape(supported), Conv1dBackend::HostFallback);
            assert_eq!(
                backend_for_shape(unaligned_sequence),
                Conv1dBackend::HostFallback
            );
            assert_eq!(
                backend_for_shape(oversized_kernel),
                Conv1dBackend::HostFallback
            );
        }
    }

    /// Verifies causal conv1d on a single channel with kernel_size=2.
    ///
    /// This catches errors in weight reversal (convolution vs correlation) and causal zero-padding.
    #[test]
    fn applies_single_channel_causal_filter() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let weights = [2.0, 1.0];
        let output =
            depthwise_causal_conv1d_host(&input, &weights, Conv1dShape::new(4, 1, 2)).unwrap();

        approx_eq_slice(&output, &[1.0, 4.0, 7.0, 10.0]);
    }

    /// Verifies that each channel is convolved independently (depthwise).
    ///
    /// This catches cross-channel mixing bugs in index calculations.
    #[test]
    fn applies_depthwise_filter_per_channel() {
        let input = [
            1.0, 10.0, //
            2.0, 20.0, //
            3.0, 30.0, //
        ];
        let weights = [
            1.0, 0.5, //
            2.0, -1.0, //
        ];

        let output =
            depthwise_causal_conv1d_host(&input, &weights, Conv1dShape::new(3, 2, 2)).unwrap();

        approx_eq_slice(
            &output,
            &[
                0.5, -10.0, //
                2.0, 0.0, //
                3.5, 10.0, //
            ],
        );
    }

    /// Verifies that kernel_size=1 degenerates to per-channel scaling (no history).
    ///
    /// This catches off-by-one in the tap loop boundary when kernel_size is minimal.
    #[test]
    fn kernel_size_one_is_per_channel_scaling() {
        let input = [
            1.0, -1.0, //
            2.0, -2.0, //
            3.0, -3.0, //
        ];
        let weights = [2.0, -0.5];

        let output =
            depthwise_causal_conv1d_host(&input, &weights, Conv1dShape::new(3, 2, 1)).unwrap();

        approx_eq_slice(
            &output,
            &[
                2.0, 0.5, //
                4.0, 1.0, //
                6.0, 1.5, //
            ],
        );
    }

    /// Verifies that the _into variant writes results into a pre-allocated buffer.
    ///
    /// This catches bugs where _into silently re-allocates instead of writing in place.
    #[test]
    fn conv1d_into_writes_existing_buffer() {
        let input = [
            1.0, 10.0, //
            2.0, 20.0, //
        ];
        let weights = [
            1.0, 1.0, //
            1.0, 0.0, //
        ];
        let mut output = [-1.0; 4];

        depthwise_causal_conv1d_into_host(&input, &weights, Conv1dShape::new(2, 2, 2), &mut output)
            .unwrap();

        approx_eq_slice(&output, &[1.0, 0.0, 3.0, 10.0]);
    }

    /// Verifies that a zero-length sequence produces an empty output without error.
    ///
    /// This catches panics on empty input slices.
    #[test]
    fn empty_sequence_produces_empty_output() {
        let output =
            depthwise_causal_conv1d_host(&[], &[1.0, 2.0], Conv1dShape::new(0, 1, 2)).unwrap();
        assert!(output.is_empty());
    }

    /// Verifies that zero channel_count is rejected as an invalid shape.
    ///
    /// This catches missing dimension validation.
    #[test]
    fn rejects_invalid_shape() {
        let error = depthwise_causal_conv1d_host(&[], &[], Conv1dShape::new(1, 0, 1)).unwrap_err();
        assert_eq!(error, Conv1dError::InvalidShape(Conv1dShape::new(1, 0, 1)));
    }

    /// Verifies that an input buffer shorter than expected is rejected.
    ///
    /// This catches missing input length validation.
    #[test]
    fn rejects_input_length_mismatch() {
        let error = depthwise_causal_conv1d_host(&[1.0], &[1.0, 2.0], Conv1dShape::new(2, 1, 2))
            .unwrap_err();

        assert_eq!(
            error,
            Conv1dError::LengthMismatch {
                argument: "input",
                expected: 2,
                actual: 1,
            }
        );
    }

    /// Verifies that a weight buffer shorter than expected is rejected.
    ///
    /// This catches missing weight length validation.
    #[test]
    fn rejects_weight_length_mismatch() {
        let error = depthwise_causal_conv1d_host(&[1.0, 2.0], &[1.0], Conv1dShape::new(2, 1, 2))
            .unwrap_err();

        assert_eq!(
            error,
            Conv1dError::LengthMismatch {
                argument: "weights",
                expected: 2,
                actual: 1,
            }
        );
    }

    /// Verifies that a too-small output buffer is rejected in the _into variant.
    ///
    /// This catches missing output length validation.
    #[test]
    fn rejects_output_length_mismatch() {
        let mut output = [0.0; 1];
        let error = depthwise_causal_conv1d_into_host(
            &[1.0, 2.0],
            &[1.0, 0.5],
            Conv1dShape::new(2, 1, 2),
            &mut output,
        )
        .unwrap_err();

        assert_eq!(
            error,
            Conv1dError::LengthMismatch {
                argument: "output",
                expected: 2,
                actual: 1,
            }
        );
    }

    /// Verifies that async GPU Conv1D preserves host-reference parity and output shape for a cutile-eligible shape.
    ///
    /// This catches regressions in the Linux cutile-backed Conv1D path while keeping the public async API stable.
    #[tokio::test]
    async fn gpu_conv1d_matches_host_reference() {
        let shape = Conv1dShape::new(16, 16, 4);
        let input = (0..shape.input_len())
            .map(|index| ((index % 23) as f32 - 11.0) * 0.125)
            .collect::<Vec<_>>();
        let weights = (0..shape.weight_len())
            .map(|index| ((index % 13) as f32 - 6.0) * 0.0625)
            .collect::<Vec<_>>();
        let expected = depthwise_causal_conv1d_host(&input, &weights, shape).unwrap();
        let gpu_input =
            GpuTensor::from_host(&input, &[shape.sequence_len, shape.channel_count]).unwrap();
        let gpu_weights =
            GpuTensor::from_host(&weights, &[shape.channel_count, shape.kernel_size]).unwrap();
        let result = super::depthwise_causal_conv1d(&gpu_input, &gpu_weights, shape)
            .await
            .unwrap();
        assert_eq!(result.shape(), gpu_input.shape());
        approx_eq_slice_with_tolerance(&result.to_host(), &expected, 5e-4);
    }

    /// Verifies that Linux keeps Conv1D shapes with unaligned sequence lengths on the host bridge instead of forcing cutile dispatch.
    ///
    /// This catches regressions where unsupported fixture-like shapes would be sent down the device path anyway.
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn linux_gpu_conv1d_falls_back_for_unaligned_sequence() {
        let shape = Conv1dShape::new(12, 16, 4);
        let input = (0..shape.input_len())
            .map(|index| ((index % 19) as f32 - 9.0) * 0.125)
            .collect::<Vec<_>>();
        let weights = (0..shape.weight_len())
            .map(|index| ((index % 11) as f32 - 5.0) * 0.0625)
            .collect::<Vec<_>>();
        let expected = depthwise_causal_conv1d_host(&input, &weights, shape).unwrap();
        let gpu_input =
            GpuTensor::from_host(&input, &[shape.sequence_len, shape.channel_count]).unwrap();
        let gpu_weights =
            GpuTensor::from_host(&weights, &[shape.channel_count, shape.kernel_size]).unwrap();

        assert_eq!(backend_for_shape(shape), Conv1dBackend::HostFallback);

        let result = super::depthwise_causal_conv1d(&gpu_input, &gpu_weights, shape)
            .await
            .unwrap();

        approx_eq_slice_with_tolerance(&result.to_host(), &expected, 1e-6);
    }
}
