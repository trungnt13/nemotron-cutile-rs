use crate::tensor::{GpuTensor, TensorError};
use crate::KernelStub;

pub const SPEC: KernelStub = KernelStub {
    name: "rms_norm_host",
    summary: "RMSNorm and gated RMSNorm kernels.",
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RmsNormBackend {
    HostFallback,
    Cutile,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RmsNormKernel {
    pub name: &'static str,
    pub backend: RmsNormBackend,
}

#[cfg(target_os = "linux")]
const RMS_NORM_BACKEND: RmsNormBackend = RmsNormBackend::Cutile;
#[cfg(not(target_os = "linux"))]
const RMS_NORM_BACKEND: RmsNormBackend = RmsNormBackend::HostFallback;

pub const RMS_NORM: RmsNormKernel = RmsNormKernel {
    name: "rms_norm_host",
    backend: RMS_NORM_BACKEND,
};

#[cfg(target_os = "linux")]
const GATED_RMS_NORM_BACKEND: RmsNormBackend = RmsNormBackend::Cutile;
#[cfg(not(target_os = "linux"))]
const GATED_RMS_NORM_BACKEND: RmsNormBackend = RmsNormBackend::HostFallback;

pub const GATED_RMS_NORM: RmsNormKernel = RmsNormKernel {
    name: "gated_rms_norm_host",
    backend: GATED_RMS_NORM_BACKEND,
};

pub fn supported_rms_norm_kernels() -> [RmsNormKernel; 2] {
    [RMS_NORM, GATED_RMS_NORM]
}

pub fn rms_norm_host(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>, RmsNormError> {
    let mut output = vec![0.0; input.len()];
    rms_norm_into_host(input, weight, epsilon, &mut output)?;
    Ok(output)
}

pub fn rms_norm_in_place_host(
    values: &mut [f32],
    weight: &[f32],
    epsilon: f32,
) -> Result<(), RmsNormError> {
    let output = rms_norm_host(values, weight, epsilon)?;
    values.copy_from_slice(&output);
    Ok(())
}

pub fn rms_norm_into_host(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    output: &mut [f32],
) -> Result<(), RmsNormError> {
    validate_lengths(input, weight, None, output)?;
    apply_rms_norm(input, weight, None, epsilon, output)
}

pub fn gated_rms_norm_host(
    input: &[f32],
    weight: &[f32],
    gate: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>, RmsNormError> {
    let mut output = vec![0.0; input.len()];
    gated_rms_norm_into_host(input, weight, gate, epsilon, &mut output)?;
    Ok(output)
}

pub fn gated_rms_norm_in_place_host(
    values: &mut [f32],
    weight: &[f32],
    gate: &[f32],
    epsilon: f32,
) -> Result<(), RmsNormError> {
    let output = gated_rms_norm_host(values, weight, gate, epsilon)?;
    values.copy_from_slice(&output);
    Ok(())
}

pub fn gated_rms_norm_into_host(
    input: &[f32],
    weight: &[f32],
    gate: &[f32],
    epsilon: f32,
    output: &mut [f32],
) -> Result<(), RmsNormError> {
    validate_lengths(input, weight, Some(gate), output)?;
    apply_rms_norm(input, weight, Some(gate), epsilon, output)
}

pub fn rms_host(input: &[f32], epsilon: f32) -> Result<f32, RmsNormError> {
    if input.is_empty() {
        return Err(RmsNormError::EmptyInput);
    }

    if epsilon < 0.0 {
        return Err(RmsNormError::NegativeEpsilon(epsilon));
    }

    let mean_square = input
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        / input.len() as f64;
    Ok((mean_square as f32 + epsilon).sqrt())
}

fn apply_rms_norm(
    input: &[f32],
    weight: &[f32],
    gate: Option<&[f32]>,
    epsilon: f32,
    output: &mut [f32],
) -> Result<(), RmsNormError> {
    let denom = rms_host(input, epsilon)?;
    let scale = denom.recip();

    for index in 0..input.len() {
        let mut value = input[index] * scale * weight[index];
        if let Some(gate) = gate {
            value *= gate[index];
        }
        output[index] = value;
    }

    Ok(())
}

fn validate_lengths(
    input: &[f32],
    weight: &[f32],
    gate: Option<&[f32]>,
    output: &mut [f32],
) -> Result<(), RmsNormError> {
    if input.is_empty() {
        return Err(RmsNormError::EmptyInput);
    }

    if input.len() != weight.len() {
        return Err(RmsNormError::LengthMismatch {
            expected: input.len(),
            actual: weight.len(),
            argument: "weight",
        });
    }

    if let Some(gate) = gate {
        if input.len() != gate.len() {
            return Err(RmsNormError::LengthMismatch {
                expected: input.len(),
                actual: gate.len(),
                argument: "gate",
            });
        }
    }

    if input.len() != output.len() {
        return Err(RmsNormError::LengthMismatch {
            expected: input.len(),
            actual: output.len(),
            argument: "output",
        });
    }

    Ok(())
}

fn validate_tensor_lengths(
    input: &GpuTensor,
    weight: &GpuTensor,
    gate: Option<&GpuTensor>,
    epsilon: f32,
) -> Result<(), RmsNormError> {
    if input.numel() == 0 {
        return Err(RmsNormError::EmptyInput);
    }

    if epsilon < 0.0 {
        return Err(RmsNormError::NegativeEpsilon(epsilon));
    }

    let row_width = input.shape()[input.ndim() - 1];

    if row_width != weight.numel() {
        return Err(RmsNormError::LengthMismatch {
            expected: row_width,
            actual: weight.numel(),
            argument: "weight",
        });
    }

    if let Some(gate) = gate {
        if input.numel() != gate.numel() {
            return Err(RmsNormError::LengthMismatch {
                expected: input.numel(),
                actual: gate.numel(),
                argument: "gate",
            });
        }
    }

    Ok(())
}

fn rms_norm_row_shape(input: &GpuTensor) -> (usize, usize) {
    let row_width = input.shape()[input.ndim() - 1];
    (input.numel() / row_width, row_width)
}

fn rms_norm_rows_host(
    input: &[f32],
    weight: &[f32],
    row_width: usize,
    epsilon: f32,
) -> Result<Vec<f32>, RmsNormError> {
    let mut output = Vec::with_capacity(input.len());
    for row in input.chunks_exact(row_width) {
        output.extend(rms_norm_host(row, weight, epsilon)?);
    }
    Ok(output)
}

fn gated_rms_norm_rows_host(
    input: &[f32],
    weight: &[f32],
    gate: &[f32],
    row_width: usize,
    epsilon: f32,
) -> Result<Vec<f32>, RmsNormError> {
    let mut output = Vec::with_capacity(input.len());
    for (row, gate_row) in input
        .chunks_exact(row_width)
        .zip(gate.chunks_exact(row_width))
    {
        output.extend(gated_rms_norm_host(row, weight, gate_row, epsilon)?);
    }
    Ok(output)
}

#[cfg(any(target_os = "linux", test))]
const CUTILE_BLOCK_SIZES: [usize; 9] = [256, 128, 64, 32, 16, 8, 4, 2, 1];

#[cfg(any(target_os = "linux", test))]
fn select_cutile_block_size(numel: usize) -> usize {
    CUTILE_BLOCK_SIZES
        .into_iter()
        .find(|block_size| numel % block_size == 0)
        .unwrap_or(1)
}

#[derive(Clone, Debug, PartialEq)]
pub enum RmsNormError {
    EmptyInput,
    NegativeEpsilon(f32),
    LengthMismatch {
        expected: usize,
        actual: usize,
        argument: &'static str,
    },
    DeviceError(String),
}

impl From<TensorError> for RmsNormError {
    fn from(e: TensorError) -> Self {
        RmsNormError::DeviceError(e.to_string())
    }
}

#[cfg(target_os = "linux")]
#[cutile::module]
mod cutile_rms_norm_kernel {
    use cutile::core::*;

    #[cutile::entry()]
    fn rms_norm<const N: i32, const BLOCK_SIZE: i32>(
        x: &Tensor<f32, { [-1, N] }>,
        w: &Tensor<f32, { [N] }>,
        out: &mut Tensor<f32, { [1, N] }>,
        epsilon: f32,
    ) {
        let tile_shape: Shape<{ [1, BLOCK_SIZE] }> = const_shape![1, BLOCK_SIZE];
        let pid = get_tile_block_id();
        let row = pid.0;
        let num_tiles: i32 = N / BLOCK_SIZE;
        let x_part: Partition<f32, { [1, BLOCK_SIZE] }> = x.partition(tile_shape);
        let mut rms: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        for tile_index in 0i32..num_tiles {
            let x_tile: Tile<f32, { [1, BLOCK_SIZE] }> = x_part.load([row, tile_index]);
            rms = rms + x_tile * x_tile;
        }
        let rms: Tile<f32, { [1] }> = reduce_sum(rms, 1i32);
        let rms: Tile<f32, { [] }> = rms.reshape(const_shape![]);
        let rms: f32 = tile_to_scalar(rms);
        let n: f32 = convert_scalar(N);
        let rms: f32 = 1.0f32 / (rms / n + epsilon);
        let rms: Tile<f32, { [] }> = sqrt(scalar_to_tile(rms), "negative_inf");
        let rms: f32 = tile_to_scalar(rms);
        let rms: Tile<f32, { [1, BLOCK_SIZE] }> = rms.broadcast(tile_shape);
        let w_part: Partition<f32, { [BLOCK_SIZE] }> = w.partition(const_shape![BLOCK_SIZE]);
        let mut out_part: PartitionMut<f32, { [1, BLOCK_SIZE] }> =
            unsafe { out.partition_mut(tile_shape) };
        for tile_index in 0i32..num_tiles {
            let x_tile: Tile<f32, { [1, BLOCK_SIZE] }> = x_part.load([row, tile_index]);
            let w_tile: Tile<f32, { [1, BLOCK_SIZE] }> =
                w_part.load([tile_index]).reshape(tile_shape);
            let out_tile: Tile<f32, { [1, BLOCK_SIZE] }> = x_tile * rms * w_tile;
            unsafe { out_part.store(out_tile, [0i32, tile_index]) };
        }
    }

    #[cutile::entry()]
    fn gated_rms_norm<const N: i32, const BLOCK_SIZE: i32>(
        x: &Tensor<f32, { [-1, N] }>,
        w: &Tensor<f32, { [N] }>,
        gate: &Tensor<f32, { [-1, N] }>,
        out: &mut Tensor<f32, { [1, N] }>,
        epsilon: f32,
    ) {
        let tile_shape: Shape<{ [1, BLOCK_SIZE] }> = const_shape![1, BLOCK_SIZE];
        let pid = get_tile_block_id();
        let row = pid.0;
        let num_tiles: i32 = N / BLOCK_SIZE;
        let x_part: Partition<f32, { [1, BLOCK_SIZE] }> = x.partition(tile_shape);
        let mut rms: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        for tile_index in 0i32..num_tiles {
            let x_tile: Tile<f32, { [1, BLOCK_SIZE] }> = x_part.load([row, tile_index]);
            rms = rms + x_tile * x_tile;
        }
        let rms: Tile<f32, { [1] }> = reduce_sum(rms, 1i32);
        let rms: Tile<f32, { [] }> = rms.reshape(const_shape![]);
        let rms: f32 = tile_to_scalar(rms);
        let n: f32 = convert_scalar(N);
        let rms: f32 = 1.0f32 / (rms / n + epsilon);
        let rms: Tile<f32, { [] }> = sqrt(scalar_to_tile(rms), "negative_inf");
        let rms: f32 = tile_to_scalar(rms);
        let rms: Tile<f32, { [1, BLOCK_SIZE] }> = rms.broadcast(tile_shape);
        let w_part: Partition<f32, { [BLOCK_SIZE] }> = w.partition(const_shape![BLOCK_SIZE]);
        let gate_part: Partition<f32, { [1, BLOCK_SIZE] }> = gate.partition(tile_shape);
        let mut out_part: PartitionMut<f32, { [1, BLOCK_SIZE] }> =
            unsafe { out.partition_mut(tile_shape) };
        for tile_index in 0i32..num_tiles {
            let x_tile: Tile<f32, { [1, BLOCK_SIZE] }> = x_part.load([row, tile_index]);
            let w_tile: Tile<f32, { [1, BLOCK_SIZE] }> =
                w_part.load([tile_index]).reshape(tile_shape);
            let gate_tile: Tile<f32, { [1, BLOCK_SIZE] }> = gate_part.load([row, tile_index]);
            let out_tile: Tile<f32, { [1, BLOCK_SIZE] }> = x_tile * rms * w_tile * gate_tile;
            unsafe { out_part.store(out_tile, [0i32, tile_index]) };
        }
    }
}

async fn rms_norm_host_bridge(
    input: &GpuTensor,
    weight: &GpuTensor,
    epsilon: f32,
) -> Result<GpuTensor, RmsNormError> {
    let input_data = input.to_host_async().await?;
    let weight_data = weight.to_host_async().await?;
    let (_, row_width) = rms_norm_row_shape(input);
    let result = rms_norm_rows_host(&input_data, &weight_data, row_width, epsilon)?;
    Ok(GpuTensor::from_host_async(&result, input.shape()).await?)
}

async fn gated_rms_norm_host_bridge(
    input: &GpuTensor,
    weight: &GpuTensor,
    gate: &GpuTensor,
    epsilon: f32,
) -> Result<GpuTensor, RmsNormError> {
    let input_data = input.to_host_async().await?;
    let weight_data = weight.to_host_async().await?;
    let gate_data = gate.to_host_async().await?;
    let (_, row_width) = rms_norm_row_shape(input);
    let result =
        gated_rms_norm_rows_host(&input_data, &weight_data, &gate_data, row_width, epsilon)?;
    Ok(GpuTensor::from_host_async(&result, input.shape()).await?)
}

#[cfg(target_os = "linux")]
async fn rms_norm_cutile(
    input: &GpuTensor,
    weight: &GpuTensor,
    epsilon: f32,
) -> Result<Option<GpuTensor>, RmsNormError> {
    use cutile::tile_kernel::{IntoDeviceOperation, IntoDeviceOperationPartition, TileKernel};

    let (row_count, row_width) = rms_norm_row_shape(input);
    let Ok(row_width_i32) = i32::try_from(row_width) else {
        return Ok(None);
    };
    let block_size = select_cutile_block_size(row_width);
    let block_size_i32 = i32::try_from(block_size)
        .map_err(|_| RmsNormError::DeviceError("cutile block size overflowed i32".to_string()))?;
    let input_tensor = input
        .cutile_tensor_for_shape(&[row_count, row_width])
        .await?;
    let weight_tensor = weight.cutile_tensor_for_shape(&[row_width]).await?;
    let output =
        cutile::api::zeros::<2, f32>([row_count, row_width]).partition([1, block_size_i32]);
    let (_input, _weight, output, _epsilon) = cutile_rms_norm_kernel::rms_norm_async(
        input_tensor.device_operation(),
        weight_tensor.device_operation(),
        output,
        epsilon.device_operation(),
    )
    .generics(vec![row_width_i32.to_string(), block_size_i32.to_string()])
    .await
    .map_err(|error| RmsNormError::DeviceError(format!("{error:?}")))?;
    let output = output.unpartition().reshape_dyn(input.shape());
    Ok(Some(GpuTensor::from_cutile_tensor(output, input.shape())?))
}

#[cfg(target_os = "linux")]
async fn gated_rms_norm_cutile(
    input: &GpuTensor,
    weight: &GpuTensor,
    gate: &GpuTensor,
    epsilon: f32,
) -> Result<Option<GpuTensor>, RmsNormError> {
    use cutile::tile_kernel::{IntoDeviceOperation, IntoDeviceOperationPartition, TileKernel};

    let (row_count, row_width) = rms_norm_row_shape(input);
    let Ok(row_width_i32) = i32::try_from(row_width) else {
        return Ok(None);
    };
    let block_size = select_cutile_block_size(row_width);
    let block_size_i32 = i32::try_from(block_size)
        .map_err(|_| RmsNormError::DeviceError("cutile block size overflowed i32".to_string()))?;
    let input_tensor = input
        .cutile_tensor_for_shape(&[row_count, row_width])
        .await?;
    let weight_tensor = weight.cutile_tensor_for_shape(&[row_width]).await?;
    let gate_tensor = gate
        .cutile_tensor_for_shape(&[row_count, row_width])
        .await?;
    let output =
        cutile::api::zeros::<2, f32>([row_count, row_width]).partition([1, block_size_i32]);
    let (_input, _weight, _gate, output, _epsilon) = cutile_rms_norm_kernel::gated_rms_norm_async(
        input_tensor.device_operation(),
        weight_tensor.device_operation(),
        gate_tensor.device_operation(),
        output,
        epsilon.device_operation(),
    )
    .generics(vec![row_width_i32.to_string(), block_size_i32.to_string()])
    .await
    .map_err(|error| RmsNormError::DeviceError(format!("{error:?}")))?;
    let output = output.unpartition().reshape_dyn(input.shape());
    Ok(Some(GpuTensor::from_cutile_tensor(output, input.shape())?))
}

// ---------------------------------------------------------------------------
// Async GPU API
// ---------------------------------------------------------------------------

/// Async GPU RMS normalization.
pub async fn rms_norm(
    input: &GpuTensor,
    weight: &GpuTensor,
    epsilon: f32,
) -> Result<GpuTensor, RmsNormError> {
    validate_tensor_lengths(input, weight, None, epsilon)?;
    #[cfg(target_os = "linux")]
    if let Some(result) = rms_norm_cutile(input, weight, epsilon).await? {
        return Ok(result);
    }
    rms_norm_host_bridge(input, weight, epsilon).await
}

/// Async GPU gated RMS normalization.
pub async fn gated_rms_norm(
    input: &GpuTensor,
    weight: &GpuTensor,
    gate: &GpuTensor,
    epsilon: f32,
) -> Result<GpuTensor, RmsNormError> {
    validate_tensor_lengths(input, weight, Some(gate), epsilon)?;
    #[cfg(target_os = "linux")]
    if let Some(result) = gated_rms_norm_cutile(input, weight, gate, epsilon).await? {
        return Ok(result);
    }
    gated_rms_norm_host_bridge(input, weight, gate, epsilon).await
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

    fn assert_slice_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len(), "slice lengths differ");
        for (index, (actual_value, expected_value)) in
            actual.iter().zip(expected.iter()).enumerate()
        {
            let diff = (actual_value - expected_value).abs();
            assert!(
                diff <= tolerance,
                "values differ at index {index}: left={actual_value:?}, right={expected_value:?}, diff={diff:?}"
            );
        }
    }

    /// Verifies that both RMSNorm kernels report the expected backend for the current platform.
    ///
    /// This catches backend registry drift when Linux cutile support or host fallback wiring changes.
    #[test]
    fn reports_expected_backends_for_platform() {
        let expected_backend = if cfg!(target_os = "linux") {
            RmsNormBackend::Cutile
        } else {
            RmsNormBackend::HostFallback
        };

        assert_eq!(
            supported_rms_norm_kernels(),
            [
                RmsNormKernel {
                    name: "rms_norm_host",
                    backend: expected_backend,
                },
                RmsNormKernel {
                    name: "gated_rms_norm_host",
                    backend: expected_backend,
                },
            ]
        );
    }

    /// Verifies that the cutile block-size selector chooses divisors of the input length when tiling device work.
    ///
    /// This catches kernel launches that would otherwise require partial tiles and out-of-bounds loads.
    #[test]
    fn selects_divisible_cutile_block_sizes() {
        assert_eq!(select_cutile_block_size(256), 256);
        assert_eq!(select_cutile_block_size(192), 64);
        assert_eq!(select_cutile_block_size(30), 2);
        assert_eq!(select_cutile_block_size(17), 1);
    }

    /// Verifies RMSNorm with unit weights against hand-computed reference values.
    ///
    /// This catches errors in the RMS denominator or normalization formula.
    #[test]
    fn rms_norm_matches_reference_values() {
        let output = rms_norm_host(&[1.0, 2.0], &[1.0, 1.0], 1e-5).unwrap();

        approx_eq(output[0], 0.6324543);
        approx_eq(output[1], 1.2649086);
    }

    /// Verifies that per-element weights scale the normalized output.
    ///
    /// This catches missing or swapped weight multiplication.
    #[test]
    fn rms_norm_applies_weights() {
        let output = rms_norm_host(&[1.0, 2.0], &[2.0, 0.5], 1e-5).unwrap();

        approx_eq(output[0], 1.2649086);
        approx_eq(output[1], 0.6324543);
    }

    /// Verifies that gated RMSNorm multiplies the gate vector after normalization.
    ///
    /// This catches missing gate multiplication or wrong application order.
    #[test]
    fn gated_rms_norm_multiplies_gate_after_normalization() {
        let output = gated_rms_norm_host(&[1.0, 2.0], &[1.0, 1.0], &[2.0, 0.5], 1e-5).unwrap();

        approx_eq(output[0], 1.2649086);
        approx_eq(output[1], 0.6324543);
    }

    /// Verifies that the in-place RMSNorm variant mutates the buffer correctly.
    ///
    /// This catches issues where in-place writes are skipped or mis-ordered.
    #[test]
    fn rms_norm_in_place_updates_buffer() {
        let mut values = [1.0, 2.0];
        rms_norm_in_place_host(&mut values, &[1.0, 1.0], 1e-5).unwrap();

        approx_eq(values[0], 0.6324543);
        approx_eq(values[1], 1.2649086);
    }

    /// Verifies that the in-place gated RMSNorm variant mutates the buffer correctly.
    ///
    /// This catches issues where in-place writes are skipped or mis-ordered.
    #[test]
    fn gated_rms_norm_in_place_updates_buffer() {
        let mut values = [1.0, 2.0];
        gated_rms_norm_in_place_host(&mut values, &[1.0, 1.0], &[0.5, 2.0], 1e-5).unwrap();

        approx_eq(values[0], 0.31622714);
        approx_eq(values[1], 2.529817);
    }

    /// Verifies that all-zero input produces all-zero output (0/sqrt(eps) * w * 0 = 0).
    ///
    /// This catches NaN or infinity from dividing zero by a near-zero RMS.
    #[test]
    fn zero_input_stays_zero() {
        let output = rms_norm_host(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0], 1e-5).unwrap();
        assert_eq!(output, vec![0.0, 0.0, 0.0]);
    }

    /// Verifies that mismatched input/weight lengths are rejected.
    ///
    /// This catches missing weight length validation.
    #[test]
    fn rejects_length_mismatch() {
        let error = rms_norm_host(&[1.0, 2.0], &[1.0], 1e-5).unwrap_err();
        assert_eq!(
            error,
            RmsNormError::LengthMismatch {
                expected: 2,
                actual: 1,
                argument: "weight",
            }
        );
    }

    /// Verifies that a mismatched gate length is rejected.
    ///
    /// This catches missing gate length validation.
    #[test]
    fn rejects_gate_length_mismatch() {
        let error = gated_rms_norm_host(&[1.0, 2.0], &[1.0, 1.0], &[1.0], 1e-5).unwrap_err();
        assert_eq!(
            error,
            RmsNormError::LengthMismatch {
                expected: 2,
                actual: 1,
                argument: "gate",
            }
        );
    }

    /// Verifies that a negative epsilon is rejected.
    ///
    /// This catches missing epsilon sign validation.
    #[test]
    fn rejects_negative_epsilon() {
        let error = rms_norm_host(&[1.0, 2.0], &[1.0, 1.0], -1.0).unwrap_err();
        assert_eq!(error, RmsNormError::NegativeEpsilon(-1.0));
    }

    /// Verifies that empty input is rejected.
    ///
    /// This catches division by zero in the mean-square computation.
    #[test]
    fn rejects_empty_input() {
        let error = rms_norm_host(&[], &[], 1e-5).unwrap_err();
        assert_eq!(error, RmsNormError::EmptyInput);
    }

    fn rms_norm_rows_reference(
        input: &[f32],
        weight: &[f32],
        row_width: usize,
        epsilon: f32,
    ) -> Vec<f32> {
        input
            .chunks_exact(row_width)
            .flat_map(|row| rms_norm_host(row, weight, epsilon).unwrap())
            .collect()
    }

    fn gated_rms_norm_rows_reference(
        input: &[f32],
        weight: &[f32],
        gate: &[f32],
        row_width: usize,
        epsilon: f32,
    ) -> Vec<f32> {
        input
            .chunks_exact(row_width)
            .zip(gate.chunks_exact(row_width))
            .flat_map(|(row, gate_row)| {
                gated_rms_norm_host(row, weight, gate_row, epsilon).unwrap()
            })
            .collect()
    }

    /// Verifies that the async GPU RMS norm preserves row-wise RMS semantics for matrix-shaped input.
    ///
    /// This catches regressions where the GPU path accidentally normalizes the flattened tensor instead of each row.
    #[tokio::test]
    async fn gpu_rms_norm_matches_host_rowwise_contract() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![0.5, 1.0];
        let epsilon = 1e-5;
        let expected = rms_norm_rows_reference(&input, &weight, 2, epsilon);
        let gpu_input = GpuTensor::from_host(&input, &[2, 2]).unwrap();
        let gpu_weight = GpuTensor::from_host(&weight, &[1, 2]).unwrap();
        let result = super::rms_norm(&gpu_input, &gpu_weight, epsilon)
            .await
            .unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_slice_close(&result.to_host(), &expected, 1e-5);
    }

    /// Verifies that async GPU gated RMSNorm preserves row-wise semantics and shape for matrix-shaped input.
    ///
    /// This catches regressions where gated GPU RMSNorm flattens rows or misaligns the gate tensor.
    #[tokio::test]
    async fn gpu_gated_rms_norm_matches_host_rowwise_contract() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![0.5, 1.0];
        let gate = vec![1.0, 0.5, 2.0, 1.5];
        let epsilon = 1e-5;
        let expected = gated_rms_norm_rows_reference(&input, &weight, &gate, 2, epsilon);
        let gpu_input = GpuTensor::from_host(&input, &[2, 2]).unwrap();
        let gpu_weight = GpuTensor::from_host(&weight, &[2]).unwrap();
        let gpu_gate = GpuTensor::from_host(&gate, &[4]).unwrap();

        let result = super::gated_rms_norm(&gpu_input, &gpu_weight, &gpu_gate, epsilon)
            .await
            .unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_slice_close(&result.to_host(), &expected, 1e-5);
    }

    /// Verifies that async GPU RMSNorm rejects weights that do not match the input row width.
    ///
    /// This catches regressions where batched GPU RMSNorm validates against total element count instead of the per-row contract.
    #[tokio::test]
    async fn gpu_rms_norm_rejects_weight_length_mismatch_for_rowwise_input() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 2.0, 3.0, 4.0];
        let gpu_input = GpuTensor::from_host(&input, &[2, 2]).unwrap();
        let gpu_weight = GpuTensor::from_host(&weight, &[4]).unwrap();

        let error = super::rms_norm(&gpu_input, &gpu_weight, 1e-5)
            .await
            .unwrap_err();

        assert_eq!(
            error,
            RmsNormError::LengthMismatch {
                expected: 2,
                actual: 4,
                argument: "weight",
            }
        );
    }
}
