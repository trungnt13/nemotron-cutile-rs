use crate::tensor::{GpuTensor, TensorError};
use crate::KernelStub;

pub const SPEC: KernelStub = KernelStub {
    name: "quantize",
    summary: "INT4 dequantization and quantization helpers.",
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantizeBackend {
    HostFallback,
    Cutile,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuantizeKernel {
    pub name: &'static str,
    pub backend: QuantizeBackend,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Int4QuantizationParams {
    pub scale: f32,
    pub zero_point: u8,
}

impl Int4QuantizationParams {
    pub const fn new(scale: f32, zero_point: u8) -> Self {
        Self { scale, zero_point }
    }
}

pub const INT4_AFFINE: QuantizeKernel = QuantizeKernel {
    name: "int4_affine",
    backend: QuantizeBackend::HostFallback,
};

pub fn supported_quantize_kernels() -> [QuantizeKernel; 1] {
    [INT4_AFFINE]
}

pub const fn packed_int4_len(value_count: usize) -> usize {
    value_count.div_ceil(2)
}

pub fn pack_int4_host(values: &[u8]) -> Result<Vec<u8>, QuantizeError> {
    let mut packed = vec![0; packed_int4_len(values.len())];
    pack_int4_into_host(values, &mut packed)?;
    Ok(packed)
}

pub fn pack_int4_into_host(values: &[u8], packed: &mut [u8]) -> Result<(), QuantizeError> {
    if packed.len() != packed_int4_len(values.len()) {
        return Err(QuantizeError::LengthMismatch {
            argument: "packed",
            expected: packed_int4_len(values.len()),
            actual: packed.len(),
        });
    }

    packed.fill(0);

    for (index, value) in values.iter().copied().enumerate() {
        if value > 0x0F {
            return Err(QuantizeError::Int4OutOfRange(value));
        }

        let byte_index = index / 2;
        let nibble = if index % 2 == 0 { value } else { value << 4 };
        packed[byte_index] |= nibble;
    }

    Ok(())
}

pub fn unpack_int4_host(packed: &[u8], value_count: usize) -> Result<Vec<u8>, QuantizeError> {
    let mut values = vec![0; value_count];
    unpack_int4_into_host(packed, value_count, &mut values)?;
    Ok(values)
}

pub fn unpack_int4_into_host(
    packed: &[u8],
    value_count: usize,
    output: &mut [u8],
) -> Result<(), QuantizeError> {
    if packed.len() != packed_int4_len(value_count) {
        return Err(QuantizeError::LengthMismatch {
            argument: "packed",
            expected: packed_int4_len(value_count),
            actual: packed.len(),
        });
    }

    if output.len() != value_count {
        return Err(QuantizeError::LengthMismatch {
            argument: "output",
            expected: value_count,
            actual: output.len(),
        });
    }

    for (index, slot) in output.iter_mut().enumerate() {
        let byte = packed[index / 2];
        *slot = if index % 2 == 0 {
            byte & 0x0F
        } else {
            (byte >> 4) & 0x0F
        };
    }

    Ok(())
}

pub fn quantize_int4_host(
    values: &[f32],
    params: Int4QuantizationParams,
) -> Result<Vec<u8>, QuantizeError> {
    let mut packed = vec![0; packed_int4_len(values.len())];
    quantize_int4_into_host(values, params, &mut packed)?;
    Ok(packed)
}

pub fn quantize_int4_into_host(
    values: &[f32],
    params: Int4QuantizationParams,
    packed: &mut [u8],
) -> Result<(), QuantizeError> {
    validate_int4_params(params)?;

    let mut quantized = vec![0; values.len()];
    for (index, value) in values.iter().copied().enumerate() {
        quantized[index] = quantize_scalar(value, params, index)?;
    }

    pack_int4_into_host(&quantized, packed)
}

pub fn dequantize_int4_host(
    packed: &[u8],
    value_count: usize,
    params: Int4QuantizationParams,
) -> Result<Vec<f32>, QuantizeError> {
    let mut output = vec![0.0; value_count];
    dequantize_int4_into_host(packed, value_count, params, &mut output)?;
    Ok(output)
}

pub fn dequantize_int4_into_host(
    packed: &[u8],
    value_count: usize,
    params: Int4QuantizationParams,
    output: &mut [f32],
) -> Result<(), QuantizeError> {
    validate_int4_params(params)?;

    if output.len() != value_count {
        return Err(QuantizeError::LengthMismatch {
            argument: "output",
            expected: value_count,
            actual: output.len(),
        });
    }

    let unpacked = unpack_int4_host(packed, value_count)?;
    for (code, slot) in unpacked.into_iter().zip(output.iter_mut()) {
        *slot = dequantize_scalar(code, params);
    }

    Ok(())
}

pub fn validate_int4_params(params: Int4QuantizationParams) -> Result<(), QuantizeError> {
    if !params.scale.is_finite() || params.scale <= 0.0 || params.zero_point > 0x0F {
        return Err(QuantizeError::InvalidParams(params));
    }

    Ok(())
}

fn quantize_scalar(
    value: f32,
    params: Int4QuantizationParams,
    index: usize,
) -> Result<u8, QuantizeError> {
    if !value.is_finite() {
        return Err(QuantizeError::NonFiniteInput { index, value });
    }

    let scaled = value / params.scale + f32::from(params.zero_point);
    let quantized = scaled.round().clamp(0.0, 15.0);
    Ok(quantized as u8)
}

fn dequantize_scalar(code: u8, params: Int4QuantizationParams) -> f32 {
    (f32::from(code) - f32::from(params.zero_point)) * params.scale
}

#[derive(Clone, Debug, PartialEq)]
pub enum QuantizeError {
    InvalidParams(Int4QuantizationParams),
    Int4OutOfRange(u8),
    LengthMismatch {
        argument: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFiniteInput {
        index: usize,
        value: f32,
    },
    DeviceError(String),
}

impl From<TensorError> for QuantizeError {
    fn from(e: TensorError) -> Self {
        QuantizeError::DeviceError(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Async GPU API
// ---------------------------------------------------------------------------

/// Async GPU INT4 dequantization.
pub async fn dequantize_int4(
    packed: &[u8],
    value_count: usize,
    params: Int4QuantizationParams,
) -> Result<GpuTensor, QuantizeError> {
    let result = dequantize_int4_host(packed, value_count, params)?;
    GpuTensor::from_host_async(&result, &[value_count])
        .await
        .map_err(|e| QuantizeError::DeviceError(e.to_string()))
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

    fn approx_eq_slice(lhs: &[f32], rhs: &[f32]) {
        assert_eq!(lhs.len(), rhs.len(), "slice lengths differ");
        for (index, (left, right)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let diff = (left - right).abs();
            assert!(
                diff <= 1e-6,
                "index {index}: left={left:?}, right={right:?}, diff={diff:?}"
            );
        }
    }

    /// Verifies that the quantize kernel reports HostFallback as its backend.
    ///
    /// This catches accidental backend tag changes before GPU kernels exist.
    #[test]
    fn reports_host_fallback_backend_for_now() {
        assert_eq!(
            supported_quantize_kernels(),
            [QuantizeKernel {
                name: "int4_affine",
                backend: QuantizeBackend::HostFallback,
            }]
        );
    }

    /// Verifies INT4 pack/unpack round-trips correctly with an even value count.
    ///
    /// This catches nibble ordering errors (low vs high nibble placement).
    #[test]
    fn packs_and_unpacks_even_number_of_values() {
        let packed = pack_int4_host(&[0x1, 0xA, 0x3, 0xF]).unwrap();
        assert_eq!(packed, vec![0xA1, 0xF3]);

        let unpacked = unpack_int4_host(&packed, 4).unwrap();
        assert_eq!(unpacked, vec![0x1, 0xA, 0x3, 0xF]);
    }

    /// Verifies INT4 pack/unpack with an odd value count (trailing nibble padding).
    ///
    /// This catches errors in the ceil-division packed length or trailing nibble handling.
    #[test]
    fn packs_and_unpacks_odd_number_of_values() {
        let packed = pack_int4_host(&[0x2, 0x4, 0x6]).unwrap();
        assert_eq!(packed, vec![0x42, 0x06]);

        let unpacked = unpack_int4_host(&packed, 3).unwrap();
        assert_eq!(unpacked, vec![0x2, 0x4, 0x6]);
    }

    /// Verifies that quantize→dequantize round-trips losslessly with chosen affine params.
    ///
    /// This catches errors in the scale/zero_point formula or nibble encoding.
    #[test]
    fn quantizes_and_dequantizes_with_affine_params() {
        let params = Int4QuantizationParams::new(0.5, 8);
        let input = [-1.0, -0.5, 0.0, 0.5, 1.0];

        let packed = quantize_int4_host(&input, params).unwrap();
        assert_eq!(packed, vec![0x76, 0x98, 0x0A]);

        let output = dequantize_int4_host(&packed, input.len(), params).unwrap();
        approx_eq_slice(&output, &input);
    }

    /// Verifies that out-of-range float values are clamped to the [0, 15] INT4 range.
    ///
    /// This catches missing clamp in the quantize scalar path.
    #[test]
    fn quantization_clamps_to_int4_range() {
        let params = Int4QuantizationParams::new(1.0, 8);
        let packed = quantize_int4_host(&[-100.0, 100.0], params).unwrap();
        assert_eq!(packed, vec![0xF0]);

        let output = dequantize_int4_host(&packed, 2, params).unwrap();
        approx_eq(output[0], -8.0);
        approx_eq(output[1], 7.0);
    }

    /// Verifies that the _into quantize variant writes into a pre-allocated buffer.
    ///
    /// This catches bugs where _into silently re-allocates instead of writing in place.
    #[test]
    fn quantize_into_writes_existing_buffer() {
        let params = Int4QuantizationParams::new(0.25, 0);
        let mut packed = [0xFF; 2];

        quantize_int4_into_host(&[0.0, 0.25, 0.5], params, &mut packed).unwrap();

        assert_eq!(packed, [0x10, 0x02]);
    }

    /// Verifies that the _into dequantize variant writes into a pre-allocated buffer.
    ///
    /// This catches bugs where _into silently re-allocates instead of writing in place.
    #[test]
    fn dequantize_into_writes_existing_buffer() {
        let params = Int4QuantizationParams::new(0.25, 2);
        let mut output = [-1.0; 3];

        dequantize_int4_into_host(&[0x31, 0x04], 3, params, &mut output).unwrap();

        approx_eq_slice(&output, &[-0.25, 0.25, 0.5]);
    }

    /// Verifies that all pack/unpack/quantize/dequantize functions accept empty inputs.
    ///
    /// This catches panics on zero-length slices.
    #[test]
    fn handles_empty_inputs() {
        let params = Int4QuantizationParams::new(0.5, 8);

        assert_eq!(pack_int4_host(&[]).unwrap(), Vec::<u8>::new());
        assert_eq!(unpack_int4_host(&[], 0).unwrap(), Vec::<u8>::new());
        assert_eq!(quantize_int4_host(&[], params).unwrap(), Vec::<u8>::new());
        assert_eq!(
            dequantize_int4_host(&[], 0, params).unwrap(),
            Vec::<f32>::new()
        );
    }

    /// Verifies that a zero scale is rejected as invalid quantization params.
    ///
    /// This catches missing scale validation (zero scale causes division by zero).
    #[test]
    fn rejects_invalid_quantization_params() {
        let error = quantize_int4_host(&[0.0], Int4QuantizationParams::new(0.0, 8)).unwrap_err();
        assert_eq!(
            error,
            QuantizeError::InvalidParams(Int4QuantizationParams::new(0.0, 8))
        );
    }

    /// Verifies that a nibble value > 15 is rejected during packing.
    ///
    /// This catches missing range check in the pack path.
    #[test]
    fn rejects_out_of_range_int4_value() {
        let error = pack_int4_host(&[16]).unwrap_err();
        assert_eq!(error, QuantizeError::Int4OutOfRange(16));
    }

    /// Verifies that a too-small packed buffer is rejected in the _into variant.
    ///
    /// This catches missing packed length validation.
    #[test]
    fn rejects_packed_length_mismatch() {
        let mut packed = [0; 0];
        let error = quantize_int4_into_host(
            &[0.0, 1.0],
            Int4QuantizationParams::new(1.0, 8),
            &mut packed,
        )
        .unwrap_err();

        assert_eq!(
            error,
            QuantizeError::LengthMismatch {
                argument: "packed",
                expected: 1,
                actual: 0,
            }
        );
    }

    /// Verifies that a too-small output buffer is rejected in dequantize_into.
    ///
    /// This catches missing output length validation.
    #[test]
    fn rejects_output_length_mismatch() {
        let mut output = [0.0; 1];
        let error =
            dequantize_int4_into_host(&[0x10], 2, Int4QuantizationParams::new(1.0, 0), &mut output)
                .unwrap_err();

        assert_eq!(
            error,
            QuantizeError::LengthMismatch {
                argument: "output",
                expected: 2,
                actual: 1,
            }
        );
    }

    /// Verifies that NaN input is rejected during quantization.
    ///
    /// This catches missing non-finite input validation.
    #[test]
    fn rejects_non_finite_input() {
        let error =
            quantize_int4_host(&[f32::NAN], Int4QuantizationParams::new(1.0, 8)).unwrap_err();

        match error {
            QuantizeError::NonFiniteInput { index, value } => {
                assert_eq!(index, 0);
                assert!(value.is_nan());
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    /// Verifies that the async GPU INT4 dequantize wrapper matches the host fallback and preserves a 1D output shape when the packed input has an odd element count. This catches regressions in nibble unpacking, result upload, and shape reconstruction for trailing padded nibbles.
    #[tokio::test]
    async fn gpu_dequantize_int4_matches_host_fallback_and_preserves_shape() {
        let packed = vec![0x76, 0x98, 0x0A];
        let value_count = 5;
        let params = Int4QuantizationParams::new(0.5, 8);
        let expected = dequantize_int4_host(&packed, value_count, params).unwrap();

        let result = super::dequantize_int4(&packed, value_count, params)
            .await
            .unwrap();

        assert_eq!(result.shape(), &[value_count]);
        approx_eq_slice(&result.to_host(), &expected);
    }

    /// Verifies that the async GPU INT4 dequantize wrapper propagates invalid quantization parameters before uploading results. This catches regressions where wrapper-side error handling could hide host-contract validation failures.
    #[tokio::test]
    async fn gpu_dequantize_int4_rejects_invalid_params() {
        let error = super::dequantize_int4(&[0x10], 2, Int4QuantizationParams::new(0.0, 8))
            .await
            .unwrap_err();

        assert_eq!(
            error,
            QuantizeError::InvalidParams(Int4QuantizationParams::new(0.0, 8))
        );
    }
}
