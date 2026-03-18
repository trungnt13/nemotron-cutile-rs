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

pub fn pack_int4(values: &[u8]) -> Result<Vec<u8>, QuantizeError> {
    let mut packed = vec![0; packed_int4_len(values.len())];
    pack_int4_into(values, &mut packed)?;
    Ok(packed)
}

pub fn pack_int4_into(values: &[u8], packed: &mut [u8]) -> Result<(), QuantizeError> {
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

pub fn unpack_int4(packed: &[u8], value_count: usize) -> Result<Vec<u8>, QuantizeError> {
    let mut values = vec![0; value_count];
    unpack_int4_into(packed, value_count, &mut values)?;
    Ok(values)
}

pub fn unpack_int4_into(
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

pub fn quantize_int4(
    values: &[f32],
    params: Int4QuantizationParams,
) -> Result<Vec<u8>, QuantizeError> {
    let mut packed = vec![0; packed_int4_len(values.len())];
    quantize_int4_into(values, params, &mut packed)?;
    Ok(packed)
}

pub fn quantize_int4_into(
    values: &[f32],
    params: Int4QuantizationParams,
    packed: &mut [u8],
) -> Result<(), QuantizeError> {
    validate_int4_params(params)?;

    let mut quantized = vec![0; values.len()];
    for (index, value) in values.iter().copied().enumerate() {
        quantized[index] = quantize_scalar(value, params, index)?;
    }

    pack_int4_into(&quantized, packed)
}

pub fn dequantize_int4(
    packed: &[u8],
    value_count: usize,
    params: Int4QuantizationParams,
) -> Result<Vec<f32>, QuantizeError> {
    let mut output = vec![0.0; value_count];
    dequantize_int4_into(packed, value_count, params, &mut output)?;
    Ok(output)
}

pub fn dequantize_int4_into(
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

    let unpacked = unpack_int4(packed, value_count)?;
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

#[derive(Clone, Copy, Debug, PartialEq)]
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

    #[test]
    fn packs_and_unpacks_even_number_of_values() {
        let packed = pack_int4(&[0x1, 0xA, 0x3, 0xF]).unwrap();
        assert_eq!(packed, vec![0xA1, 0xF3]);

        let unpacked = unpack_int4(&packed, 4).unwrap();
        assert_eq!(unpacked, vec![0x1, 0xA, 0x3, 0xF]);
    }

    #[test]
    fn packs_and_unpacks_odd_number_of_values() {
        let packed = pack_int4(&[0x2, 0x4, 0x6]).unwrap();
        assert_eq!(packed, vec![0x42, 0x06]);

        let unpacked = unpack_int4(&packed, 3).unwrap();
        assert_eq!(unpacked, vec![0x2, 0x4, 0x6]);
    }

    #[test]
    fn quantizes_and_dequantizes_with_affine_params() {
        let params = Int4QuantizationParams::new(0.5, 8);
        let input = [-1.0, -0.5, 0.0, 0.5, 1.0];

        let packed = quantize_int4(&input, params).unwrap();
        assert_eq!(packed, vec![0x76, 0x98, 0x0A]);

        let output = dequantize_int4(&packed, input.len(), params).unwrap();
        approx_eq_slice(&output, &input);
    }

    #[test]
    fn quantization_clamps_to_int4_range() {
        let params = Int4QuantizationParams::new(1.0, 8);
        let packed = quantize_int4(&[-100.0, 100.0], params).unwrap();
        assert_eq!(packed, vec![0xF0]);

        let output = dequantize_int4(&packed, 2, params).unwrap();
        approx_eq(output[0], -8.0);
        approx_eq(output[1], 7.0);
    }

    #[test]
    fn quantize_into_writes_existing_buffer() {
        let params = Int4QuantizationParams::new(0.25, 0);
        let mut packed = [0xFF; 2];

        quantize_int4_into(&[0.0, 0.25, 0.5], params, &mut packed).unwrap();

        assert_eq!(packed, [0x10, 0x02]);
    }

    #[test]
    fn dequantize_into_writes_existing_buffer() {
        let params = Int4QuantizationParams::new(0.25, 2);
        let mut output = [-1.0; 3];

        dequantize_int4_into(&[0x31, 0x04], 3, params, &mut output).unwrap();

        approx_eq_slice(&output, &[-0.25, 0.25, 0.5]);
    }

    #[test]
    fn handles_empty_inputs() {
        let params = Int4QuantizationParams::new(0.5, 8);

        assert_eq!(pack_int4(&[]).unwrap(), Vec::<u8>::new());
        assert_eq!(unpack_int4(&[], 0).unwrap(), Vec::<u8>::new());
        assert_eq!(quantize_int4(&[], params).unwrap(), Vec::<u8>::new());
        assert_eq!(dequantize_int4(&[], 0, params).unwrap(), Vec::<f32>::new());
    }

    #[test]
    fn rejects_invalid_quantization_params() {
        let error = quantize_int4(&[0.0], Int4QuantizationParams::new(0.0, 8)).unwrap_err();
        assert_eq!(
            error,
            QuantizeError::InvalidParams(Int4QuantizationParams::new(0.0, 8))
        );
    }

    #[test]
    fn rejects_out_of_range_int4_value() {
        let error = pack_int4(&[16]).unwrap_err();
        assert_eq!(error, QuantizeError::Int4OutOfRange(16));
    }

    #[test]
    fn rejects_packed_length_mismatch() {
        let mut packed = [0; 0];
        let error = quantize_int4_into(
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

    #[test]
    fn rejects_output_length_mismatch() {
        let mut output = [0.0; 1];
        let error =
            dequantize_int4_into(&[0x10], 2, Int4QuantizationParams::new(1.0, 0), &mut output)
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

    #[test]
    fn rejects_non_finite_input() {
        let error = quantize_int4(&[f32::NAN], Int4QuantizationParams::new(1.0, 8)).unwrap_err();

        match error {
            QuantizeError::NonFiniteInput { index, value } => {
                assert_eq!(index, 0);
                assert!(value.is_nan());
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
