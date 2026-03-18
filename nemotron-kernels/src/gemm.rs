use crate::KernelStub;

pub const SPEC: KernelStub = KernelStub {
    name: "gemm",
    summary: "Matrix multiply kernels adapted from cutile-rs examples.",
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GemmBackend {
    HostFallback,
    Cutile,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GemmKernel {
    pub name: &'static str,
    pub backend: GemmBackend,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GemmShape {
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

impl GemmShape {
    pub const fn new(m: usize, k: usize, n: usize) -> Self {
        Self { m, k, n }
    }

    pub const fn lhs_len(self) -> usize {
        self.m * self.k
    }

    pub const fn rhs_len(self) -> usize {
        self.k * self.n
    }

    pub const fn output_len(self) -> usize {
        self.m * self.n
    }
}

pub const GEMM: GemmKernel = GemmKernel {
    name: "gemm",
    backend: GemmBackend::HostFallback,
};

pub fn supported_gemm_kernels() -> [GemmKernel; 1] {
    [GEMM]
}

pub fn gemm(lhs: &[f32], rhs: &[f32], shape: GemmShape) -> Result<Vec<f32>, GemmError> {
    let mut output = vec![0.0; shape.output_len()];
    gemm_into(lhs, rhs, shape, &mut output)?;
    Ok(output)
}

pub fn gemm_into(
    lhs: &[f32],
    rhs: &[f32],
    shape: GemmShape,
    output: &mut [f32],
) -> Result<(), GemmError> {
    validate_shape(lhs, rhs, shape, output)?;

    for row in 0..shape.m {
        for col in 0..shape.n {
            let mut acc = 0.0_f64;
            for depth in 0..shape.k {
                let lhs_index = row * shape.k + depth;
                let rhs_index = depth * shape.n + col;
                acc += f64::from(lhs[lhs_index]) * f64::from(rhs[rhs_index]);
            }
            output[row * shape.n + col] = acc as f32;
        }
    }

    Ok(())
}

fn validate_shape(
    lhs: &[f32],
    rhs: &[f32],
    shape: GemmShape,
    output: &mut [f32],
) -> Result<(), GemmError> {
    if shape.m == 0 || shape.k == 0 || shape.n == 0 {
        return Err(GemmError::InvalidShape(shape));
    }

    if lhs.len() != shape.lhs_len() {
        return Err(GemmError::LengthMismatch {
            argument: "lhs",
            expected: shape.lhs_len(),
            actual: lhs.len(),
        });
    }

    if rhs.len() != shape.rhs_len() {
        return Err(GemmError::LengthMismatch {
            argument: "rhs",
            expected: shape.rhs_len(),
            actual: rhs.len(),
        });
    }

    if output.len() != shape.output_len() {
        return Err(GemmError::LengthMismatch {
            argument: "output",
            expected: shape.output_len(),
            actual: output.len(),
        });
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GemmError {
    InvalidShape(GemmShape),
    LengthMismatch {
        argument: &'static str,
        expected: usize,
        actual: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

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
            supported_gemm_kernels(),
            [GemmKernel {
                name: "gemm",
                backend: GemmBackend::HostFallback,
            }]
        );
    }

    #[test]
    fn multiplies_square_matrices() {
        let lhs = [1.0, 2.0, 3.0, 4.0];
        let rhs = [5.0, 6.0, 7.0, 8.0];
        let output = gemm(&lhs, &rhs, GemmShape::new(2, 2, 2)).unwrap();

        approx_eq_slice(&output, &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn multiplies_rectangular_matrices() {
        let lhs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let output = gemm(&lhs, &rhs, GemmShape::new(2, 3, 2)).unwrap();

        approx_eq_slice(&output, &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn identity_matrix_preserves_input() {
        let lhs = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0];
        let rhs = [1.0, 0.0, 0.0, 1.0];
        let output = gemm(&lhs, &rhs, GemmShape::new(3, 2, 2)).unwrap();

        approx_eq_slice(&output, &lhs);
    }

    #[test]
    fn accumulates_in_f32_output_with_f64_intermediate() {
        let lhs = [0.1, 0.2, 0.3, 0.4];
        let rhs = [0.5, 0.6, 0.7, 0.8];
        let output = gemm(&lhs, &rhs, GemmShape::new(2, 2, 2)).unwrap();

        approx_eq_slice(&output, &[0.19, 0.22, 0.43, 0.50]);
    }

    #[test]
    fn gemm_into_writes_existing_buffer() {
        let lhs = [1.0, 2.0, 3.0, 4.0];
        let rhs = [1.0, 0.0, 0.0, 1.0];
        let mut output = [-1.0; 4];

        gemm_into(&lhs, &rhs, GemmShape::new(2, 2, 2), &mut output).unwrap();

        approx_eq_slice(&output, &lhs);
    }

    #[test]
    fn rejects_invalid_shape() {
        let error = gemm(&[1.0], &[1.0], GemmShape::new(0, 1, 1)).unwrap_err();
        assert_eq!(error, GemmError::InvalidShape(GemmShape::new(0, 1, 1)));
    }

    #[test]
    fn rejects_lhs_length_mismatch() {
        let error = gemm(&[1.0], &[1.0, 2.0], GemmShape::new(1, 2, 1)).unwrap_err();
        assert_eq!(
            error,
            GemmError::LengthMismatch {
                argument: "lhs",
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn rejects_rhs_length_mismatch() {
        let error = gemm(&[1.0, 2.0], &[1.0], GemmShape::new(1, 2, 1)).unwrap_err();
        assert_eq!(
            error,
            GemmError::LengthMismatch {
                argument: "rhs",
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn rejects_output_length_mismatch() {
        let lhs = [1.0, 2.0];
        let rhs = [3.0, 4.0, 5.0, 6.0];
        let mut output = [0.0; 1];
        let error = gemm_into(&lhs, &rhs, GemmShape::new(1, 2, 2), &mut output).unwrap_err();

        assert_eq!(
            error,
            GemmError::LengthMismatch {
                argument: "output",
                expected: 2,
                actual: 1,
            }
        );
    }
}
