use crate::tensor::{GpuTensor, TensorError};
use crate::KernelStub;

pub const SPEC: KernelStub = KernelStub {
    name: "gemm_host",
    summary: "Matrix multiply kernels adapted from cutile-rs examples.",
};

#[cfg(any(target_os = "linux", test))]
const CUTILE_TILE_M: usize = 16;
#[cfg(any(target_os = "linux", test))]
const CUTILE_TILE_N: usize = 16;
#[cfg(any(target_os = "linux", test))]
const CUTILE_TILE_K: usize = 8;

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

    pub const fn checked_lhs_len(self) -> Option<usize> {
        self.m.checked_mul(self.k)
    }

    pub const fn checked_rhs_len(self) -> Option<usize> {
        self.k.checked_mul(self.n)
    }

    pub const fn checked_output_len(self) -> Option<usize> {
        self.m.checked_mul(self.n)
    }

    pub fn lhs_len(self) -> usize {
        self.checked_lhs_len()
            .expect("GemmShape::lhs_len overflowed usize")
    }

    pub fn rhs_len(self) -> usize {
        self.checked_rhs_len()
            .expect("GemmShape::rhs_len overflowed usize")
    }

    pub fn output_len(self) -> usize {
        self.checked_output_len()
            .expect("GemmShape::output_len overflowed usize")
    }
}

#[cfg(target_os = "linux")]
pub const GEMM: GemmKernel = GemmKernel {
    name: "gemm",
    backend: GemmBackend::Cutile,
};

#[cfg(not(target_os = "linux"))]
pub const GEMM: GemmKernel = GemmKernel {
    name: "gemm_host",
    backend: GemmBackend::HostFallback,
};

pub fn supported_gemm_kernels() -> [GemmKernel; 1] {
    [GEMM]
}

pub fn gemm_host(lhs: &[f32], rhs: &[f32], shape: GemmShape) -> Result<Vec<f32>, GemmError> {
    let output_len = checked_shape_lengths(shape)?.2;
    let mut output = vec![0.0; output_len];
    gemm_into_host(lhs, rhs, shape, &mut output)?;
    Ok(output)
}

pub fn gemm_into_host(
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
    let (lhs_len, rhs_len, output_len) = checked_shape_lengths(shape)?;

    if lhs.len() != lhs_len {
        return Err(GemmError::LengthMismatch {
            argument: "lhs",
            expected: lhs_len,
            actual: lhs.len(),
        });
    }

    if rhs.len() != rhs_len {
        return Err(GemmError::LengthMismatch {
            argument: "rhs",
            expected: rhs_len,
            actual: rhs.len(),
        });
    }

    if output.len() != output_len {
        return Err(GemmError::LengthMismatch {
            argument: "output",
            expected: output_len,
            actual: output.len(),
        });
    }

    Ok(())
}

fn validate_tensor_lengths(
    lhs: &GpuTensor,
    rhs: &GpuTensor,
    shape: GemmShape,
) -> Result<(), GemmError> {
    let (lhs_len, rhs_len, _) = checked_shape_lengths(shape)?;

    if lhs.numel() != lhs_len {
        return Err(GemmError::LengthMismatch {
            argument: "lhs",
            expected: lhs_len,
            actual: lhs.numel(),
        });
    }

    if rhs.numel() != rhs_len {
        return Err(GemmError::LengthMismatch {
            argument: "rhs",
            expected: rhs_len,
            actual: rhs.numel(),
        });
    }

    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GemmError {
    InvalidShape(GemmShape),
    LengthMismatch {
        argument: &'static str,
        expected: usize,
        actual: usize,
    },
    DeviceError(String),
}

impl From<TensorError> for GemmError {
    fn from(e: TensorError) -> Self {
        GemmError::DeviceError(e.to_string())
    }
}

fn checked_shape_lengths(shape: GemmShape) -> Result<(usize, usize, usize), GemmError> {
    if shape.m == 0 || shape.k == 0 || shape.n == 0 {
        return Err(GemmError::InvalidShape(shape));
    }

    let lhs_len = shape
        .checked_lhs_len()
        .ok_or(GemmError::InvalidShape(shape))?;
    let rhs_len = shape
        .checked_rhs_len()
        .ok_or(GemmError::InvalidShape(shape))?;
    let output_len = shape
        .checked_output_len()
        .ok_or(GemmError::InvalidShape(shape))?;

    Ok((lhs_len, rhs_len, output_len))
}

#[cfg(target_os = "linux")]
mod cutile_impl {
    use super::{GemmError, GemmShape, CUTILE_TILE_K, CUTILE_TILE_M, CUTILE_TILE_N};
    use crate::tensor::GpuTensor;
    use cutile::tensor::Unpartition;
    use cutile::tile_kernel::{
        zip, DeviceOperation, IntoDeviceOperation, IntoDeviceOperationPartition, TileKernel,
        Unzippable3, Zippable,
    };

    #[cutile::module]
    mod cutile_gemm_kernel {
        use cutile::core::*;

        #[cutile::entry()]
        fn gemm<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
            output: &mut Tensor<f32, { [BM, BN] }>,
            lhs: &Tensor<f32, { [-1, K] }>,
            rhs: &Tensor<f32, { [K, -1] }>,
        ) {
            let lhs_tiles = lhs.partition(const_shape![BM, BK]);
            let rhs_tiles = rhs.partition(const_shape![BK, BN]);
            let tile_block: (i32, i32, i32) = get_tile_block_id();
            let mut tile_output = output.load();

            for depth in 0i32..(K / BK) {
                let lhs_tile = lhs_tiles.load([tile_block.0, depth]);
                let rhs_tile = rhs_tiles.load([depth, tile_block.1]);
                tile_output = mma(lhs_tile, rhs_tile, tile_output);
            }

            output.store(tile_output);
        }
    }

    use cutile_gemm_kernel::gemm_apply;

    pub(super) fn supports_shape(shape: GemmShape) -> bool {
        shape.m <= i32::MAX as usize
            && shape.k <= i32::MAX as usize
            && shape.n <= i32::MAX as usize
            && shape.m % CUTILE_TILE_M == 0
            && shape.k % CUTILE_TILE_K == 0
            && shape.n % CUTILE_TILE_N == 0
    }

    fn device_error(prefix: &str, error: impl std::fmt::Debug) -> GemmError {
        GemmError::DeviceError(format!("{prefix}: {error:?}"))
    }

    pub(super) async fn gemm(
        lhs: &GpuTensor,
        rhs: &GpuTensor,
        shape: GemmShape,
    ) -> Result<GpuTensor, GemmError> {
        let output = cutile::api::zeros::<2, f32>([shape.m, shape.n]);
        let generics = vec![
            CUTILE_TILE_M.to_string(),
            CUTILE_TILE_N.to_string(),
            CUTILE_TILE_K.to_string(),
            shape.k.to_string(),
        ];
        let args = zip!(
            output.partition([CUTILE_TILE_M as i32, CUTILE_TILE_N as i32]),
            lhs.cutile_tensor_for_shape(&[shape.m, shape.k])
                .await?
                .device_operation(),
            rhs.cutile_tensor_for_shape(&[shape.k, shape.n])
                .await?
                .device_operation()
        );
        let (output, _lhs, _rhs) = args.apply(gemm_apply).generics(generics).unzip();
        let output = output
            .unpartition()
            .await
            .map_err(|error| device_error("cutile GEMM unpartition failed", error))?;
        GpuTensor::from_cutile_tensor(output, &[shape.m, shape.n]).map_err(Into::into)
    }
}

fn backend_for_shape(_shape: GemmShape) -> GemmBackend {
    #[cfg(target_os = "linux")]
    {
        if cutile_impl::supports_shape(_shape) {
            return GemmBackend::Cutile;
        }
    }

    GemmBackend::HostFallback
}

async fn gemm_via_host_bridge(
    lhs: &GpuTensor,
    rhs: &GpuTensor,
    shape: GemmShape,
) -> Result<GpuTensor, GemmError> {
    let lhs_data = lhs.to_host_async().await?;
    let rhs_data = rhs.to_host_async().await?;
    let result = gemm_host(&lhs_data, &rhs_data, shape)?;
    Ok(GpuTensor::from_host_async(&result, &[shape.m, shape.n]).await?)
}

// ---------------------------------------------------------------------------
// Async GPU API
// ---------------------------------------------------------------------------

/// Async GPU matrix multiplication. On Linux+CUDA dispatches to a cutile
/// tile kernel for aligned shapes; elsewhere delegates to the host fallback.
pub async fn gemm(
    lhs: &GpuTensor,
    rhs: &GpuTensor,
    shape: GemmShape,
) -> Result<GpuTensor, GemmError> {
    validate_tensor_lengths(lhs, rhs, shape)?;

    match backend_for_shape(shape) {
        GemmBackend::Cutile => {
            #[cfg(target_os = "linux")]
            {
                cutile_impl::gemm(lhs, rhs, shape).await
            }
            #[cfg(not(target_os = "linux"))]
            {
                unreachable!("non-Linux platforms never select the cutile GEMM backend")
            }
        }
        GemmBackend::HostFallback => gemm_via_host_bridge(lhs, rhs, shape).await,
    }
}

/// Async GPU matrix multiplication writing into an existing tensor.
pub async fn gemm_into(
    lhs: &GpuTensor,
    rhs: &GpuTensor,
    shape: GemmShape,
    output: &mut GpuTensor,
) -> Result<(), GemmError> {
    validate_tensor_lengths(lhs, rhs, shape)?;
    let output_len = checked_shape_lengths(shape)?.2;

    if output.numel() != output_len {
        return Err(GemmError::LengthMismatch {
            argument: "output",
            expected: output_len,
            actual: output.numel(),
        });
    }

    let output_shape = output.shape().to_vec();
    let mut result = gemm(lhs, rhs, shape).await?;
    result.reshape(&output_shape)?;
    *output = result;
    Ok(())
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
                "index {index}: left={left:?}, right={right:?}, diff={diff:?}"
            );
        }
    }

    fn approx_eq_slice(lhs: &[f32], rhs: &[f32]) {
        approx_eq_slice_with_tolerance(lhs, rhs, 1e-6);
    }

    /// Verifies that GEMM advertises the active platform backend while keeping the
    /// public registry shape stable. This catches accidental backend metadata
    /// regressions when the Linux cutile path is enabled.
    #[test]
    fn reports_platform_gemm_backend() {
        #[cfg(target_os = "linux")]
        let expected = [GemmKernel {
            name: "gemm",
            backend: GemmBackend::Cutile,
        }];

        #[cfg(not(target_os = "linux"))]
        let expected = [GemmKernel {
            name: "gemm_host",
            backend: GemmBackend::HostFallback,
        }];

        assert_eq!(supported_gemm_kernels(), expected);
    }

    /// Verifies that GEMM selects the cutile backend only when the matrix shape
    /// is tile-aligned on Linux. This catches accidental dispatch of unsupported
    /// shapes into the device kernel.
    #[test]
    fn selects_backend_from_shape_constraints() {
        let aligned = GemmShape::new(CUTILE_TILE_M * 2, CUTILE_TILE_K * 2, CUTILE_TILE_N * 2);
        let unaligned = GemmShape::new(CUTILE_TILE_M + 1, CUTILE_TILE_K, CUTILE_TILE_N);

        #[cfg(target_os = "linux")]
        {
            assert_eq!(backend_for_shape(aligned), GemmBackend::Cutile);
            assert_eq!(backend_for_shape(unaligned), GemmBackend::HostFallback);
        }

        #[cfg(not(target_os = "linux"))]
        {
            assert_eq!(backend_for_shape(aligned), GemmBackend::HostFallback);
            assert_eq!(backend_for_shape(unaligned), GemmBackend::HostFallback);
        }
    }

    /// Verifies 2×2 matrix multiplication against hand-computed results.
    ///
    /// This catches errors in the row/column accumulation loop.
    #[test]
    fn multiplies_square_matrices() {
        let lhs = [1.0, 2.0, 3.0, 4.0];
        let rhs = [5.0, 6.0, 7.0, 8.0];
        let output = gemm_host(&lhs, &rhs, GemmShape::new(2, 2, 2)).unwrap();

        approx_eq_slice(&output, &[19.0, 22.0, 43.0, 50.0]);
    }

    /// Verifies GEMM with non-square dimensions (2×3 × 3×2 → 2×2).
    ///
    /// This catches indexing bugs that only appear with m ≠ k ≠ n.
    #[test]
    fn multiplies_rectangular_matrices() {
        let lhs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let output = gemm_host(&lhs, &rhs, GemmShape::new(2, 3, 2)).unwrap();

        approx_eq_slice(&output, &[58.0, 64.0, 139.0, 154.0]);
    }

    /// Verifies that multiplying by the identity matrix preserves the input.
    ///
    /// This catches systematic bias in the accumulation (e.g. forgotten zero-init).
    #[test]
    fn identity_matrix_preserves_input() {
        let lhs = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0];
        let rhs = [1.0, 0.0, 0.0, 1.0];
        let output = gemm_host(&lhs, &rhs, GemmShape::new(3, 2, 2)).unwrap();

        approx_eq_slice(&output, &lhs);
    }

    /// Verifies that f64 intermediate accumulation avoids f32 rounding drift.
    ///
    /// This catches accidental use of f32 accumulation in the inner loop.
    #[test]
    fn accumulates_in_f32_output_with_f64_intermediate() {
        let lhs = [0.1, 0.2, 0.3, 0.4];
        let rhs = [0.5, 0.6, 0.7, 0.8];
        let output = gemm_host(&lhs, &rhs, GemmShape::new(2, 2, 2)).unwrap();

        approx_eq_slice(&output, &[0.19, 0.22, 0.43, 0.50]);
    }

    /// Verifies that the _into variant writes results into a pre-allocated buffer.
    ///
    /// This catches bugs where _into silently re-allocates instead of writing in place.
    #[test]
    fn gemm_into_writes_existing_buffer() {
        let lhs = [1.0, 2.0, 3.0, 4.0];
        let rhs = [1.0, 0.0, 0.0, 1.0];
        let mut output = [-1.0; 4];

        gemm_into_host(&lhs, &rhs, GemmShape::new(2, 2, 2), &mut output).unwrap();

        approx_eq_slice(&output, &lhs);
    }

    /// Verifies that a zero dimension (m=0) is rejected as an invalid shape.
    ///
    /// This catches missing dimension validation.
    #[test]
    fn rejects_invalid_shape() {
        let error = gemm_host(&[1.0], &[1.0], GemmShape::new(0, 1, 1)).unwrap_err();
        assert_eq!(error, GemmError::InvalidShape(GemmShape::new(0, 1, 1)));
    }

    /// Verifies that overflowing GEMM dimensions are rejected as invalid shapes
    /// instead of wrapping in release builds. This catches integer-overflow
    /// regressions in length validation before buffers are trusted.
    #[test]
    fn rejects_overflowing_shape_lengths() {
        let shape = GemmShape::new(2, usize::MAX, 1);
        let error = gemm_host(&[1.0], &[1.0], shape).unwrap_err();
        assert_eq!(error, GemmError::InvalidShape(shape));
        assert_eq!(shape.checked_lhs_len(), None);
    }

    /// Verifies that a too-short lhs buffer is rejected.
    ///
    /// This catches missing lhs length validation.
    #[test]
    fn rejects_lhs_length_mismatch() {
        let error = gemm_host(&[1.0], &[1.0, 2.0], GemmShape::new(1, 2, 1)).unwrap_err();
        assert_eq!(
            error,
            GemmError::LengthMismatch {
                argument: "lhs",
                expected: 2,
                actual: 1,
            }
        );
    }

    /// Verifies that a too-short rhs buffer is rejected.
    ///
    /// This catches missing rhs length validation.
    #[test]
    fn rejects_rhs_length_mismatch() {
        let error = gemm_host(&[1.0, 2.0], &[1.0], GemmShape::new(1, 2, 1)).unwrap_err();
        assert_eq!(
            error,
            GemmError::LengthMismatch {
                argument: "rhs",
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
        let lhs = [1.0, 2.0];
        let rhs = [3.0, 4.0, 5.0, 6.0];
        let mut output = [0.0; 1];
        let error = gemm_into_host(&lhs, &rhs, GemmShape::new(1, 2, 2), &mut output).unwrap_err();

        assert_eq!(
            error,
            GemmError::LengthMismatch {
                argument: "output",
                expected: 2,
                actual: 1,
            }
        );
    }

    /// Verifies that async GPU GEMM preserves host parity when an unaligned shape
    /// forces the wrapper back through the host fallback path. This catches
    /// regressions in the fallback bridge while real cutile support is partial.
    #[tokio::test]
    async fn gpu_gemm_matches_host_fallback() {
        let lhs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let shape = GemmShape::new(2, 3, 2);
        let expected = gemm_host(&lhs, &rhs, shape).unwrap();
        let gpu_lhs = GpuTensor::from_host(&lhs, &[2, 3]).unwrap();
        let gpu_rhs = GpuTensor::from_host(&rhs, &[3, 2]).unwrap();
        let result = super::gemm(&gpu_lhs, &gpu_rhs, shape).await.unwrap();
        assert_eq!(result.to_host(), expected);
    }

    /// Verifies that async GPU GEMM-into preserves the destination shape when the
    /// output tensor has the correct element count. This catches regressions
    /// where the wrapper silently reshapes caller-owned tensors.
    #[tokio::test]
    async fn gpu_gemm_into_preserves_existing_output_shape() {
        let lhs = vec![1.0, 2.0, 3.0, 4.0];
        let rhs = vec![1.0, 0.0, 0.0, 1.0];
        let shape = GemmShape::new(2, 2, 2);
        let expected = gemm_host(&lhs, &rhs, shape).unwrap();
        let gpu_lhs = GpuTensor::from_host(&lhs, &[2, 2]).unwrap();
        let gpu_rhs = GpuTensor::from_host(&rhs, &[2, 2]).unwrap();
        let mut output = GpuTensor::zeros(&[1, 4]).unwrap();

        super::gemm_into(&gpu_lhs, &gpu_rhs, shape, &mut output)
            .await
            .unwrap();

        assert_eq!(output.shape(), &[1, 4]);
        assert_eq!(output.to_host(), expected);
    }

    /// Verifies that async GPU GEMM-into rejects a mismatched output tensor when
    /// its element count is wrong. This catches regressions where the wrapper
    /// silently replaces invalid outputs instead of matching host validation.
    #[tokio::test]
    async fn gpu_gemm_into_rejects_output_length_mismatch() {
        let lhs = vec![1.0, 2.0, 3.0, 4.0];
        let rhs = vec![1.0, 0.0, 0.0, 1.0];
        let shape = GemmShape::new(2, 2, 2);
        let gpu_lhs = GpuTensor::from_host(&lhs, &[2, 2]).unwrap();
        let gpu_rhs = GpuTensor::from_host(&rhs, &[2, 2]).unwrap();
        let mut output = GpuTensor::zeros(&[3]).unwrap();

        let error = super::gemm_into(&gpu_lhs, &gpu_rhs, shape, &mut output)
            .await
            .unwrap_err();

        assert_eq!(
            error,
            GemmError::LengthMismatch {
                argument: "output",
                expected: 4,
                actual: 3,
            }
        );
    }

    /// Verifies that async GPU GEMM rejects overflowing shape products before it
    /// trusts tensor lengths. This catches release-build wraparound bugs in the
    /// cutile and host-fallback dispatch paths.
    #[tokio::test]
    async fn gpu_gemm_rejects_overflowing_shape_lengths() {
        let shape = GemmShape::new(2, usize::MAX, 1);
        let gpu_lhs = GpuTensor::from_host(&[1.0], &[1]).unwrap();
        let gpu_rhs = GpuTensor::from_host(&[1.0], &[1]).unwrap();

        let error = super::gemm(&gpu_lhs, &gpu_rhs, shape).await.unwrap_err();

        assert_eq!(error, GemmError::InvalidShape(shape));
    }

    /// Verifies that the Linux cutile GEMM path matches host GEMM for a
    /// tile-aligned shape. This catches regressions that would silently route
    /// aligned matrices back through the old host bridge.
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn gpu_gemm_uses_cutile_for_aligned_shapes() {
        let shape = GemmShape::new(CUTILE_TILE_M * 2, CUTILE_TILE_K * 4, CUTILE_TILE_N * 2);
        assert_eq!(backend_for_shape(shape), GemmBackend::Cutile);

        let lhs = (0..shape.lhs_len())
            .map(|index| ((index % 11) as f32 - 5.0) * 0.25)
            .collect::<Vec<_>>();
        let rhs = (0..shape.rhs_len())
            .map(|index| ((index % 7) as f32 - 3.0) * 0.5)
            .collect::<Vec<_>>();
        let expected = gemm_host(&lhs, &rhs, shape).unwrap();
        let gpu_lhs = GpuTensor::from_host(&lhs, &[shape.m, shape.k]).unwrap();
        let gpu_rhs = GpuTensor::from_host(&rhs, &[shape.k, shape.n]).unwrap();

        let result = super::gemm(&gpu_lhs, &gpu_rhs, shape).await.unwrap();

        approx_eq_slice_with_tolerance(&result.to_host(), &expected, 1e-4);
    }
}
