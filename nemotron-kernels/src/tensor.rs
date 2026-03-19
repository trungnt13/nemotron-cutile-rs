//! GPU tensor abstraction over cutile's `Tensor<T>`.
//!
//! On Linux with CUDA, wraps a real `cutile::tensor::Tensor<f32>` living in
//! device memory. On other platforms, wraps a `Vec<f32>` for host-fallback
//! development and testing.

use std::fmt;

#[cfg(target_os = "linux")]
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum TensorError {
    ShapeMismatch {
        expected_numel: usize,
        got_numel: usize,
    },
    EmptyShape,
    DeviceError(String),
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch {
                expected_numel,
                got_numel,
            } => write!(
                f,
                "tensor shape expects {expected_numel} elements but got {got_numel}"
            ),
            Self::EmptyShape => write!(f, "tensor shape must have at least one dimension"),
            Self::DeviceError(msg) => write!(f, "device error: {msg}"),
        }
    }
}

impl std::error::Error for TensorError {}

// ---------------------------------------------------------------------------
// Platform-specific inner storage
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
mod inner {
    use super::TensorError;
    use std::fmt;
    use std::sync::{Arc, OnceLock};

    use cutile::cuda_async::device_operation::DeviceOperation;
    use cutile::cuda_core::{CudaContext, CudaStream};
    use cutile::tensor::ToHostVec;

    /// Lazily-initialized CUDA context + stream for synchronous tensor ops.
    /// Using `sync_on(&stream)` avoids requiring a tokio runtime, so
    /// `GpuTensor::from_host()` / `to_host()` work from plain `#[test]`
    /// functions as well as from inside async contexts.
    struct SyncCudaRuntime {
        _context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
    }

    // SAFETY: CudaContext and CudaStream are Send+Sync (verified from cutile source).
    unsafe impl Send for SyncCudaRuntime {}
    unsafe impl Sync for SyncCudaRuntime {}

    static SYNC_RUNTIME: OnceLock<SyncCudaRuntime> = OnceLock::new();

    fn sync_stream() -> &'static Arc<CudaStream> {
        &SYNC_RUNTIME
            .get_or_init(|| {
                let context = CudaContext::new(0).expect("CUDA context init for sync tensor ops");
                let stream = context
                    .new_stream()
                    .expect("CUDA stream for sync tensor ops");
                SyncCudaRuntime {
                    _context: context,
                    stream,
                }
            })
            .stream
    }

    /// Wraps `cutile::tensor::Tensor<f32>` on Linux.
    pub struct TensorInner {
        pub(crate) tensor: Arc<cutile::tensor::Tensor<f32>>,
    }

    impl TensorInner {
        pub fn zeros(numel: usize) -> Self {
            let data = Arc::new(vec![0.0f32; numel]);
            let tensor = cutile::api::copy_host_vec_to_device(&data)
                .sync_on(sync_stream())
                .expect("cutile zeros allocation");
            Self {
                tensor: Arc::new(tensor),
            }
        }

        pub fn from_host(data: &[f32]) -> Self {
            let data = Arc::new(data.to_vec());
            let tensor = cutile::api::copy_host_vec_to_device(&data)
                .sync_on(sync_stream())
                .expect("cutile host-to-device copy");
            Self {
                tensor: Arc::new(tensor),
            }
        }

        pub async fn from_host_async(data: &[f32]) -> Result<Self, TensorError> {
            let data = Arc::new(data.to_vec());
            let tensor = cutile::api::copy_host_vec_to_device(&data)
                .await
                .map_err(|e| TensorError::DeviceError(format!("{e:?}")))?;
            Ok(Self {
                tensor: Arc::new(tensor),
            })
        }

        pub fn to_host(&self) -> Vec<f32> {
            (&self.tensor)
                .to_host_vec()
                .sync_on(sync_stream())
                .expect("cutile device-to-host copy")
        }

        pub async fn to_host_async(&self) -> Result<Vec<f32>, TensorError> {
            self.tensor
                .clone()
                .to_host_vec()
                .await
                .map_err(|e| TensorError::DeviceError(format!("{e:?}")))
        }

        pub fn cutile_tensor(&self) -> &Arc<cutile::tensor::Tensor<f32>> {
            &self.tensor
        }
    }

    impl Clone for TensorInner {
        fn clone(&self) -> Self {
            Self {
                tensor: self.tensor.clone(),
            }
        }
    }

    impl fmt::Debug for TensorInner {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "TensorInner(cutile)")
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod inner {
    use super::TensorError;

    /// Host-fallback: wraps a `Vec<f32>` on non-Linux platforms.
    #[derive(Clone)]
    pub struct TensorInner {
        pub(crate) data: Vec<f32>,
    }

    impl TensorInner {
        pub fn zeros(numel: usize) -> Self {
            Self {
                data: vec![0.0f32; numel],
            }
        }

        pub fn from_host(data: &[f32]) -> Self {
            Self {
                data: data.to_vec(),
            }
        }

        pub async fn from_host_async(data: &[f32]) -> Result<Self, TensorError> {
            Ok(Self::from_host(data))
        }

        pub fn to_host(&self) -> Vec<f32> {
            self.data.clone()
        }

        pub async fn to_host_async(&self) -> Result<Vec<f32>, TensorError> {
            Ok(self.data.clone())
        }
    }

    impl std::fmt::Debug for TensorInner {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "TensorInner(host, {} elements)", self.data.len())
        }
    }
}

// ---------------------------------------------------------------------------
// GpuTensor — the public API
// ---------------------------------------------------------------------------

/// A tensor that lives on the GPU (Linux+CUDA) or falls back to host memory.
///
/// # Shape
///
/// Shape is tracked as `Vec<usize>` at runtime. The total number of elements
/// (`numel`) must equal the product of all dimensions. Operations that change
/// shape (like `reshape`) are zero-copy when the underlying storage is
/// contiguous.
///
/// # Thread Safety
///
/// `GpuTensor` is `Send + Sync`. On Linux it wraps an `Arc<cutile::Tensor<f32>>`
/// which is already reference-counted and thread-safe.
#[derive(Clone, Debug)]
pub struct GpuTensor {
    inner: inner::TensorInner,
    shape: Vec<usize>,
}

impl GpuTensor {
    // -- Constructors -------------------------------------------------------

    /// Create a zero-filled tensor with the given shape.
    pub fn zeros(shape: &[usize]) -> Result<Self, TensorError> {
        let numel = shape_numel(shape)?;
        Ok(Self {
            inner: inner::TensorInner::zeros(numel),
            shape: shape.to_vec(),
        })
    }

    /// Create a tensor from host data, copying it to the device.
    pub fn from_host(data: &[f32], shape: &[usize]) -> Result<Self, TensorError> {
        let numel = shape_numel(shape)?;
        if data.len() != numel {
            return Err(TensorError::ShapeMismatch {
                expected_numel: numel,
                got_numel: data.len(),
            });
        }
        Ok(Self {
            inner: inner::TensorInner::from_host(data),
            shape: shape.to_vec(),
        })
    }

    /// Async variant of [`from_host`](Self::from_host).
    pub async fn from_host_async(data: &[f32], shape: &[usize]) -> Result<Self, TensorError> {
        let numel = shape_numel(shape)?;
        if data.len() != numel {
            return Err(TensorError::ShapeMismatch {
                expected_numel: numel,
                got_numel: data.len(),
            });
        }
        Ok(Self {
            inner: inner::TensorInner::from_host_async(data).await?,
            shape: shape.to_vec(),
        })
    }

    // -- Data transfer ------------------------------------------------------

    /// Copy tensor data back to host memory.
    pub fn to_host(&self) -> Vec<f32> {
        self.inner.to_host()
    }

    /// Async variant of [`to_host`](Self::to_host).
    pub async fn to_host_async(&self) -> Result<Vec<f32>, TensorError> {
        self.inner.to_host_async().await
    }

    // -- Shape queries ------------------------------------------------------

    /// The tensor's shape as a slice.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Size of dimension `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.ndim()`.
    pub fn dim(&self, i: usize) -> usize {
        self.shape[i]
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().copied().product::<usize>().max(1)
    }

    // -- Reshaping ----------------------------------------------------------

    /// Reshape the tensor (zero-copy). The new shape must have the same
    /// total number of elements.
    pub fn reshape(&mut self, new_shape: &[usize]) -> Result<(), TensorError> {
        let new_numel = shape_numel(new_shape)?;
        if new_numel != self.numel() {
            return Err(TensorError::ShapeMismatch {
                expected_numel: self.numel(),
                got_numel: new_numel,
            });
        }
        self.shape = new_shape.to_vec();
        Ok(())
    }

    // -- Platform access ----------------------------------------------------

    /// Access the underlying platform-specific storage. Callers using this
    /// must be behind the appropriate `cfg` gate.
    #[cfg(target_os = "linux")]
    pub fn cutile_tensor(&self) -> &Arc<cutile::tensor::Tensor<f32>> {
        self.inner.cutile_tensor()
    }

    /// On non-Linux platforms, return a reference to the host data buffer.
    #[cfg(not(target_os = "linux"))]
    pub fn as_host_slice(&self) -> &[f32] {
        &self.inner.data
    }

    /// On non-Linux platforms, return a mutable reference to the host data.
    #[cfg(not(target_os = "linux"))]
    pub fn as_host_slice_mut(&mut self) -> &mut [f32] {
        &mut self.inner.data
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn shape_numel(shape: &[usize]) -> Result<usize, TensorError> {
    if shape.is_empty() {
        return Err(TensorError::EmptyShape);
    }
    Ok(shape.iter().copied().product::<usize>())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that a zero-filled tensor has the correct shape and all-zero data.
    /// This catches regressions in tensor allocation and shape tracking.
    #[test]
    fn zeros_creates_correct_shape_and_data() {
        let t = GpuTensor::zeros(&[3, 4]).unwrap();
        assert_eq!(t.shape(), &[3, 4]);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.dim(0), 3);
        assert_eq!(t.dim(1), 4);
        assert_eq!(t.numel(), 12);
        let host = t.to_host();
        assert_eq!(host.len(), 12);
        assert!(host.iter().all(|&v| v == 0.0));
    }

    /// Verifies that from_host correctly transfers data and preserves shape.
    /// This catches regressions in host-to-device copy.
    #[test]
    fn from_host_preserves_data_and_shape() {
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let t = GpuTensor::from_host(&data, &[2, 3]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_host(), data);
    }

    /// Verifies that from_host rejects data whose length doesn't match the shape.
    /// This catches missing validation in the constructor.
    #[test]
    fn from_host_rejects_mismatched_length() {
        let data = vec![1.0, 2.0, 3.0];
        let err = GpuTensor::from_host(&data, &[2, 3]).unwrap_err();
        assert!(matches!(err, TensorError::ShapeMismatch { .. }));
    }

    /// Verifies that reshape changes the shape without modifying data.
    /// This catches regressions in zero-copy reshape logic.
    #[test]
    fn reshape_changes_shape_preserves_data() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let mut t = GpuTensor::from_host(&data, &[3, 4]).unwrap();
        t.reshape(&[4, 3]).unwrap();
        assert_eq!(t.shape(), &[4, 3]);
        assert_eq!(t.to_host(), data);
    }

    /// Verifies that reshape rejects incompatible shapes.
    /// This catches missing numel validation in reshape.
    #[test]
    fn reshape_rejects_incompatible_shape() {
        let mut t = GpuTensor::zeros(&[3, 4]).unwrap();
        let err = t.reshape(&[5, 3]).unwrap_err();
        assert!(matches!(err, TensorError::ShapeMismatch { .. }));
    }

    /// Verifies that empty shape is rejected.
    /// This catches missing edge-case validation.
    #[test]
    fn rejects_empty_shape() {
        let err = GpuTensor::zeros(&[]).unwrap_err();
        assert!(matches!(err, TensorError::EmptyShape));
    }

    /// Verifies that the async constructor works identically to the sync one.
    /// This catches regressions in the async code path.
    #[tokio::test]
    async fn async_from_host_matches_sync() {
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let t = GpuTensor::from_host_async(&data, &[2, 4]).await.unwrap();
        assert_eq!(t.shape(), &[2, 4]);
        let host = t.to_host_async().await.unwrap();
        assert_eq!(host, data);
    }
}
