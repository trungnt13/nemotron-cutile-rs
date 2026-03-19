//! GPU device context for managing CUDA resources.
//!
//! On Linux with CUDA, wraps cutile's `CudaContext` and `CudaStream`.
//! On other platforms, provides a no-op stub so the same code compiles
//! everywhere.

use std::fmt;
use std::sync::Arc;

use crate::tensor::TensorError;

// ---------------------------------------------------------------------------
// Platform-specific device implementation
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
mod inner {
    use super::TensorError;
    use std::sync::Arc;

    pub struct DeviceInner {
        pub(crate) context: Arc<cuda_core::CudaContext>,
        pub(crate) stream: Arc<cuda_core::CudaStream>,
        pub(crate) ordinal: usize,
    }

    impl DeviceInner {
        pub fn new(ordinal: usize) -> Result<Self, TensorError> {
            let context = cuda_core::CudaContext::new(ordinal)
                .map_err(|e| TensorError::DeviceError(format!("failed to create context: {e:?}")))?;
            let stream = context
                .new_stream()
                .map_err(|e| TensorError::DeviceError(format!("failed to create stream: {e:?}")))?;
            // Initialize cutile async device contexts
            cuda_async::device_context::init_device_contexts(ordinal, 1)
                .map_err(|e| TensorError::DeviceError(format!("failed to init device: {e:?}")))?;
            Ok(Self {
                context,
                stream,
                ordinal,
            })
        }

        pub fn synchronize(&self) -> Result<(), TensorError> {
            self.stream
                .synchronize()
                .map_err(|e| TensorError::DeviceError(format!("sync failed: {e:?}")))
        }

        pub fn ordinal(&self) -> usize {
            self.ordinal
        }
    }

    impl std::fmt::Debug for DeviceInner {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "DeviceInner(cuda:{})", self.ordinal)
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod inner {
    use super::TensorError;

    /// No-op device on non-Linux platforms.
    #[derive(Debug)]
    pub struct DeviceInner {
        ordinal: usize,
    }

    impl DeviceInner {
        pub fn new(ordinal: usize) -> Result<Self, TensorError> {
            Ok(Self { ordinal })
        }

        pub fn synchronize(&self) -> Result<(), TensorError> {
            Ok(())
        }

        pub fn ordinal(&self) -> usize {
            self.ordinal
        }
    }
}

// ---------------------------------------------------------------------------
// GpuDevice — the public API
// ---------------------------------------------------------------------------

/// Handle to a GPU device (or host-fallback stub on non-CUDA platforms).
///
/// On Linux with CUDA, initializes the cutile async device context and creates
/// a CUDA stream. On macOS/Windows, all operations are no-ops.
///
/// # Usage
///
/// Create once at application startup and pass through the model:
///
/// ```ignore
/// let device = GpuDevice::new(0)?;  // GPU 0
/// // ... pass device to model layers ...
/// device.synchronize()?;
/// ```
pub struct GpuDevice {
    inner: inner::DeviceInner,
}

impl GpuDevice {
    /// Initialize a GPU device by ordinal (0 = first GPU).
    ///
    /// On Linux, this creates a CUDA context and stream. On non-Linux
    /// platforms this is a no-op that always succeeds.
    pub fn new(ordinal: usize) -> Result<Self, TensorError> {
        Ok(Self {
            inner: inner::DeviceInner::new(ordinal)?,
        })
    }

    /// Block until all previously submitted GPU work completes.
    pub fn synchronize(&self) -> Result<(), TensorError> {
        self.inner.synchronize()
    }

    /// The device ordinal (0-indexed GPU ID).
    pub fn ordinal(&self) -> usize {
        self.inner.ordinal()
    }

    /// Access the underlying CUDA context (Linux only).
    #[cfg(target_os = "linux")]
    pub fn cuda_context(&self) -> &Arc<cuda_core::CudaContext> {
        &self.inner.context
    }

    /// Access the default CUDA stream (Linux only).
    #[cfg(target_os = "linux")]
    pub fn cuda_stream(&self) -> &Arc<cuda_core::CudaStream> {
        &self.inner.stream
    }
}

impl fmt::Debug for GpuDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GpuDevice({:?})", self.inner)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that a device can be created on any platform.
    /// This catches regressions in platform-conditional construction.
    #[test]
    fn creates_device_without_error() {
        let device = GpuDevice::new(0).unwrap();
        assert_eq!(device.ordinal(), 0);
    }

    /// Verifies that synchronize succeeds on any platform.
    /// This catches regressions in the no-op stub.
    #[test]
    fn synchronize_succeeds() {
        let device = GpuDevice::new(0).unwrap();
        device.synchronize().unwrap();
    }
}
