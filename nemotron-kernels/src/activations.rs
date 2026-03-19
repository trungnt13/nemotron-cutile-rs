use crate::tensor::{GpuTensor, TensorError};
use crate::KernelStub;

pub const SPEC: KernelStub = KernelStub {
    name: "activations",
    summary: "SiLU, sigmoid_host, and ReLU² GPU kernels.",
};

#[cfg(any(target_os = "linux", test))]
const CUTILE_ACTIVATION_MAX_WIDTH: usize = 4096;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActivationBackend {
    HostFallback,
    Cutile,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActivationKernel {
    pub name: &'static str,
    pub backend: ActivationBackend,
}

#[cfg(target_os = "linux")]
pub const SILU: ActivationKernel = ActivationKernel {
    name: "silu_cutile",
    backend: ActivationBackend::Cutile,
};

#[cfg(not(target_os = "linux"))]
pub const SILU: ActivationKernel = ActivationKernel {
    name: "silu_host",
    backend: ActivationBackend::HostFallback,
};

#[cfg(target_os = "linux")]
pub const RELU2: ActivationKernel = ActivationKernel {
    name: "relu2_cutile",
    backend: ActivationBackend::Cutile,
};

#[cfg(not(target_os = "linux"))]
pub const RELU2: ActivationKernel = ActivationKernel {
    name: "relu2_host",
    backend: ActivationBackend::HostFallback,
};

#[cfg(target_os = "linux")]
pub const SIGMOID: ActivationKernel = ActivationKernel {
    name: "sigmoid_cutile",
    backend: ActivationBackend::Cutile,
};

#[cfg(not(target_os = "linux"))]
pub const SIGMOID: ActivationKernel = ActivationKernel {
    name: "sigmoid_host",
    backend: ActivationBackend::HostFallback,
};

pub fn supported_activations() -> [ActivationKernel; 3] {
    [SILU, RELU2, SIGMOID]
}

pub fn silu_scalar(x: f32) -> f32 {
    x * sigmoid_scalar(x)
}

pub fn relu2_scalar(x: f32) -> f32 {
    let relu = x.max(0.0);
    relu * relu
}

pub fn sigmoid_scalar(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

pub fn silu_host(values: &[f32]) -> Vec<f32> {
    map_activation(values, silu_scalar)
}

pub fn relu2_host(values: &[f32]) -> Vec<f32> {
    map_activation(values, relu2_scalar)
}

pub fn sigmoid_host(values: &[f32]) -> Vec<f32> {
    map_activation(values, sigmoid_scalar)
}

pub fn silu_in_place_host(values: &mut [f32]) {
    map_activation_in_place(values, silu_scalar);
}

pub fn relu2_in_place_host(values: &mut [f32]) {
    map_activation_in_place(values, relu2_scalar);
}

pub fn sigmoid_in_place_host(values: &mut [f32]) {
    map_activation_in_place(values, sigmoid_scalar);
}

fn map_activation(values: &[f32], activation: fn(f32) -> f32) -> Vec<f32> {
    values.iter().copied().map(activation).collect()
}

fn map_activation_in_place(values: &mut [f32], activation: fn(f32) -> f32) {
    for value in values {
        *value = activation(*value);
    }
}

#[cfg(any(target_os = "linux", test))]
fn select_cutile_block_size(numel: usize) -> Option<usize> {
    if numel == 0
        || numel > CUTILE_ACTIVATION_MAX_WIDTH
        || !numel.is_power_of_two()
        || i32::try_from(numel).is_err()
    {
        return None;
    }

    Some(numel)
}

#[cfg(any(target_os = "linux", test))]
fn supports_cutile_activations(numel: usize) -> bool {
    select_cutile_block_size(numel).is_some()
}

#[cfg(target_os = "linux")]
fn active_backend(input: &GpuTensor) -> ActivationBackend {
    if supports_cutile_activations(input.numel()) {
        ActivationBackend::Cutile
    } else {
        ActivationBackend::HostFallback
    }
}

#[cfg(not(target_os = "linux"))]
fn active_backend(_input: &GpuTensor) -> ActivationBackend {
    ActivationBackend::HostFallback
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CutileActivation {
    Silu,
    Relu2,
    Sigmoid,
}

#[cfg(target_os = "linux")]
#[cutile::module]
mod cutile_activations {
    use cutile::core::*;

    #[cutile::entry()]
    fn silu<const S: [i32; 1]>(x: &Tensor<f32, { [-1] }>, y: &mut Tensor<f32, S>) {
        let ones: Tile<f32, S> = constant(1.0f32, const_shape!(S));
        let neg_x: Tile<f32, S> = negf(load_tile_like_1d(x, y));
        let denom: Tile<f32, S> = constant(1.0f32, const_shape!(S)) + exp(neg_x);
        let sigmoid: Tile<f32, S> = ones / denom;
        let input_tile: Tile<f32, S> = load_tile_like_1d(x, y);
        let result = input_tile * sigmoid;
        y.store(result);
    }

    #[cutile::entry()]
    fn relu2<const S: [i32; 1]>(x: &Tensor<f32, { [-1] }>, y: &mut Tensor<f32, S>) {
        let zeros: Tile<f32, S> = constant(0.0f32, const_shape!(S));
        let relu = max_tile(load_tile_like_1d(x, y), zeros);
        let zeros: Tile<f32, S> = constant(0.0f32, const_shape!(S));
        let result = relu * max_tile(load_tile_like_1d(x, y), zeros);
        y.store(result);
    }

    #[cutile::entry()]
    fn sigmoid<const S: [i32; 1]>(x: &Tensor<f32, { [-1] }>, y: &mut Tensor<f32, S>) {
        let ones: Tile<f32, S> = constant(1.0f32, const_shape!(S));
        let neg_x: Tile<f32, S> = negf(load_tile_like_1d(x, y));
        let denom: Tile<f32, S> = constant(1.0f32, const_shape!(S)) + exp(neg_x);
        let result: Tile<f32, S> = ones / denom;
        y.store(result);
    }
}

// ---------------------------------------------------------------------------
// Async GPU API
// ---------------------------------------------------------------------------

async fn activation_host_bridge(
    input: &GpuTensor,
    activation: fn(&[f32]) -> Vec<f32>,
) -> Result<GpuTensor, TensorError> {
    let data = input.to_host_async().await?;
    let result = activation(&data);
    GpuTensor::from_host_async(&result, input.shape()).await
}

#[cfg(target_os = "linux")]
async fn activation_cutile(
    input: &GpuTensor,
    activation: CutileActivation,
) -> Result<GpuTensor, TensorError> {
    use cutile::api;
    use cutile::tile_kernel::{IntoDeviceOperation, IntoDeviceOperationPartition};

    let numel = input.numel();
    let block_size = select_cutile_block_size(numel).ok_or_else(|| {
        TensorError::DeviceError(format!(
            "cutile activation dispatch does not support flattened width {numel}"
        ))
    })?;
    let block_size_i32 = i32::try_from(block_size).map_err(|_| {
        TensorError::DeviceError(format!(
            "cutile activation block size {block_size} exceeds the supported i32 launch bound"
        ))
    })?;
    let flattened_input = input.cutile_tensor_for_shape(&[numel]).await?;
    let output = api::zeros::<1, f32>([numel]).partition([block_size_i32]);
    let (activation_name, output) = match activation {
        CutileActivation::Silu => {
            let (_input, output) =
                cutile_activations::silu_async(flattened_input.device_operation(), output)
                    .await
                    .map_err(|error| {
                        TensorError::DeviceError(format!("cutile silu launch failed: {error:?}"))
                    })?;
            ("silu", output)
        }
        CutileActivation::Relu2 => {
            let (_input, output) =
                cutile_activations::relu2_async(flattened_input.device_operation(), output)
                    .await
                    .map_err(|error| {
                        TensorError::DeviceError(format!("cutile relu2 launch failed: {error:?}"))
                    })?;
            ("relu2", output)
        }
        CutileActivation::Sigmoid => {
            let (_input, output) =
                cutile_activations::sigmoid_async(flattened_input.device_operation(), output)
                    .await
                    .map_err(|error| {
                        TensorError::DeviceError(format!("cutile sigmoid launch failed: {error:?}"))
                    })?;
            ("sigmoid", output)
        }
    };

    GpuTensor::from_cutile_tensor(output.unpartition(), input.shape()).map_err(|error| {
        TensorError::DeviceError(format!(
            "cutile {activation_name} result wrap failed: {error}"
        ))
    })
}

/// Async GPU SiLU activation.
pub async fn silu(input: &GpuTensor) -> Result<GpuTensor, TensorError> {
    match active_backend(input) {
        #[cfg(target_os = "linux")]
        ActivationBackend::Cutile => activation_cutile(input, CutileActivation::Silu).await,
        ActivationBackend::HostFallback => activation_host_bridge(input, silu_host).await,
        #[cfg(not(target_os = "linux"))]
        ActivationBackend::Cutile => unreachable!("cutile backend is unavailable on this platform"),
    }
}

/// Async GPU ReLU² activation.
pub async fn relu2(input: &GpuTensor) -> Result<GpuTensor, TensorError> {
    match active_backend(input) {
        #[cfg(target_os = "linux")]
        ActivationBackend::Cutile => activation_cutile(input, CutileActivation::Relu2).await,
        ActivationBackend::HostFallback => activation_host_bridge(input, relu2_host).await,
        #[cfg(not(target_os = "linux"))]
        ActivationBackend::Cutile => unreachable!("cutile backend is unavailable on this platform"),
    }
}

/// Async GPU sigmoid activation.
pub async fn sigmoid(input: &GpuTensor) -> Result<GpuTensor, TensorError> {
    match active_backend(input) {
        #[cfg(target_os = "linux")]
        ActivationBackend::Cutile => activation_cutile(input, CutileActivation::Sigmoid).await,
        ActivationBackend::HostFallback => activation_host_bridge(input, sigmoid_host).await,
        #[cfg(not(target_os = "linux"))]
        ActivationBackend::Cutile => unreachable!("cutile backend is unavailable on this platform"),
    }
}

/// Async GPU SiLU in-place activation.
pub async fn silu_in_place(tensor: &mut GpuTensor) -> Result<(), TensorError> {
    let result = silu(&*tensor).await?;
    *tensor = result;
    Ok(())
}

/// Async GPU ReLU² in-place activation.
pub async fn relu2_in_place(tensor: &mut GpuTensor) -> Result<(), TensorError> {
    let result = relu2(&*tensor).await?;
    *tensor = result;
    Ok(())
}

/// Async GPU sigmoid in-place activation.
pub async fn sigmoid_in_place(tensor: &mut GpuTensor) -> Result<(), TensorError> {
    let result = sigmoid(&*tensor).await?;
    *tensor = result;
    Ok(())
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

    /// Verifies that the activation kernel registry reports the platform primary backends when Linux enables cutile and other targets keep host fallback.
    /// This catches accidental backend tag regressions as real device kernels land.
    #[test]
    fn reports_platform_primary_backends() {
        #[cfg(target_os = "linux")]
        let expected = [
            ActivationKernel {
                name: "silu_cutile",
                backend: ActivationBackend::Cutile,
            },
            ActivationKernel {
                name: "relu2_cutile",
                backend: ActivationBackend::Cutile,
            },
            ActivationKernel {
                name: "sigmoid_cutile",
                backend: ActivationBackend::Cutile,
            },
        ];

        #[cfg(not(target_os = "linux"))]
        let expected = [
            ActivationKernel {
                name: "silu_host",
                backend: ActivationBackend::HostFallback,
            },
            ActivationKernel {
                name: "relu2_host",
                backend: ActivationBackend::HostFallback,
            },
            ActivationKernel {
                name: "sigmoid_host",
                backend: ActivationBackend::HostFallback,
            },
        ];

        assert_eq!(supported_activations(), expected);
    }

    /// Verifies that the cutile activation dispatch heuristic only accepts flattened widths within the current power-of-two safety envelope.
    /// This catches unsupported Linux launches before they can bypass the host fallback.
    #[test]
    fn cutile_support_heuristic_respects_length_boundaries() {
        assert_eq!(select_cutile_block_size(1), Some(1));
        assert_eq!(select_cutile_block_size(8), Some(8));
        assert_eq!(
            select_cutile_block_size(CUTILE_ACTIVATION_MAX_WIDTH),
            Some(CUTILE_ACTIVATION_MAX_WIDTH)
        );
        assert!(!supports_cutile_activations(0));
        assert!(!supports_cutile_activations(6));
        assert!(!supports_cutile_activations(
            CUTILE_ACTIVATION_MAX_WIDTH + 1
        ));
        assert_eq!(select_cutile_block_size(i32::MAX as usize + 1), None);
    }

    /// Verifies sigmoid_host at zero, positive, and negative inputs against known values.
    /// This catches regressions in the numerically stable split-branch sigmoid_host.
    #[test]
    fn sigmoid_matches_reference_values() {
        approx_eq(sigmoid_scalar(0.0), 0.5);
        approx_eq(sigmoid_scalar(2.0), 0.880797);
        approx_eq(sigmoid_scalar(-2.0), 0.11920292);
    }

    /// Verifies that SiLU computes x·σ(x) at zero, positive, and negative inputs.
    /// This catches errors in the SiLU composition with sigmoid_host.
    #[test]
    fn silu_matches_reference_values() {
        approx_eq(silu_scalar(0.0), 0.0);
        approx_eq(silu_scalar(1.0), 0.7310586);
        approx_eq(silu_scalar(-1.0), -0.26894143);
    }

    /// Verifies that ReLU² clamps negatives to zero and squares positive values when applying the scalar formula.
    /// This catches sign errors or missing squaring in the relu2_host formula.
    #[test]
    fn relu2_matches_reference_values() {
        approx_eq(relu2_scalar(-3.0), 0.0);
        approx_eq(relu2_scalar(0.0), 0.0);
        approx_eq(relu2_scalar(1.5), 2.25);
    }

    /// Verifies that the vector API allocates correctly sized outputs when calling the allocating wrappers.
    /// This catches length miscalculation in the host convenience functions.
    #[test]
    fn vector_api_allocates_output() {
        assert_eq!(sigmoid_host(&[0.0, 1.0]).len(), 2);
        assert_eq!(relu2_host(&[-1.0, 2.0]), vec![0.0, 4.0]);
    }

    /// Verifies that the in-place host API mutates the buffer with the expected SiLU values when writing back into the same slice.
    /// This catches issues where in-place writes are skipped or mis-ordered.
    #[test]
    fn in_place_api_updates_buffer() {
        let mut values = [-1.0, 0.0, 2.0];
        silu_in_place_host(&mut values);

        approx_eq(values[0], -0.26894143);
        approx_eq(values[1], 0.0);
        approx_eq(values[2], 1.761594);
    }

    /// Verifies that Linux selects the cutile activation backend when the flattened width fits the current launch heuristic.
    /// This catches regressions that would route supported device launches back through the host bridge.
    #[cfg(target_os = "linux")]
    #[test]
    fn active_backend_uses_cutile_for_supported_lengths() {
        let input = GpuTensor::from_host(&[0.0; 8], &[2, 4]).unwrap();
        assert_eq!(active_backend(&input), ActivationBackend::Cutile);
    }

    /// Verifies that the async GPU SiLU matches the host fallback when the device wrapper runs on the current platform.
    /// This catches regressions in cutile parity and the host-bridge fallback path.
    #[tokio::test]
    async fn gpu_silu_matches_host_fallback() {
        let data = vec![-1.0, 0.0, 1.0, 2.0, -3.5, 7.25, -8.0, 6.0];
        let expected = silu_host(&data);
        let gpu_input = GpuTensor::from_host(&data, &[2, 4]).unwrap();
        let result = super::silu(&gpu_input).await.unwrap();

        assert_eq!(result.shape(), &[2, 4]);
        approx_slice_eq(&result.to_host(), &expected, 5e-5);
    }

    /// Verifies that the async GPU ReLU² matches the host fallback when the input carries higher-rank shape metadata.
    /// This catches regressions where the wrapper changes activation values or reshapes tensors unexpectedly.
    #[tokio::test]
    async fn gpu_relu2_matches_host_fallback() {
        let data = vec![-1.0, 0.0, 1.0, 2.0, -3.0, 4.5, -8.0, 6.0];
        let expected = relu2_host(&data);
        let gpu_input = GpuTensor::from_host(&data, &[2, 4]).unwrap();
        let result = super::relu2(&gpu_input).await.unwrap();

        assert_eq!(result.shape(), &[2, 4]);
        approx_slice_eq(&result.to_host(), &expected, 5e-6);
    }

    /// Verifies that async GPU sigmoid preserves shape metadata and numerical parity when the input tensor is multi-dimensional.
    /// This catches regressions in the sigmoid device dispatch and host-fallback bridge.
    #[tokio::test]
    async fn gpu_sigmoid_matches_host_fallback() {
        let data = vec![-1.0, 0.0, 1.0, 2.0, -8.0, 6.0, -3.5, 7.25];
        let expected = sigmoid_host(&data);
        let gpu_input = GpuTensor::from_host(&data, &[2, 4]).unwrap();

        let result = super::sigmoid(&gpu_input).await.unwrap();

        assert_eq!(result.shape(), &[2, 4]);
        approx_slice_eq(&result.to_host(), &expected, 5e-5);
    }

    /// Verifies that async GPU sigmoid-in-place preserves shape metadata while mutating a tensor in place.
    /// This catches regressions where the device wrapper drops tensor metadata or skips the write-back.
    #[tokio::test]
    async fn gpu_sigmoid_in_place_matches_host_fallback() {
        let data = vec![-1.0, 0.0, 1.0, 2.0, -8.0, 6.0, -3.5, 7.25];
        let mut expected = data.clone();
        sigmoid_in_place_host(&mut expected);
        let mut gpu_tensor = GpuTensor::from_host(&data, &[2, 4]).unwrap();

        super::sigmoid_in_place(&mut gpu_tensor).await.unwrap();

        assert_eq!(gpu_tensor.shape(), &[2, 4]);
        approx_slice_eq(&gpu_tensor.to_host(), &expected, 5e-5);
    }
}
