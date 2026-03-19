use crate::KernelStub;

pub const SPEC: KernelStub = KernelStub {
    name: "activations",
    summary: "SiLU, sigmoid_host, and ReLU² GPU kernels.",
};

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

pub const SILU: ActivationKernel = ActivationKernel {
    name: "silu_host",
    backend: ActivationBackend::HostFallback,
};

pub const RELU2: ActivationKernel = ActivationKernel {
    name: "relu2_host",
    backend: ActivationBackend::HostFallback,
};

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

    /// Verifies that all activation kernels report HostFallback as their backend.
    ///
    /// This catches accidental backend tag changes before GPU kernels exist.
    #[test]
    fn reports_host_fallback_backend_for_now() {
        assert_eq!(
            supported_activations(),
            [
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
            ]
        );
    }

    /// Verifies sigmoid_host at zero, positive, and negative inputs against known values.
    ///
    /// This catches regressions in the numerically stable split-branch sigmoid_host.
    #[test]
    fn sigmoid_matches_reference_values() {
        approx_eq(sigmoid_scalar(0.0), 0.5);
        approx_eq(sigmoid_scalar(2.0), 0.880797);
        approx_eq(sigmoid_scalar(-2.0), 0.11920292);
    }

    /// Verifies SiLU (x·σ(x)) at zero, positive, and negative inputs.
    ///
    /// This catches errors in the SiLU composition with sigmoid_host.
    #[test]
    fn silu_matches_reference_values() {
        approx_eq(silu_scalar(0.0), 0.0);
        approx_eq(silu_scalar(1.0), 0.7310586);
        approx_eq(silu_scalar(-1.0), -0.26894143);
    }

    /// Verifies ReLU² clamps negatives to zero and squares positive values.
    ///
    /// This catches sign errors or missing squaring in the relu2_host formula.
    #[test]
    fn relu2_matches_reference_values() {
        approx_eq(relu2_scalar(-3.0), 0.0);
        approx_eq(relu2_scalar(0.0), 0.0);
        approx_eq(relu2_scalar(1.5), 2.25);
    }

    /// Verifies that the vector API allocates a correctly sized output.
    ///
    /// This catches length miscalculation in the allocating wrappers.
    #[test]
    fn vector_api_allocates_output() {
        assert_eq!(sigmoid_host(&[0.0, 1.0]).len(), 2);
        assert_eq!(relu2_host(&[-1.0, 2.0]), vec![0.0, 4.0]);
    }

    /// Verifies that the in-place API mutates the buffer with correct SiLU values.
    ///
    /// This catches issues where in-place writes are skipped or mis-ordered.
    #[test]
    fn in_place_api_updates_buffer() {
        let mut values = [-1.0, 0.0, 2.0];
        silu_in_place_host(&mut values);

        approx_eq(values[0], -0.26894143);
        approx_eq(values[1], 0.0);
        approx_eq(values[2], 1.761594);
    }
}
