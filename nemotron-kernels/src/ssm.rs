use crate::tensor::{GpuTensor, TensorError};
use crate::KernelStub;

pub const SPEC: KernelStub = KernelStub {
    name: "ssm",
    summary: "Selective scan kernels for chunked Mamba-2 state updates.",
};

#[cfg(any(target_os = "linux", test))]
const CUTILE_SSM_MAX_SEQUENCE_LEN: usize = 1024;
#[cfg(target_os = "linux")]
const CUTILE_SSM_BLOCK_SIZES: [usize; 9] = [256, 128, 64, 32, 16, 8, 4, 2, 1];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SsmBackend {
    HostFallback,
    Cutile,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SsmKernel {
    pub name: &'static str,
    pub backend: SsmBackend,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SelectiveScanShape {
    pub sequence_len: usize,
    pub channel_count: usize,
    pub state_size: usize,
}

impl SelectiveScanShape {
    pub const fn new(sequence_len: usize, channel_count: usize, state_size: usize) -> Self {
        Self {
            sequence_len,
            channel_count,
            state_size,
        }
    }

    pub const fn input_len(self) -> usize {
        self.sequence_len * self.channel_count
    }

    pub const fn state_matrix_len(self) -> usize {
        self.channel_count * self.state_size
    }

    pub const fn initial_state_len(self) -> usize {
        self.state_matrix_len()
    }

    pub const fn dt_len(self) -> usize {
        self.input_len()
    }

    pub const fn b_len(self) -> usize {
        self.sequence_len * self.channel_count * self.state_size
    }

    pub const fn c_len(self) -> usize {
        self.b_len()
    }

    pub const fn d_len(self) -> usize {
        self.channel_count
    }

    pub const fn output_len(self) -> usize {
        self.input_len()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelectiveScanParams<'a> {
    pub input: &'a [f32],
    pub delta_t: &'a [f32],
    pub a: &'a [f32],
    pub b: &'a [f32],
    pub c: &'a [f32],
    pub d: Option<&'a [f32]>,
    pub initial_state: Option<&'a [f32]>,
    pub delta_bias: f32,
    pub apply_softplus_to_dt: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectiveScanOutput {
    pub values: Vec<f32>,
    pub final_state: Vec<f32>,
}

#[cfg(target_os = "linux")]
pub const SELECTIVE_SCAN: SsmKernel = SsmKernel {
    name: "selective_scan_cutile",
    backend: SsmBackend::Cutile,
};

#[cfg(not(target_os = "linux"))]
pub const SELECTIVE_SCAN: SsmKernel = SsmKernel {
    name: "selective_scan_host",
    backend: SsmBackend::HostFallback,
};

pub fn supported_ssm_kernels() -> [SsmKernel; 1] {
    [SELECTIVE_SCAN]
}

pub fn selective_scan_host(
    params: SelectiveScanParams<'_>,
    shape: SelectiveScanShape,
) -> Result<SelectiveScanOutput, SsmError> {
    let mut values = vec![0.0; shape.output_len()];
    let mut final_state = vec![0.0; shape.initial_state_len()];
    selective_scan_into_host(params, shape, &mut values, &mut final_state)?;
    Ok(SelectiveScanOutput {
        values,
        final_state,
    })
}

pub fn selective_scan_into_host(
    params: SelectiveScanParams<'_>,
    shape: SelectiveScanShape,
    output: &mut [f32],
    final_state: &mut [f32],
) -> Result<(), SsmError> {
    validate_params(&params, shape, output, final_state)?;

    if let Some(initial_state) = params.initial_state {
        final_state.copy_from_slice(initial_state);
    } else {
        final_state.fill(0.0);
    }

    for timestep in 0..shape.sequence_len {
        for channel in 0..shape.channel_count {
            let input_index = timestep * shape.channel_count + channel;
            let input_value = params.input[input_index];
            let dt = validated_delta_t(
                params.delta_t[input_index],
                timestep,
                channel,
                params.delta_bias,
                params.apply_softplus_to_dt,
            )?;
            let state_offset = channel * shape.state_size;
            let mut output_acc = 0.0_f64;

            for state_index in 0..shape.state_size {
                let matrix_index = state_offset + state_index;
                let transition = (dt * params.a[matrix_index]).exp();
                let input_mix = dt
                    * params.b[ssm_tensor_offset(timestep, channel, state_index, shape)]
                    * input_value;

                let next_state = transition * final_state[matrix_index] + input_mix;
                final_state[matrix_index] = next_state;

                let projection =
                    params.c[ssm_tensor_offset(timestep, channel, state_index, shape)] * next_state;
                output_acc += f64::from(projection);
            }

            if let Some(direct) = params.d {
                output_acc += f64::from(direct[channel] * input_value);
            }

            output[input_index] = output_acc as f32;
        }
    }

    Ok(())
}

fn validate_params(
    params: &SelectiveScanParams<'_>,
    shape: SelectiveScanShape,
    output: &mut [f32],
    final_state: &mut [f32],
) -> Result<(), SsmError> {
    validate_shape(shape)?;
    validate_len("input", params.input.len(), shape.input_len())?;
    validate_len("delta_t", params.delta_t.len(), shape.dt_len())?;
    validate_len("a", params.a.len(), shape.state_matrix_len())?;
    validate_len("b", params.b.len(), shape.b_len())?;
    validate_len("c", params.c.len(), shape.c_len())?;

    if let Some(direct) = params.d {
        validate_len("d", direct.len(), shape.d_len())?;
    }

    if let Some(initial_state) = params.initial_state {
        validate_len(
            "initial_state",
            initial_state.len(),
            shape.initial_state_len(),
        )?;
    }

    validate_len("output", output.len(), shape.output_len())?;
    validate_len("final_state", final_state.len(), shape.initial_state_len())?;

    Ok(())
}

fn validate_tensor_params(
    params: &GpuSelectiveScanParams<'_>,
    shape: SelectiveScanShape,
) -> Result<(), SsmError> {
    validate_shape(shape)?;
    validate_len("input", params.input.numel(), shape.input_len())?;
    validate_len("delta_t", params.delta_t.numel(), shape.dt_len())?;
    validate_len("a", params.a.numel(), shape.state_matrix_len())?;
    validate_len("b", params.b.numel(), shape.b_len())?;
    validate_len("c", params.c.numel(), shape.c_len())?;

    if let Some(direct) = params.d {
        validate_len("d", direct.numel(), shape.d_len())?;
    }

    if let Some(initial_state) = params.initial_state {
        validate_len(
            "initial_state",
            initial_state.numel(),
            shape.initial_state_len(),
        )?;
    }

    Ok(())
}

fn validate_shape(shape: SelectiveScanShape) -> Result<(), SsmError> {
    if shape.sequence_len == 0 || shape.channel_count == 0 || shape.state_size == 0 {
        return Err(SsmError::InvalidShape(shape));
    }

    Ok(())
}

fn validate_len(argument: &'static str, actual: usize, expected: usize) -> Result<(), SsmError> {
    if actual != expected {
        return Err(SsmError::LengthMismatch {
            argument,
            expected,
            actual,
        });
    }

    Ok(())
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        (1.0 + value.exp()).ln()
    }
}

fn validated_delta_t(
    raw_dt: f32,
    timestep: usize,
    channel: usize,
    delta_bias: f32,
    apply_softplus_to_dt: bool,
) -> Result<f32, SsmError> {
    let raw_dt = raw_dt + delta_bias;
    let dt = if apply_softplus_to_dt {
        softplus(raw_dt)
    } else {
        raw_dt
    };

    if !dt.is_finite() || dt < 0.0 {
        return Err(SsmError::InvalidDeltaT {
            timestep,
            channel,
            value: dt,
        });
    }

    Ok(dt)
}

#[cfg(target_os = "linux")]
async fn materialize_validated_delta_t(
    delta_t: &GpuTensor,
    shape: SelectiveScanShape,
    delta_bias: f32,
    apply_softplus_to_dt: bool,
) -> Result<GpuTensor, SsmError> {
    let delta_t = delta_t.to_host_async().await?;
    let mut validated = Vec::with_capacity(delta_t.len());
    for timestep in 0..shape.sequence_len {
        for channel in 0..shape.channel_count {
            let index = timestep * shape.channel_count + channel;
            validated.push(validated_delta_t(
                delta_t[index],
                timestep,
                channel,
                delta_bias,
                apply_softplus_to_dt,
            )?);
        }
    }
    GpuTensor::from_host_async(&validated, &[shape.sequence_len, shape.channel_count])
        .await
        .map_err(Into::into)
}

#[cfg(target_os = "linux")]
fn select_cutile_block_size(numel: usize) -> usize {
    CUTILE_SSM_BLOCK_SIZES
        .into_iter()
        .find(|block_size| numel % block_size == 0)
        .unwrap_or(1)
}

#[cfg(any(target_os = "linux", test))]
fn supports_cutile_selective_scan(shape: SelectiveScanShape) -> bool {
    if validate_shape(shape).is_err() {
        return false;
    }

    let Some(sequence_channels) = shape.sequence_len.checked_mul(shape.channel_count) else {
        return false;
    };

    shape.state_size == 1
        && shape.sequence_len <= CUTILE_SSM_MAX_SEQUENCE_LEN
        && sequence_channels <= i32::MAX as usize
        && shape.sequence_len <= i32::MAX as usize
        && shape.channel_count <= i32::MAX as usize
}

#[cfg(target_os = "linux")]
fn transpose_output_by_channel(values: &[f32], shape: SelectiveScanShape) -> Vec<f32> {
    let mut output = vec![0.0; shape.output_len()];
    for channel in 0..shape.channel_count {
        for timestep in 0..shape.sequence_len {
            output[timestep * shape.channel_count + channel] =
                values[channel * shape.sequence_len + timestep];
        }
    }
    output
}

fn ssm_tensor_offset(
    timestep: usize,
    channel: usize,
    state_index: usize,
    shape: SelectiveScanShape,
) -> usize {
    ((timestep * shape.channel_count + channel) * shape.state_size) + state_index
}

#[derive(Clone, Debug, PartialEq)]
pub enum SsmError {
    InvalidShape(SelectiveScanShape),
    LengthMismatch {
        argument: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidDeltaT {
        timestep: usize,
        channel: usize,
        value: f32,
    },
    DeviceError(String),
}

impl From<TensorError> for SsmError {
    fn from(e: TensorError) -> Self {
        SsmError::DeviceError(e.to_string())
    }
}

#[cfg(target_os = "linux")]
#[cutile::module]
mod cutile_ssm_kernel {
    use cutile::core::*;

    #[cutile::entry()]
    fn selective_scan<const SEQ_LEN: i32>(
        output: &mut Tensor<f32, { [1, SEQ_LEN] }>,
        final_state: &mut Tensor<f32, { [1] }>,
        input: &Tensor<f32, { [-1, -1] }>,
        delta_t: &Tensor<f32, { [-1, -1] }>,
        a: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
        c: &Tensor<f32, { [-1] }>,
        d: &Tensor<f32, { [-1] }>,
        initial_state: &Tensor<f32, { [-1] }>,
    ) {
        let scalar_tile_shape: Shape<{ [1, 1] }> = const_shape![1, 1];
        let channel = get_tile_block_id().0;
        let channel_count = get_shape_dim(input.shape(), 1i32);
        let input_part: Partition<f32, { [1, 1] }> = input.partition(scalar_tile_shape);
        let delta_t_part: Partition<f32, { [1, 1] }> = delta_t.partition(scalar_tile_shape);
        let a_part: Partition<f32, { [1] }> = a.partition(const_shape![1]);
        let b_part: Partition<f32, { [1] }> = b.partition(const_shape![1]);
        let c_part: Partition<f32, { [1] }> = c.partition(const_shape![1]);
        let d_part: Partition<f32, { [1] }> = d.partition(const_shape![1]);
        let initial_state_part: Partition<f32, { [1] }> = initial_state.partition(const_shape![1]);
        let mut final_state_part: PartitionMut<f32, { [1] }> =
            unsafe { final_state.partition_mut(const_shape![1]) };
        let mut output_part: PartitionMut<f32, { [1, 1] }> =
            unsafe { output.partition_mut(scalar_tile_shape) };

        let initial_state_tile: Tile<f32, { [1] }> = initial_state_part.load([channel]);
        unsafe { final_state_part.store(initial_state_tile, [0i32]) };
        let a_value: f32 =
            tile_to_scalar::<f32, f32>(a_part.load([channel]).reshape(const_shape![]));
        let direct_tile: Tile<f32, { [1] }> = d_part.load([channel]);
        let direct_term: f32 = tile_to_scalar::<f32, f32>(direct_tile.reshape(const_shape![]));
        let mut state_value: f32 =
            tile_to_scalar::<f32, f32>(initial_state_tile.reshape(const_shape![]));

        for timestep in 0i32..SEQ_LEN {
            let input_tile: Tile<f32, { [] }> =
                input_part.load([timestep, channel]).reshape(const_shape![]);
            let input_value: f32 = tile_to_scalar::<f32, f32>(input_tile);
            let raw_dt_tile: Tile<f32, { [] }> = delta_t_part
                .load([timestep, channel])
                .reshape(const_shape![]);
            let dt_value: f32 = tile_to_scalar::<f32, f32>(raw_dt_tile);
            let bc_row = timestep * channel_count + channel;
            let transition_input_tile: Tile<f32, { [] }> = scalar_to_tile(dt_value * a_value);
            let transition_tile: Tile<f32, { [] }> = exp(transition_input_tile);
            let transition: f32 = tile_to_scalar::<f32, f32>(transition_tile);
            let b_value: f32 =
                tile_to_scalar::<f32, f32>(b_part.load([bc_row]).reshape(const_shape![]));
            let c_value: f32 =
                tile_to_scalar::<f32, f32>(c_part.load([bc_row]).reshape(const_shape![]));
            state_value = transition * state_value + dt_value * b_value * input_value;
            let next_state_scalar_tile: Tile<f32, { [] }> = scalar_to_tile(state_value);
            let next_state_tile: Tile<f32, { [1] }> =
                next_state_scalar_tile.reshape(const_shape![1]);
            unsafe { final_state_part.store(next_state_tile, [0i32]) };
            let output_acc = c_value * state_value + direct_term * input_value;

            let output_scalar_tile: Tile<f32, { [] }> = scalar_to_tile(output_acc);
            let output_tile: Tile<f32, { [1, 1] }> = output_scalar_tile.reshape(scalar_tile_shape);
            unsafe { output_part.store(output_tile, [0i32, timestep]) };
        }
    }
}

// ---------------------------------------------------------------------------
// Async GPU API
// ---------------------------------------------------------------------------

/// GPU-side selective scan parameters using GpuTensors.
pub struct GpuSelectiveScanParams<'a> {
    pub input: &'a GpuTensor,
    pub delta_t: &'a GpuTensor,
    pub a: &'a GpuTensor,
    pub b: &'a GpuTensor,
    pub c: &'a GpuTensor,
    pub d: Option<&'a GpuTensor>,
    pub initial_state: Option<&'a GpuTensor>,
    pub delta_bias: f32,
    pub apply_softplus_to_dt: bool,
}

/// Output of the async GPU selective scan.
pub struct GpuSelectiveScanOutput {
    pub output: GpuTensor,
    pub final_state: GpuTensor,
}

async fn selective_scan_host_bridge(
    params: GpuSelectiveScanParams<'_>,
    shape: SelectiveScanShape,
) -> Result<GpuSelectiveScanOutput, SsmError> {
    let input = params.input.to_host_async().await?;
    let delta_t = params.delta_t.to_host_async().await?;
    let a = params.a.to_host_async().await?;
    let b = params.b.to_host_async().await?;
    let c = params.c.to_host_async().await?;
    let d = match params.d {
        Some(t) => Some(t.to_host_async().await?),
        None => None,
    };
    let initial_state = match params.initial_state {
        Some(t) => Some(t.to_host_async().await?),
        None => None,
    };

    let host_params = SelectiveScanParams {
        input: &input,
        delta_t: &delta_t,
        a: &a,
        b: &b,
        c: &c,
        d: d.as_deref(),
        initial_state: initial_state.as_deref(),
        delta_bias: params.delta_bias,
        apply_softplus_to_dt: params.apply_softplus_to_dt,
    };

    let host_result = selective_scan_host(host_params, shape)?;
    let output_shape = &[shape.sequence_len, shape.channel_count];
    let state_shape = &[shape.channel_count, shape.state_size];
    Ok(GpuSelectiveScanOutput {
        output: GpuTensor::from_host_async(&host_result.values, output_shape).await?,
        final_state: GpuTensor::from_host_async(&host_result.final_state, state_shape).await?,
    })
}

#[cfg(target_os = "linux")]
async fn selective_scan_cutile(
    params: &GpuSelectiveScanParams<'_>,
    shape: SelectiveScanShape,
) -> Result<Option<GpuSelectiveScanOutput>, SsmError> {
    use cutile::tensor::ToHostVec;
    use cutile::tile_kernel::{IntoDeviceOperation, IntoDeviceOperationPartition, TileKernel};

    if !supports_cutile_selective_scan(shape) {
        return Ok(None);
    }

    let validated_delta_t = materialize_validated_delta_t(
        params.delta_t,
        shape,
        params.delta_bias,
        params.apply_softplus_to_dt,
    )
    .await?;

    let seq_block_size = select_cutile_block_size(shape.sequence_len);
    let seq_len_i32 = i32::try_from(shape.sequence_len).map_err(|_| {
        SsmError::DeviceError("cutile SSM sequence length overflowed i32".to_string())
    })?;
    let seq_block_size_i32 = i32::try_from(seq_block_size).map_err(|_| {
        SsmError::DeviceError("cutile SSM sequence block size overflowed i32".to_string())
    })?;
    let bc_rows = shape
        .sequence_len
        .checked_mul(shape.channel_count)
        .ok_or_else(|| {
            SsmError::DeviceError("cutile SSM B/C row count overflowed usize".to_string())
        })?;
    let direct_term = match params.d {
        Some(direct_term) => direct_term.clone(),
        None => GpuTensor::zeros(&[shape.channel_count])?,
    };
    let initial_state = match params.initial_state {
        Some(initial_state) => initial_state.clone(),
        None => GpuTensor::zeros(&[shape.channel_count, shape.state_size])?,
    };
    let output = cutile::api::zeros::<2, f32>([shape.channel_count, shape.sequence_len])
        .partition([1, seq_block_size_i32]);
    let final_state = cutile::api::zeros::<1, f32>([shape.channel_count]).partition([1]);
    let input = params
        .input
        .cutile_tensor_for_shape(&[shape.sequence_len, shape.channel_count])
        .await?;
    let delta_t = validated_delta_t
        .cutile_tensor_for_shape(&[shape.sequence_len, shape.channel_count])
        .await?;
    let a = params
        .a
        .cutile_tensor_for_shape(&[shape.channel_count])
        .await?;
    let b = params.b.cutile_tensor_for_shape(&[bc_rows]).await?;
    let c = params.c.cutile_tensor_for_shape(&[bc_rows]).await?;
    let d = direct_term
        .cutile_tensor_for_shape(&[shape.channel_count])
        .await?;
    let initial_state = initial_state
        .cutile_tensor_for_shape(&[shape.channel_count])
        .await?;
    let generics = vec![seq_len_i32.to_string()];
    let (output, final_state, _input, _delta_t, _a, _b, _c, _d, _initial_state) =
        cutile_ssm_kernel::selective_scan_async(
            output,
            final_state,
            input.device_operation(),
            delta_t.device_operation(),
            a.device_operation(),
            b.device_operation(),
            c.device_operation(),
            d.device_operation(),
            initial_state.device_operation(),
        )
        .generics(generics)
        .await
        .map_err(|error| {
            SsmError::DeviceError(format!("cutile selective scan launch failed: {error:?}"))
        })?;
    let output_by_channel = output.unpartition();
    let final_state = final_state.unpartition();
    let output_by_channel_host = output_by_channel.to_host_vec().await.map_err(|error| {
        SsmError::DeviceError(format!(
            "cutile selective scan output copy failed: {error:?}"
        ))
    })?;
    let output = transpose_output_by_channel(&output_by_channel_host, shape);
    Ok(Some(GpuSelectiveScanOutput {
        output: GpuTensor::from_host_async(&output, &[shape.sequence_len, shape.channel_count])
            .await?,
        final_state: GpuTensor::from_cutile_tensor(
            final_state.reshape_dyn(&[shape.channel_count, shape.state_size]),
            &[shape.channel_count, shape.state_size],
        )?,
    }))
}

/// Async GPU selective scan (Mamba-2 SSM kernel).
pub async fn selective_scan(
    params: GpuSelectiveScanParams<'_>,
    shape: SelectiveScanShape,
) -> Result<GpuSelectiveScanOutput, SsmError> {
    validate_tensor_params(&params, shape)?;
    #[cfg(target_os = "linux")]
    if let Some(result) = selective_scan_cutile(&params, shape).await? {
        return Ok(result);
    }
    selective_scan_host_bridge(params, shape).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_slice(lhs: &[f32], rhs: &[f32], tolerance: f32) {
        assert_eq!(lhs.len(), rhs.len(), "slice lengths differ");
        for (index, (left, right)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let diff = (left - right).abs();
            assert!(
                diff <= tolerance,
                "index {index}: left={left:?}, right={right:?}, diff={diff:?}, tolerance={tolerance:?}"
            );
        }
    }

    /// Verifies that the SSM kernel registry reports the active primary backend for the current platform.
    ///
    /// This catches accidental backend tag changes as Linux adds a real cutile path while other platforms keep host fallback.
    #[test]
    fn reports_platform_primary_backend() {
        #[cfg(target_os = "linux")]
        let expected = [SsmKernel {
            name: "selective_scan_cutile",
            backend: SsmBackend::Cutile,
        }];

        #[cfg(not(target_os = "linux"))]
        let expected = [SsmKernel {
            name: "selective_scan_host",
            backend: SsmBackend::HostFallback,
        }];

        assert_eq!(supported_ssm_kernels(), expected);
    }

    /// Verifies that the cutile dispatch heuristic only accepts shapes inside the current bounded safety envelope.
    ///
    /// This catches regressions where Linux would try to compile the first selective-scan kernel for oversized shapes instead of falling back safely.
    #[test]
    fn cutile_support_heuristic_respects_bounds() {
        assert!(supports_cutile_selective_scan(SelectiveScanShape::new(
            8, 4, 1
        )));
        assert!(supports_cutile_selective_scan(SelectiveScanShape::new(
            CUTILE_SSM_MAX_SEQUENCE_LEN,
            1,
            1,
        )));
        assert!(!supports_cutile_selective_scan(SelectiveScanShape::new(
            CUTILE_SSM_MAX_SEQUENCE_LEN + 1,
            1,
            1,
        )));
        assert!(!supports_cutile_selective_scan(SelectiveScanShape::new(
            8, 1, 2,
        )));
    }

    /// Verifies SSM output and final state for a 2-step, 1-channel, 2-state sequence.
    ///
    /// This catches errors in the discretization formula or state accumulation.
    #[test]
    fn selective_scan_updates_state_and_output() {
        let shape = SelectiveScanShape::new(2, 1, 2);
        let output = selective_scan_host(
            SelectiveScanParams {
                input: &[1.0, 2.0],
                delta_t: &[1.0, 1.0],
                a: &[0.0, 0.0],
                b: &[
                    1.0, 0.5, //
                    1.0, 0.5, //
                ],
                c: &[
                    0.25, 1.0, //
                    0.25, 1.0, //
                ],
                d: Some(&[0.1]),
                initial_state: None,
                delta_bias: 0.0,
                apply_softplus_to_dt: false,
            },
            shape,
        )
        .unwrap();

        approx_eq_slice(&output.values, &[0.85, 2.45], 1e-5);
        approx_eq_slice(&output.final_state, &[3.0, 1.5], 1e-5);
    }

    /// Verifies that a non-zero initial state is carried into the first timestep.
    ///
    /// This catches bugs where initial_state is ignored or zeroed.
    #[test]
    fn selective_scan_uses_initial_state() {
        let shape = SelectiveScanShape::new(1, 1, 1);
        let output = selective_scan_host(
            SelectiveScanParams {
                input: &[2.0],
                delta_t: &[1.0],
                a: &[0.0],
                b: &[0.5],
                c: &[2.0],
                d: None,
                initial_state: Some(&[3.0]),
                delta_bias: 0.0,
                apply_softplus_to_dt: false,
            },
            shape,
        )
        .unwrap();

        approx_eq_slice(&output.values, &[8.0], 1e-5);
        approx_eq_slice(&output.final_state, &[4.0], 1e-5);
    }

    /// Verifies that softplus is applied to delta_t when the flag is set.
    ///
    /// This catches missing or incorrect softplus gating of the time step.
    #[test]
    fn selective_scan_applies_softplus_to_delta_t() {
        let shape = SelectiveScanShape::new(1, 1, 1);
        let output = selective_scan_host(
            SelectiveScanParams {
                input: &[2.0],
                delta_t: &[-1.0],
                a: &[0.0],
                b: &[1.0],
                c: &[1.0],
                d: None,
                initial_state: None,
                delta_bias: 0.0,
                apply_softplus_to_dt: true,
            },
            shape,
        )
        .unwrap();

        approx_eq_slice(&output.values, &[0.6265234], 1e-5);
        approx_eq_slice(&output.final_state, &[0.6265234], 1e-5);
    }

    /// Verifies that the _into variant writes into pre-allocated output and state buffers.
    ///
    /// This catches bugs where _into silently re-allocates instead of writing in place.
    #[test]
    fn selective_scan_into_writes_existing_buffers() {
        let shape = SelectiveScanShape::new(2, 1, 1);
        let params = SelectiveScanParams {
            input: &[1.0, 1.0],
            delta_t: &[1.0, 1.0],
            a: &[0.0],
            b: &[1.0, 1.0],
            c: &[1.0, 1.0],
            d: None,
            initial_state: None,
            delta_bias: 0.0,
            apply_softplus_to_dt: false,
        };
        let mut output = [-1.0; 2];
        let mut final_state = [-1.0; 1];

        selective_scan_into_host(params, shape, &mut output, &mut final_state).unwrap();

        approx_eq_slice(&output, &[1.0, 2.0], 1e-5);
        approx_eq_slice(&final_state, &[2.0], 1e-5);
    }

    /// Verifies that a zero sequence_len is rejected as an invalid shape.
    ///
    /// This catches missing dimension validation.
    #[test]
    fn rejects_invalid_shape() {
        let shape = SelectiveScanShape::new(0, 1, 1);
        let error = selective_scan_host(
            SelectiveScanParams {
                input: &[],
                delta_t: &[],
                a: &[1.0],
                b: &[],
                c: &[],
                d: Some(&[1.0]),
                initial_state: Some(&[0.0]),
                delta_bias: 0.0,
                apply_softplus_to_dt: false,
            },
            shape,
        )
        .unwrap_err();

        assert_eq!(error, SsmError::InvalidShape(shape));
    }

    /// Verifies that a mismatched B parameter length is rejected.
    ///
    /// This catches missing parameter length validation.
    #[test]
    fn rejects_length_mismatch() {
        let shape = SelectiveScanShape::new(1, 1, 1);
        let error = selective_scan_host(
            SelectiveScanParams {
                input: &[1.0],
                delta_t: &[1.0],
                a: &[1.0],
                b: &[],
                c: &[1.0],
                d: None,
                initial_state: None,
                delta_bias: 0.0,
                apply_softplus_to_dt: false,
            },
            shape,
        )
        .unwrap_err();

        assert_eq!(
            error,
            SsmError::LengthMismatch {
                argument: "b",
                expected: 1,
                actual: 0,
            }
        );
    }

    /// Verifies that a negative delta_t (without softplus) is rejected at runtime.
    ///
    /// This catches missing dt sign validation, which would produce unstable SSM dynamics.
    #[test]
    fn rejects_negative_delta_t_without_softplus() {
        let shape = SelectiveScanShape::new(1, 1, 1);
        let error = selective_scan_host(
            SelectiveScanParams {
                input: &[1.0],
                delta_t: &[-0.5],
                a: &[0.0],
                b: &[1.0],
                c: &[1.0],
                d: None,
                initial_state: None,
                delta_bias: 0.0,
                apply_softplus_to_dt: false,
            },
            shape,
        )
        .unwrap_err();

        assert_eq!(
            error,
            SsmError::InvalidDeltaT {
                timestep: 0,
                channel: 0,
                value: -0.5,
            }
        );
    }

    /// Verifies that the async GPU selective scan matches the host implementation and preserves tensor shapes when optional tensors are provided.
    ///
    /// This catches regressions in the device path, optional-state materialization, and final output shape reconstruction.
    #[tokio::test]
    async fn gpu_selective_scan_matches_host_and_preserves_shapes() {
        let shape = SelectiveScanShape::new(2, 1, 1);
        let input = vec![1.0, 2.0];
        let delta_t = vec![1.0, 1.0];
        let a = vec![0.0];
        let b = vec![
            1.0, //
            1.0, //
        ];
        let c = vec![
            0.25, //
            0.25, //
        ];
        let d = vec![0.1];
        let initial_state = vec![0.5];
        let expected = selective_scan_host(
            SelectiveScanParams {
                input: &input,
                delta_t: &delta_t,
                a: &a,
                b: &b,
                c: &c,
                d: Some(&d),
                initial_state: Some(&initial_state),
                delta_bias: 0.0,
                apply_softplus_to_dt: false,
            },
            shape,
        )
        .unwrap();
        let gpu_input = GpuTensor::from_host(&input, &[2, 1]).unwrap();
        let gpu_delta_t = GpuTensor::from_host(&delta_t, &[2, 1]).unwrap();
        let gpu_a = GpuTensor::from_host(&a, &[1, 1]).unwrap();
        let gpu_b = GpuTensor::from_host(&b, &[2, 1, 1]).unwrap();
        let gpu_c = GpuTensor::from_host(&c, &[2, 1, 1]).unwrap();
        let gpu_d = GpuTensor::from_host(&d, &[1]).unwrap();
        let gpu_initial_state = GpuTensor::from_host(&initial_state, &[1, 1]).unwrap();

        let result = super::selective_scan(
            GpuSelectiveScanParams {
                input: &gpu_input,
                delta_t: &gpu_delta_t,
                a: &gpu_a,
                b: &gpu_b,
                c: &gpu_c,
                d: Some(&gpu_d),
                initial_state: Some(&gpu_initial_state),
                delta_bias: 0.0,
                apply_softplus_to_dt: false,
            },
            shape,
        )
        .await
        .unwrap();

        assert_eq!(result.output.shape(), &[2, 1]);
        assert_eq!(result.final_state.shape(), &[1, 1]);
        approx_eq_slice(&result.output.to_host(), &expected.values, 1e-4);
        approx_eq_slice(&result.final_state.to_host(), &expected.final_state, 1e-4);
    }

    /// Verifies that the async GPU selective scan falls back cleanly for unsupported state sizes while preserving host parity.
    ///
    /// This catches regressions where Linux would try to launch the minimal cutile path outside its documented `state_size == 1` support envelope.
    #[tokio::test]
    async fn gpu_selective_scan_falls_back_for_unsupported_state_size() {
        let shape = SelectiveScanShape::new(2, 1, 2);
        let input = vec![1.0, 2.0];
        let delta_t = vec![1.0, 1.0];
        let a = vec![0.0, 0.0];
        let b = vec![
            1.0, 0.5, //
            1.0, 0.5, //
        ];
        let c = vec![
            0.25, 1.0, //
            0.25, 1.0, //
        ];
        let expected = selective_scan_host(
            SelectiveScanParams {
                input: &input,
                delta_t: &delta_t,
                a: &a,
                b: &b,
                c: &c,
                d: None,
                initial_state: None,
                delta_bias: 0.0,
                apply_softplus_to_dt: false,
            },
            shape,
        )
        .unwrap();
        let gpu_input = GpuTensor::from_host(&input, &[2, 1]).unwrap();
        let gpu_delta_t = GpuTensor::from_host(&delta_t, &[2, 1]).unwrap();
        let gpu_a = GpuTensor::from_host(&a, &[1, 2]).unwrap();
        let gpu_b = GpuTensor::from_host(&b, &[2, 1, 2]).unwrap();
        let gpu_c = GpuTensor::from_host(&c, &[2, 1, 2]).unwrap();

        let result = super::selective_scan(
            GpuSelectiveScanParams {
                input: &gpu_input,
                delta_t: &gpu_delta_t,
                a: &gpu_a,
                b: &gpu_b,
                c: &gpu_c,
                d: None,
                initial_state: None,
                delta_bias: 0.0,
                apply_softplus_to_dt: false,
            },
            shape,
        )
        .await
        .unwrap();

        approx_eq_slice(&result.output.to_host(), &expected.values, 1e-4);
        approx_eq_slice(&result.final_state.to_host(), &expected.final_state, 1e-4);
    }

    /// Verifies that the async GPU selective scan matches the host implementation when optional direct/state tensors are absent.
    ///
    /// This catches regressions in the zero-materialized cutile inputs and fallback parity for the minimal supported path.
    #[tokio::test]
    async fn gpu_selective_scan_matches_host_without_optional_tensors() {
        let shape = SelectiveScanShape::new(2, 2, 1);
        let input = vec![1.0, -2.0, 0.5, 3.0];
        let delta_t = vec![0.2, 0.1, 0.3, 0.4];
        let a = vec![-0.2, -0.1];
        let b = vec![
            0.5, //
            0.1, //
            0.3, //
            0.6, //
        ];
        let c = vec![
            0.6, //
            0.3, //
            0.5, //
            0.2, //
        ];
        let expected = selective_scan_host(
            SelectiveScanParams {
                input: &input,
                delta_t: &delta_t,
                a: &a,
                b: &b,
                c: &c,
                d: None,
                initial_state: None,
                delta_bias: 0.0,
                apply_softplus_to_dt: false,
            },
            shape,
        )
        .unwrap();
        let gpu_input = GpuTensor::from_host(&input, &[2, 2]).unwrap();
        let gpu_delta_t = GpuTensor::from_host(&delta_t, &[2, 2]).unwrap();
        let gpu_a = GpuTensor::from_host(&a, &[2, 1]).unwrap();
        let gpu_b = GpuTensor::from_host(&b, &[2, 2, 1]).unwrap();
        let gpu_c = GpuTensor::from_host(&c, &[2, 2, 1]).unwrap();

        let result = super::selective_scan(
            GpuSelectiveScanParams {
                input: &gpu_input,
                delta_t: &gpu_delta_t,
                a: &gpu_a,
                b: &gpu_b,
                c: &gpu_c,
                d: None,
                initial_state: None,
                delta_bias: 0.0,
                apply_softplus_to_dt: false,
            },
            shape,
        )
        .await
        .unwrap();

        approx_eq_slice(&result.output.to_host(), &expected.values, 1e-4);
        approx_eq_slice(&result.final_state.to_host(), &expected.final_state, 1e-4);
    }

    /// Verifies that the async GPU selective scan propagates validation errors when delta_t is negative without softplus.
    ///
    /// This catches regressions where the device path might skip the required dt stability checks before launching cutile.
    #[tokio::test]
    async fn gpu_selective_scan_propagates_invalid_delta_t() {
        let shape = SelectiveScanShape::new(1, 1, 1);
        let gpu_input = GpuTensor::from_host(&[1.0], &[1, 1]).unwrap();
        let gpu_delta_t = GpuTensor::from_host(&[-0.5], &[1, 1]).unwrap();
        let gpu_a = GpuTensor::from_host(&[0.0], &[1, 1]).unwrap();
        let gpu_b = GpuTensor::from_host(&[1.0], &[1, 1, 1]).unwrap();
        let gpu_c = GpuTensor::from_host(&[1.0], &[1, 1, 1]).unwrap();

        let error = match super::selective_scan(
            GpuSelectiveScanParams {
                input: &gpu_input,
                delta_t: &gpu_delta_t,
                a: &gpu_a,
                b: &gpu_b,
                c: &gpu_c,
                d: None,
                initial_state: None,
                delta_bias: 0.0,
                apply_softplus_to_dt: false,
            },
            shape,
        )
        .await
        {
            Ok(_) => panic!("expected InvalidDeltaT error"),
            Err(error) => error,
        };

        assert_eq!(
            error,
            SsmError::InvalidDeltaT {
                timestep: 0,
                channel: 0,
                value: -0.5,
            }
        );
    }
}
