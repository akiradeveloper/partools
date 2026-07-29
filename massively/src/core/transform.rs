//! Materialization and unary transform dispatch.

use cubecl::prelude::*;

use crate::core::bindings::Bindings;
use crate::core::eval::Eval13;
use crate::core::output::{
    LowerOutputExpression, OutputExpression, PaddedOutputSlots, StageOutput,
};
use crate::core::read::{
    Env0, LowerReadExpression, PaddedReadSlots, ReadExpression, StageRead, Transform,
};
use crate::core::storage::{Decompose, StorageLayout, StorePadded12, StorePadded12Expand};
use crate::op::UnaryOp;
use crate::{Error, Executor};

const BLOCK_SIZE: u32 = 256;

macro_rules! define_padded_materialize_kernel {
    ($name:ident, $eval:ident, $method:ident; [$( $leaf:ident : $slot:ident ),+]) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $leaf: CubePrimitive + cubecl::frontend::Scalar, )+
            O0: CubePrimitive + cubecl::frontend::Scalar,
            O1: CubePrimitive + cubecl::frontend::Scalar,
            O2: CubePrimitive + cubecl::frontend::Scalar,
            O3: CubePrimitive + cubecl::frontend::Scalar,
            O4: CubePrimitive + cubecl::frontend::Scalar,
            O5: CubePrimitive + cubecl::frontend::Scalar,
            O6: CubePrimitive + cubecl::frontend::Scalar,
            O7: CubePrimitive + cubecl::frontend::Scalar,
            O8: CubePrimitive + cubecl::frontend::Scalar,
            O9: CubePrimitive + cubecl::frontend::Scalar,
            O10: CubePrimitive + cubecl::frontend::Scalar,
            O11: CubePrimitive + cubecl::frontend::Scalar,
            Leaves: CubeType + Send + Sync + 'static
                + StorePadded12<
                    O0 = O0, O1 = O1, O2 = O2, O3 = O3, O4 = O4, O5 = O5,
                    O6 = O6, O7 = O7, O8 = O8, O9 = O9, O10 = O10, O11 = O11,
                >,
            Expr: $eval<Item, $( $leaf ),+>,
            Layout: Decompose<Item, Leaves = Leaves>,
        >(
            $( $slot: &[$leaf], )+
            read_offsets: &[u32],
            len: &[u32],
            out0: &mut [O0],
            out1: &mut [O1],
            out2: &mut [O2],
            out3: &mut [O3],
            out4: &mut [O4],
            out5: &mut [O5],
            out6: &mut [O6],
            out7: &mut [O7],
            out8: &mut [O8],
            out9: &mut [O9],
            out10: &mut [O10],
            out11: &mut [O11],
            write_offsets: &[u32],
        ) {
            let index = ABSOLUTE_POS as usize;
            if index < len[0] as usize {
                let value = Expr::$method($( $slot, )+ read_offsets, index);
                Layout::decompose(value).store_padded(
                    out0, out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11,
                    write_offsets, index,
                );
            }
        }
    };
}

define_padded_materialize_kernel!(materialize_a13, Eval13, eval13; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7,L8:slot8,L9:slot9,L10:slot10,L11:slot11,L12:slot12]);

pub(crate) fn materialize_fixed<R, Input, Output>(
    exec: &Executor<R>,
    input: &Input,
    output: &Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Input::Item: StorageLayout,
    <Input::Item as StorageLayout>::StorageLeaves: StorePadded12,
    <<Input::Item as StorageLayout>::StorageLeaves as CubeType>::ExpandType: StorePadded12Expand,
    Output: OutputExpression<Item = Input::Item> + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Slots: PaddedOutputSlots<Leaves = <Input::Item as StorageLayout>::StorageLeaves>,
{
    materialize_prefix_fixed(exec, input, None, output)
}

/// Evaluates at most the device-resident prefix of `input`.
///
/// The host-known input length is the physical dispatch capacity.  `active_len`
/// only narrows that range in the kernel and is never copied to the host.
pub(crate) fn materialize_prefix_fixed<R, Input, Output>(
    exec: &Executor<R>,
    input: &Input,
    active_len: Option<&crate::DeviceVec<R, u32>>,
    output: &Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Input::Item: StorageLayout,
    <Input::Item as StorageLayout>::StorageLeaves: StorePadded12,
    <<Input::Item as StorageLayout>::StorageLeaves as CubeType>::ExpandType: StorePadded12Expand,
    Output: OutputExpression<Item = Input::Item> + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Slots: PaddedOutputSlots<Leaves = <Input::Item as StorageLayout>::StorageLeaves>,
{
    let input_len = input.physical_len()?;
    let output_len = output.physical_len()?;
    if output_len < input_len {
        return Err(Error::OutputTooShort {
            input: input_len,
            output: output_len,
        });
    }
    if input_len == 0 {
        return Ok(());
    }
    let len = u32::try_from(input_len).map_err(|_| Error::LengthTooLarge { len: input_len })?;
    let reads = Bindings::read(exec, input)?;
    let writes = Bindings::write(exec, output)?;
    let read_offsets = exec
        .client()
        .create_from_slice(u32::as_bytes(&reads.offsets));
    let write_offsets = exec
        .client()
        .create_from_slice(u32::as_bytes(&writes.offsets));
    let logical_len = match active_len {
        Some(active_len) => crate::core::extent::LogicalExtent::from_device(active_len, input_len),
        None => input.logical_extent()?,
    }
    .materialize(exec)?;
    let cubes = crate::core::launch::cube_count_1d((len as usize).div_ceil(BLOCK_SIZE as usize))?;
    unsafe {
        materialize_a13::launch_unchecked::<
            Input::Item,
            <Input::Slots as PaddedReadSlots>::L0,
            <Input::Slots as PaddedReadSlots>::L1,
            <Input::Slots as PaddedReadSlots>::L2,
            <Input::Slots as PaddedReadSlots>::L3,
            <Input::Slots as PaddedReadSlots>::L4,
            <Input::Slots as PaddedReadSlots>::L5,
            <Input::Slots as PaddedReadSlots>::L6,
            <Input::Slots as PaddedReadSlots>::L7,
            <Input::Slots as PaddedReadSlots>::L8,
            <Input::Slots as PaddedReadSlots>::L9,
            <Input::Slots as PaddedReadSlots>::L10,
            <Input::Slots as PaddedReadSlots>::L11,
            <Input::Slots as PaddedReadSlots>::L12,
            <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O0,
            <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O1,
            <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O2,
            <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O3,
            <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O4,
            <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O5,
            <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O6,
            <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O7,
            <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O8,
            <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O9,
            <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O10,
            <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O11,
            <Output::Item as StorageLayout>::StorageLeaves,
            Input::DeviceExpr,
            <Output::Item as StorageLayout>::DeviceLayout,
            R,
        >(
            exec.client(),
            cubes,
            CubeDim::new_1d(BLOCK_SIZE),
            BufferArg::from_raw_parts(reads.slots[0].0.clone(), reads.slots[0].1),
            BufferArg::from_raw_parts(reads.slots[1].0.clone(), reads.slots[1].1),
            BufferArg::from_raw_parts(reads.slots[2].0.clone(), reads.slots[2].1),
            BufferArg::from_raw_parts(reads.slots[3].0.clone(), reads.slots[3].1),
            BufferArg::from_raw_parts(reads.slots[4].0.clone(), reads.slots[4].1),
            BufferArg::from_raw_parts(reads.slots[5].0.clone(), reads.slots[5].1),
            BufferArg::from_raw_parts(reads.slots[6].0.clone(), reads.slots[6].1),
            BufferArg::from_raw_parts(reads.slots[7].0.clone(), reads.slots[7].1),
            BufferArg::from_raw_parts(reads.slots[8].0.clone(), reads.slots[8].1),
            BufferArg::from_raw_parts(reads.slots[9].0.clone(), reads.slots[9].1),
            BufferArg::from_raw_parts(reads.slots[10].0.clone(), reads.slots[10].1),
            BufferArg::from_raw_parts(reads.slots[11].0.clone(), reads.slots[11].1),
            BufferArg::from_raw_parts(reads.slots[12].0.clone(), reads.slots[12].1),
            BufferArg::from_raw_parts(read_offsets, reads.offsets.len()),
            BufferArg::from_raw_parts(logical_len.handle.clone(), 1),
            BufferArg::from_raw_parts(writes.slots[0].0.clone(), writes.slots[0].1),
            BufferArg::from_raw_parts(writes.slots[1].0.clone(), writes.slots[1].1),
            BufferArg::from_raw_parts(writes.slots[2].0.clone(), writes.slots[2].1),
            BufferArg::from_raw_parts(writes.slots[3].0.clone(), writes.slots[3].1),
            BufferArg::from_raw_parts(writes.slots[4].0.clone(), writes.slots[4].1),
            BufferArg::from_raw_parts(writes.slots[5].0.clone(), writes.slots[5].1),
            BufferArg::from_raw_parts(writes.slots[6].0.clone(), writes.slots[6].1),
            BufferArg::from_raw_parts(writes.slots[7].0.clone(), writes.slots[7].1),
            BufferArg::from_raw_parts(writes.slots[8].0.clone(), writes.slots[8].1),
            BufferArg::from_raw_parts(writes.slots[9].0.clone(), writes.slots[9].1),
            BufferArg::from_raw_parts(writes.slots[10].0.clone(), writes.slots[10].1),
            BufferArg::from_raw_parts(writes.slots[11].0.clone(), writes.slots[11].1),
            BufferArg::from_raw_parts(write_offsets, writes.offsets.len()),
        );
    }
    Ok(())
}

/// Evaluates a lazy read expression and writes it to a preallocated output tree.
pub(crate) fn materialize<R, Input, Output>(
    exec: &Executor<R>,
    input: Input,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Input::Item: StorageLayout,
    <Input::Item as StorageLayout>::StorageLeaves: StorePadded12,
    <<Input::Item as StorageLayout>::StorageLeaves as CubeType>::ExpandType: StorePadded12Expand,
    Output: OutputExpression<Item = Input::Item> + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Slots: PaddedOutputSlots<Leaves = <Input::Item as StorageLayout>::StorageLeaves>,
{
    materialize_fixed(exec, &input, &output)
}

/// Applies a unary operation through the fixed thirteen-read/twelve-write ABI
/// without selecting an arity-specific dispatch implementation.
pub(crate) fn transform_fixed<R, Input, Output, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: ReadExpression,
    Op: UnaryOp<Input::Item>,
    Op::Output: crate::api::iter::KernelRow,
    Transform<Input, Op>:
        ReadExpression<Item = Op::Output> + LowerReadExpression + StageRead<R, Env0>,
    Output: OutputExpression<Item = <Op as UnaryOp<Input::Item>>::Output>
        + LowerOutputExpression
        + StageOutput<R, Env0>,
    Output::Slots: PaddedOutputSlots<Leaves = <Output::Item as StorageLayout>::StorageLeaves>,
    <Output::Item as StorageLayout>::StorageLeaves: StorePadded12,
    <<Output::Item as StorageLayout>::StorageLeaves as CubeType>::ExpandType: StorePadded12Expand,
{
    let input = Transform::new(input, op);
    materialize_fixed(exec, &input, &output)
}

/// Applies a unary operation to a device-resident prefix through the fixed
/// read/write ABI.  The physical input length remains the dispatch bound.
pub(crate) fn transform_prefix_fixed<R, Input, Output, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
    active_len: &crate::DeviceVec<R, u32>,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: ReadExpression,
    Op: UnaryOp<Input::Item>,
    Op::Output: crate::api::iter::KernelRow,
    Transform<Input, Op>:
        ReadExpression<Item = Op::Output> + LowerReadExpression + StageRead<R, Env0>,
    Output: OutputExpression<Item = <Op as UnaryOp<Input::Item>>::Output>
        + LowerOutputExpression
        + StageOutput<R, Env0>,
    Output::Slots: PaddedOutputSlots<Leaves = <Output::Item as StorageLayout>::StorageLeaves>,
    <Output::Item as StorageLayout>::StorageLeaves: StorePadded12,
    <<Output::Item as StorageLayout>::StorageLeaves as CubeType>::ExpandType: StorePadded12Expand,
{
    let input = Transform::new(input, op);
    materialize_prefix_fixed(exec, &input, Some(active_len), &output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DeviceVec;
    use crate::core::iter::Zip;
    use crate::core::read::{Counting, Permute};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    struct Double;

    #[cubecl::cube]
    impl UnaryOp<u32> for Double {
        type Output = u32;
        fn apply(input: u32) -> u32 {
            input * 2
        }
    }

    #[test]
    fn generated_materialization_fits_the_binding_budget() {
        type ScalarLeaves = <u32 as StorageLayout>::StorageLeaves;
        type ScalarExpr = <crate::core::read::Column<u32> as LowerReadExpression>::DeviceExpr;
        type ScalarLayout = <u32 as StorageLayout>::DeviceLayout;
        type Kernel = materialize_a13::MaterializeA13<
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            ScalarLeaves,
            ScalarExpr,
            ScalarLayout,
            WgpuRuntime,
        >;

        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let settings = KernelSettings::new(
            CubeDim::new_1d(BLOCK_SIZE).into(),
            ExecutionMode::Unchecked,
            AddressType::U32,
        );
        let mut launcher = KernelLauncher::<WgpuRuntime>::new(settings.clone());
        let handle = exec.client().empty(core::mem::size_of::<u32>());
        let arg = unsafe {
            <[u32] as LaunchArg>::register(BufferArg::from_raw_parts(handle, 1), &mut launcher)
        };
        let kernel = crate::core::launch::kernel_with_max_explicit_storage_bindings!(
            Kernel,
            settings,
            exec.client().clone(),
            arg
        );
        crate::core::launch::assert_binding_budget("materialization", &kernel);
    }

    #[test]
    fn transform_a1_s1_writes_preallocated_slice() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input: DeviceVec<_, u32> = exec.to_device(&[1_u32, 2, 3, 4]);
        let output = exec.to_device(&[99_u32; 6]);

        transform_fixed(
            &exec,
            input.slice_usize(1..4),
            Double,
            output.slice_mut_usize(2..5),
        )
        .unwrap();

        assert_eq!(exec.to_host(&output).unwrap(), vec![99, 99, 4, 6, 8, 99]);
    }

    #[test]
    fn materialize_a3_s3_writes_flat_rows() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let a = exec.to_device(&[1_u32, 2, 3]);
        let b = exec.to_device(&[10_f32, 20.0, 30.0]);
        let c = exec.to_device(&[-1_i32, -2, -3]);
        let out_a = exec.to_device(&[0_u32; 3]);
        let out_b = exec.to_device(&[0_f32; 3]);
        let out_c = exec.to_device(&[0_i32; 3]);

        let input = Zip::new(a.column(), Zip::new(b.column(), c.column()));
        let output = Zip::new(
            Zip::new(out_a.slice_mut(..), out_b.slice_mut(..)),
            out_c.slice_mut(..),
        );
        materialize(&exec, input, output).unwrap();

        assert_eq!(exec.to_host(&out_a).unwrap(), vec![1, 2, 3]);
        assert_eq!(exec.to_host(&out_b).unwrap(), vec![10.0, 20.0, 30.0]);
        assert_eq!(exec.to_host(&out_c).unwrap(), vec![-1, -2, -3]);
    }

    #[test]
    fn materialize_a8_s7_uses_the_flat_evaluator_and_writer() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let columns: Vec<_> = (0_u32..7)
            .map(|column| exec.to_device(&[column * 10 + 1, column * 10 + 2, column * 10 + 3]))
            .collect();
        let outputs: Vec<_> = (0..7).map(|_| exec.to_device(&[0_u32; 3])).collect();
        let seven = Zip::new(
            columns[0].column(),
            Zip::new(
                columns[1].column(),
                Zip::new(
                    columns[2].column(),
                    Zip::new(
                        columns[3].column(),
                        Zip::new(
                            columns[4].column(),
                            Zip::new(columns[5].column(), columns[6].column()),
                        ),
                    ),
                ),
            ),
        );
        let input = Permute::new(seven, Counting::new(0, 3));
        let output = Zip::new(
            Zip::new(
                Zip::new(
                    Zip::new(
                        Zip::new(
                            Zip::new(outputs[0].slice_mut(..), outputs[1].slice_mut(..)),
                            outputs[2].slice_mut(..),
                        ),
                        outputs[3].slice_mut(..),
                    ),
                    outputs[4].slice_mut(..),
                ),
                outputs[5].slice_mut(..),
            ),
            outputs[6].slice_mut(..),
        );

        materialize(&exec, input, output).unwrap();

        for (column, output) in outputs.iter().enumerate() {
            assert_eq!(
                exec.to_host(output).unwrap(),
                vec![
                    column as u32 * 10 + 1,
                    column as u32 * 10 + 2,
                    column as u32 * 10 + 3
                ]
            );
        }
    }
}
