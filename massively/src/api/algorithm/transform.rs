use cubecl::prelude::{CubeType, Runtime};

use crate::api::iter::MStorageExtent;
use crate::op::UnaryOp;
use crate::{Error, Executor, MAlloc, MIter, MIterMut, MStorage, MVec};

struct TransformOperation<'a, R: Runtime, Input, Op> {
    exec: &'a Executor<R>,
    input: Input,
    op: Op,
    active_len: Option<&'a crate::DeviceVec<R, u32>>,
}

impl<R, Item, Input, Op> crate::api::iter::OutputOperation<R, Item>
    for TransformOperation<'_, R, Input, Op>
where
    R: Runtime,
    Item: CubeType + Send + Sync + 'static,
    Input: MIter<R>,
    Op: UnaryOp<Input::Item, Output = Item>,
{
    type Result = Result<(), Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: crate::api::iter::KernelRow + crate::core::allocation::ScratchStorage<R>,
        Output: crate::api::iter::ConcreteOutput<R, Item>,
    {
        let input = crate::api::iter::lower_fixed::<R, _>(self.input);
        match self.active_len {
            Some(active_len) => crate::core::transform::transform_prefix_fixed(
                self.exec, input, self.op, active_len, output,
            ),
            None => crate::core::transform::transform_fixed(self.exec, input, self.op, output),
        }
    }
}

/// Copies every input row into caller-provided storage.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, vector::copy};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3]);
/// let output = exec.alloc::<u32>(3);
///
/// copy(&exec, input.slice(..), output.slice_mut(..)).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2, 3]);
/// ```
pub fn copy<R, Input, Output>(exec: &Executor<R>, input: Input, output: Output) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R, Item = Output::Item>,
    Output: MIterMut<R>,
{
    transform_into(exec, input, crate::op::Identity, output)
}

/// Applies a unary operation and returns its result in owned device storage.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, op, vector::map};
///
/// struct AddOne;
///
/// #[cubecl::cube]
/// impl op::UnaryOp<u32> for AddOne {
///     type Output = u32;
///
///     fn apply(value: u32) -> u32 {
///         value + 1
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3]);
/// let output = map(&exec, input.slice(..), AddOne).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![2, 3, 4]);
/// ```
pub fn map<R, Input, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
) -> Result<MVec<R, Op::Output>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Op: UnaryOp<Input::Item>,
    Op::Output: MAlloc<R>,
{
    let len = input.capacity()?;
    let extent = input.logical_extent()?;
    let mut output = exec.alloc::<Op::Output>(len);
    transform_into(exec, input, op, output.slice_mut(..))?;
    output.set_logical_extent(extent);
    Ok(output)
}

/// Applies a unary operation into caller-provided storage.
#[doc(hidden)]
pub(crate) fn transform_into<R, Input, Output, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Output: MIterMut<R>,
    Op: UnaryOp<Input::Item, Output = Output::Item>,
{
    output.run_output_operation(TransformOperation {
        exec,
        input,
        op,
        active_len: None,
    })
}

/// Applies a unary operation to the prefix selected by a device-resident
/// length.  This is the internal bridge used by bounded algorithm results.
pub(crate) fn transform_prefix_into<R, Input, Output, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
    active_len: &crate::DeviceVec<R, u32>,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Output: MIterMut<R>,
    Op: UnaryOp<Input::Item, Output = Output::Item>,
{
    output.run_output_operation(TransformOperation {
        exec,
        input,
        op,
        active_len: Some(active_len),
    })
}
