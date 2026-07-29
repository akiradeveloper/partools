use cubecl::prelude::{CubeType, Runtime};

use crate::api::iter::MStorageExtent;
use crate::op::{PredicateOp, UnaryOp};
use crate::{Error, Executor, MAlloc, MFlag, MIndex, MIter, MIterMut, MStorage, MVec};

struct CopyWhereOperation<'a, R: Runtime, Input, Stencil, const REMOVE: bool> {
    exec: &'a Executor<R>,
    input: Input,
    stencil: Stencil,
}

impl<R, Item, Input, Stencil, const REMOVE: bool> crate::api::iter::OutputOperation<R, Item>
    for CopyWhereOperation<'_, R, Input, Stencil, REMOVE>
where
    R: Runtime,
    Item: CubeType + Send + Sync + 'static,
    Input: MIter<R, Item = Item>,
    Stencil: MIter<R, Item = MFlag>,
{
    type Result = Result<crate::DeviceVec<R, u32>, Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: crate::api::iter::KernelRow + crate::core::allocation::ScratchStorage<R>,
        Output: crate::api::iter::ConcreteOutput<R, Item>,
    {
        let input = crate::api::iter::lower::<R, _>(self.input);
        let stencil = crate::api::iter::lower::<R, _>(self.stencil);
        if REMOVE {
            crate::core::selection::remove_where(self.exec, input, stencil, output)
        } else {
            crate::core::selection::copy_where(self.exec, input, stencil, output)
        }
    }
}

struct PartitionOperation<'a, R: Runtime, Input, Pred> {
    exec: &'a Executor<R>,
    input: Input,
    pred: Pred,
}

struct ReplaceWhereOperation<'a, R: Runtime, Item: MAlloc<R>, Stencil> {
    exec: &'a Executor<R>,
    value: MVec<R, Item>,
    stencil: Stencil,
}

impl<R, Item, Stencil> crate::api::iter::OutputOperation<R, Item>
    for ReplaceWhereOperation<'_, R, Item, Stencil>
where
    R: Runtime,
    Item: MAlloc<R>,
    Stencil: MIter<R, Item = MFlag>,
{
    type Result = Result<(), Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: crate::api::iter::KernelRow + crate::core::allocation::ScratchStorage<R>,
        Output: crate::api::iter::ConcreteOutput<R, Item>,
    {
        crate::core::selection::replace_where(
            self.exec,
            crate::api::value::into_scratch::<R, Item>(self.value),
            crate::api::iter::lower::<R, _>(self.stencil),
            output,
        )
    }
}

impl<R, Item, Input, Pred> crate::api::iter::OutputOperation<R, Item>
    for PartitionOperation<'_, R, Input, Pred>
where
    R: Runtime,
    Item: CubeType + Send + Sync + 'static,
    Input: MIter<R, Item = Item>,
    Pred: PredicateOp<Item>,
{
    type Result = Result<crate::DeviceVec<R, u32>, Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: crate::api::iter::KernelRow + crate::core::allocation::ScratchStorage<R>,
        Output: crate::api::iter::ConcreteOutput<R, Item>,
    {
        crate::core::selection::partition(
            self.exec,
            crate::api::iter::lower_fixed::<R, _>(self.input),
            self.pred,
            output,
        )
    }
}

struct TransformWhereOperation<'a, R: Runtime, Input, Op, Stencil> {
    exec: &'a Executor<R>,
    input: Input,
    op: Op,
    stencil: Stencil,
}

impl<R, Item, Input, Op, Stencil> crate::api::iter::OutputOperation<R, Item>
    for TransformWhereOperation<'_, R, Input, Op, Stencil>
where
    R: Runtime,
    Item: CubeType + Send + Sync + 'static,
    Input: MIter<R>,
    Op: UnaryOp<Input::Item, Output = Item>,
    Stencil: MIter<R, Item = MFlag>,
{
    type Result = Result<(), Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: crate::api::iter::KernelRow + crate::core::allocation::ScratchStorage<R>,
        Output: crate::api::iter::ConcreteOutput<R, Item>,
    {
        crate::core::selection::transform_where(
            self.exec,
            crate::api::iter::lower::<R, _>(self.input),
            self.op,
            crate::api::iter::lower::<R, _>(self.stencil),
            output,
        )
    }
}

/// Copies rows whose stencil is true.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, vector::copy_where};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[10_u32, 20, 30, 40]);
/// let stencil = exec.to_device(&[1_u32, 0, 1, 0]);
/// let output = copy_where(&exec, input.slice(..), stencil.slice(..)).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![10, 30]);
/// ```
pub fn copy_where<R, Input, Item, Stencil>(
    exec: &Executor<R>,
    input: Input,
    stencil: Stencil,
) -> Result<MVec<R, Item>, Error>
where
    R: Runtime,
    Input: MIter<R, Item = Item>,
    Item: MAlloc<R>,
    Stencil: MIter<R, Item = MFlag>,
{
    let capacity = input.capacity()?;
    let mut output = exec.alloc::<Item>(capacity);
    let len = copy_where_into(exec, input, stencil, output.slice_mut(..))?;
    output.set_logical_extent(crate::core::extent::LogicalExtent::from_device(
        &len,
        capacity as usize,
    ));
    Ok(output)
}

/// Copies rows whose stencil is true into caller-provided storage.
#[doc(hidden)]
pub(crate) fn copy_where_into<R, Input, Stencil, Output>(
    exec: &Executor<R>,
    input: Input,
    stencil: Stencil,
    output: Output,
) -> Result<MVec<R, MIndex>, Error>
where
    R: Runtime,
    Input: MIter<R, Item = Output::Item>,
    Stencil: MIter<R, Item = MFlag>,
    Output: MIterMut<R>,
{
    output.run_output_operation(CopyWhereOperation::<_, _, _, false> {
        exec,
        input,
        stencil,
    })
}

/// Copies rows whose stencil is false.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, vector::remove_where};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[10_u32, 20, 30, 40]);
/// let stencil = exec.to_device(&[1_u32, 0, 1, 0]);
/// let output = remove_where(&exec, input.slice(..), stencil.slice(..)).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![20, 40]);
/// ```
pub fn remove_where<R, Input, Item, Stencil>(
    exec: &Executor<R>,
    input: Input,
    stencil: Stencil,
) -> Result<MVec<R, Item>, Error>
where
    R: Runtime,
    Input: MIter<R, Item = Item>,
    Item: MAlloc<R>,
    Stencil: MIter<R, Item = MFlag>,
{
    let capacity = input.capacity()?;
    let mut output = exec.alloc::<Item>(capacity);
    let len = remove_where_into(exec, input, stencil, output.slice_mut(..))?;
    output.set_logical_extent(crate::core::extent::LogicalExtent::from_device(
        &len,
        capacity as usize,
    ));
    Ok(output)
}

/// Copies rows whose stencil is false into caller-provided storage.
#[doc(hidden)]
pub(crate) fn remove_where_into<R, Input, Stencil, Output>(
    exec: &Executor<R>,
    input: Input,
    stencil: Stencil,
    output: Output,
) -> Result<MVec<R, MIndex>, Error>
where
    R: Runtime,
    Input: MIter<R, Item = Output::Item>,
    Stencil: MIter<R, Item = MFlag>,
    Output: MIterMut<R>,
{
    output.run_output_operation(CopyWhereOperation::<_, _, _, true> {
        exec,
        input,
        stencil,
    })
}

/// Stably partitions passing items before failing items.
///
/// The return value is the boundary between the two partitions.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, op, vector::partition};
///
/// struct Even;
///
/// #[cubecl::cube]
/// impl op::PredicateOp<u32> for Even {
///     fn apply(value: u32) -> massively::MFlag {
///         massively::flag::from_bool(value % 2 == 0)
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3, 4]);
/// let (output, boundary) = partition(&exec, input.slice(..), Even).unwrap();
///
/// assert_eq!(boundary, 2);
/// assert_eq!(exec.to_host(&output).unwrap(), vec![2, 4, 1, 3]);
/// ```
pub fn partition<R, Input, Item, Pred>(
    exec: &Executor<R>,
    input: Input,
    pred: Pred,
) -> Result<(MVec<R, Item>, MIndex), Error>
where
    R: Runtime,
    Input: MIter<R, Item = Item>,
    Item: MAlloc<R>,
    Pred: PredicateOp<Item>,
{
    let len = input.capacity()?;
    let extent = input.logical_extent()?;
    let mut output = exec.alloc::<Item>(len);
    let boundary = partition_into(exec, input, pred, output.slice_mut(..))?;
    let boundary = crate::api::value::read::<R, MIndex>(exec, &boundary)?;
    output.set_logical_extent(extent);
    Ok((output, boundary))
}

/// Stably partitions into caller-provided storage.
#[doc(hidden)]
pub(crate) fn partition_into<R, Input, Output, Pred>(
    exec: &Executor<R>,
    input: Input,
    pred: Pred,
    output: Output,
) -> Result<MVec<R, MIndex>, Error>
where
    R: Runtime,
    Input: MIter<R, Item = Output::Item>,
    Output: MIterMut<R>,
    Pred: PredicateOp<Input::Item>,
{
    output.run_output_operation(PartitionOperation { exec, input, pred })
}

/// Fills every output item with one value.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, vector::fill};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let output = exec.alloc::<u32>(4);
/// fill(&exec, 7_u32, output.slice_mut(..)).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![7, 7, 7, 7]);
/// ```
pub fn fill<R, Output>(exec: &Executor<R>, value: Output::Item, output: Output) -> Result<(), Error>
where
    R: Runtime,
    Output: MIterMut<R>,
    Output::Item: MAlloc<R>,
{
    let value = crate::api::value::store(exec, value)?;
    fill_value(exec, &value, output)
}

pub(crate) fn fill_value<R, Output>(
    exec: &Executor<R>,
    value: &MVec<R, Output::Item>,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Output: MIterMut<R>,
    Output::Item: MAlloc<R>,
{
    output.run_output_operation(crate::api::iter::FillValueOperation {
        exec,
        value: value.slice(..),
    })
}

/// Replaces output items whose stencil is true.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, vector::replace_where};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let stencil = exec.to_device(&[0_u32, 1, 0, 1]);
/// let output = exec.to_device(&[10_u32, 20, 30, 40]);
/// replace_where(&exec, 99_u32, stencil.slice(..), output.slice_mut(..))
/// .unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![10, 99, 30, 99]);
/// ```
pub fn replace_where<R, Stencil, Output>(
    exec: &Executor<R>,
    value: Output::Item,
    stencil: Stencil,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Stencil: MIter<R, Item = MFlag>,
    Output: MIterMut<R>,
    Output::Item: MAlloc<R>,
{
    let value = crate::api::value::store(exec, value)?;
    replace_where_value(exec, value, stencil, output)
}

fn replace_where_value<R, Stencil, Output>(
    exec: &Executor<R>,
    value: MVec<R, Output::Item>,
    stencil: Stencil,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Stencil: MIter<R, Item = MFlag>,
    Output: MIterMut<R>,
    Output::Item: MAlloc<R>,
{
    output.run_output_operation(ReplaceWhereOperation {
        exec,
        value,
        stencil,
    })
}

/// Applies an operation where the stencil is true.
///
/// A false stencil leaves the corresponding output item unchanged.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, op, vector::transform_where};
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
/// let stencil = exec.to_device(&[1_u32, 0, 1]);
/// let output = exec.to_device(&[100_u32, 100, 100]);
///
/// transform_where(
///     &exec,
///     input.slice(..),
///     AddOne,
///     stencil.slice(..),
///     output.slice_mut(..),
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![2, 100, 4]);
/// ```
pub fn transform_where<R, Input, Stencil, Output, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
    stencil: Stencil,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Stencil: MIter<R, Item = MFlag>,
    Output: MIterMut<R>,
    Op: UnaryOp<Input::Item, Output = Output::Item>,
{
    output.run_output_operation(TransformWhereOperation {
        exec,
        input,
        op,
        stencil,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn copy_where_keeps_physical_storage_private_and_tracks_logical_output() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input = exec.to_device(&[10_u32, 20, 30, 40, 50]);
        let flags = exec.to_device(&[1_u32, 0, 1, 0, 1]);
        let output = copy_where(&exec, input.slice(..), flags.slice(..)).unwrap();
        let bytes = exec.client().read_one(output.handle.clone()).unwrap();

        assert_eq!(bytes.len(), 5 * core::mem::size_of::<u32>());
        assert_eq!(exec.to_host(&output).unwrap(), vec![10, 30, 50]);
    }
}
