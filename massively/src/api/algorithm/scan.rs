#![allow(private_bounds)]

use cubecl::prelude::{CubeType, Runtime};

use crate::api::iter::MStorageExtent;
use crate::op::ReductionOp;
use crate::{Error, Executor, MAlloc, MIter, MIterMut, MStorage, MVec};

struct ScanOperation<'a, R: Runtime, Input, Op, const ADJACENT: bool> {
    exec: &'a Executor<R>,
    input: Input,
    op: Op,
}

struct ExclusiveScanOperation<'a, R: Runtime, Input, Item: MAlloc<R>, Op> {
    exec: &'a Executor<R>,
    input: Input,
    init: MVec<R, Item>,
    op: Op,
}

impl<R, Item, Input, Op> crate::api::iter::OutputOperation<R, Item>
    for ExclusiveScanOperation<'_, R, Input, Item, Op>
where
    R: Runtime,
    Item: MAlloc<R>,
    Input: MIter<R, Item = Item>,
    Op: ReductionOp<Item>,
{
    type Result = Result<(), Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: crate::api::iter::KernelRow + crate::core::allocation::ScratchStorage<R>,
        Output: crate::api::iter::ConcreteOutput<R, Item>,
    {
        crate::core::scan::exclusive_scan(
            self.exec,
            crate::api::iter::lower_fixed::<R, _>(self.input),
            crate::api::value::into_scratch::<R, Item>(self.init),
            self.op,
            output,
        )
    }
}

impl<R, Item, Input, Op, const ADJACENT: bool> crate::api::iter::OutputOperation<R, Item>
    for ScanOperation<'_, R, Input, Op, ADJACENT>
where
    R: Runtime,
    Item: CubeType + Send + Sync + 'static,
    Input: MIter<R, Item = Item>,
    Op: ReductionOp<Item>,
{
    type Result = Result<(), Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: crate::api::iter::KernelRow + crate::core::allocation::ScratchStorage<R>,
        Output: crate::api::iter::ConcreteOutput<R, Item>,
    {
        let input = crate::api::iter::lower_fixed::<R, _>(self.input);
        if ADJACENT {
            crate::core::scan::adjacent_difference(self.exec, input, self.op, output)
        } else {
            crate::core::scan::inclusive_scan(self.exec, input, self.op, output)
        }
    }
}

/// Computes an inclusive scan and returns owned device storage.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, op, vector::inclusive_scan};
///
/// struct Add;
///
/// #[cubecl::cube]
/// impl op::ReductionOp<u32> for Add {
///     fn apply(lhs: u32, rhs: u32) -> u32 {
///         lhs + rhs
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3, 4]);
/// let output = inclusive_scan(&exec, input.slice(..), Add).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![1, 3, 6, 10]);
/// ```
pub fn inclusive_scan<R, Input, Item, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
) -> Result<MVec<R, Item>, Error>
where
    R: Runtime,
    Input: MIter<R, Item = Item>,
    Item: MAlloc<R>,
    Op: ReductionOp<Item>,
{
    let len = input.capacity()?;
    let extent = input.logical_extent()?;
    let mut output = exec.alloc::<Item>(len);
    inclusive_scan_into(exec, input, op, output.slice_mut(..))?;
    output.set_logical_extent(extent);
    Ok(output)
}

/// Computes an inclusive scan into caller-provided storage.
#[doc(hidden)]
pub(crate) fn inclusive_scan_into<R, Input, Output, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R, Item = Output::Item>,
    Output: MIterMut<R>,
    Op: ReductionOp<Input::Item>,
{
    output.run_output_operation(ScanOperation::<_, _, _, false> { exec, input, op })
}

/// Computes adjacent reductions while preserving the first item.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, op, vector::adjacent_difference};
///
/// struct Add;
///
/// #[cubecl::cube]
/// impl op::ReductionOp<u32> for Add {
///     fn apply(lhs: u32, rhs: u32) -> u32 {
///         lhs + rhs
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3, 4]);
/// let output = adjacent_difference(&exec, input.slice(..), Add).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![1, 3, 5, 7]);
/// ```
pub fn adjacent_difference<R, Input, Item, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
) -> Result<MVec<R, Item>, Error>
where
    R: Runtime,
    Input: MIter<R, Item = Item>,
    Item: MAlloc<R>,
    Op: ReductionOp<Item>,
{
    let len = input.capacity()?;
    let extent = input.logical_extent()?;
    let mut output = exec.alloc::<Item>(len);
    adjacent_difference_into(exec, input, op, output.slice_mut(..))?;
    output.set_logical_extent(extent);
    Ok(output)
}

/// Computes adjacent reductions into caller-provided storage.
#[doc(hidden)]
pub(crate) fn adjacent_difference_into<R, Input, Output, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R, Item = Output::Item>,
    Output: MIterMut<R>,
    Op: ReductionOp<Input::Item>,
{
    output.run_output_operation(ScanOperation::<_, _, _, true> { exec, input, op })
}

/// Computes an exclusive scan and returns owned device storage.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, op, vector::exclusive_scan};
///
/// struct Add;
///
/// #[cubecl::cube]
/// impl op::ReductionOp<u32> for Add {
///     fn apply(lhs: u32, rhs: u32) -> u32 {
///         lhs + rhs
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3, 4]);
/// let output = exclusive_scan(&exec, input.slice(..), 0_u32, Add).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![0, 1, 3, 6]);
/// ```
pub fn exclusive_scan<R, Input, Item, Op>(
    exec: &Executor<R>,
    input: Input,
    init: Item,
    op: Op,
) -> Result<MVec<R, Item>, Error>
where
    R: Runtime,
    Input: MIter<R, Item = Item>,
    Item: MAlloc<R>,
    Op: ReductionOp<Item>,
{
    let init = crate::api::value::store(exec, init)?;
    exclusive_scan_value(exec, input, init, op)
}

fn exclusive_scan_value<R, Input, Item, Op>(
    exec: &Executor<R>,
    input: Input,
    init: MVec<R, Item>,
    op: Op,
) -> Result<MVec<R, Item>, Error>
where
    R: Runtime,
    Input: MIter<R, Item = Item>,
    Item: MAlloc<R>,
    Op: ReductionOp<Item>,
{
    let len = input.capacity()?;
    let extent = input.logical_extent()?;
    let mut output = exec.alloc::<Item>(len);
    output
        .slice_mut(..)
        .run_output_operation(ExclusiveScanOperation {
            exec,
            input,
            init,
            op,
        })?;
    output.set_logical_extent(extent);
    Ok(output)
}
