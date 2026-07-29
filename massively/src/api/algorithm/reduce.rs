#![allow(private_bounds)]

use cubecl::prelude::Runtime;

use crate::op::ReductionOp;
use crate::{Error, Executor, MAlloc, MIter};

/// Reduces all input items, starting from `init`.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, op, vector::reduce};
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
///
/// let sum = reduce(&exec, input.slice(..), 0_u32, Add).unwrap();
///
/// assert_eq!(sum, 10);
/// ```
pub fn reduce<R, Input, Op>(
    exec: &Executor<R>,
    input: Input,
    init: Input::Item,
    op: Op,
) -> Result<Input::Item, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Input::Item: MAlloc<R>,
    Op: ReductionOp<Input::Item>,
{
    let init = crate::api::value::store(exec, init)?;
    let storage =
        <<Input::Item as MAlloc<R>>::Dispatch as crate::api::iter::ItemDispatch<R>>::reduce(
            exec, input, init, op,
        )?;
    crate::api::value::read::<R, Input::Item>(exec, &storage)
}
