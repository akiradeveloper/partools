#![allow(private_bounds)]

use cubecl::prelude::{CubeType, Runtime};

use crate::api::iter::MStorageExtent;
use crate::core::extent::LogicalExtent;
use crate::op::BinaryPredicateOp;
use crate::{Error, Executor, MAlloc, MIter, MIterMut, MStorage, MVec};

struct MergeOperation<'a, R: Runtime, Left, Right, Less> {
    exec: &'a Executor<R>,
    left: Left,
    right: Right,
    less: Less,
}

impl<R, Item, Left, Right, Less> crate::api::iter::OutputOperation<R, Item>
    for MergeOperation<'_, R, Left, Right, Less>
where
    R: Runtime,
    Item: CubeType + Send + Sync + 'static,
    Left: MIter<R, Item = Item>,
    Right: MIter<R, Item = Item>,
    Less: BinaryPredicateOp<Item>,
{
    type Result = Result<(), Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: crate::api::iter::KernelRow + crate::core::allocation::ScratchStorage<R>,
        Output: crate::api::iter::ConcreteOutput<R, Item>,
    {
        crate::core::merge::merge_direct(
            self.exec,
            crate::api::iter::lower_fixed::<R, _>(self.left),
            crate::api::iter::lower_fixed::<R, _>(self.right),
            self.less,
            output,
        )
    }
}

struct MergeByKeyOperation<'a, R: Runtime, LeftKeys, LeftValues, RightKeys, RightValues, Less> {
    exec: &'a Executor<R>,
    left_keys: LeftKeys,
    left_values: LeftValues,
    right_keys: RightKeys,
    right_values: RightValues,
    less: Less,
}

impl<R, Item, LeftKeys, LeftValues, RightKeys, RightValues, Less>
    crate::api::iter::OutputOperation<R, Item>
    for MergeByKeyOperation<'_, R, LeftKeys, LeftValues, RightKeys, RightValues, Less>
where
    R: Runtime,
    Item: CubeType + Send + Sync + 'static,
    LeftKeys: MIter<R>,
    LeftValues: MIter<R, Item = Item>,
    RightKeys: MIter<R, Item = LeftKeys::Item>,
    RightValues: MIter<R, Item = Item>,
    Less: BinaryPredicateOp<LeftKeys::Item>,
{
    type Result = Result<(), Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: crate::api::iter::KernelRow + crate::core::allocation::ScratchStorage<R>,
        Output: crate::api::iter::ConcreteOutput<R, Item>,
    {
        let left_extent = self
            .left_keys
            .logical_extent()?
            .zipped(&self.left_values.logical_extent()?)?;
        let right_extent = self
            .right_keys
            .logical_extent()?
            .zipped(&self.right_values.logical_extent()?)?;
        let control = crate::core::merge::merge_control_fixed(
            self.exec,
            crate::api::iter::lower_fixed::<R, _>(self.left_keys)
                .with_logical_extent(left_extent.clone()),
            crate::api::iter::lower_fixed::<R, _>(self.right_keys)
                .with_logical_extent(right_extent.clone()),
            self.less,
        )?;
        crate::core::merge::apply_fixed(
            self.exec,
            crate::api::iter::lower_fixed::<R, _>(self.left_values)
                .with_logical_extent(left_extent),
            crate::api::iter::lower_fixed::<R, _>(self.right_values)
                .with_logical_extent(right_extent),
            &control,
            output,
        )
    }
}

/// Stably merges two sorted ranges.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::merge};
///
/// struct Less;
///
/// #[cubecl::cube]
/// impl op::BinaryPredicateOp<u32> for Less {
///     fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
///         massively::flag::from_bool(lhs < rhs)
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let left = exec.to_device(&[1_u32, 3, 5]);
/// let right = exec.to_device(&[2_u32, 4, 6]);
/// let output = merge(&exec, left.slice(..), right.slice(..), Less).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2, 3, 4, 5, 6]);
/// ```
pub fn merge<R, Left, Right, Item, Less>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    less: Less,
) -> Result<MVec<R, Item>, Error>
where
    R: Runtime,
    Left: MIter<R, Item = Item>,
    Right: MIter<R, Item = Item>,
    Item: MAlloc<R>,
    Less: BinaryPredicateOp<Item>,
{
    let left_len = left.capacity()?;
    let right_len = right.capacity()?;
    let len = left_len
        .checked_add(right_len)
        .ok_or(Error::LengthTooLarge {
            len: left_len as usize + right_len as usize,
        })?;
    let extent = LogicalExtent::add(
        exec,
        &left.logical_extent()?,
        &right.logical_extent()?,
        len as usize,
    )?;
    let mut output = exec.alloc::<Item>(len);
    merge_into(exec, left, right, less, output.slice_mut(..))?;
    output.set_logical_extent(extent);
    Ok(output)
}

/// Stably merges two sorted ranges into caller-provided storage.
#[doc(hidden)]
pub(crate) fn merge_into<R, Left, Right, Less, Output>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    less: Less,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Left: MIter<R, Item = Output::Item>,
    Right: MIter<R, Item = Output::Item>,
    Less: BinaryPredicateOp<Left::Item>,
    Output: MIterMut<R>,
{
    output.run_output_operation(MergeOperation {
        exec,
        left,
        right,
        less,
    })
}

/// Stably merges key/value ranges using the ordering of the keys.
///
/// Keys are compared directly and are not materialized into owned storage.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::merge_by_key};
///
/// struct Less;
///
/// #[cubecl::cube]
/// impl op::BinaryPredicateOp<u32> for Less {
///     fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
///         massively::flag::from_bool(lhs < rhs)
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let left_keys = exec.to_device(&[1_u32, 3]);
/// let left_values = exec.to_device(&[10_u32, 30]);
/// let right_keys = exec.to_device(&[2_u32, 4]);
/// let right_values = exec.to_device(&[20_u32, 40]);
/// let output = merge_by_key(
///     &exec,
///     left_keys.slice(..),
///     left_values.slice(..),
///     right_keys.slice(..),
///     right_values.slice(..),
///     Less,
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![10, 20, 30, 40]);
/// ```
pub fn merge_by_key<R, LeftKeys, LeftValues, RightKeys, RightValues, Less>(
    exec: &Executor<R>,
    left_keys: LeftKeys,
    left_values: LeftValues,
    right_keys: RightKeys,
    right_values: RightValues,
    less: Less,
) -> Result<MVec<R, LeftValues::Item>, Error>
where
    R: Runtime,
    LeftKeys: MIter<R>,
    LeftValues: MIter<R>,
    RightKeys: MIter<R, Item = LeftKeys::Item>,
    RightValues: MIter<R, Item = LeftValues::Item>,
    LeftValues::Item: MAlloc<R>,
    Less: BinaryPredicateOp<LeftKeys::Item>,
{
    let left_len = left_keys.capacity()?;
    let left_value_len = left_values.capacity()?;
    if left_len != left_value_len {
        return Err(Error::LengthMismatch {
            left: left_len as usize,
            right: left_value_len as usize,
        });
    }
    let right_len = right_keys.capacity()?;
    let right_value_len = right_values.capacity()?;
    if right_len != right_value_len {
        return Err(Error::LengthMismatch {
            left: right_len as usize,
            right: right_value_len as usize,
        });
    }
    let len = left_len
        .checked_add(right_len)
        .ok_or(Error::LengthTooLarge {
            len: left_len as usize + right_len as usize,
        })?;
    let left_extent = left_keys
        .logical_extent()?
        .zipped(&left_values.logical_extent()?)?;
    let right_extent = right_keys
        .logical_extent()?
        .zipped(&right_values.logical_extent()?)?;
    let extent = LogicalExtent::add(exec, &left_extent, &right_extent, len as usize)?;
    let mut value_output = exec.alloc::<LeftValues::Item>(len);
    merge_by_key_into(
        exec,
        left_keys,
        left_values,
        right_keys,
        right_values,
        less,
        value_output.slice_mut(..),
    )?;
    value_output.set_logical_extent(extent);
    Ok(value_output)
}

/// Stably merges values into caller-provided storage using key-derived control.
#[doc(hidden)]
pub(crate) fn merge_by_key_into<
    R,
    LeftKeys,
    LeftValues,
    RightKeys,
    RightValues,
    Less,
    ValueOutput,
>(
    exec: &Executor<R>,
    left_keys: LeftKeys,
    left_values: LeftValues,
    right_keys: RightKeys,
    right_values: RightValues,
    less: Less,
    value_output: ValueOutput,
) -> Result<(), Error>
where
    R: Runtime,
    LeftKeys: MIter<R>,
    LeftValues: MIter<R, Item = ValueOutput::Item>,
    RightKeys: MIter<R, Item = LeftKeys::Item>,
    RightValues: MIter<R, Item = ValueOutput::Item>,
    Less: BinaryPredicateOp<LeftKeys::Item>,
    ValueOutput: MIterMut<R>,
{
    value_output.run_output_operation(MergeByKeyOperation {
        exec,
        left_keys,
        left_values,
        right_keys,
        right_values,
        less,
    })
}
