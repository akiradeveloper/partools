#![allow(private_bounds)]

use cubecl::prelude::{CubeType, Runtime};

use crate::api::iter::MStorageExtent;
use crate::op::{BinaryPredicateOp, ReductionOp};
use crate::{Error, Executor, MAlloc, MIndex, MIter, MIterMut, MRadix, MStorage, MVec};

struct ByKeyScanOperation<'a, R: Runtime, Keys, Values, Equal, Item: MAlloc<R>, Op> {
    exec: &'a Executor<R>,
    keys: Keys,
    values: Values,
    equal: Equal,
    init: Option<MVec<R, Item>>,
    op: Op,
}

impl<R, Item, Keys, Values, Equal, Op> crate::api::iter::OutputOperation<R, Item>
    for ByKeyScanOperation<'_, R, Keys, Values, Equal, Item, Op>
where
    R: Runtime,
    Item: MAlloc<R>,
    Keys: MIter<R>,
    Values: MIter<R, Item = Item>,
    Equal: BinaryPredicateOp<Keys::Item>,
    Op: ReductionOp<Item>,
{
    type Result = Result<(), Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: crate::api::iter::KernelRow + crate::core::allocation::ScratchStorage<R>,
        Output: crate::api::iter::ConcreteOutput<R, Item>,
    {
        let extent = self
            .keys
            .logical_extent()?
            .zipped(&self.values.logical_extent()?)?;
        let keys =
            crate::api::iter::lower_fixed::<R, _>(self.keys).with_logical_extent(extent.clone());
        let values = crate::api::iter::lower_fixed::<R, _>(self.values).with_logical_extent(extent);
        if let Some(init) = self.init {
            crate::core::by_key::exclusive_scan_by_key_lowered(
                self.exec,
                keys,
                values,
                self.equal,
                crate::api::value::into_scratch::<R, Item>(init),
                self.op,
                output,
            )
        } else {
            crate::core::by_key::inclusive_scan_by_key_lowered(
                self.exec, keys, values, self.equal, self.op, output,
            )
        }
    }
}

struct ReduceByKeyValueOperation<
    'a,
    R: Runtime,
    Keys,
    Values,
    Equal,
    ValueItem: MAlloc<R>,
    Op,
    KeyOutput,
> {
    exec: &'a Executor<R>,
    keys: Keys,
    values: Values,
    equal: Equal,
    init: MVec<R, ValueItem>,
    op: Op,
    key_output: KeyOutput,
}

struct ReduceByKeyKeyOperation<
    'a,
    R: Runtime,
    Keys,
    Values,
    Equal,
    ValueItem: MAlloc<R>,
    Op,
    ValueOutput,
> {
    exec: &'a Executor<R>,
    keys: Keys,
    values: Values,
    equal: Equal,
    init: MVec<R, ValueItem>,
    op: Op,
    value_output: ValueOutput,
}

impl<R, KeyItem, ValueItem, Keys, Values, Equal, Op, ValueOutput>
    crate::api::iter::OutputOperation<R, KeyItem>
    for ReduceByKeyKeyOperation<'_, R, Keys, Values, Equal, ValueItem, Op, ValueOutput>
where
    R: Runtime,
    KeyItem: CubeType + Send + Sync + 'static,
    ValueItem: MAlloc<R> + crate::api::iter::KernelRow + crate::core::allocation::ScratchStorage<R>,
    Keys: MIter<R, Item = KeyItem>,
    Values: MIter<R, Item = ValueItem>,
    Equal: BinaryPredicateOp<KeyItem>,
    Op: ReductionOp<ValueItem>,
    ValueOutput: crate::api::iter::ConcreteOutput<R, ValueItem>,
{
    type Result = Result<crate::DeviceVec<R, u32>, Error>;

    fn run<KeyOutput>(self, key_output: KeyOutput) -> Self::Result
    where
        KeyItem: crate::api::iter::KernelRow + crate::core::allocation::ScratchStorage<R>,
        KeyOutput: crate::api::iter::ConcreteOutput<R, KeyItem>,
    {
        let extent = self
            .keys
            .logical_extent()?
            .zipped(&self.values.logical_extent()?)?;
        crate::core::by_key::reduce_by_key_lowered(
            self.exec,
            crate::api::iter::lower_fixed::<R, _>(self.keys).with_logical_extent(extent.clone()),
            crate::api::iter::lower_fixed::<R, _>(self.values).with_logical_extent(extent),
            self.equal,
            crate::api::value::into_scratch::<R, ValueItem>(self.init),
            self.op,
            key_output,
            self.value_output,
        )
    }
}

impl<R, KeyItem, ValueItem, Keys, Values, Equal, Op, KeyOutput>
    crate::api::iter::OutputOperation<R, ValueItem>
    for ReduceByKeyValueOperation<'_, R, Keys, Values, Equal, ValueItem, Op, KeyOutput>
where
    R: Runtime,
    KeyItem: CubeType + Send + Sync + 'static,
    ValueItem: MAlloc<R>,
    Keys: MIter<R, Item = KeyItem>,
    Values: MIter<R, Item = ValueItem>,
    Equal: BinaryPredicateOp<KeyItem>,
    Op: ReductionOp<ValueItem>,
    KeyOutput: MIterMut<R, Item = KeyItem>,
{
    type Result = Result<crate::DeviceVec<R, u32>, Error>;

    fn run<ValueOutput>(self, value_output: ValueOutput) -> Self::Result
    where
        ValueItem: crate::api::iter::KernelRow + crate::core::allocation::ScratchStorage<R>,
        ValueOutput: crate::api::iter::ConcreteOutput<R, ValueItem>,
    {
        self.key_output
            .run_output_operation(ReduceByKeyKeyOperation {
                exec: self.exec,
                keys: self.keys,
                values: self.values,
                equal: self.equal,
                init: self.init,
                op: self.op,
                value_output,
            })
    }
}

struct UniqueByKeyOperation<'a, R: Runtime, Keys, Values, Equal> {
    exec: &'a Executor<R>,
    keys: Keys,
    values: Values,
    equal: Equal,
}

impl<R, Item, Keys, Values, Equal> crate::api::iter::OutputOperation<R, Item>
    for UniqueByKeyOperation<'_, R, Keys, Values, Equal>
where
    R: Runtime,
    Item: CubeType + Send + Sync + 'static,
    Keys: MIter<R>,
    Values: MIter<R, Item = Item>,
    Equal: BinaryPredicateOp<Keys::Item>,
{
    type Result = Result<crate::DeviceVec<R, u32>, Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: crate::api::iter::KernelRow + crate::core::allocation::ScratchStorage<R>,
        Output: crate::api::iter::ConcreteOutput<R, Item>,
    {
        let extent = self
            .keys
            .logical_extent()?
            .zipped(&self.values.logical_extent()?)?;
        crate::core::by_key::unique_by_key(
            self.exec,
            crate::api::iter::lower_fixed::<R, _>(self.keys).with_logical_extent(extent.clone()),
            crate::api::iter::lower_fixed::<R, _>(self.values).with_logical_extent(extent),
            self.equal,
            output,
        )
    }
}

/// Stably sorts keys and applies the same ordering to values.
///
/// Keys are compared directly and are not materialized into owned storage.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::sort_by_key};
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
/// let keys = exec.to_device(&[3_u32, 1, 2]);
/// let values = exec.to_device(&[30_u32, 10, 20]);
/// let output = sort_by_key(
///     &exec,
///     keys.slice(..),
///     values.slice(..),
///     Less,
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![10, 20, 30]);
/// ```
pub fn sort_by_key<R, Keys, Values, Less>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    less: Less,
) -> Result<MVec<R, Values::Item>, Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Values: MIter<R>,
    Values::Item: MAlloc<R>,
    Less: BinaryPredicateOp<Keys::Item>,
{
    let len = keys.capacity()?;
    let value_len = values.capacity()?;
    if len != value_len {
        return Err(Error::LengthMismatch {
            left: len as usize,
            right: value_len as usize,
        });
    }
    let extent = keys.logical_extent()?.zipped(&values.logical_extent()?)?;
    let mut value_output = exec.alloc::<Values::Item>(len);
    sort_values_by_key_into(exec, keys, values, less, value_output.slice_mut(..))?;
    value_output.set_logical_extent(extent);
    Ok(value_output)
}

/// Stably radix-sorts keys in ascending order and applies the same ordering to values.
///
/// Integer keys use their numeric order. Floating-point keys use the same total
/// ordering as `f32::total_cmp` and `f64::total_cmp`. Keys may contain up to
/// three columns; compound keys produced by `zip2` or `zip3` are ordered
/// lexicographically from left to right.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, vector::radix_sort_by_key};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let keys = exec.to_device(&[3_i32, -1, 2, -1]);
/// let values = exec.to_device(&[30_u32, 10, 20, 11]);
/// let output = radix_sort_by_key(&exec, keys.slice(..), values.slice(..)).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![10, 11, 20, 30]);
/// ```
pub fn radix_sort_by_key<R, Keys, Values>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
) -> Result<MVec<R, Values::Item>, Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Keys::Item: MRadix<R>,
    Values: MIter<R>,
    Values::Item: MAlloc<R>,
{
    let len = keys.capacity()?;
    let value_len = values.capacity()?;
    if len != value_len {
        return Err(Error::LengthMismatch {
            left: len as usize,
            right: value_len as usize,
        });
    }
    let extent = keys.logical_extent()?.zipped(&values.logical_extent()?)?;
    let mut value_output = exec.alloc::<Values::Item>(len);
    radix_sort_values_by_key_into(exec, keys, values, value_output.slice_mut(..))?;
    value_output.set_logical_extent(extent);
    Ok(value_output)
}

/// Stably radix-sorts values into caller-provided storage.
#[doc(hidden)]
pub(crate) fn radix_sort_values_by_key_into<R, Keys, Values, ValueOutput>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    value_output: ValueOutput,
) -> Result<(), Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Keys::Item: MRadix<R>,
    Values: MIter<R, Item = ValueOutput::Item>,
    ValueOutput: MIterMut<R>,
{
    let key_len = keys.capacity()?;
    let value_len = values.capacity()?;
    if key_len != value_len {
        return Err(Error::LengthMismatch {
            left: key_len as usize,
            right: value_len as usize,
        });
    }
    let extent = keys.logical_extent()?.zipped(&values.logical_extent()?)?;
    let mut key_storage = crate::api::algorithm::transform::map(exec, keys, crate::op::Identity)?;
    key_storage.set_logical_extent(extent);
    let permutation =
        <Keys::Item as MRadix<R>>::radix_permutation(exec, &key_storage, key_len as usize)?;
    crate::api::algorithm::indexed::apply_permutation_into(
        exec,
        crate::api::iter::lower_fixed::<R, _>(values),
        permutation.column(),
        value_output,
    )
}

/// Stably sorts values using a key-derived permutation without materializing keys.
#[doc(hidden)]
pub(crate) fn sort_values_by_key_into<R, Keys, Values, Less, ValueOutput>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    less: Less,
    value_output: ValueOutput,
) -> Result<(), Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Values: MIter<R, Item = ValueOutput::Item>,
    Less: BinaryPredicateOp<Keys::Item>,
    ValueOutput: MIterMut<R>,
{
    let key_len = keys.capacity()?;
    let value_len = values.capacity()?;
    if key_len != value_len {
        return Err(Error::LengthMismatch {
            left: key_len as usize,
            right: value_len as usize,
        });
    }
    let extent = keys.logical_extent()?.zipped(&values.logical_extent()?)?;
    let permutation = crate::core::ordering::sort_control_with(
        exec,
        crate::api::iter::lower_fixed::<R, _>(keys).with_logical_extent(extent),
        less,
    )?;
    crate::api::algorithm::indexed::apply_permutation_into(
        exec,
        crate::api::iter::lower_fixed::<R, _>(values),
        permutation.column(),
        value_output,
    )
}

/// Computes an inclusive scan within each adjacent equal-key segment.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::inclusive_scan_by_key};
///
/// struct Equal;
/// struct Add;
///
/// #[cubecl::cube]
/// impl op::BinaryPredicateOp<u32> for Equal {
///     fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
///         massively::flag::from_bool(lhs == rhs)
///     }
/// }
///
/// #[cubecl::cube]
/// impl op::ReductionOp<u32> for Add {
///     fn apply(lhs: u32, rhs: u32) -> u32 {
///         lhs + rhs
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let keys = exec.to_device(&[1_u32, 1, 2, 2]);
/// let values = exec.to_device(&[10_u32, 20, 30, 40]);
/// let output = inclusive_scan_by_key(
///     &exec,
///     keys.slice(..),
///     values.slice(..),
///     Equal,
///     Add,
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![10, 30, 30, 70]);
/// ```
pub fn inclusive_scan_by_key<R, Keys, Values, Equal, Op>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    equal: Equal,
    op: Op,
) -> Result<MVec<R, Values::Item>, Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Values: MIter<R>,
    Values::Item: MAlloc<R>,
    Equal: BinaryPredicateOp<Keys::Item>,
    Op: ReductionOp<Values::Item>,
{
    let len = keys.capacity()?;
    let value_len = values.capacity()?;
    if len != value_len {
        return Err(Error::LengthMismatch {
            left: len as usize,
            right: value_len as usize,
        });
    }
    let extent = keys.logical_extent()?.zipped(&values.logical_extent()?)?;
    let mut output = exec.alloc::<Values::Item>(len);
    inclusive_scan_by_key_into(exec, keys, values, equal, op, output.slice_mut(..))?;
    output.set_logical_extent(extent);
    Ok(output)
}

/// Computes an inclusive scan by key into caller-provided storage.
#[doc(hidden)]
pub(crate) fn inclusive_scan_by_key_into<R, Keys, Values, Item, Equal, Op, Output>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    equal: Equal,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Values: MIter<R, Item = Item>,
    Item: MAlloc<R>,
    Equal: BinaryPredicateOp<Keys::Item>,
    Op: ReductionOp<Item>,
    Output: MIterMut<R, Item = Item>,
{
    output.run_output_operation(ByKeyScanOperation {
        exec,
        keys,
        values,
        equal,
        init: None,
        op,
    })
}

/// Computes an exclusive scan within each adjacent equal-key segment.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::exclusive_scan_by_key};
///
/// struct Equal;
/// struct Add;
///
/// #[cubecl::cube]
/// impl op::BinaryPredicateOp<u32> for Equal {
///     fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
///         massively::flag::from_bool(lhs == rhs)
///     }
/// }
///
/// #[cubecl::cube]
/// impl op::ReductionOp<u32> for Add {
///     fn apply(lhs: u32, rhs: u32) -> u32 {
///         lhs + rhs
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let keys = exec.to_device(&[1_u32, 1, 2, 2]);
/// let values = exec.to_device(&[10_u32, 20, 30, 40]);
/// let output = exclusive_scan_by_key(
///     &exec,
///     keys.slice(..),
///     values.slice(..),
///     Equal,
///     0_u32,
///     Add,
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![0, 10, 0, 30]);
/// ```
pub fn exclusive_scan_by_key<R, Keys, Values, Equal, Op>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    equal: Equal,
    init: Values::Item,
    op: Op,
) -> Result<MVec<R, Values::Item>, Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Values: MIter<R>,
    Values::Item: MAlloc<R>,
    Equal: BinaryPredicateOp<Keys::Item>,
    Op: ReductionOp<Values::Item>,
{
    let init = crate::api::value::store(exec, init)?;
    exclusive_scan_by_key_value(exec, keys, values, equal, init, op)
}

fn exclusive_scan_by_key_value<R, Keys, Values, Equal, Op>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    equal: Equal,
    init: MVec<R, Values::Item>,
    op: Op,
) -> Result<MVec<R, Values::Item>, Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Values: MIter<R>,
    Values::Item: MAlloc<R>,
    Equal: BinaryPredicateOp<Keys::Item>,
    Op: ReductionOp<Values::Item>,
{
    let len = keys.capacity()?;
    let value_len = values.capacity()?;
    if len != value_len {
        return Err(Error::LengthMismatch {
            left: len as usize,
            right: value_len as usize,
        });
    }
    let extent = keys.logical_extent()?.zipped(&values.logical_extent()?)?;
    let mut output = exec.alloc::<Values::Item>(len);
    exclusive_scan_by_key_into(exec, keys, values, equal, init, op, output.slice_mut(..))?;
    output.set_logical_extent(extent);
    Ok(output)
}

/// Computes an exclusive scan by key into caller-provided storage.
#[doc(hidden)]
pub(crate) fn exclusive_scan_by_key_into<R, Keys, Values, Item, Equal, Op, Output>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    equal: Equal,
    init: MVec<R, Values::Item>,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Values: MIter<R, Item = Item>,
    Item: MAlloc<R>,
    Equal: BinaryPredicateOp<Keys::Item>,
    Op: ReductionOp<Item>,
    Output: MIterMut<R, Item = Item>,
{
    output.run_output_operation(ByKeyScanOperation {
        exec,
        keys,
        values,
        equal,
        init: Some(init),
        op,
    })
}

/// Reduces each adjacent equal-key segment.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::reduce_by_key};
///
/// struct Equal;
/// struct Add;
///
/// #[cubecl::cube]
/// impl op::BinaryPredicateOp<u32> for Equal {
///     fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
///         massively::flag::from_bool(lhs == rhs)
///     }
/// }
///
/// #[cubecl::cube]
/// impl op::ReductionOp<u32> for Add {
///     fn apply(lhs: u32, rhs: u32) -> u32 {
///         lhs + rhs
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let keys = exec.to_device(&[1_u32, 1, 2, 2]);
/// let values = exec.to_device(&[10_u32, 20, 30, 40]);
/// let (key_output, value_output) = reduce_by_key(
///     &exec,
///     keys.slice(..),
///     values.slice(..),
///     Equal,
///     0_u32,
///     Add,
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&key_output).unwrap(), vec![1, 2]);
/// assert_eq!(exec.to_host(&value_output).unwrap(), vec![30, 70]);
/// ```
pub fn reduce_by_key<R, Keys, Values, KeyItem, ValueItem, Equal, Op>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    equal: Equal,
    init: ValueItem,
    op: Op,
) -> Result<(MVec<R, KeyItem>, MVec<R, ValueItem>), Error>
where
    R: Runtime,
    Keys: MIter<R, Item = KeyItem>,
    KeyItem: MAlloc<R>,
    Values: MIter<R, Item = ValueItem>,
    ValueItem: MAlloc<R>,
    Equal: BinaryPredicateOp<KeyItem>,
    Op: ReductionOp<ValueItem>,
{
    let init = crate::api::value::store(exec, init)?;
    reduce_by_key_value(exec, keys, values, equal, init, op)
}

fn reduce_by_key_value<R, Keys, Values, KeyItem, ValueItem, Equal, Op>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    equal: Equal,
    init: MVec<R, ValueItem>,
    op: Op,
) -> Result<(MVec<R, KeyItem>, MVec<R, ValueItem>), Error>
where
    R: Runtime,
    Keys: MIter<R, Item = KeyItem>,
    KeyItem: MAlloc<R>,
    Values: MIter<R, Item = ValueItem>,
    ValueItem: MAlloc<R>,
    Equal: BinaryPredicateOp<KeyItem>,
    Op: ReductionOp<ValueItem>,
{
    let capacity = keys.capacity()?;
    let value_len = values.capacity()?;
    if capacity != value_len {
        return Err(Error::LengthMismatch {
            left: capacity as usize,
            right: value_len as usize,
        });
    }
    let mut key_output = exec.alloc::<KeyItem>(capacity);
    let mut value_output = exec.alloc::<ValueItem>(capacity);
    let len = reduce_by_key_into(
        exec,
        keys,
        values,
        equal,
        init,
        op,
        key_output.slice_mut(..),
        value_output.slice_mut(..),
    )?;
    let extent = crate::core::extent::LogicalExtent::from_device(&len, capacity as usize);
    key_output.set_logical_extent(extent.clone());
    value_output.set_logical_extent(extent);
    Ok((key_output, value_output))
}

/// Reduces by key into caller-provided storage.
#[doc(hidden)]
pub(crate) fn reduce_by_key_into<
    R,
    Keys,
    Values,
    KeyItem,
    ValueItem,
    Equal,
    Op,
    KeyOutput,
    ValueOutput,
>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    equal: Equal,
    init: MVec<R, ValueItem>,
    op: Op,
    key_output: KeyOutput,
    value_output: ValueOutput,
) -> Result<MVec<R, MIndex>, Error>
where
    R: Runtime,
    Keys: MIter<R, Item = KeyItem>,
    KeyItem: CubeType + Send + Sync + 'static,
    Values: MIter<R, Item = ValueItem>,
    ValueItem: MAlloc<R>,
    Equal: BinaryPredicateOp<KeyItem>,
    Op: ReductionOp<ValueItem>,
    KeyOutput: MIterMut<R, Item = KeyItem>,
    ValueOutput: MIterMut<R, Item = ValueItem>,
{
    value_output.run_output_operation(ReduceByKeyValueOperation {
        exec,
        keys,
        values,
        equal,
        init,
        op,
        key_output,
    })
}

/// Keeps the first value of every adjacent equal-key run.
///
/// Keys are compared directly and are not materialized into owned storage.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::unique_by_key};
///
/// struct Equal;
///
/// #[cubecl::cube]
/// impl op::BinaryPredicateOp<u32> for Equal {
///     fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
///         massively::flag::from_bool(lhs == rhs)
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let keys = exec.to_device(&[1_u32, 1, 2, 2, 3]);
/// let values = exec.to_device(&[10_u32, 11, 20, 21, 30]);
/// let output = unique_by_key(
///     &exec,
///     keys.slice(..),
///     values.slice(..),
///     Equal,
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![10, 20, 30]);
/// ```
pub fn unique_by_key<R, Keys, Values, Equal>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    equal: Equal,
) -> Result<MVec<R, Values::Item>, Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Values: MIter<R>,
    Values::Item: MAlloc<R>,
    Equal: BinaryPredicateOp<Keys::Item>,
{
    let capacity = keys.capacity()?;
    let value_len = values.capacity()?;
    if capacity != value_len {
        return Err(Error::LengthMismatch {
            left: capacity as usize,
            right: value_len as usize,
        });
    }
    let mut value_output = exec.alloc::<Values::Item>(capacity);
    let len = unique_by_key_into(exec, keys, values, equal, value_output.slice_mut(..))?;
    value_output.set_logical_extent(crate::core::extent::LogicalExtent::from_device(
        &len,
        capacity as usize,
    ));
    Ok(value_output)
}

/// Keeps the first value of each unique key run in caller-provided storage.
#[doc(hidden)]
pub(crate) fn unique_by_key_into<R, Keys, Values, Equal, ValueOutput>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    equal: Equal,
    value_output: ValueOutput,
) -> Result<MVec<R, MIndex>, Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Values: MIter<R, Item = ValueOutput::Item>,
    Equal: BinaryPredicateOp<Keys::Item>,
    ValueOutput: MIterMut<R>,
{
    value_output.run_output_operation(UniqueByKeyOperation {
        exec,
        keys,
        values,
        equal,
    })
}
