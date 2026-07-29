#![allow(private_bounds)]

use cubecl::prelude::*;

use crate::api::iter::MStorageExtent;
use crate::op::BinaryPredicateOp;
use crate::{Error, Executor, MAlloc, MFlag, MIndex, MIter, MIterMut, MStorage, MVec};

struct UniqueOperation<'a, R: Runtime, Input, Equal> {
    exec: &'a Executor<R>,
    input: Input,
    equal: Equal,
}

impl<R, Item, Input, Equal> crate::api::iter::OutputOperation<R, Item>
    for UniqueOperation<'_, R, Input, Equal>
where
    R: Runtime,
    Item: CubeType + Send + Sync + 'static,
    Input: MIter<R, Item = Item>,
    Equal: BinaryPredicateOp<Item>,
{
    type Result = Result<crate::DeviceVec<R, u32>, Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: crate::api::iter::KernelRow + crate::core::allocation::ScratchStorage<R>,
        Output: crate::api::iter::ConcreteOutput<R, Item>,
    {
        crate::core::ordering::unique(
            self.exec,
            crate::api::iter::lower_fixed::<R, _>(self.input),
            self.equal,
            output,
        )
    }
}

/// Stably sorts an input and returns owned device storage.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::sort};
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
/// let input = exec.to_device(&[3_u32, 1, 2]);
/// let output = sort(&exec, input.slice(..), Less).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2, 3]);
/// ```
pub fn sort<R, Input, Item, Less>(
    exec: &Executor<R>,
    input: Input,
    less: Less,
) -> Result<MVec<R, Item>, Error>
where
    R: Runtime,
    Input: MIter<R, Item = Item>,
    Item: MAlloc<R>,
    Less: BinaryPredicateOp<Item>,
{
    <Item::Dispatch as crate::api::iter::ItemDispatch<R>>::sort_owned(exec, input, less)
}

/// Finds the first accepted adjacent pair.
///
/// This operation synchronizes to resolve the optional index on the host.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::adjacent_find};
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
/// let input = exec.to_device(&[1_u32, 2, 2, 3]);
///
/// assert_eq!(
///     adjacent_find(&exec, input.slice(..), Equal)
///         .unwrap(),
///     Some(1)
/// );
/// ```
pub fn adjacent_find<R, Input, Equal>(
    exec: &Executor<R>,
    input: Input,
    equal: Equal,
) -> Result<Option<MIndex>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Equal: BinaryPredicateOp<Input::Item>,
{
    let value = crate::core::ordering::adjacent_find(
        exec,
        crate::api::iter::lower_fixed::<R, _>(input),
        equal,
    )?;
    crate::api::value::read_optional_index(exec, &value)
}

/// Removes consecutive duplicates.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::unique};
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
/// let input = exec.to_device(&[1_u32, 1, 2, 2, 3]);
/// let output = unique(&exec, input.slice(..), Equal).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2, 3]);
/// ```
pub fn unique<R, Input, Item, Equal>(
    exec: &Executor<R>,
    input: Input,
    equal: Equal,
) -> Result<MVec<R, Item>, Error>
where
    R: Runtime,
    Input: MIter<R, Item = Item>,
    Item: MAlloc<R>,
    Equal: BinaryPredicateOp<Item>,
{
    let capacity = input.capacity()?;
    let mut output = exec.alloc::<Item>(capacity);
    let len = unique_into(exec, input, equal, output.slice_mut(..))?;
    output.set_logical_extent(crate::core::extent::LogicalExtent::from_device(
        &len,
        capacity as usize,
    ));
    Ok(output)
}

/// Removes consecutive duplicates into caller-provided storage.
///
/// Returns the number of items written at the beginning of `output`.
#[doc(hidden)]
pub(crate) fn unique_into<R, Input, Output, Equal>(
    exec: &Executor<R>,
    input: Input,
    equal: Equal,
    output: Output,
) -> Result<MVec<R, MIndex>, Error>
where
    R: Runtime,
    Input: MIter<R, Item = Output::Item>,
    Output: MIterMut<R>,
    Equal: BinaryPredicateOp<Input::Item>,
{
    output.run_output_operation(UniqueOperation { exec, input, equal })
}

/// Returns the first index at which the input ceases to be sorted.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::is_sorted_until};
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
/// let input = exec.to_device(&[1_u32, 2, 4, 3, 5]);
///
/// assert_eq!(
///     is_sorted_until(&exec, input.slice(..), Less)
///         .unwrap(),
///     3
/// );
/// ```
pub fn is_sorted_until<R, Input, Less>(
    exec: &Executor<R>,
    input: Input,
    less: Less,
) -> Result<MIndex, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Less: BinaryPredicateOp<Input::Item>,
{
    let len = input.capacity()?;
    let index = crate::core::ordering::is_sorted_until(
        exec,
        crate::api::iter::lower_fixed::<R, _>(input),
        less,
    )?;
    let index = crate::api::value::read::<R, MIndex>(exec, &index)?;
    Ok(if index == MIndex::MAX { len } else { index })
}

/// Returns whether the input is sorted.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::is_sorted};
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
/// let input = exec.to_device(&[1_u32, 2, 3, 4]);
///
/// assert!(massively::flag::is_set(
///     is_sorted(&exec, input.slice(..), Less)
///         .unwrap()
/// ));
/// ```
pub fn is_sorted<R, Input, Less>(
    exec: &Executor<R>,
    input: Input,
    less: Less,
) -> Result<MFlag, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Less: BinaryPredicateOp<Input::Item>,
{
    let index =
        crate::core::ordering::is_sorted(exec, crate::api::iter::lower_fixed::<R, _>(input), less)?;
    let index = crate::api::value::read::<R, MIndex>(exec, &index)?;
    Ok(crate::flag::from_bool(index == MIndex::MAX))
}

macro_rules! extremum_api {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        pub fn $name<R, Input, Less>(
            exec: &Executor<R>,
            input: Input,
            less: Less,
        ) -> Result<Option<MIndex>, Error>
        where
            R: Runtime,
            Input: MIter<R>,
            Less: BinaryPredicateOp<Input::Item>,
        {
            let value = crate::core::ordering::$name(
                exec,
                crate::api::iter::lower_fixed::<R, _>(input),
                less,
            )?;
            crate::api::value::read_optional_index(exec, &value)
        }
    };
}

extremum_api!(
    min_element,
    r#"Returns the first minimum element index.

This operation synchronizes to resolve the optional index on the host.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{op, Executor, vector::min_element};

struct Less;

#[cubecl::cube]
impl op::BinaryPredicateOp<u32> for Less {
    fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
        massively::flag::from_bool(lhs < rhs)
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[3_u32, 1, 2, 1]);

assert_eq!(
    min_element(&exec, input.slice(..), Less)
        .unwrap(),
    Some(1)
);
```
"#
);
extremum_api!(
    max_element,
    r#"Returns the first maximum element index.

This operation synchronizes to resolve the optional index on the host.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{op, Executor, vector::max_element};

struct Less;

#[cubecl::cube]
impl op::BinaryPredicateOp<u32> for Less {
    fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
        massively::flag::from_bool(lhs < rhs)
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[3_u32, 1, 3, 2]);

assert_eq!(
    max_element(&exec, input.slice(..), Less)
        .unwrap(),
    Some(0)
);
```
"#
);
macro_rules! minmax_api {
    ($doc:literal) => {
        #[doc = $doc]
        pub fn minmax_element<R, Input, Less>(
            exec: &Executor<R>,
            input: Input,
            less: Less,
        ) -> Result<Option<(MIndex, MIndex)>, Error>
        where
            R: Runtime,
            Input: MIter<R>,
            Less: BinaryPredicateOp<Input::Item>,
        {
            let (minimum, maximum) = crate::core::ordering::minmax_element(
                exec,
                crate::api::iter::lower_fixed::<R, _>(input),
                less,
            )?;
            let pair: MVec<R, (MIndex, MIndex)> = crate::core::iter::Zip::new(minimum, maximum);
            crate::api::value::read_optional_index_pair(exec, &pair)
        }
    };
}

minmax_api!(
    r#"Returns the minimum and maximum element indices.

This operation synchronizes to resolve the optional indices on the host.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{op, Executor, vector::minmax_element};

struct Less;

#[cubecl::cube]
impl op::BinaryPredicateOp<u32> for Less {
    fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
        massively::flag::from_bool(lhs < rhs)
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[3_u32, 1, 4, 2]);

assert_eq!(
    minmax_element(&exec, input.slice(..), Less)
        .unwrap(),
    Some((1, 2))
);
```
"#
);
