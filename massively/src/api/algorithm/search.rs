use cubecl::prelude::*;

use crate::op::BinaryPredicateOp;
use crate::{Error, Executor, MFlag, MIndex, MIter, MVec};

/// Finds the first source item equal to any needle.
///
/// This operation synchronizes to resolve the optional index on the host.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::find_first_of};
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
/// let source = exec.to_device(&[1_u32, 2, 3, 4]);
/// let needles = exec.to_device(&[7_u32, 3]);
///
/// let index = find_first_of(&exec, source.slice(..), needles.slice(..), Equal).unwrap();
///
/// assert_eq!(index, Some(2));
/// ```
pub fn find_first_of<R, Source, Needles, Equal>(
    exec: &Executor<R>,
    source: Source,
    needles: Needles,
    equal: Equal,
) -> Result<Option<MIndex>, Error>
where
    R: Runtime,
    Source: MIter<R>,
    Needles: MIter<R, Item = Source::Item>,
    Equal: BinaryPredicateOp<Source::Item>,
{
    let value = crate::core::search::find_first_of(
        exec,
        crate::api::iter::lower_fixed::<R, _>(source),
        crate::api::iter::lower_fixed::<R, _>(needles),
        equal,
    )?;
    crate::api::value::read_optional_index(exec, &value)
}

/// Finds the lower bound of each value.
///
/// `source` must be sorted according to `less`.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::lower_bound};
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
/// let source = exec.to_device(&[1_u32, 3, 5, 7]);
/// let values = exec.to_device(&[0_u32, 3, 6, 8]);
/// let output = lower_bound(&exec, source.slice(..), values.slice(..), Less).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![0, 1, 3, 4]);
/// ```
pub fn lower_bound<R, Source, Values, Less>(
    exec: &Executor<R>,
    source: Source,
    values: Values,
    less: Less,
) -> Result<MVec<R, MIndex>, Error>
where
    R: Runtime,
    Source: MIter<R>,
    Values: MIter<R, Item = Source::Item>,
    Less: BinaryPredicateOp<Source::Item>,
{
    crate::core::search::lower_bounds_storage(
        exec,
        crate::api::iter::lower_fixed::<R, _>(source),
        crate::api::iter::lower_fixed::<R, _>(values),
        less,
    )
}

/// Finds the upper bound of each value.
///
/// `source` must be sorted according to `less`.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::upper_bound};
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
/// let source = exec.to_device(&[1_u32, 3, 5, 7]);
/// let values = exec.to_device(&[0_u32, 3, 6, 8]);
/// let output = upper_bound(&exec, source.slice(..), values.slice(..), Less).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![0, 2, 3, 4]);
/// ```
pub fn upper_bound<R, Source, Values, Less>(
    exec: &Executor<R>,
    source: Source,
    values: Values,
    less: Less,
) -> Result<MVec<R, MIndex>, Error>
where
    R: Runtime,
    Source: MIter<R>,
    Values: MIter<R, Item = Source::Item>,
    Less: BinaryPredicateOp<Source::Item>,
{
    crate::core::search::upper_bounds_storage(
        exec,
        crate::api::iter::lower_fixed::<R, _>(source),
        crate::api::iter::lower_fixed::<R, _>(values),
        less,
    )
}

/// Returns whether two ranges contain equal items.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::equal};
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
/// let left = exec.to_device(&[1_u32, 2, 3]);
/// let right = exec.to_device(&[1_u32, 2, 3]);
///
/// assert!(massively::flag::is_set(
///     equal(&exec, left.slice(..), right.slice(..), Equal)
///         .unwrap()
/// ));
/// ```
pub fn equal<R, Left, Right, Equal>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    equal: Equal,
) -> Result<MFlag, Error>
where
    R: Runtime,
    Left: MIter<R>,
    Right: MIter<R, Item = Left::Item>,
    Equal: BinaryPredicateOp<Left::Item>,
{
    let left_len = left.capacity()?;
    let right_len = right.capacity()?;
    if left_len != right_len {
        return Ok(crate::flag::from_bool(false));
    }
    let mismatch = crate::core::search::equal(
        exec,
        crate::api::iter::lower_fixed::<R, _>(left),
        crate::api::iter::lower_fixed::<R, _>(right),
        equal,
    )?;
    let mismatch = crate::api::value::read::<R, MIndex>(exec, &mismatch)?;
    Ok(crate::flag::from_bool(mismatch >= left_len))
}

/// Returns the first mismatch.
///
/// This operation synchronizes to resolve the optional index on the host.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::mismatch};
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
/// let left = exec.to_device(&[1_u32, 2, 3]);
/// let right = exec.to_device(&[1_u32, 4, 3]);
///
/// assert_eq!(
///     mismatch(&exec, left.slice(..), right.slice(..), Equal).unwrap(),
///     Some(1)
/// );
/// ```
pub fn mismatch<R, Left, Right, Equal>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    equal: Equal,
) -> Result<Option<MIndex>, Error>
where
    R: Runtime,
    Left: MIter<R>,
    Right: MIter<R, Item = Left::Item>,
    Equal: BinaryPredicateOp<Left::Item>,
{
    let left_len = left.capacity()?;
    let right_len = right.capacity()?;
    let shared_len = left_len.min(right_len);
    let index = crate::core::search::mismatch(
        exec,
        crate::api::iter::lower_fixed::<R, _>(left),
        crate::api::iter::lower_fixed::<R, _>(right),
        equal,
    )?;
    let index = crate::api::value::read::<R, MIndex>(exec, &index)?;
    if index < shared_len {
        Ok(Some(index))
    } else if left_len == right_len {
        Ok(None)
    } else {
        Ok(Some(shared_len))
    }
}

/// Lexicographically compares two ranges.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::lexicographical_compare};
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
/// let left = exec.to_device(&[1_u32, 2, 3]);
/// let right = exec.to_device(&[1_u32, 3, 0]);
///
/// assert!(massively::flag::is_set(
///     lexicographical_compare(&exec, left.slice(..), right.slice(..), Less)
///         .unwrap()
/// ));
/// ```
pub fn lexicographical_compare<R, Left, Right, Less>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    less: Less,
) -> Result<MFlag, Error>
where
    R: Runtime,
    Left: MIter<R>,
    Right: MIter<R, Item = Left::Item>,
    Less: BinaryPredicateOp<Left::Item>,
{
    let value = crate::core::search::lexicographical_compare(
        exec,
        crate::api::iter::lower_fixed::<R, _>(left),
        crate::api::iter::lower_fixed::<R, _>(right),
        less,
    )?;
    crate::api::value::read::<R, MFlag>(exec, &value)
}
