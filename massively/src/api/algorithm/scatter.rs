use cubecl::prelude::*;

use crate::core::allocation::RowStorage;
use crate::op::{BinaryPredicateOp, ReductionOp};
use crate::{Error, Executor, MAlloc, MFlag, MIter, MIterMut, MVec};

struct IndexLess;

#[cubecl::cube]
impl BinaryPredicateOp<crate::MIndex> for IndexLess {
    fn apply(lhs: crate::MIndex, rhs: crate::MIndex) -> MFlag {
        crate::flag::from_bool(lhs < rhs)
    }
}

struct IndexEqual;

#[cubecl::cube]
impl BinaryPredicateOp<crate::MIndex> for IndexEqual {
    fn apply(lhs: crate::MIndex, rhs: crate::MIndex) -> MFlag {
        crate::flag::from_bool(lhs == rhs)
    }
}

struct ScatterOperation<'a, R: Runtime, Values, Indices> {
    exec: &'a Executor<R>,
    values: Values,
    indices: Indices,
}

struct ScatterPrefixOperation<'a, R: Runtime, Values, Indices> {
    exec: &'a Executor<R>,
    values: Values,
    indices: Indices,
    active_len: &'a crate::DeviceVec<R, u32>,
}

impl<R, Item, Values, Indices> crate::api::iter::OutputOperation<R, Item>
    for ScatterPrefixOperation<'_, R, Values, Indices>
where
    R: Runtime,
    Item: CubeType + Send + Sync + 'static,
    Values: MIter<R, Item = Item>,
    Indices: MIter<R, Item = crate::MIndex>,
{
    type Result = Result<(), Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: crate::api::iter::KernelRow + crate::core::allocation::ScratchStorage<R>,
        Output: crate::api::iter::ConcreteOutput<R, Item>,
    {
        crate::core::indexed::IndexedCopyInput::indexed_copy_selected(
            crate::api::iter::lower::<R, _>(self.values),
            self.exec,
            crate::api::iter::lower::<R, _>(self.indices),
            None,
            Some(self.active_len),
            false,
            output,
        )
    }
}

impl<R, Item, Values, Indices> crate::api::iter::OutputOperation<R, Item>
    for ScatterOperation<'_, R, Values, Indices>
where
    R: Runtime,
    Item: CubeType + Send + Sync + 'static,
    Values: MIter<R, Item = Item>,
    Indices: MIter<R, Item = crate::MIndex>,
{
    type Result = Result<(), Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: crate::api::iter::KernelRow + crate::core::allocation::ScratchStorage<R>,
        Output: crate::api::iter::ConcreteOutput<R, Item>,
    {
        crate::core::scatter::scatter(
            self.exec,
            crate::api::iter::lower::<R, _>(self.values),
            crate::api::iter::lower::<R, _>(self.indices),
            output,
        )
    }
}

struct ScatterWhereOperation<'a, R: Runtime, Values, Indices, Stencil> {
    exec: &'a Executor<R>,
    values: Values,
    indices: Indices,
    stencil: Stencil,
}

impl<R, Item, Values, Indices, Stencil> crate::api::iter::OutputOperation<R, Item>
    for ScatterWhereOperation<'_, R, Values, Indices, Stencil>
where
    R: Runtime,
    Item: CubeType + Send + Sync + 'static,
    Values: MIter<R, Item = Item>,
    Indices: MIter<R, Item = crate::MIndex>,
    Stencil: MIter<R, Item = MFlag>,
{
    type Result = Result<(), Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: crate::api::iter::KernelRow + crate::core::allocation::ScratchStorage<R>,
        Output: crate::api::iter::ConcreteOutput<R, Item>,
    {
        crate::core::scatter::scatter_where(
            self.exec,
            crate::api::iter::lower::<R, _>(self.values),
            crate::api::iter::lower::<R, _>(self.indices),
            crate::api::iter::lower::<R, _>(self.stencil),
            output,
        )
    }
}

struct ScatterReduceOperation<'a, R: Runtime, Values, Indices, Item: MAlloc<R>, Op> {
    exec: &'a Executor<R>,
    values: Values,
    indices: Indices,
    init: MVec<R, Item>,
    op: Op,
}

impl<R, Item, Values, Indices, Op> crate::api::iter::OutputOperation<R, Item>
    for ScatterReduceOperation<'_, R, Values, Indices, Item, Op>
where
    R: Runtime,
    Item: MAlloc<R>,
    Values: MIter<R, Item = Item>,
    Indices: MIter<R, Item = crate::MIndex>,
    Op: ReductionOp<Item>,
{
    type Result = Result<(), Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: crate::api::iter::KernelRow + crate::core::allocation::ScratchStorage<R>,
        Output: crate::api::iter::ConcreteOutput<R, Item>,
    {
        let len = self.values.capacity()?;
        let indices_len = self.indices.capacity()?;
        if len != indices_len {
            return Err(Error::LengthMismatch {
                left: len as usize,
                right: indices_len as usize,
            });
        }
        if len == 0 {
            return Ok(());
        }
        let extent = self
            .values
            .logical_extent()?
            .zipped(&self.indices.logical_extent()?)?;

        let indices = crate::api::iter::lower::<R, _>(self.indices);
        let mut sorted_values = <Item as crate::core::allocation::ScratchStorage<R>>::alloc_scratch(
            self.exec,
            len as usize,
        );
        RowStorage::set_logical_extent(&mut sorted_values, extent);
        let values = crate::api::iter::lower_fixed::<R, _>(self.values);
        let permutation =
            crate::core::ordering::sort_control_with(self.exec, indices.clone(), IndexLess)?;
        crate::core::indexed::PermutationCopyInput::permutation_copy(
            values,
            self.exec,
            permutation.column(),
            None,
            <<Item as crate::core::allocation::ScratchStorage<R>>::Storage as RowStorage<R>>::write(
                &sorted_values,
            ),
        )?;

        let heads = crate::core::ordering::unique_head_flags_ordered::<R, _, IndexEqual>(
            self.exec,
            indices.clone(),
            &permutation,
        )?;
        let head_control =
            crate::core::selection::SelectionControl::from_flags(self.exec, heads.clone())?;
        let head_indices = head_control.materialize_indices(self.exec)?;
        let reduced_extent = head_indices.logical_extent();

        let sorted_values =
            crate::core::read::FixedRead::new(<<Item as crate::core::allocation::ScratchStorage<
                R,
            >>::Storage as RowStorage<R>>::read(
                &sorted_values
            ));
        let mut reduced_values =
            <Item as crate::core::allocation::ScratchStorage<R>>::alloc_scratch(
                self.exec,
                len as usize,
            );
        crate::core::by_key::reduce_values_by_heads_lowered(
            self.exec,
            sorted_values,
            &heads,
            &head_control,
            crate::api::value::into_scratch::<R, Item>(self.init),
            self.op,
            <<Item as crate::core::allocation::ScratchStorage<R>>::Storage as RowStorage<R>>::write(
                &reduced_values,
            ),
        )?;
        RowStorage::set_logical_extent(&mut reduced_values, reduced_extent);

        let reduced_values =
            crate::core::read::FixedRead::new(<<Item as crate::core::allocation::ScratchStorage<
                R,
            >>::Storage as RowStorage<R>>::read(
                &reduced_values
            ));
        crate::core::scatter_reduce::apply::<R, _, _, _, Op>(
            self.exec,
            reduced_values,
            indices,
            &head_indices,
            &permutation,
            head_control.count(),
            output,
        )
    }
}

/// Writes each input item to the position given by its index.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, vector::scatter};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let values = exec.to_device(&[10_u32, 20, 30]);
/// let indices = exec.to_device(&[2_u32, 0, 1]);
/// let output = exec.alloc::<u32>(3);
///
/// scatter(
///     &exec,
///     values.slice(..),
///     indices.slice(..),
///     output.slice_mut(..),
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![20, 30, 10]);
/// ```
pub fn scatter<R, Values, Indices, Output>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: MIter<R, Item = Output::Item>,
    Indices: MIter<R, Item = crate::MIndex>,
    Output: MIterMut<R>,
{
    output.run_output_operation(ScatterOperation {
        exec,
        values,
        indices,
    })
}

pub(crate) fn scatter_prefix<R, Values, Indices, Output>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    active_len: &crate::DeviceVec<R, u32>,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: MIter<R, Item = Output::Item>,
    Indices: MIter<R, Item = crate::MIndex>,
    Output: MIterMut<R>,
{
    output.run_output_operation(ScatterPrefixOperation {
        exec,
        values,
        indices,
        active_len,
    })
}

/// Scatters selected rows while preserving other output rows.
///
/// A false stencil leaves the indexed destination unchanged.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, vector::scatter_where};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let values = exec.to_device(&[10_u32, 20, 30]);
/// let indices = exec.to_device(&[2_u32, 0, 1]);
/// let stencil = exec.to_device(&[1_u32, 0, 1]);
/// let output = exec.to_device(&[99_u32, 99, 99]);
///
/// scatter_where(
///     &exec,
///     values.slice(..),
///     indices.slice(..),
///     stencil.slice(..),
///     output.slice_mut(..),
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![99, 30, 10]);
/// ```
pub fn scatter_where<R, Values, Indices, Stencil, Output>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    stencil: Stencil,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: MIter<R, Item = Output::Item>,
    Indices: MIter<R, Item = crate::MIndex>,
    Stencil: MIter<R, Item = MFlag>,
    Output: MIterMut<R>,
{
    output.run_output_operation(ScatterWhereOperation {
        exec,
        values,
        indices,
        stencil,
    })
}

/// Reduces colliding scatter proposals and combines each result with its destination.
///
/// `init` must be the identity of `op`. Proposals targeting the same index are reduced in an
/// unspecified order; consequently `op` must be associative and commutative. Destinations not
/// present in `indices` are left unchanged.
///
/// This implementation preserves semantic rows, including multi-column tuple rows. It sorts and
/// reduces proposals before the final write, so the last phase has exactly one writer per
/// destination and does not require an atomic implementation for every item type.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, op, vector::scatter_reduce};
///
/// struct Add;
///
/// #[cubecl::cube]
/// impl op::ReductionOp<u32> for Add {
///     fn apply(lhs: u32, rhs: u32) -> u32 { lhs + rhs }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let values = exec.to_device(&[2_u32, 3, 5, 7]);
/// let indices = exec.to_device(&[1_u32, 0, 1, 1]);
/// let output = exec.to_device(&[10_u32, 20, 30]);
/// scatter_reduce(
///     &exec,
///     values.slice(..),
///     indices.slice(..),
///     0_u32,
///     Add,
///     output.slice_mut(..),
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![13, 34, 30]);
/// ```
pub fn scatter_reduce<R, Values, Indices, Item, Output, Op>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    init: Item,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: MIter<R, Item = Item>,
    Item: MAlloc<R>,
    Indices: MIter<R, Item = crate::MIndex>,
    Output: MIterMut<R, Item = Item>,
    Op: ReductionOp<Item>,
{
    let init = crate::api::value::store(exec, init)?;
    scatter_reduce_value(exec, values, indices, init, op, output)
}

pub(crate) fn scatter_reduce_value<R, Values, Indices, Item, Output, Op>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    init: MVec<R, Item>,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: MIter<R, Item = Item>,
    Item: MAlloc<R>,
    Indices: MIter<R, Item = crate::MIndex>,
    Output: MIterMut<R, Item = Item>,
    Op: ReductionOp<Item>,
{
    output.run_output_operation(ScatterReduceOperation {
        exec,
        values,
        indices,
        init,
        op,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{zip2, zip3};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    struct PairAdd;

    #[cubecl::cube]
    impl ReductionOp<(u32, u32)> for PairAdd {
        fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> (u32, u32) {
            (lhs.0 + rhs.0, lhs.1 + rhs.1)
        }
    }

    struct TripleAdd;

    #[cubecl::cube]
    impl ReductionOp<(u32, u32, u32)> for TripleAdd {
        fn apply(lhs: (u32, u32, u32), rhs: (u32, u32, u32)) -> (u32, u32, u32) {
            (lhs.0 + rhs.0, lhs.1 + rhs.1, lhs.2 + rhs.2)
        }
    }

    #[test]
    fn scatter_reduce_preserves_multi_column_rows() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let left = exec.to_device(&[1_u32, 2, 3]);
        let right = exec.to_device(&[10_u32, 20, 30]);
        let indices = exec.to_device(&[1_u32, 0, 1]);
        let output_left = exec.to_device(&[100_u32, 200]);
        let output_right = exec.to_device(&[1000_u32, 2000]);
        scatter_reduce(
            &exec,
            zip2(left.slice(..), right.slice(..)),
            indices.slice(..),
            (0_u32, 0_u32),
            PairAdd,
            zip2(output_left.slice_mut(..), output_right.slice_mut(..)),
        )
        .unwrap();

        assert_eq!(exec.to_host(&output_left).unwrap(), vec![102, 204]);
        assert_eq!(exec.to_host(&output_right).unwrap(), vec![1020, 2040]);
    }

    #[test]
    fn scatter_reduce_preserves_three_flat_columns() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let first = exec.to_device(&[1_u32, 2, 3]);
        let second = exec.to_device(&[10_u32, 20, 30]);
        let third = exec.to_device(&[100_u32, 200, 300]);
        let indices = exec.to_device(&[1_u32, 0, 1]);
        let output_first = exec.to_device(&[100_u32, 200]);
        let output_second = exec.to_device(&[1000_u32, 2000]);
        let output_third = exec.to_device(&[10000_u32, 20000]);
        scatter_reduce(
            &exec,
            zip3(first.slice(..), second.slice(..), third.slice(..)),
            indices.slice(..),
            (0_u32, 0_u32, 0_u32),
            TripleAdd,
            zip3(
                output_first.slice_mut(..),
                output_second.slice_mut(..),
                output_third.slice_mut(..),
            ),
        )
        .unwrap();

        assert_eq!(exec.to_host(&output_first).unwrap(), vec![102, 204]);
        assert_eq!(exec.to_host(&output_second).unwrap(), vec![1020, 2040]);
        assert_eq!(exec.to_host(&output_third).unwrap(), vec![10200, 20400]);
    }
}
