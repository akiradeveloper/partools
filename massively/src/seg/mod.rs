//! Algorithms applied independently to offset-delimited segments.
//!
//! A [`SegmentIterator`] value combines a flat [`MIter`] with offsets. Segment `i` is
//! the half-open range `offsets[i]..offsets[i + 1]` in the flat values.
//!
//! Segment algorithms return owned device storage. Length-preserving algorithms
//! return new flat values with the original offsets, length-changing algorithms
//! return values with a rebuilt [`Segmentation`], and reductions return one
//! value per segment.

pub(crate) mod control;
mod segment;
mod segmentation;

pub use segment::Segment;
pub(crate) use segment::{SegmentExpand, SegmentReader};
pub use segmentation::Segmentation;

use cubecl::prelude::*;
use std::{marker::PhantomData, ops::RangeBounds};

use crate::api::iter::{MIterExtent, MStorageExtent};
use crate::op::{BinaryPredicateOp, ExpandOp, PredicateOp, ReductionOp, UnaryOp};
use crate::{Error, Executor, MAlloc, MFlag, MIndex, MIter, MIterMut, MStorage, MVec};

pub(crate) fn segment_count(offset_count: usize) -> Result<usize, Error> {
    offset_count
        .checked_sub(1)
        .ok_or(Error::InvalidSegmentation)
}

/// A flat value stream and the offsets delimiting its segments.
///
/// When `Values` and `Offsets` are logical iterators, this is itself an
/// [`MIter`] whose item is [`Segment<Values::Item>`](Segment). Each item is a
/// read-only adapter over the shared value stream and supports `len()` and
/// unchecked `at()` access inside CubeCL operations.
///
/// [`Segmentation::segments`] uses a cloned [`Segmentation`] as `Offsets`, so
/// validated partition metadata remains available after length-preserving
/// algorithms. Length-changing owned results rebuild and retain one as well.
#[derive(Clone, Copy, Debug)]
pub struct SegmentIterator<Values, Offsets> {
    values: Values,
    offsets: Offsets,
}

impl<Values, Offsets> SegmentIterator<Values, Offsets> {
    /// Combines flat values and offsets into a segment iterator.
    pub const fn new(values: Values, offsets: Offsets) -> Self {
        Self { values, offsets }
    }

    /// Returns the flat values.
    pub const fn values(&self) -> &Values {
        &self.values
    }

    /// Returns the iterator that supplies the segment offsets.
    pub const fn offsets(&self) -> &Offsets {
        &self.offsets
    }

    /// Decomposes this input into its flat values and offset iterator.
    pub fn into_parts(self) -> (Values, Offsets) {
        (self.values, self.offsets)
    }
}

/// Physical segmented read created when a logical [`SegmentIterator`] is consumed.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct SegmentRead<Values, Offsets> {
    values: Values,
    offsets: Offsets,
}

impl<Values, Offsets> SegmentRead<Values, Offsets> {
    pub(crate) const fn new(values: Values, offsets: Offsets) -> Self {
        Self { values, offsets }
    }

    pub(crate) const fn values(&self) -> &Values {
        &self.values
    }

    pub(crate) const fn offsets(&self) -> &Offsets {
        &self.offsets
    }
}

#[doc(hidden)]
impl<R, Values, Offsets> MIter<R> for SegmentIterator<Values, Offsets>
where
    R: Runtime,
    Values: MIter<R>,
    Offsets: MIter<R, Item = MIndex>,
    SegmentRead<Values::Read, Offsets::Read>: crate::core::facade::KernelInput<R, Item = Segment<Values::Item>>
        + crate::core::read::SliceExpression,
{
    type Item = Segment<Values::Item>;
    type Read = SegmentRead<Values::Read, Offsets::Read>;
    type Slice = crate::core::read::Slice<R, Self::Read>;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<MIndex>,
    {
        let input = self.clone().lower_read();
        let len = crate::core::facade::physical_len::<R, _>(&input)
            .expect("cannot slice segmented input with an invalid length");
        let (start, count) = crate::core::read::resolve_mindex_slice_range(len, range);
        crate::core::read::Slice::new(crate::core::read::SliceExpression::slice_expression(
            &input, start, count,
        ))
    }

    fn lower_read(self) -> Self::Read {
        SegmentRead::new(self.values.lower_read(), self.offsets.lower_read())
    }
}

impl<R, Values, Offsets> MIterExtent<R> for SegmentIterator<Values, Offsets>
where
    R: Runtime,
    Values: MIter<R>,
    Offsets: MIter<R, Item = MIndex>,
{
    fn capacity(&self) -> Result<crate::MIndex, Error> {
        crate::api::iter::checked_len(segment_count(self.offsets.capacity()? as usize)?)
    }
}

/// Mutable flat values and offsets for a length-changing segmented result.
#[derive(Debug)]
pub(crate) struct SegmentedMut<Values, Offsets> {
    values: Values,
    offsets: Offsets,
}

impl<Values, Offsets> SegmentedMut<Values, Offsets> {
    /// Combines preallocated flat value and offset outputs.
    pub(crate) const fn new(values: Values, offsets: Offsets) -> Self {
        Self { values, offsets }
    }

    /// Decomposes this output into its flat values and offsets.
    pub(crate) fn into_parts(self) -> (Values, Offsets) {
        (self.values, self.offsets)
    }
}

/// Lifts one ordinary algorithm so it runs independently for every segment.
#[derive(Clone, Copy, Debug)]
pub struct ForEachSegment<Algorithm>(pub Algorithm);

/// Maps every item and returns the new values with the original offsets.
#[derive(Clone, Copy, Debug)]
pub struct Map<Op>(pub Op);

/// Expands every item and rebuilds the offsets for each segment.
#[derive(Clone, Copy, Debug)]
pub struct FlatMap<Op>(pub Op);

/// Stably sorts the items within every segment and preserves the offsets.
#[derive(Clone, Copy, Debug)]
pub struct Sort<Less>(pub Less);

/// Reverses the items within every segment and preserves the offsets.
#[derive(Clone, Copy, Debug, Default)]
pub struct Reverse;

/// Computes an inclusive scan within every segment and preserves the offsets.
#[derive(Clone, Copy, Debug)]
pub struct InclusiveScan<Op>(pub Op);

/// Computes an exclusive scan within every segment and preserves the offsets.
///
/// Each segment starts from `init`.
#[derive(Clone, Copy, Debug)]
pub struct ExclusiveScan<Op, Init>(pub Op, pub Init);

/// Computes adjacent differences within every segment and preserves the offsets.
#[derive(Clone, Copy, Debug)]
pub struct AdjacentDifference<Op>(pub Op);

/// Keeps the first item of each adjacent-equal run within every segment.
#[derive(Clone, Copy, Debug)]
pub struct Unique<Equal>(pub Equal);

/// Keeps the items satisfying a predicate within every segment.
#[derive(Clone, Copy, Debug)]
pub struct Filter<Pred>(pub Pred);

/// Keeps at most the first `n` items from every segment.
#[derive(Clone, Copy, Debug)]
pub struct Take(pub MIndex);

/// Reduces every segment to one item, starting from `init`.
#[derive(Clone, Copy, Debug)]
pub struct Reduce<Op, Init>(pub Op, pub Init);

/// Counts the items satisfying a predicate within every segment.
#[derive(Clone, Copy, Debug)]
pub struct CountIf<Pred>(pub Pred);

/// Finds the first matching local index in every segment.
///
/// A segment with no match returns its length.
#[derive(Clone, Copy, Debug)]
pub struct FindIf<Pred>(pub Pred);

/// Finds the first local index whose item equals its successor in every segment.
///
/// A segment with no adjacent equal pair returns its length.
#[derive(Clone, Copy, Debug)]
pub struct AdjacentFind<Equal>(pub Equal);

/// Tests whether every item satisfies a predicate within every segment.
#[derive(Clone, Copy, Debug)]
pub struct AllOf<Pred>(pub Pred);

/// Tests whether any item satisfies a predicate within every segment.
#[derive(Clone, Copy, Debug)]
pub struct AnyOf<Pred>(pub Pred);

/// Tests whether no item satisfies a predicate within every segment.
#[derive(Clone, Copy, Debug)]
pub struct NoneOf<Pred>(pub Pred);

/// Tests whether every segment is sorted.
#[derive(Clone, Copy, Debug)]
pub struct IsSorted<Less>(pub Less);

/// Finds the sorted prefix length of every segment.
#[derive(Clone, Copy, Debug)]
pub struct IsSortedUntil<Less>(pub Less);

/// Execution contract for segment algorithms that preserve segment lengths.
///
/// The output contains only flat values. Its segment offsets are the input
/// offsets and therefore do not need to be copied.
#[doc(hidden)]
pub(crate) trait LengthPreservingExecutableInto<R, Input, InputOffsets, Output>:
    Sized
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
{
    /// Executes the algorithm for every segment.
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error>;
}

/// Execution contract for algorithms that compact each segment independently.
///
/// Output segment lengths never exceed their corresponding input lengths. The
/// caller provides capacity for both flat values and rebuilt offsets, and the
/// returned index is the total number of flat values written.
#[doc(hidden)]
pub(crate) trait CompactingExecutableInto<R, Input, InputOffsets, Output, OutputOffsets>:
    Sized
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    OutputOffsets: MIterMut<R, Item = MIndex>,
{
    /// Executes the algorithm for every segment.
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: SegmentedMut<Output, OutputOffsets>,
    ) -> Result<crate::DeviceVec<R, MIndex>, Error>;
}

/// Execution contract for segment algorithms that write one item per segment.
#[doc(hidden)]
pub(crate) trait SummarizingExecutableInto<R, Input, InputOffsets, Output>: Sized
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
{
    /// Executes the algorithm for every segment.
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error>;
}

/// Owned-result execution contract for segment algorithms.
///
/// Each algorithm selects its complete output shape. Length-preserving
/// algorithms return new values with the input offsets, compacting algorithms
/// return rebuilt segments backed by a [`Segmentation`], and summarizing
/// algorithms return one value per segment.
pub trait Executable<R, Input, InputOffsets>: Sized
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
{
    /// Complete owned result selected by the segment algorithm.
    type Output;

    /// Executes this algorithm independently for every input segment.
    fn run(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
    ) -> Result<Self::Output, Error>;
}

impl<R, Input, InputOffsets, Op> Executable<R, Input, InputOffsets> for ForEachSegment<Map<Op>>
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Op: UnaryOp<Input::Item>,
    Op::Output: MAlloc<R>,
{
    type Output = SegmentIterator<MVec<R, Op::Output>, InputOffsets>;

    fn run(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
    ) -> Result<Self::Output, Error> {
        let Map(op) = self.0;
        let (values, offsets) = input.into_parts();
        let output = crate::vector::map(exec, values, op)?;
        Ok(SegmentIterator::new(output, offsets))
    }
}

impl<R, Input, InputOffsets, Op> Executable<R, Input, InputOffsets> for ForEachSegment<FlatMap<Op>>
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Op: ExpandOp<Input::Item>,
    Op::Output: MAlloc<R>,
{
    type Output = SegmentIterator<MVec<R, Op::Output>, Segmentation<R>>;

    fn run(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
    ) -> Result<Self::Output, Error> {
        let FlatMap(op) = self.0;
        let (values, offsets) = input.into_parts();
        let (output, element_offsets) = crate::api::algorithm::flat_map::expand(exec, values, op)?;
        let output_offsets = crate::vector::gather(exec, element_offsets.slice(..), offsets)?;
        let segmentation =
            Segmentation::from_generated_offsets(output_offsets, output.capacity()?)?;
        Ok(SegmentIterator::new(output, segmentation))
    }
}

impl<R, Input, InputOffsets, Output, Item, Less>
    LengthPreservingExecutableInto<R, Input, InputOffsets, Output> for ForEachSegment<Sort<Less>>
where
    R: Runtime,
    Input: MIter<R, Item = Item>,
    Item: MAlloc<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R, Item = Item>,
    Less: BinaryPredicateOp<Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let Sort(less) = self.0;
        let (values, offsets) = input.into_parts();
        let value_len = values.capacity()?;
        let control = control::SegmentControl::new(exec, offsets, value_len)?;
        let ids = control.ids(exec)?;

        let values = crate::api::iter::lower_fixed::<R, _>(values);
        let ordering = crate::core::ordering::sort_control_with(exec, values.clone(), less)?;
        let sorted_ids = exec.alloc::<MIndex>(value_len);
        crate::api::algorithm::apply_permutation_into(
            exec,
            crate::api::iter::lower_fixed::<R, _>(ids.slice(..)),
            ordering.column(),
            sorted_ids.slice_mut(..),
        )?;

        let id_ordering = crate::core::ordering::sort_control_with(
            exec,
            crate::api::iter::lower_fixed::<R, _>(sorted_ids.slice(..)),
            control::LessU32,
        )?;
        let permutation = exec.alloc::<MIndex>(value_len);
        crate::api::algorithm::apply_permutation_into(
            exec,
            crate::api::iter::lower_fixed::<R, _>(ordering.slice(..)),
            id_ordering.column(),
            permutation.slice_mut(..),
        )?;
        crate::api::algorithm::apply_permutation_into(exec, values, permutation.column(), output)
    }
}

impl<R, Input, InputOffsets, Output, Op>
    LengthPreservingExecutableInto<R, Input, InputOffsets, Output>
    for ForEachSegment<AdjacentDifference<Op>>
where
    R: Runtime,
    Input: MIter<R, Item = Output::Item> + Clone,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Output::SliceMut: MIterMut<R, Item = Output::Item>,
    Op: ReductionOp<Input::Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let AdjacentDifference(op) = self.0;
        let (values, offsets) = input.into_parts();
        let control = control::SegmentControl::new(exec, offsets, values.capacity()?)?;
        crate::vector::adjacent_difference_into(exec, values.clone(), op, output.slice_mut(..))?;
        crate::vector::transform_where(
            exec,
            values,
            crate::op::Identity,
            control.heads.slice(..),
            output,
        )
    }
}

struct SegmentScanOperation<'a, R: Runtime, Values, Item: MAlloc<R>, Op> {
    exec: &'a Executor<R>,
    values: Values,
    heads: crate::DeviceVec<R, u32>,
    init: Option<MVec<R, Item>>,
    _op: Op,
}

impl<R, Values, Item, Op> crate::api::iter::OutputOperation<R, Item>
    for SegmentScanOperation<'_, R, Values, Item, Op>
where
    R: Runtime,
    Values: MIter<R, Item = Item>,
    Item: MAlloc<R>,
    Op: ReductionOp<Item>,
{
    type Result = Result<(), Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: crate::api::iter::KernelRow + crate::core::allocation::ScratchStorage<R>,
        Output: crate::api::iter::ConcreteOutput<R, Item>,
    {
        let values = crate::api::iter::lower_fixed::<R, _>(self.values);
        if let Some(init) = self.init {
            crate::core::segmented::segmented_exclusive::<R, _, _, Item, Op>(
                self.exec,
                &values,
                &self.heads,
                crate::api::value::into_scratch::<R, Item>(init),
                &output,
            )
        } else {
            crate::core::segmented::segmented_inclusive::<R, _, _, Item, Op>(
                self.exec,
                &values,
                &self.heads,
                &output,
            )
        }
    }
}

fn scan_segments_into<R, Values, Output, Item, Op>(
    exec: &Executor<R>,
    values: Values,
    heads: crate::DeviceVec<R, u32>,
    init: Option<MVec<R, Item>>,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: MIter<R, Item = Item>,
    Output: MIterMut<R, Item = Item>,
    Item: MAlloc<R>,
    Op: ReductionOp<Item>,
{
    output.run_output_operation(SegmentScanOperation {
        exec,
        values,
        heads,
        init,
        _op: op,
    })
}

impl<R, Input, InputOffsets, Output, Item, Op>
    LengthPreservingExecutableInto<R, Input, InputOffsets, Output>
    for ForEachSegment<InclusiveScan<Op>>
where
    R: Runtime,
    Input: MIter<R, Item = Item>,
    Item: MAlloc<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R, Item = Item>,
    Op: ReductionOp<Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let InclusiveScan(op) = self.0;
        let (values, offsets) = input.into_parts();
        let control = control::SegmentControl::new(exec, offsets, values.capacity()?)?;
        scan_segments_into(exec, values, control.heads, None, op, output)
    }
}

impl<R, Input, InputOffsets, Output, Op, Item>
    LengthPreservingExecutableInto<R, Input, InputOffsets, Output>
    for ForEachSegment<ExclusiveScan<Op, Item>>
where
    R: Runtime,
    Input: MIter<R, Item = Item>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R, Item = Item>,
    Item: MAlloc<R>,
    Op: ReductionOp<Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let ExclusiveScan(op, init) = self.0;
        let init = crate::api::value::store(exec, init)?;
        let (values, offsets) = input.into_parts();
        let control = control::SegmentControl::new(exec, offsets, values.capacity()?)?;
        scan_segments_into(exec, values, control.heads, Some(init), op, output)
    }
}

impl<R, Input, InputOffsets, Output> LengthPreservingExecutableInto<R, Input, InputOffsets, Output>
    for ForEachSegment<Reverse>
where
    R: Runtime,
    Input: MIter<R, Item = Output::Item>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let (values, offsets) = input.into_parts();
        let control = control::SegmentControl::new(exec, offsets, values.capacity()?)?;
        let indices = control.reverse_indices(exec)?;
        crate::vector::gather_into(exec, values, indices.slice(..), output)
    }
}

impl<R, Input, InputOffsets, Output, OutputOffsets, Equal>
    CompactingExecutableInto<R, Input, InputOffsets, Output, OutputOffsets>
    for ForEachSegment<Unique<Equal>>
where
    R: Runtime,
    Input: MIter<R, Item = Output::Item>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    OutputOffsets: MIterMut<R, Item = MIndex>,
    Equal: BinaryPredicateOp<Input::Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: SegmentedMut<Output, OutputOffsets>,
    ) -> Result<crate::DeviceVec<R, MIndex>, Error> {
        let Unique(_equal) = self.0;
        let (values, offsets) = input.into_parts();
        let value_len = values.capacity()?;
        let control = control::SegmentControl::new(exec, offsets, value_len)?;
        let values = crate::api::iter::lower_fixed::<R, _>(values);
        let flags = crate::core::ordering::unique_head_flags::<R, _, Equal>(exec, values.clone())?;
        control.merge_heads(exec, &flags)?;
        let (output_values, output_offsets) = output.into_parts();
        control.compact(exec, values, flags, output_values, output_offsets)
    }
}

struct PredicateMap<Pred>(PhantomData<fn() -> Pred>);

#[cubecl::cube]
impl<Item, Pred> UnaryOp<Item> for PredicateMap<Pred>
where
    Item: CubeType + 'static,
    Pred: PredicateOp<Item>,
{
    type Output = MFlag;

    fn apply(input: Item) -> MFlag {
        crate::flag::from_bool(crate::flag::is_set(Pred::apply(input)))
    }
}

struct NegatedPredicateMap<Pred>(PhantomData<fn() -> Pred>);

#[cubecl::cube]
impl<Item, Pred> UnaryOp<Item> for NegatedPredicateMap<Pred>
where
    Item: CubeType + 'static,
    Pred: PredicateOp<Item>,
{
    type Output = MFlag;

    fn apply(input: Item) -> MFlag {
        crate::flag::from_bool(!crate::core::predicate::predicate::<Item, Pred>(input))
    }
}

struct Decrement;

#[cubecl::cube]
impl UnaryOp<MIndex> for Decrement {
    type Output = MIndex;

    fn apply(input: MIndex) -> MIndex {
        input - 1u32
    }
}

struct InvertFlag;

#[cubecl::cube]
impl UnaryOp<MFlag> for InvertFlag {
    type Output = MFlag;

    fn apply(input: MFlag) -> MFlag {
        crate::flag::from_bool(!crate::flag::is_set(input))
    }
}

struct SegmentReduceOperation<'a, R: Runtime, Values, Item: MAlloc<R>, Op> {
    exec: &'a Executor<R>,
    values: Values,
    heads: &'a crate::DeviceVec<R, u32>,
    head_control: &'a crate::core::selection::SelectionControl<R>,
    init: MVec<R, Item>,
    _op: Op,
}

impl<R, Values, Item, Op> crate::api::iter::OutputOperation<R, Item>
    for SegmentReduceOperation<'_, R, Values, Item, Op>
where
    R: Runtime,
    Values: MIter<R, Item = Item>,
    Item: MAlloc<R>,
    Op: ReductionOp<Item>,
{
    type Result = Result<(), Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: crate::api::iter::KernelRow + crate::core::allocation::ScratchStorage<R>,
        Output: crate::api::iter::ConcreteOutput<R, Item>,
    {
        crate::core::by_key::reduce_values_by_heads_lowered(
            self.exec,
            crate::api::iter::lower_fixed::<R, _>(self.values),
            self.heads,
            self.head_control,
            crate::api::value::into_scratch::<R, Item>(self.init),
            self._op,
            output,
        )
    }
}

fn reduce_segments_with_control<R, Values, Output, Op>(
    exec: &Executor<R>,
    values: Values,
    control: &control::SegmentControl<R>,
    init: MVec<R, Values::Item>,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: MIter<R>,
    Values::Item: MAlloc<R>,
    Output: MIterMut<R, Item = Values::Item>,
    Op: ReductionOp<Values::Item>,
{
    let segment_count = control.segment_count;
    let result = exec.alloc::<Values::Item>(segment_count);
    crate::api::algorithm::fill_value(exec, &init, result.slice_mut(..))?;
    if control.value_len == 0 {
        return crate::api::algorithm::transform::transform_into(
            exec,
            result.slice(..),
            crate::op::Identity,
            output,
        );
    }

    let head_control = crate::core::selection::FlagInput::selected_control(
        crate::api::iter::lower::<R, _>(control.heads.slice(..)),
        exec,
    )?;
    let head_indices = head_control.materialize_indices(exec)?;

    let reduced = exec.alloc::<Values::Item>(segment_count);
    reduced
        .slice_mut(..)
        .run_output_operation(SegmentReduceOperation {
            exec,
            values,
            heads: &control.heads,
            head_control: &head_control,
            init,
            _op: op,
        })?;

    let keys = exec.alloc::<MIndex>(segment_count);
    crate::api::algorithm::apply_permutation_prefix_into(
        exec,
        crate::api::iter::lower_fixed::<R, _>(control.heads.slice(..)),
        head_indices.column(),
        head_control.count(),
        keys.slice_mut(..),
    )?;
    let indices = exec.alloc::<MIndex>(segment_count);
    crate::api::algorithm::transform::transform_prefix_into(
        exec,
        keys.slice(..),
        Decrement,
        head_control.count(),
        indices.slice_mut(..),
    )?;
    crate::vector::scatter_prefix(
        exec,
        reduced.slice(..),
        indices.slice(..),
        head_control.count(),
        result.slice_mut(..),
    )?;
    crate::api::algorithm::transform::transform_into(
        exec,
        result.slice(..),
        crate::op::Identity,
        output,
    )
}

pub(crate) fn reduce_segments<R, Values, Offsets, Output, Op>(
    exec: &Executor<R>,
    input: SegmentIterator<Values, Offsets>,
    init: MVec<R, Values::Item>,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: MIter<R>,
    Values::Item: MAlloc<R>,
    Offsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R, Item = Values::Item>,
    Op: ReductionOp<Values::Item>,
{
    let (values, offsets) = input.into_parts();
    let value_len = values.capacity()?;
    let control = control::SegmentControl::new(exec, offsets, value_len)?;
    reduce_segments_with_control(exec, values, &control, init, op, output)
}

impl<R, Input, InputOffsets, Output, OutputOffsets, Pred>
    CompactingExecutableInto<R, Input, InputOffsets, Output, OutputOffsets>
    for ForEachSegment<Filter<Pred>>
where
    R: Runtime,
    Input: MIter<R, Item = Output::Item> + Clone,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    OutputOffsets: MIterMut<R, Item = MIndex>,
    Pred: PredicateOp<Input::Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: SegmentedMut<Output, OutputOffsets>,
    ) -> Result<crate::DeviceVec<R, MIndex>, Error> {
        let (values, offsets) = input.into_parts();
        let value_len = values.capacity()?;
        let control = control::SegmentOffsets::new(exec, offsets, value_len)?;
        let flags = exec.alloc::<u32>(value_len);
        crate::api::algorithm::transform::transform_into(
            exec,
            values.clone(),
            PredicateMap::<Pred>(PhantomData),
            flags.slice_mut(..),
        )?;
        let (output_values, output_offsets) = output.into_parts();
        control.compact(
            exec,
            crate::api::iter::lower_fixed::<R, _>(values),
            flags,
            output_values,
            output_offsets,
        )
    }
}

impl<R, Input, InputOffsets, Output, OutputOffsets>
    CompactingExecutableInto<R, Input, InputOffsets, Output, OutputOffsets> for ForEachSegment<Take>
where
    R: Runtime,
    Input: MIter<R, Item = Output::Item>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    OutputOffsets: MIterMut<R, Item = MIndex>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: SegmentedMut<Output, OutputOffsets>,
    ) -> Result<crate::DeviceVec<R, MIndex>, Error> {
        let Take(count) = self.0;
        let (values, offsets) = input.into_parts();
        let value_len = values.capacity()?;
        let control = control::SegmentControl::new(exec, offsets, value_len)?;
        let flags = control.take_flags(exec, count)?;
        let (output_values, output_offsets) = output.into_parts();
        control.compact(
            exec,
            crate::api::iter::lower_fixed::<R, _>(values),
            flags,
            output_values,
            output_offsets,
        )
    }
}

impl<R, Input, InputOffsets, Output, Op, Item>
    SummarizingExecutableInto<R, Input, InputOffsets, Output> for ForEachSegment<Reduce<Op, Item>>
where
    R: Runtime,
    Input: MIter<R, Item = Item>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R, Item = Item>,
    Item: MAlloc<R>,
    Op: ReductionOp<Input::Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let Reduce(op, init) = self.0;
        let init = crate::api::value::store(exec, init)?;
        reduce_segments(exec, input, init, op, output)
    }
}

macro_rules! impl_predicate_summary {
    ($algorithm:ident, $map:ident, $result:ty, $reducer:expr, $init:expr) => {
        impl<R, Input, InputOffsets, Output, Pred>
            SummarizingExecutableInto<R, Input, InputOffsets, Output>
            for ForEachSegment<$algorithm<Pred>>
        where
            R: Runtime,
            Input: MIter<R>,
            InputOffsets: MIter<R, Item = MIndex>,
            Output: MIterMut<R, Item = $result>,
            Pred: PredicateOp<Input::Item>,
        {
            fn run_into(
                self,
                exec: &Executor<R>,
                input: SegmentIterator<Input, InputOffsets>,
                output: Output,
            ) -> Result<(), Error> {
                let (values, offsets) = input.into_parts();
                let value_len = values.capacity()?;
                let mapped = exec.alloc::<$result>(value_len);
                crate::api::algorithm::transform::transform_into(
                    exec,
                    values,
                    $map::<Pred>(PhantomData),
                    mapped.slice_mut(..),
                )?;
                reduce_segments(
                    exec,
                    SegmentIterator::new(mapped.slice(..), offsets),
                    crate::api::value::store(exec, $init)?,
                    $reducer,
                    output,
                )
            }
        }
    };
}

impl_predicate_summary!(CountIf, PredicateMap, MIndex, control::SumU32, 0u32);
impl_predicate_summary!(AllOf, PredicateMap, MFlag, control::MinU32, 1u32);
impl_predicate_summary!(AnyOf, PredicateMap, MFlag, control::MaxU32, 0u32);
impl_predicate_summary!(NoneOf, NegatedPredicateMap, MFlag, control::MinU32, 1u32);

fn finish_match_positions<R, Output>(
    exec: &Executor<R>,
    control: &control::SegmentControl<R>,
    candidates: crate::DeviceVec<R, MIndex>,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Output: MIterMut<R, Item = MIndex>,
{
    let reduced = exec.alloc::<u32>(control.segment_count);
    reduce_segments_with_control(
        exec,
        candidates.slice(..),
        control,
        crate::api::value::store(exec, u32::MAX)?,
        control::MinU32,
        reduced.slice_mut(..),
    )?;
    let result = control.finish_sorted_until(exec, &reduced)?;
    crate::api::algorithm::transform::transform_into(
        exec,
        result.slice(..),
        crate::op::Identity,
        output,
    )
}

impl<R, Input, InputOffsets, Output, Pred> SummarizingExecutableInto<R, Input, InputOffsets, Output>
    for ForEachSegment<FindIf<Pred>>
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R, Item = MIndex>,
    Pred: PredicateOp<Input::Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let FindIf(_pred) = self.0;
        let (values, offsets) = input.into_parts();
        let value_len = values.capacity()?;
        let control = control::SegmentControl::new(exec, offsets, value_len)?;
        let matches = exec.alloc::<MFlag>(value_len);
        crate::api::algorithm::transform::transform_into(
            exec,
            values,
            PredicateMap::<Pred>(PhantomData),
            matches.slice_mut(..),
        )?;
        let candidates = control.match_candidates(exec, &matches, 0)?;
        finish_match_positions(exec, &control, candidates, output)
    }
}

impl<R, Input, InputOffsets, Output, Equal>
    SummarizingExecutableInto<R, Input, InputOffsets, Output>
    for ForEachSegment<AdjacentFind<Equal>>
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R, Item = MIndex>,
    Equal: BinaryPredicateOp<Input::Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let AdjacentFind(_equal) = self.0;
        let (values, offsets) = input.into_parts();
        let value_len = values.capacity()?;
        let control = control::SegmentControl::new(exec, offsets, value_len)?;
        let different = crate::core::ordering::unique_head_flags::<R, _, Equal>(
            exec,
            crate::api::iter::lower_fixed::<R, _>(values),
        )?;
        control.merge_heads(exec, &different)?;
        let matches = exec.alloc::<MFlag>(value_len);
        crate::api::algorithm::transform::transform_into(
            exec,
            different.slice(..),
            InvertFlag,
            matches.slice_mut(..),
        )?;
        let candidates = control.match_candidates(exec, &matches, 1)?;
        finish_match_positions(exec, &control, candidates, output)
    }
}

impl<R, Input, InputOffsets, Output, Less> SummarizingExecutableInto<R, Input, InputOffsets, Output>
    for ForEachSegment<IsSorted<Less>>
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R, Item = MFlag>,
    Less: BinaryPredicateOp<Input::Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let IsSorted(_less) = self.0;
        let (values, offsets) = input.into_parts();
        let value_len = values.capacity()?;
        let control = control::SegmentControl::new(exec, offsets, value_len)?;
        let breaks = crate::core::ordering::sorted_break_flags::<R, _, Less>(
            exec,
            crate::api::iter::lower_fixed::<R, _>(values),
        )?;
        control.clear_heads(exec, &breaks)?;
        let ordered = exec.alloc::<MFlag>(value_len);
        crate::api::algorithm::transform::transform_into(
            exec,
            breaks.slice(..),
            InvertFlag,
            ordered.slice_mut(..),
        )?;
        reduce_segments_with_control(
            exec,
            ordered.slice(..),
            &control,
            crate::api::value::store(exec, 1u32)?,
            control::MinU32,
            output,
        )
    }
}

impl<R, Input, InputOffsets, Output, Less> SummarizingExecutableInto<R, Input, InputOffsets, Output>
    for ForEachSegment<IsSortedUntil<Less>>
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R, Item = MIndex>,
    Less: BinaryPredicateOp<Input::Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let IsSortedUntil(_less) = self.0;
        let (values, offsets) = input.into_parts();
        let value_len = values.capacity()?;
        let control = control::SegmentControl::new(exec, offsets, value_len)?;
        let breaks = crate::core::ordering::sorted_break_flags::<R, _, Less>(
            exec,
            crate::api::iter::lower_fixed::<R, _>(values),
        )?;
        let candidates = control.sorted_until_candidates(exec, &breaks)?;
        let reduced = exec.alloc::<u32>(control.segment_count);
        reduce_segments_with_control(
            exec,
            candidates.slice(..),
            &control,
            crate::api::value::store(exec, u32::MAX)?,
            control::MinU32,
            reduced.slice_mut(..),
        )?;
        let result = control.finish_sorted_until(exec, &reduced)?;
        crate::api::algorithm::transform::transform_into(
            exec,
            result.slice(..),
            crate::op::Identity,
            output,
        )
    }
}

macro_rules! impl_owned_length_preserving_input {
    ($algorithm:ty, $( $bound:tt )+) => {
        impl<R, Input, InputOffsets, Op> Executable<R, Input, InputOffsets>
            for ForEachSegment<$algorithm>
        where
            R: Runtime,
            Input: MIter<R>,
            Input::Item: MAlloc<R>,
            InputOffsets: MIter<R, Item = MIndex>,
            $( $bound )+
        {
            type Output = SegmentIterator<MVec<R, Input::Item>, InputOffsets>;

            fn run(
                self,
                exec: &Executor<R>,
                input: SegmentIterator<Input, InputOffsets>,
            ) -> Result<Self::Output, Error> {
                let (values, offsets) = input.into_parts();
                let output = exec.alloc::<Input::Item>(values.capacity()?);
                LengthPreservingExecutableInto::run_into(
                    self,
                    exec,
                    SegmentIterator::new(values, offsets.clone()),
                    output.slice_mut(..),
                )?;
                Ok(SegmentIterator::new(output, offsets))
            }
        }
    };
}

impl_owned_length_preserving_input!(Sort<Op>, Op: BinaryPredicateOp<Input::Item>);
impl_owned_length_preserving_input!(InclusiveScan<Op>,
    Op: ReductionOp<Input::Item>
);

impl<R, Input, InputOffsets, Op, Item> Executable<R, Input, InputOffsets>
    for ForEachSegment<ExclusiveScan<Op, Item>>
where
    R: Runtime,
    Input: MIter<R, Item = Item>,
    InputOffsets: MIter<R, Item = MIndex>,
    Item: MAlloc<R>,
    Op: ReductionOp<Item>,
{
    type Output = SegmentIterator<MVec<R, Input::Item>, InputOffsets>;

    fn run(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
    ) -> Result<Self::Output, Error> {
        let (values, offsets) = input.into_parts();
        let output = exec.alloc::<Input::Item>(values.capacity()?);
        LengthPreservingExecutableInto::run_into(
            self,
            exec,
            SegmentIterator::new(values, offsets.clone()),
            output.slice_mut(..),
        )?;
        Ok(SegmentIterator::new(output, offsets))
    }
}

impl<R, Input, InputOffsets> Executable<R, Input, InputOffsets> for ForEachSegment<Reverse>
where
    R: Runtime,
    Input: MIter<R>,
    Input::Item: MAlloc<R>,
    InputOffsets: MIter<R, Item = MIndex>,
{
    type Output = SegmentIterator<MVec<R, Input::Item>, InputOffsets>;

    fn run(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
    ) -> Result<Self::Output, Error> {
        let (values, offsets) = input.into_parts();
        let output = exec.alloc::<Input::Item>(values.capacity()?);
        LengthPreservingExecutableInto::run_into(
            self,
            exec,
            SegmentIterator::new(values, offsets.clone()),
            output.slice_mut(..),
        )?;
        Ok(SegmentIterator::new(output, offsets))
    }
}

impl<R, Input, InputOffsets, Op> Executable<R, Input, InputOffsets>
    for ForEachSegment<AdjacentDifference<Op>>
where
    R: Runtime,
    Input: MIter<R> + Clone,
    Input::Item: MAlloc<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Op: ReductionOp<Input::Item>,
{
    type Output = SegmentIterator<MVec<R, Input::Item>, InputOffsets>;

    fn run(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
    ) -> Result<Self::Output, Error> {
        let AdjacentDifference(op) = self.0;
        let (values, offsets) = input.into_parts();
        let control = control::SegmentControl::new(exec, offsets.clone(), values.capacity()?)?;
        let output = crate::vector::adjacent_difference(exec, values.clone(), op)?;
        crate::vector::transform_where(
            exec,
            values,
            crate::op::Identity,
            control.heads.column(),
            output.slice_mut(..),
        )?;
        Ok(SegmentIterator::new(output, offsets))
    }
}

macro_rules! impl_owned_compacting {
    ($algorithm:ty, $( $bound:tt )+) => {
        impl<R, Input, InputOffsets, Op> Executable<R, Input, InputOffsets>
            for ForEachSegment<$algorithm>
        where
            R: Runtime,
            Input: MIter<R> + Clone,
            Input::Item: MAlloc<R>,
            InputOffsets: MIter<R, Item = MIndex>,
            $( $bound )+
        {
            type Output = SegmentIterator<MVec<R, Input::Item>, Segmentation<R>>;

            fn run(
                self,
                exec: &Executor<R>,
                input: SegmentIterator<Input, InputOffsets>,
            ) -> Result<Self::Output, Error> {
                let values = exec.alloc::<Input::Item>(input.values().capacity()?);
                let offsets = exec.alloc::<MIndex>(input.offsets().capacity()?);
                let written = CompactingExecutableInto::run_into(
                    self,
                    exec,
                    input,
                    SegmentedMut::new(values.slice_mut(..), offsets.slice_mut(..)),
                )?;
                let written = crate::api::value::read::<R, MIndex>(exec, &written)?;
                let values =
                    crate::api::materialize::into_exact_prefix::<R, Input::Item>(exec, values, written)?;
                let segmentation = Segmentation::from_generated_offsets(offsets, written)?;
                Ok(SegmentIterator::new(values, segmentation))
            }
        }
    };
}

impl_owned_compacting!(Unique<Op>, Op: BinaryPredicateOp<Input::Item>);
impl_owned_compacting!(Filter<Op>, Op: PredicateOp<Input::Item>);

impl<R, Input, InputOffsets> Executable<R, Input, InputOffsets> for ForEachSegment<Take>
where
    R: Runtime,
    Input: MIter<R>,
    Input::Item: MAlloc<R>,
    InputOffsets: MIter<R, Item = MIndex>,
{
    type Output = SegmentIterator<MVec<R, Input::Item>, Segmentation<R>>;

    fn run(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
    ) -> Result<Self::Output, Error> {
        let values = exec.alloc::<Input::Item>(input.values().capacity()?);
        let offsets = exec.alloc::<MIndex>(input.offsets().capacity()?);
        let written = CompactingExecutableInto::run_into(
            self,
            exec,
            input,
            SegmentedMut::new(values.slice_mut(..), offsets.slice_mut(..)),
        )?;
        let written = crate::api::value::read::<R, MIndex>(exec, &written)?;
        let values =
            crate::api::materialize::into_exact_prefix::<R, Input::Item>(exec, values, written)?;
        let segmentation = Segmentation::from_generated_offsets(offsets, written)?;
        Ok(SegmentIterator::new(values, segmentation))
    }
}

impl<R, Input, InputOffsets, Op, Item> Executable<R, Input, InputOffsets>
    for ForEachSegment<Reduce<Op, Item>>
where
    R: Runtime,
    Input: MIter<R, Item = Item>,
    InputOffsets: MIter<R, Item = MIndex>,
    Item: MAlloc<R>,
    Op: ReductionOp<Item>,
{
    type Output = MVec<R, Input::Item>;

    fn run(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
    ) -> Result<Self::Output, Error> {
        let segment_count = segment_count(input.offsets().capacity()? as usize)?;
        let segment_count = crate::api::iter::checked_len(segment_count)?;
        let output = exec.alloc::<Input::Item>(segment_count);
        SummarizingExecutableInto::run_into(self, exec, input, output.slice_mut(..))?;
        Ok(output)
    }
}

macro_rules! impl_owned_summary {
    ($algorithm:ty, $result:ty, $( $bound:tt )+) => {
        impl<R, Input, InputOffsets, Op> Executable<R, Input, InputOffsets>
            for ForEachSegment<$algorithm>
        where
            R: Runtime,
            Input: MIter<R>,
            InputOffsets: MIter<R, Item = MIndex>,
            $( $bound )+
        {
            type Output = MVec<R, $result>;

            fn run(
                self,
                exec: &Executor<R>,
                input: SegmentIterator<Input, InputOffsets>,
            ) -> Result<Self::Output, Error> {
                let segment_count = segment_count(input.offsets().capacity()? as usize)?;
                let segment_count = crate::api::iter::checked_len(segment_count)?;
                let output = exec.alloc::<$result>(segment_count);
                SummarizingExecutableInto::run_into(self, exec, input, output.slice_mut(..))?;
                Ok(output)
            }
        }
    };
}

impl_owned_summary!(CountIf<Op>, MIndex, Op: PredicateOp<Input::Item>);
impl_owned_summary!(AllOf<Op>, MFlag, Op: PredicateOp<Input::Item>);
impl_owned_summary!(AnyOf<Op>, MFlag, Op: PredicateOp<Input::Item>);
impl_owned_summary!(NoneOf<Op>, MFlag, Op: PredicateOp<Input::Item>);
impl_owned_summary!(FindIf<Op>, MIndex, Op: PredicateOp<Input::Item>);
impl_owned_summary!(AdjacentFind<Op>, MIndex, Op: BinaryPredicateOp<Input::Item>);
impl_owned_summary!(IsSorted<Op>, MFlag, Op: BinaryPredicateOp<Input::Item>);
impl_owned_summary!(IsSortedUntil<Op>, MIndex, Op: BinaryPredicateOp<Input::Item>);
