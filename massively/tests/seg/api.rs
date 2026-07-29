use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{
    Executor, MIter, MStorage, lazy,
    op::BinaryPredicateOp,
    op::ExpandOp,
    op::PredicateOp,
    op::ReductionOp,
    op::UnaryOp,
    seg::{
        AdjacentFind, AllOf, ExclusiveScan, Executable, FindIf, FlatMap, ForEachSegment, Map,
        Reduce, Segment, SegmentIterator, Segmentation, Take, Unique,
    },
    vector::{copy_where, equal, is_sorted, map as vector_map},
    zip2, zip3,
};

struct CastU64;

fn assert_validated_segmentation<R: Runtime, Values>(
    _input: &SegmentIterator<Values, Segmentation<R>>,
) {
}

#[cubecl::cube]
impl UnaryOp<u32> for CastU64 {
    type Output = u64;

    fn apply(value: u32) -> u64 {
        u64::cast_from(value)
    }
}

struct Equal;
struct Even;
struct RawFlag;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for Equal {
    fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
        massively::flag::from_bool(lhs == rhs)
    }
}

#[cubecl::cube]
impl PredicateOp<u32> for Even {
    fn apply(value: u32) -> massively::MFlag {
        massively::flag::from_bool(value % 2u32 == 0u32)
    }
}

#[cubecl::cube]
impl PredicateOp<u32> for RawFlag {
    fn apply(value: u32) -> massively::MFlag {
        value
    }
}

struct RepeatValue;

#[cubecl::cube]
impl ExpandOp<u32> for RepeatValue {
    type Output = u32;

    fn count(input: u32) -> u32 {
        input
    }

    fn generate(input: u32, local_index: u32) -> u32 {
        input * 10 + local_index
    }
}

struct Add;

#[cubecl::cube]
impl ReductionOp<u32> for Add {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

struct LexicographicalBytes;

struct SliceLength;
struct PairSliceLength;
struct SegmentPairLengths;
struct ContextForEmptySegment;
struct AddU32Pair;
struct EmptyContextExpand;
struct CyclicPredecessorIndex;
struct AddU32Triple;

#[cubecl::cube]
impl UnaryOp<Segment<u32>> for SliceLength {
    type Output = u32;

    fn apply(value: Segment<u32>) -> u32 {
        value.len()
    }
}

#[cubecl::cube]
impl UnaryOp<Segment<(u32, u32)>> for PairSliceLength {
    type Output = u32;

    fn apply(value: Segment<(u32, u32)>) -> u32 {
        value.len()
    }
}

#[cubecl::cube]
impl UnaryOp<(Segment<u32>, Segment<u32>)> for SegmentPairLengths {
    type Output = u32;

    fn apply(value: (Segment<u32>, Segment<u32>)) -> u32 {
        value.0.len() * 10u32 + value.1.len()
    }
}

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for ContextForEmptySegment {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        if input.0 == 0u32 { input.1 } else { 0u32 }
    }
}

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for AddU32Pair {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        input.0 + input.1
    }
}

#[cubecl::cube]
impl ExpandOp<(u32, u32)> for EmptyContextExpand {
    type Output = u32;

    fn count(input: (u32, u32)) -> u32 {
        if input.0 == 0u32 {
            input.1 % 3u32
        } else {
            0u32
        }
    }

    fn generate(input: (u32, u32), local_index: u32) -> u32 {
        input.1 + local_index
    }
}

#[cubecl::cube]
impl UnaryOp<(u32, u32, u32)> for CyclicPredecessorIndex {
    type Output = u32;

    fn apply(input: (u32, u32, u32)) -> u32 {
        if input.0 == input.1 {
            input.2 - 1u32
        } else {
            input.0 - 1u32
        }
    }
}

#[cubecl::cube]
impl UnaryOp<(u32, u32, u32)> for AddU32Triple {
    type Output = u32;

    fn apply(input: (u32, u32, u32)) -> u32 {
        input.0 + input.1 + input.2
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<Segment<u32>> for LexicographicalBytes {
    fn apply(lhs: Segment<u32>, rhs: Segment<u32>) -> massively::MFlag {
        let lhs_len = lhs.len();
        let rhs_len = rhs.len();
        let common_len = if lhs_len < rhs_len { lhs_len } else { rhs_len };
        let cursor = RuntimeCell::<u32>::new(0u32);
        let ordering = RuntimeCell::<u32>::new(0u32);

        while cursor.read() < common_len {
            let lhs_item = lhs.at(cursor.read());
            let rhs_item = rhs.at(cursor.read());
            if lhs_item < rhs_item {
                ordering.store(1u32);
                cursor.store(common_len);
            } else if rhs_item < lhs_item {
                ordering.store(2u32);
                cursor.store(common_len);
            } else {
                cursor.store(cursor.read() + 1u32);
            }
        }

        massively::flag::from_bool(if ordering.read() == 1u32 {
            true
        } else if ordering.read() == 2u32 {
            false
        } else {
            lhs_len < rhs_len
        })
    }
}

#[derive(CubeType, Clone, Copy)]
struct Code {
    value: u32,
}

struct WrapCode;

#[cubecl::cube]
impl UnaryOp<u32> for WrapCode {
    type Output = Code;

    fn apply(value: u32) -> Code {
        Code { value }
    }
}

struct LexicographicalCodes;

#[cubecl::cube]
impl BinaryPredicateOp<Segment<Code>> for LexicographicalCodes {
    fn apply(lhs: Segment<Code>, rhs: Segment<Code>) -> massively::MFlag {
        let lhs_len = lhs.len();
        let rhs_len = rhs.len();
        let common_len = if lhs_len < rhs_len { lhs_len } else { rhs_len };
        let cursor = RuntimeCell::<u32>::new(0u32);
        let ordering = RuntimeCell::<u32>::new(0u32);

        while cursor.read() < common_len {
            let lhs_item = lhs.at(cursor.read()).value;
            let rhs_item = rhs.at(cursor.read()).value;
            if lhs_item < rhs_item {
                ordering.store(1u32);
                cursor.store(common_len);
            } else if rhs_item < lhs_item {
                ordering.store(2u32);
                cursor.store(common_len);
            } else {
                cursor.store(cursor.read() + 1u32);
            }
        }

        massively::flag::from_bool(if ordering.read() == 1u32 {
            true
        } else if ordering.read() == 2u32 {
            false
        } else {
            lhs_len < rhs_len
        })
    }
}

struct SlicesEqual;

#[cubecl::cube]
impl BinaryPredicateOp<Segment<u32>> for SlicesEqual {
    fn apply(lhs: Segment<u32>, rhs: Segment<u32>) -> massively::MFlag {
        massively::flag::from_bool(if lhs.len() != rhs.len() {
            false
        } else {
            let cursor = RuntimeCell::<u32>::new(0u32);
            let equal = RuntimeCell::<u32>::new(1u32);
            while cursor.read() < lhs.len() {
                if lhs.at(cursor.read()) != rhs.at(cursor.read()) {
                    equal.store(0u32);
                    cursor.store(lhs.len());
                } else {
                    cursor.store(cursor.read() + 1u32);
                }
            }
            equal.read() != 0u32
        })
    }
}

#[test]
fn segmented_wrappers_expose_their_parts() {
    let input = SegmentIterator::new([1_u32, 2], [0_u32, 2]);
    assert_eq!(input.values(), &[1, 2]);
    assert_eq!(input.offsets(), &[0, 2]);
    assert_eq!(input.into_parts(), ([1, 2], [0, 2]));
}

#[test]
fn segmented_algorithms_return_owned_device_results() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let values = exec.to_device(&[1_u32, 1, 2, 3, 3]);
    let offsets = exec.to_device(&[0_u32, 3, 5]);

    let mapped = ForEachSegment(Map(CastU64))
        .run(
            &exec,
            SegmentIterator::new(values.slice(..), offsets.slice(..)),
        )
        .unwrap();
    assert_eq!(
        exec.to_host(mapped.values()).unwrap(),
        vec![1_u64, 1, 2, 3, 3]
    );
    assert_eq!(exec.to_host(mapped.offsets()).unwrap(), vec![0, 3, 5]);

    let unique = ForEachSegment(Unique(Equal))
        .run(
            &exec,
            SegmentIterator::new(values.slice(..), offsets.slice(..)),
        )
        .unwrap();
    assert_validated_segmentation(&unique);
    assert_eq!(exec.to_host(unique.values()).unwrap(), vec![1, 2, 3]);
    assert_eq!(
        exec.to_host(&unique.offsets().offsets()).unwrap(),
        vec![0, 2, 3]
    );

    let reduced = ForEachSegment(Reduce(Add, 0_u32))
        .run(
            &exec,
            SegmentIterator::new(values.slice(..), offsets.slice(..)),
        )
        .unwrap();
    assert_eq!(exec.to_host(&reduced).unwrap(), vec![4, 6]);
}

#[test]
fn segmented_single_value_inputs_accept_host_results() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let empty = exec.alloc::<u32>(0);
    let zero = massively::vector::reduce(&exec, empty.slice(..), 0_u32, Add).unwrap();
    let values = exec.to_device(&[1_u32, 2, 4]);
    let offsets = exec.to_device(&[0_u32, 2, 3]);

    let scanned = ForEachSegment(ExclusiveScan(Add, zero.clone()))
        .run(
            &exec,
            SegmentIterator::new(values.slice(..), offsets.slice(..)),
        )
        .unwrap();
    assert_eq!(exec.to_host(scanned.values()).unwrap(), vec![0, 1, 0]);

    let reduced = ForEachSegment(Reduce(Add, zero))
        .run(
            &exec,
            SegmentIterator::new(values.slice(..), offsets.slice(..)),
        )
        .unwrap();
    assert_eq!(exec.to_host(&reduced).unwrap(), vec![3, 4]);
}

#[test]
fn segmented_flat_map_rebuilds_offsets_and_preserves_empty_segments() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let values = exec.to_device(&[2_u32, 0, 1, 3]);
    let offsets = exec.to_device(&[0_u32, 2, 2, 4]);

    let output = ForEachSegment(FlatMap(RepeatValue))
        .run(
            &exec,
            SegmentIterator::new(values.slice(..), offsets.slice(..)),
        )
        .unwrap();
    assert_validated_segmentation(&output);

    assert_eq!(
        exec.to_host(output.values()).unwrap(),
        vec![20, 21, 10, 30, 31, 32]
    );
    assert_eq!(
        exec.to_host(&output.offsets().offsets()).unwrap(),
        vec![0, 2, 2, 6]
    );
}

#[test]
fn segmentation_representations_are_interchangeable() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);

    let lengths = exec.to_device(&[1_u32, 2, 3]);
    let from_lengths = Segmentation::from_lengths(&exec, lengths.slice(..)).unwrap();
    assert_eq!(from_lengths.segment_count(), 3);
    assert_eq!(from_lengths.value_count(), 6);
    assert_eq!(
        exec.to_host(&from_lengths.offsets()).unwrap(),
        vec![0, 1, 3, 6]
    );
    assert_eq!(
        exec.to_host(&from_lengths.segment_ids(&exec).unwrap())
            .unwrap(),
        vec![0, 1, 1, 2, 2, 2]
    );
    assert_eq!(
        exec.to_host(&from_lengths.local_indices(&exec).unwrap())
            .unwrap(),
        vec![0, 0, 1, 0, 1, 2]
    );

    let ids = exec.to_device(&[0_u32, 1, 1, 2, 2, 2]);
    let segment_count = 3_u32;
    let from_ids = Segmentation::from_segment_ids(&exec, ids.slice(..), segment_count).unwrap();
    assert_eq!(exec.to_host(&from_ids.offsets()).unwrap(), vec![0, 1, 3, 6]);
    assert_eq!(
        exec.to_host(&from_ids.lengths(&exec).unwrap()).unwrap(),
        vec![1, 2, 3]
    );

    let offsets = exec.to_device(&[0_u32, 1, 3, 6]);
    let from_offsets = Segmentation::from_offsets(&exec, offsets.slice(..)).unwrap();
    massively::vector::fill(&exec, 9u32, offsets.slice_mut(..)).unwrap();
    assert_eq!(
        exec.to_host(&from_offsets.lengths(&exec).unwrap()).unwrap(),
        vec![1, 2, 3]
    );

    let from_lazy_lengths =
        Segmentation::from_lengths(&exec, lazy::constant(2u32).take(3)).unwrap();
    assert_eq!(
        exec.to_host(&from_lazy_lengths.offsets()).unwrap(),
        vec![0, 2, 4, 6]
    );
}

#[test]
fn segmentation_uses_selected_offsets_length() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let offsets = exec.to_device(&[0_u32, 1, 3, 6, 99, 99]);
    let flags = exec.to_device(&[1_u32, 1, 1, 1, 0, 0]);
    let selected = copy_where(&exec, offsets.slice(..), flags.slice(..)).unwrap();
    let segmentation = Segmentation::from_offsets(&exec, selected.slice(..)).unwrap();

    assert_eq!(segmentation.segment_count(), 3);
    assert_eq!(segmentation.value_count(), 6);
    assert_eq!(
        exec.to_host(&segmentation.offsets()).unwrap(),
        vec![0, 1, 3, 6]
    );
}

#[test]
fn segmentation_uses_selected_lengths_length() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let lengths = exec.to_device(&[1_u32, 2, 3, 99, 99]);
    let flags = exec.to_device(&[1_u32, 1, 1, 0, 0]);
    let selected = copy_where(&exec, lengths.slice(..), flags.slice(..)).unwrap();
    let segmentation = Segmentation::from_lengths(&exec, selected.slice(..)).unwrap();

    assert_eq!(segmentation.segment_count(), 3);
    assert_eq!(segmentation.value_count(), 6);
    assert_eq!(
        exec.to_host(&segmentation.offsets()).unwrap(),
        vec![0, 1, 3, 6]
    );
}

#[test]
fn segmentation_uses_selected_ids_length() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let ids = exec.to_device(&[0_u32, 1, 1, 2, 2, 2, 99, 99]);
    let flags = exec.to_device(&[1_u32, 1, 1, 1, 1, 1, 0, 0]);
    let selected = copy_where(&exec, ids.slice(..), flags.slice(..)).unwrap();
    let segmentation = Segmentation::from_segment_ids(&exec, selected.slice(..), 4).unwrap();

    assert_eq!(segmentation.segment_count(), 4);
    assert_eq!(segmentation.value_count(), 6);
    assert_eq!(
        exec.to_host(&segmentation.offsets()).unwrap(),
        vec![0, 1, 3, 6, 6]
    );
}

#[test]
fn segmentation_handles_empty_selected_representations() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[99_u32, 99, 99]);
    let flags = exec.to_device(&[0_u32, 0, 0]);
    let selected = copy_where(&exec, input.slice(..), flags.slice(..)).unwrap();

    assert!(matches!(
        Segmentation::from_offsets(&exec, selected.slice(..)),
        Err(massively::Error::InvalidSegmentation)
    ));
    let lengths = Segmentation::from_lengths(&exec, selected.slice(..)).unwrap();
    assert_eq!(lengths.segment_count(), 0);
    assert_eq!(lengths.value_count(), 0);
    assert_eq!(exec.to_host(&lengths.offsets()).unwrap(), vec![0]);

    let ids = Segmentation::from_segment_ids(&exec, selected.slice(..), 3).unwrap();
    assert_eq!(ids.segment_count(), 3);
    assert_eq!(ids.value_count(), 0);
    assert_eq!(exec.to_host(&ids.offsets()).unwrap(), vec![0, 0, 0, 0]);
}

#[test]
fn segmentation_preserves_empty_segments() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let lengths = exec.to_device(&[1_u32, 0, 2, 0]);
    let segmentation = Segmentation::from_lengths(&exec, lengths.slice(..)).unwrap();

    assert_eq!(
        exec.to_host(&segmentation.offsets()).unwrap(),
        vec![0, 1, 1, 3, 3]
    );
    assert_eq!(
        exec.to_host(&segmentation.segment_ids(&exec).unwrap())
            .unwrap(),
        vec![0, 2, 2]
    );

    let ids = exec.to_device(&[0_u32, 2, 2]);
    let round_trip = Segmentation::from_segment_ids(&exec, ids.slice(..), 4).unwrap();
    assert_eq!(
        exec.to_host(&round_trip.lengths(&exec).unwrap()).unwrap(),
        vec![1, 0, 2, 0]
    );

    let no_lengths = exec.to_device(&[] as &[u32]);
    let empty = Segmentation::from_lengths(&exec, no_lengths.slice(..)).unwrap();
    assert_eq!(empty.segment_count(), 0);
    assert_eq!(empty.value_count(), 0);
    assert_eq!(exec.to_host(&empty.offsets()).unwrap(), vec![0]);
    assert!(
        exec.to_host(&empty.segment_ids(&exec).unwrap())
            .unwrap()
            .is_empty()
    );
    assert!(
        exec.to_host(&empty.local_indices(&exec).unwrap())
            .unwrap()
            .is_empty()
    );

    let zero_offsets = exec.to_device(&[0_u32]);
    let empty_from_offsets = Segmentation::from_offsets(&exec, zero_offsets.slice(..)).unwrap();
    assert_eq!(empty_from_offsets.segment_count(), 0);
    assert_eq!(empty_from_offsets.value_count(), 0);
    assert_eq!(
        exec.to_host(&empty_from_offsets.offsets()).unwrap(),
        vec![0]
    );

    let empty_from_ids = Segmentation::from_segment_ids(&exec, no_lengths.slice(..), 0).unwrap();
    assert_eq!(empty_from_ids.segment_count(), 0);
    assert_eq!(empty_from_ids.value_count(), 0);
    assert_eq!(exec.to_host(&empty_from_ids.offsets()).unwrap(), vec![0]);

    let all_empty = Segmentation::from_segment_ids(&exec, no_lengths.slice(..), 3).unwrap();
    assert_eq!(
        exec.to_host(&all_empty.offsets()).unwrap(),
        vec![0, 0, 0, 0]
    );
    assert!(
        exec.to_host(&all_empty.local_indices(&exec).unwrap())
            .unwrap()
            .is_empty()
    );

    let leading_empty_ids = exec.to_device(&[2_u32]);
    let leading_empty =
        Segmentation::from_segment_ids(&exec, leading_empty_ids.slice(..), 4).unwrap();
    assert_eq!(
        exec.to_host(&leading_empty.lengths(&exec).unwrap())
            .unwrap(),
        vec![0, 0, 1, 0]
    );
    assert_eq!(
        exec.to_host(&leading_empty.segment_ids(&exec).unwrap())
            .unwrap(),
        vec![2]
    );
    assert_eq!(
        exec.to_host(&leading_empty.local_indices(&exec).unwrap())
            .unwrap(),
        vec![0]
    );
}

#[test]
fn segmentation_rejects_invalid_representations_and_overflow() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);

    for offsets in [&[][..], &[1_u32, 1][..], &[0_u32, 2, 1][..]] {
        let offsets = exec.to_device(offsets);
        assert!(matches!(
            Segmentation::from_offsets(&exec, offsets.slice(..)),
            Err(massively::Error::InvalidSegmentation)
        ));
    }

    let no_values = exec.to_device(&[] as &[u32]);
    let no_offsets = exec.to_device(&[] as &[u32]);
    let raw = SegmentIterator::new(no_values.slice(..), no_offsets.slice(..));
    assert!(matches!(
        vector_map(&exec, raw, SliceLength),
        Err(massively::Error::InvalidSegmentation)
    ));

    for ids in [&[0_u32, 2, 1][..], &[0_u32, 3][..]] {
        let ids = exec.to_device(ids);
        assert!(matches!(
            Segmentation::from_segment_ids(&exec, ids.slice(..), 3),
            Err(massively::Error::InvalidSegmentation)
        ));
    }

    let overflowing = exec.to_device(&[u32::MAX, 1]);
    assert!(matches!(
        Segmentation::from_lengths(&exec, overflowing.slice(..)),
        Err(massively::Error::LengthTooLarge { .. })
    ));

    let no_ids = exec.to_device(&[] as &[u32]);
    assert!(matches!(
        Segmentation::from_segment_ids(&exec, no_ids.slice(..), u32::MAX),
        Err(massively::Error::LengthTooLarge { .. })
    ));
}

#[test]
fn segmentation_applies_to_values_and_broadcasts_multicolumn_context() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let lengths = exec.to_device(&[1_u32, 2, 3]);
    let segmentation = Segmentation::from_lengths(&exec, lengths.slice(..)).unwrap();
    let values = exec.to_device(&[10_u32, 20, 21, 30, 31, 32]);

    let segments = segmentation.segments(values.slice(..)).unwrap();
    assert_validated_segmentation(&segments);
    let observed_lengths = vector_map(&exec, segments, SliceLength).unwrap();
    assert_eq!(exec.to_host(&observed_lengths).unwrap(), vec![1, 2, 3]);

    let mapped = ForEachSegment(Map(CastU64))
        .run(&exec, segmentation.segments(values.slice(..)).unwrap())
        .unwrap();
    assert_validated_segmentation(&mapped);

    let values_y = exec.to_device(&[100_u32, 200, 201, 300, 301, 302]);
    let pair_segments = segmentation
        .segments(zip2(values.slice(..), values_y.slice(..)))
        .unwrap();
    let pair_lengths = vector_map(&exec, pair_segments, PairSliceLength).unwrap();
    assert_eq!(exec.to_host(&pair_lengths).unwrap(), vec![1, 2, 3]);

    let short_values = exec.to_device(&[1_u32, 2]);
    assert!(matches!(
        segmentation.segments(short_values.slice(..)),
        Err(massively::Error::LengthMismatch { left: 2, right: 6 })
    ));

    let context_x = exec.to_device(&[7_u32, 8, 9]);
    let context_y = exec.to_device(&[70_u32, 80, 90]);
    let ids = segmentation.segment_ids(&exec).unwrap();
    let broadcast = vector_map(
        &exec,
        lazy::permute(
            zip2(context_x.slice(..), context_y.slice(..)),
            ids.slice(..),
        ),
        massively::op::Identity,
    )
    .unwrap();
    let (broadcast_x, broadcast_y) = MStorage::into_columns(broadcast);
    assert_eq!(exec.to_host(&broadcast_x).unwrap(), vec![7, 8, 8, 9, 9, 9]);
    assert_eq!(
        exec.to_host(&broadcast_y).unwrap(),
        vec![70, 80, 80, 90, 90, 90]
    );

    let other = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    assert!(matches!(
        segmentation.lengths(&other),
        Err(massively::Error::ForeignExecutor)
    ));
    assert!(matches!(
        segmentation.segment_ids(&other),
        Err(massively::Error::ForeignExecutor)
    ));
    assert!(matches!(
        segmentation.local_indices(&other),
        Err(massively::Error::ForeignExecutor)
    ));
    assert!(matches!(
        Segmentation::from_lengths(&other, lengths.slice(..)),
        Err(massively::Error::ForeignExecutor)
    ));
    assert!(matches!(
        Segmentation::from_offsets(&other, segmentation.offsets()),
        Err(massively::Error::ForeignExecutor)
    ));
    assert!(matches!(
        Segmentation::from_segment_ids(&other, ids.slice(..), 3),
        Err(massively::Error::ForeignExecutor)
    ));
}

#[test]
fn segmentation_context_is_composed_from_general_primitives() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let host_lengths = exec.to_device(&[2_u32, 0, 1]);
    let segmentation = Segmentation::from_lengths(&exec, host_lengths.slice(..)).unwrap();
    let values = exec.to_device(&[1_u32, 2, 3]);
    let contexts = exec.to_device(&[10_u32, 20, 30]);

    let ids = segmentation.segment_ids(&exec).unwrap();
    let entry_contexts = lazy::permute(contexts.slice(..), ids.slice(..));
    let decorated_values = zip2(values.slice(..), entry_contexts);
    let entry_totals = vector_map(&exec, decorated_values, AddU32Pair).unwrap();
    let segment_totals = ForEachSegment(Reduce(Add, 0u32))
        .run(
            &exec,
            segmentation.segments(entry_totals.slice(..)).unwrap(),
        )
        .unwrap();

    // Empty segments have no entry on which to broadcast. Handle a fixed-size
    // segment-level contribution in a separate parallel pass.
    let lengths = segmentation.lengths(&exec).unwrap();
    let empty_contributions = vector_map(
        &exec,
        zip2(lengths.slice(..), contexts.slice(..)),
        ContextForEmptySegment,
    )
    .unwrap();
    let with_empty_segments = vector_map(
        &exec,
        zip2(segment_totals.slice(..), empty_contributions.slice(..)),
        AddU32Pair,
    )
    .unwrap();
    assert_eq!(
        exec.to_host(&with_empty_segments).unwrap(),
        vec![23, 20, 33]
    );

    // Variable output from an empty segment is the same general FlatMap over
    // one context row per segment. Counting supplies singleton CSR offsets.
    let singleton_offsets = lazy::counting(0).take(segmentation.segment_count() + 1);
    let empty_outputs = ForEachSegment(FlatMap(EmptyContextExpand))
        .run(
            &exec,
            SegmentIterator::new(
                zip2(lengths.slice(..), contexts.slice(..)),
                singleton_offsets,
            ),
        )
        .unwrap();
    assert_eq!(exec.to_host(empty_outputs.values()).unwrap(), vec![20, 21]);
    assert_eq!(
        exec.to_host(&empty_outputs.offsets().offsets()).unwrap(),
        vec![0, 0, 2, 2]
    );

    let uniform_context = lazy::constant(5u32).take(segmentation.value_count());
    let with_uniform_context =
        vector_map(&exec, zip2(values.slice(..), uniform_context), AddU32Pair).unwrap();
    assert_eq!(exec.to_host(&with_uniform_context).unwrap(), vec![6, 7, 8]);
}

#[test]
fn cyclic_segment_context_is_an_entry_parallel_composition() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let lengths = exec.to_device(&[3_u32, 0, 2]);
    let segmentation = Segmentation::from_lengths(&exec, lengths.slice(..)).unwrap();
    let values = exec.to_device(&[10_u32, 11, 12, 20, 21]);
    let contexts = exec.to_device(&[100_u32, 200, 300]);

    let ids = segmentation.segment_ids(&exec).unwrap();
    let offsets = segmentation.offsets();
    let starts = lazy::permute(offsets.slice(..segmentation.segment_count()), ids.slice(..));
    let ends = lazy::permute(offsets.slice(1..), ids.slice(..));
    let positions = lazy::counting(0).take(segmentation.value_count());
    let predecessor_indices =
        vector_map(&exec, zip3(positions, starts, ends), CyclicPredecessorIndex).unwrap();

    let previous = lazy::permute(values.slice(..), predecessor_indices.slice(..));
    let entry_contexts = lazy::permute(contexts.slice(..), ids.slice(..));
    let output = vector_map(
        &exec,
        zip3(previous, values.slice(..), entry_contexts),
        AddU32Triple,
    )
    .unwrap();

    assert_eq!(
        exec.to_host(&output).unwrap(),
        vec![122, 121, 123, 341, 341]
    );
}

#[test]
fn segmented_boolean_results_are_flag_iterators() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let values = exec.to_device(&[2_u32, 4, 3, 6]);
    let offsets = exec.to_device(&[0_u32, 2, 4]);
    let flags = ForEachSegment(AllOf(Even))
        .run(
            &exec,
            SegmentIterator::new(values.slice(..), offsets.slice(..)),
        )
        .unwrap();

    fn assert_flag_iter<R: Runtime, Input: MIter<R, Item = massively::MFlag>>(_input: &Input) {}
    assert_flag_iter::<WgpuRuntime, _>(&flags.slice(..));
    assert_eq!(
        exec.to_host(&flags).unwrap(),
        vec![
            massively::flag::from_bool(true),
            massively::flag::from_bool(false)
        ]
    );

    let input = exec.to_device(&[10_u32, 20]);
    let selected = copy_where(&exec, input.slice(..), flags.slice(..)).unwrap();
    assert_eq!(exec.to_host(&selected).unwrap(), vec![10]);
}

#[test]
fn segmented_predicate_summaries_normalize_nonzero_flags() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let values = exec.to_device(&[7_u32, 3, 0]);
    let offsets = exec.to_device(&[0_u32, 2, 3]);
    let flags = ForEachSegment(AllOf(RawFlag))
        .run(
            &exec,
            SegmentIterator::new(values.slice(..), offsets.slice(..)),
        )
        .unwrap();

    assert_eq!(
        exec.to_host(&flags).unwrap(),
        vec![
            massively::flag::from_bool(true),
            massively::flag::from_bool(false)
        ]
    );
}

#[test]
fn segmented_take_find_if_and_adjacent_find_cover_empty_segments() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    // [], [2, 4, 6], [1, 1, 3], [5], [7, 8, 8]
    let values = exec.to_device(&[2_u32, 4, 6, 1, 1, 3, 5, 7, 8, 8]);
    let offsets = exec.to_device(&[0_u32, 0, 3, 6, 7, 10]);

    let taken = ForEachSegment(Take(2))
        .run(
            &exec,
            SegmentIterator::new(values.slice(..), offsets.slice(..)),
        )
        .unwrap();
    assert_eq!(
        exec.to_host(taken.values()).unwrap(),
        vec![2, 4, 1, 1, 5, 7, 8]
    );
    assert_eq!(
        exec.to_host(&taken.offsets().offsets()).unwrap(),
        vec![0, 0, 2, 4, 5, 7]
    );

    let taken_zero = ForEachSegment(Take(0))
        .run(
            &exec,
            SegmentIterator::new(values.slice(..), offsets.slice(..)),
        )
        .unwrap();
    assert!(exec.to_host(taken_zero.values()).unwrap().is_empty());
    assert_eq!(
        exec.to_host(&taken_zero.offsets().offsets()).unwrap(),
        vec![0, 0, 0, 0, 0, 0]
    );

    let found = ForEachSegment(FindIf(Even))
        .run(
            &exec,
            SegmentIterator::new(values.slice(..), offsets.slice(..)),
        )
        .unwrap();
    assert_eq!(exec.to_host(&found).unwrap(), vec![0, 0, 3, 1, 1]);

    let adjacent = ForEachSegment(AdjacentFind(Equal))
        .run(
            &exec,
            SegmentIterator::new(values.slice(..), offsets.slice(..)),
        )
        .unwrap();
    assert_eq!(exec.to_host(&adjacent).unwrap(), vec![0, 3, 0, 1, 1]);
}

#[test]
fn segment_iterator_is_an_miter_of_shared_read_only_segments() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    // [], [a], [aa], [ab], [b]
    let values = exec.to_device(&[
        b'a' as u32,
        b'a' as u32,
        b'a' as u32,
        b'a' as u32,
        b'b' as u32,
        b'b' as u32,
    ]);
    let offsets = exec.to_device(&[0_u32, 0, 1, 3, 5, 6]);
    let rows = SegmentIterator::new(values.slice(..), offsets.slice(..));

    fn assert_item<R: Runtime, Input: MIter<R, Item = Segment<u32>>>(_input: &Input) {}
    assert_item::<WgpuRuntime, _>(&rows);
    let lengths = vector_map(&exec, rows.clone(), SliceLength).unwrap();
    assert_eq!(exec.to_host(&lengths).unwrap(), vec![0, 1, 2, 2, 1]);
    assert_eq!(
        is_sorted(&exec, rows.clone(), LexicographicalBytes).unwrap(),
        massively::flag::from_bool(true)
    );

    let lazy_rows = SegmentIterator::new(
        massively::lazy::map(values.slice(..), WrapCode),
        offsets.slice(..),
    );
    fn assert_lazy_item<R: Runtime, Input: MIter<R, Item = Segment<Code>>>(_input: &Input) {}
    assert_lazy_item::<WgpuRuntime, _>(&lazy_rows);
    assert_eq!(
        is_sorted(&exec, lazy_rows, LexicographicalCodes).unwrap(),
        massively::flag::from_bool(true)
    );

    let middle = rows.slice(1..4);
    assert_eq!(
        is_sorted(&exec, middle, LexicographicalBytes).unwrap(),
        massively::flag::from_bool(true)
    );

    // [ab], [aa]
    let unsorted_values = exec.to_device(&[b'a' as u32, b'b' as u32, b'a' as u32, b'a' as u32]);
    let unsorted_offsets = exec.to_device(&[0_u32, 2, 4]);
    let unsorted = SegmentIterator::new(unsorted_values.slice(..), unsorted_offsets.slice(..));
    assert_eq!(
        is_sorted(&exec, unsorted, LexicographicalBytes).unwrap(),
        massively::flag::from_bool(false)
    );

    // The two operands use independent backing expressions and absolute offsets.
    let left_values = exec.to_device(&[1_u32, 2, 3]);
    let left_offsets = exec.to_device(&[0_u32, 1, 3]);
    let right_values = exec.to_device(&[99_u32, 1, 2, 3]);
    let right_offsets = exec.to_device(&[1_u32, 2, 4]);
    let left = SegmentIterator::new(left_values.slice(..), left_offsets.slice(..));
    let right = SegmentIterator::new(
        massively::lazy::map(right_values.slice(..), massively::op::Identity),
        right_offsets.slice(..),
    );
    assert_eq!(
        equal(&exec, left, right, SlicesEqual).unwrap(),
        massively::flag::from_bool(true)
    );
}

#[test]
fn segment_iterators_zip_into_a_flat_read_only_row() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let left_values = exec.to_device(&[1_u32, 2, 3]);
    let left_offsets = exec.to_device(&[0_u32, 1, 3]);
    let right_values = exec.to_device(&[10_u32, 20, 30, 40]);
    let right_offsets = exec.to_device(&[0_u32, 3, 4]);

    let rows = zip2(
        SegmentIterator::new(left_values.slice(..), left_offsets.slice(..)),
        SegmentIterator::new(right_values.slice(..), right_offsets.slice(..)),
    );

    fn assert_item<R: Runtime, Input: MIter<R, Item = (Segment<u32>, Segment<u32>)>>(
        _input: &Input,
    ) {
    }
    assert_item::<WgpuRuntime, _>(&rows);

    let lengths = vector_map(&exec, rows, SegmentPairLengths).unwrap();
    assert_eq!(exec.to_host(&lengths).unwrap(), vec![13, 21]);
}
