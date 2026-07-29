//! Device kernels for segmented operations.

use super::*;

#[cubecl::cube(launch_unchecked, explicit_define)]
pub(super) fn prepare_segment_control(flags: &[u32], len: &[u32], control: &mut [u32]) {
    let index = ABSOLUTE_POS as usize;
    let metadata = control.len() - 1usize;
    if index == 0usize {
        control[metadata] = len[0];
    }
    if index < metadata {
        control[index] = if index < len[0] as usize {
            flags[index]
        } else {
            0u32
        };
    }
}

/// Builds the per-block segment-continuation mask used by the summary and
/// prefix stages.  A zero means that no segment head has appeared between the
/// beginning of the block and this item.
#[cubecl::cube(launch_unchecked, explicit_define)]
pub(super) fn segment_continuation_kernel(heads: &[u32], continuation: &mut [u32]) {
    let unit = UNIT_POS as usize;
    let block = CUBE_POS as usize;
    let global = block * BLOCK_SIZE as usize + unit;
    let metadata = heads.len() - 1usize;
    let logical_len = heads[metadata] as usize;
    let value = RuntimeCell::<u32>::new(if global < logical_len {
        heads[global]
    } else {
        0u32
    });

    let offset = RuntimeCell::<u32>::new(1u32);
    while offset.read() < PLANE_DIM {
        let left = plane_shuffle_up(value.read(), offset.read());
        if UNIT_POS_PLANE >= offset.read() {
            value.store(left | value.read());
        }
        offset.store(offset.read() * 2u32);
    }

    let mut plane_values = Shared::<[u32]>::new_slice(BLOCK_SIZE as usize);
    if UNIT_POS_PLANE + 1u32 == PLANE_DIM {
        plane_values[PLANE_POS as usize] = value.read();
    }
    sync_cube();

    if unit == 0usize {
        let plane_count = CUBE_DIM.div_ceil(PLANE_DIM);
        let prefix = RuntimeCell::<u32>::new(plane_values[0]);
        let plane = RuntimeCell::<u32>::new(1u32);
        while plane.read() < plane_count {
            let index = plane.read() as usize;
            prefix.store(prefix.read() | plane_values[index]);
            plane_values[index] = prefix.read();
            plane.store(plane.read() + 1u32);
        }
    }
    sync_cube();

    if PLANE_POS > 0u32 {
        value.store(plane_values[PLANE_POS as usize - 1usize] | value.read());
    }
    if global < logical_len {
        continuation[global] = value.read();
    }
    if global == 0usize {
        continuation[metadata] = logical_len as u32;
    }
}

#[cubecl::cube(launch_unchecked, explicit_define)]
#[allow(clippy::too_many_arguments)]
pub(super) fn segmented_local_scan_a13<
    Item: CubeType + Send + Sync + 'static,
    L0: CubePrimitive + cubecl::frontend::Scalar,
    L1: CubePrimitive + cubecl::frontend::Scalar,
    L2: CubePrimitive + cubecl::frontend::Scalar,
    L3: CubePrimitive + cubecl::frontend::Scalar,
    L4: CubePrimitive + cubecl::frontend::Scalar,
    L5: CubePrimitive + cubecl::frontend::Scalar,
    L6: CubePrimitive + cubecl::frontend::Scalar,
    L7: CubePrimitive + cubecl::frontend::Scalar,
    L8: CubePrimitive + cubecl::frontend::Scalar,
    L9: CubePrimitive + cubecl::frontend::Scalar,
    L10: CubePrimitive + cubecl::frontend::Scalar,
    L11: CubePrimitive + cubecl::frontend::Scalar,
    L12: CubePrimitive + cubecl::frontend::Scalar,
    O0: CubePrimitive + cubecl::frontend::Scalar,
    O1: CubePrimitive + cubecl::frontend::Scalar,
    O2: CubePrimitive + cubecl::frontend::Scalar,
    O3: CubePrimitive + cubecl::frontend::Scalar,
    O4: CubePrimitive + cubecl::frontend::Scalar,
    O5: CubePrimitive + cubecl::frontend::Scalar,
    O6: CubePrimitive + cubecl::frontend::Scalar,
    O7: CubePrimitive + cubecl::frontend::Scalar,
    O8: CubePrimitive + cubecl::frontend::Scalar,
    O9: CubePrimitive + cubecl::frontend::Scalar,
    O10: CubePrimitive + cubecl::frontend::Scalar,
    O11: CubePrimitive + cubecl::frontend::Scalar,
    Leaves: SharedLeaves
        + MutableLeaves
        + PlaneShuffleLeaves
        + StorePadded12<
            O0 = O0,
            O1 = O1,
            O2 = O2,
            O3 = O3,
            O4 = O4,
            O5 = O5,
            O6 = O6,
            O7 = O7,
            O8 = O8,
            O9 = O9,
            O10 = O10,
            O11 = O11,
        > + Send
        + Sync
        + 'static,
    Expr: Eval13<Item, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
    Op: ReductionOp<Item>,
>(
    input0: &[L0],
    input1: &[L1],
    input2: &[L2],
    input3: &[L3],
    input4: &[L4],
    input5: &[L5],
    input6: &[L6],
    input7: &[L7],
    input8: &[L8],
    input9: &[L9],
    input10: &[L10],
    input11: &[L11],
    input12: &[L12],
    control: &[u32],
    input_offsets: &[u32],
    output0: &mut [O0],
    output1: &mut [O1],
    output2: &mut [O2],
    output3: &mut [O3],
    output4: &mut [O4],
    output5: &mut [O5],
    output6: &mut [O6],
    output7: &mut [O7],
    output8: &mut [O8],
    output9: &mut [O9],
    output10: &mut [O10],
    output11: &mut [O11],
    output_offsets: &[u32],
    #[comptime] plane_capacity: usize,
) {
    let unit = UNIT_POS as usize;
    let block = CUBE_POS as usize;
    let cube_dim = BLOCK_SIZE as usize;
    let global = block * cube_dim + unit;
    let logical_len = control[control.len() - 1usize] as usize;
    if block * cube_dim >= logical_len {
        terminate!();
    }
    let safe_global = if global < logical_len { global } else { 0usize };
    let cells = Leaves::into_cells(Layout::decompose(Expr::eval13(
        input0,
        input1,
        input2,
        input3,
        input4,
        input5,
        input6,
        input7,
        input8,
        input9,
        input10,
        input11,
        input12,
        input_offsets,
        safe_global,
    )));
    let segment = RuntimeCell::<u32>::new(if global < logical_len {
        control[global]
    } else {
        0u32
    });
    let valid = RuntimeCell::<u32>::new(if global < logical_len { 1u32 } else { 0u32 });

    let offset = RuntimeCell::<u32>::new(1u32);
    while offset.read() < PLANE_DIM {
        let left_cells = Leaves::into_cells(Leaves::shuffle_leaves_up(
            Leaves::read(&cells),
            offset.read(),
        ));
        let left_segment = plane_shuffle_up(segment.read(), offset.read());
        let left_valid = plane_shuffle_up(valid.read(), offset.read());
        if UNIT_POS_PLANE >= offset.read() && left_valid != 0u32 {
            if valid.read() != 0u32 {
                if segment.read() == 0u32 {
                    let combined = Layout::decompose(Op::apply(
                        Layout::recompose(Leaves::read(&left_cells)),
                        Layout::recompose(Leaves::read(&cells)),
                    ));
                    Leaves::store(&cells, combined);
                }
                segment.store(left_segment | segment.read());
            } else {
                Leaves::store(&cells, Leaves::read(&left_cells));
                segment.store(left_segment);
                valid.store(1u32);
            }
        }
        offset.store(offset.read() * 2u32);
    }

    let mut shared = Leaves::new_shared(plane_capacity);
    let mut shared_segments = Shared::<[u32]>::new_slice(plane_capacity);
    let mut shared_valid = Shared::<[u32]>::new_slice(plane_capacity);
    if UNIT_POS_PLANE + 1u32 == PLANE_DIM {
        Leaves::read(&cells).store_shared(&mut shared, PLANE_POS as usize);
        shared_segments[PLANE_POS as usize] = segment.read();
        shared_valid[PLANE_POS as usize] = valid.read();
    }
    sync_cube();

    if unit == 0usize {
        let plane_count = CUBE_DIM.div_ceil(PLANE_DIM);
        let plane_cells = Leaves::into_cells(Leaves::load_shared(&shared, 0usize));
        let plane_segment = RuntimeCell::<u32>::new(shared_segments[0]);
        let plane_valid = RuntimeCell::<u32>::new(shared_valid[0]);
        let plane = RuntimeCell::<u32>::new(1u32);
        while plane.read() < plane_count {
            let index = plane.read() as usize;
            if shared_valid[index] != 0u32 {
                if plane_valid.read() != 0u32 {
                    if shared_segments[index] == 0u32 {
                        let combined = Layout::decompose(Op::apply(
                            Layout::recompose(Leaves::read(&plane_cells)),
                            Layout::recompose(Leaves::load_shared(&shared, index)),
                        ));
                        Leaves::store(&plane_cells, combined);
                    } else {
                        Leaves::store(&plane_cells, Leaves::load_shared(&shared, index));
                    }
                    plane_segment.store(plane_segment.read() | shared_segments[index]);
                } else {
                    Leaves::store(&plane_cells, Leaves::load_shared(&shared, index));
                    plane_segment.store(shared_segments[index]);
                    plane_valid.store(1u32);
                }
            }
            Leaves::read(&plane_cells).store_shared(&mut shared, index);
            shared_segments[index] = plane_segment.read();
            plane.store(plane.read() + 1u32);
        }
    }
    sync_cube();

    if PLANE_POS > 0u32 && valid.read() != 0u32 {
        let prefix_index = PLANE_POS as usize - 1usize;
        if segment.read() == 0u32 {
            let combined = Layout::decompose(Op::apply(
                Layout::recompose(Leaves::load_shared(&shared, prefix_index)),
                Layout::recompose(Leaves::read(&cells)),
            ));
            Leaves::store(&cells, combined);
        }
        segment.store(shared_segments[prefix_index] | segment.read());
    }

    if global < logical_len {
        Leaves::read(&cells).store_padded(
            output0,
            output1,
            output2,
            output3,
            output4,
            output5,
            output6,
            output7,
            output8,
            output9,
            output10,
            output11,
            output_offsets,
            global,
        );
    }
}

#[cubecl::cube(launch_unchecked, explicit_define)]
#[allow(clippy::too_many_arguments)]
pub(super) fn summarize_segment_blocks_padded12<
    O0: CubePrimitive + cubecl::frontend::Scalar,
    O1: CubePrimitive + cubecl::frontend::Scalar,
    O2: CubePrimitive + cubecl::frontend::Scalar,
    O3: CubePrimitive + cubecl::frontend::Scalar,
    O4: CubePrimitive + cubecl::frontend::Scalar,
    O5: CubePrimitive + cubecl::frontend::Scalar,
    O6: CubePrimitive + cubecl::frontend::Scalar,
    O7: CubePrimitive + cubecl::frontend::Scalar,
    O8: CubePrimitive + cubecl::frontend::Scalar,
    O9: CubePrimitive + cubecl::frontend::Scalar,
    O10: CubePrimitive + cubecl::frontend::Scalar,
    O11: CubePrimitive + cubecl::frontend::Scalar,
    Leaves: LoadPadded12<
            O0 = O0,
            O1 = O1,
            O2 = O2,
            O3 = O3,
            O4 = O4,
            O5 = O5,
            O6 = O6,
            O7 = O7,
            O8 = O8,
            O9 = O9,
            O10 = O10,
            O11 = O11,
        > + StorePadded12<
            O0 = O0,
            O1 = O1,
            O2 = O2,
            O3 = O3,
            O4 = O4,
            O5 = O5,
            O6 = O6,
            O7 = O7,
            O8 = O8,
            O9 = O9,
            O10 = O10,
            O11 = O11,
        > + Send
        + Sync
        + 'static,
>(
    input0: &[O0],
    input1: &[O1],
    input2: &[O2],
    input3: &[O3],
    input4: &[O4],
    input5: &[O5],
    input6: &[O6],
    input7: &[O7],
    input8: &[O8],
    input9: &[O9],
    input10: &[O10],
    input11: &[O11],
    input_offsets: &[u32],
    control: &[u32],
    sum0: &mut [O0],
    sum1: &mut [O1],
    sum2: &mut [O2],
    sum3: &mut [O3],
    sum4: &mut [O4],
    sum5: &mut [O5],
    sum6: &mut [O6],
    sum7: &mut [O7],
    sum8: &mut [O8],
    sum9: &mut [O9],
    sum10: &mut [O10],
    sum11: &mut [O11],
    sum_offsets: &[u32],
    block_flags: &mut [u32],
) {
    let block = CUBE_POS as usize;
    let logical_len = control[control.len() - 1usize] as usize;
    let block_start = block * BLOCK_SIZE as usize;
    if UNIT_POS == 0u32 && block_start < logical_len {
        let block_end = usize::min(block_start + BLOCK_SIZE as usize, logical_len);
        let last = block_end - 1usize;
        Leaves::load_padded(
            input0,
            input1,
            input2,
            input3,
            input4,
            input5,
            input6,
            input7,
            input8,
            input9,
            input10,
            input11,
            input_offsets,
            last,
        )
        .store_padded(
            sum0,
            sum1,
            sum2,
            sum3,
            sum4,
            sum5,
            sum6,
            sum7,
            sum8,
            sum9,
            sum10,
            sum11,
            sum_offsets,
            block,
        );
        block_flags[block] = control[last];
    }
}
#[cubecl::cube(launch_unchecked, explicit_define)]
#[allow(clippy::too_many_arguments)]
pub(super) fn segmented_prefix_padded12<
    Item: CubeType + Send + Sync + 'static,
    O0: CubePrimitive + cubecl::frontend::Scalar,
    O1: CubePrimitive + cubecl::frontend::Scalar,
    O2: CubePrimitive + cubecl::frontend::Scalar,
    O3: CubePrimitive + cubecl::frontend::Scalar,
    O4: CubePrimitive + cubecl::frontend::Scalar,
    O5: CubePrimitive + cubecl::frontend::Scalar,
    O6: CubePrimitive + cubecl::frontend::Scalar,
    O7: CubePrimitive + cubecl::frontend::Scalar,
    O8: CubePrimitive + cubecl::frontend::Scalar,
    O9: CubePrimitive + cubecl::frontend::Scalar,
    O10: CubePrimitive + cubecl::frontend::Scalar,
    O11: CubePrimitive + cubecl::frontend::Scalar,
    Leaves: LoadPadded12<
            O0 = O0,
            O1 = O1,
            O2 = O2,
            O3 = O3,
            O4 = O4,
            O5 = O5,
            O6 = O6,
            O7 = O7,
            O8 = O8,
            O9 = O9,
            O10 = O10,
            O11 = O11,
        > + LoadMutPadded12
        + Send
        + Sync
        + 'static,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
    Op: ReductionOp<Item>,
>(
    prefix0: &[O0],
    prefix1: &[O1],
    prefix2: &[O2],
    prefix3: &[O3],
    prefix4: &[O4],
    prefix5: &[O5],
    prefix6: &[O6],
    prefix7: &[O7],
    prefix8: &[O8],
    prefix9: &[O9],
    prefix10: &[O10],
    prefix11: &[O11],
    control: &[u32],
    prefix_offsets: &[u32],
    output_offsets: &[u32],
    output0: &mut [O0],
    output1: &mut [O1],
    output2: &mut [O2],
    output3: &mut [O3],
    output4: &mut [O4],
    output5: &mut [O5],
    output6: &mut [O6],
    output7: &mut [O7],
    output8: &mut [O8],
    output9: &mut [O9],
    output10: &mut [O10],
    output11: &mut [O11],
) {
    let block = CUBE_POS as usize;
    let index = block * BLOCK_SIZE as usize + UNIT_POS as usize;
    let logical_len = control[control.len() - 1usize] as usize;
    if block > 0usize && index < logical_len && control[index] == 0u32 {
        let prefix = Layout::recompose(Leaves::load_padded(
            prefix0,
            prefix1,
            prefix2,
            prefix3,
            prefix4,
            prefix5,
            prefix6,
            prefix7,
            prefix8,
            prefix9,
            prefix10,
            prefix11,
            prefix_offsets,
            block - 1usize,
        ));
        let current = Layout::recompose(Leaves::load_mut_padded(
            output0,
            output1,
            output2,
            output3,
            output4,
            output5,
            output6,
            output7,
            output8,
            output9,
            output10,
            output11,
            output_offsets,
            index,
        ));
        Layout::decompose(Op::apply(prefix, current)).store_padded(
            output0,
            output1,
            output2,
            output3,
            output4,
            output5,
            output6,
            output7,
            output8,
            output9,
            output10,
            output11,
            output_offsets,
            index,
        );
    }
}

#[cubecl::cube(launch_unchecked, explicit_define)]
#[allow(clippy::too_many_arguments)]
pub(super) fn segmented_exclusive_padded12<
    Item: CubeType + Send + Sync + 'static,
    O0: CubePrimitive + cubecl::frontend::Scalar,
    O1: CubePrimitive + cubecl::frontend::Scalar,
    O2: CubePrimitive + cubecl::frontend::Scalar,
    O3: CubePrimitive + cubecl::frontend::Scalar,
    O4: CubePrimitive + cubecl::frontend::Scalar,
    O5: CubePrimitive + cubecl::frontend::Scalar,
    O6: CubePrimitive + cubecl::frontend::Scalar,
    O7: CubePrimitive + cubecl::frontend::Scalar,
    O8: CubePrimitive + cubecl::frontend::Scalar,
    O9: CubePrimitive + cubecl::frontend::Scalar,
    O10: CubePrimitive + cubecl::frontend::Scalar,
    O11: CubePrimitive + cubecl::frontend::Scalar,
    Leaves: LoadPadded12<
            O0 = O0,
            O1 = O1,
            O2 = O2,
            O3 = O3,
            O4 = O4,
            O5 = O5,
            O6 = O6,
            O7 = O7,
            O8 = O8,
            O9 = O9,
            O10 = O10,
            O11 = O11,
        > + Send
        + Sync
        + 'static,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
    Op: ReductionOp<Item>,
>(
    seeded0: &[O0],
    seeded1: &[O1],
    seeded2: &[O2],
    seeded3: &[O3],
    seeded4: &[O4],
    seeded5: &[O5],
    seeded6: &[O6],
    seeded7: &[O7],
    seeded8: &[O8],
    seeded9: &[O9],
    seeded10: &[O10],
    seeded11: &[O11],
    flags: &[u32],
    len: &[u32],
    seeded_offsets: &[u32],
    output_offsets: &[u32],
    output0: &mut [O0],
    output1: &mut [O1],
    output2: &mut [O2],
    output3: &mut [O3],
    output4: &mut [O4],
    output5: &mut [O5],
    output6: &mut [O6],
    output7: &mut [O7],
    output8: &mut [O8],
    output9: &mut [O9],
    output10: &mut [O10],
    output11: &mut [O11],
) {
    let index = ABSOLUTE_POS as usize;
    if index < len[0] as usize {
        if index == 0usize || flags[index] != 0u32 {
            Leaves::load_padded(
                seeded0,
                seeded1,
                seeded2,
                seeded3,
                seeded4,
                seeded5,
                seeded6,
                seeded7,
                seeded8,
                seeded9,
                seeded10,
                seeded11,
                seeded_offsets,
                0usize,
            )
            .store_padded(
                output0,
                output1,
                output2,
                output3,
                output4,
                output5,
                output6,
                output7,
                output8,
                output9,
                output10,
                output11,
                output_offsets,
                index,
            );
        } else {
            Layout::decompose(Op::apply(
                Layout::recompose(Leaves::load_padded(
                    seeded0,
                    seeded1,
                    seeded2,
                    seeded3,
                    seeded4,
                    seeded5,
                    seeded6,
                    seeded7,
                    seeded8,
                    seeded9,
                    seeded10,
                    seeded11,
                    seeded_offsets,
                    0usize,
                )),
                Layout::recompose(Leaves::load_padded(
                    seeded0,
                    seeded1,
                    seeded2,
                    seeded3,
                    seeded4,
                    seeded5,
                    seeded6,
                    seeded7,
                    seeded8,
                    seeded9,
                    seeded10,
                    seeded11,
                    seeded_offsets,
                    index,
                )),
            ))
            .store_padded(
                output0,
                output1,
                output2,
                output3,
                output4,
                output5,
                output6,
                output7,
                output8,
                output9,
                output10,
                output11,
                output_offsets,
                index,
            );
        }
    }
}

#[cubecl::cube(launch_unchecked, explicit_define)]
pub(super) fn resolve_segment_tails(
    positions: &[u32],
    count: &[u32],
    source_len: &[u32],
    resolved: &mut [u32],
) {
    let source_position = ABSOLUTE_POS as usize;
    if source_position < source_len[0] as usize {
        let previous = if source_position == 0usize {
            0u32
        } else {
            positions[source_position - 1usize]
        };
        let rank = positions[source_position];
        // A new head closes the preceding segment.  `source_position` is
        // exactly tail + 1, which is also its index in the seeded buffer.
        if rank != previous && rank > 1u32 {
            let previous_rank = (rank - 2u32) as usize;
            if previous_rank < resolved.len() {
                resolved[previous_rank] = source_position as u32;
            }
        }
        // The final source row closes the final segment.
        if source_position + 1usize == source_len[0] as usize && rank != 0u32 {
            let final_rank = (rank - 1u32) as usize;
            if final_rank < resolved.len() && final_rank < count[0] as usize {
                resolved[final_rank] = source_len[0];
            }
        }
    }
}

#[cubecl::cube(launch_unchecked, explicit_define)]
#[allow(clippy::too_many_arguments)]
pub(super) fn segmented_reduce_selected_padded12<
    Item: CubeType + Send + Sync + 'static,
    O0: CubePrimitive + cubecl::frontend::Scalar,
    O1: CubePrimitive + cubecl::frontend::Scalar,
    O2: CubePrimitive + cubecl::frontend::Scalar,
    O3: CubePrimitive + cubecl::frontend::Scalar,
    O4: CubePrimitive + cubecl::frontend::Scalar,
    O5: CubePrimitive + cubecl::frontend::Scalar,
    O6: CubePrimitive + cubecl::frontend::Scalar,
    O7: CubePrimitive + cubecl::frontend::Scalar,
    O8: CubePrimitive + cubecl::frontend::Scalar,
    O9: CubePrimitive + cubecl::frontend::Scalar,
    O10: CubePrimitive + cubecl::frontend::Scalar,
    O11: CubePrimitive + cubecl::frontend::Scalar,
    Leaves: LoadPadded12<
            O0 = O0,
            O1 = O1,
            O2 = O2,
            O3 = O3,
            O4 = O4,
            O5 = O5,
            O6 = O6,
            O7 = O7,
            O8 = O8,
            O9 = O9,
            O10 = O10,
            O11 = O11,
        > + StorePadded12<
            O0 = O0,
            O1 = O1,
            O2 = O2,
            O3 = O3,
            O4 = O4,
            O5 = O5,
            O6 = O6,
            O7 = O7,
            O8 = O8,
            O9 = O9,
            O10 = O10,
            O11 = O11,
        > + Send
        + Sync
        + 'static,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
    Op: ReductionOp<Item>,
>(
    seeded0: &[O0],
    seeded1: &[O1],
    seeded2: &[O2],
    seeded3: &[O3],
    seeded4: &[O4],
    seeded5: &[O5],
    seeded6: &[O6],
    seeded7: &[O7],
    seeded8: &[O8],
    seeded9: &[O9],
    seeded10: &[O10],
    seeded11: &[O11],
    resolved: &[u32],
    count: &[u32],
    seeded_offsets: &[u32],
    output_offsets: &[u32],
    output0: &mut [O0],
    output1: &mut [O1],
    output2: &mut [O2],
    output3: &mut [O3],
    output4: &mut [O4],
    output5: &mut [O5],
    output6: &mut [O6],
    output7: &mut [O7],
    output8: &mut [O8],
    output9: &mut [O9],
    output10: &mut [O10],
    output11: &mut [O11],
) {
    let rank = ABSOLUTE_POS as usize;
    if rank < count[0] as usize {
        let initial = Layout::recompose(Leaves::load_padded(
            seeded0,
            seeded1,
            seeded2,
            seeded3,
            seeded4,
            seeded5,
            seeded6,
            seeded7,
            seeded8,
            seeded9,
            seeded10,
            seeded11,
            seeded_offsets,
            0usize,
        ));
        let value = Layout::recompose(Leaves::load_padded(
            seeded0,
            seeded1,
            seeded2,
            seeded3,
            seeded4,
            seeded5,
            seeded6,
            seeded7,
            seeded8,
            seeded9,
            seeded10,
            seeded11,
            seeded_offsets,
            resolved[rank] as usize,
        ));
        Layout::decompose(Op::apply(initial, value)).store_padded(
            output0,
            output1,
            output2,
            output3,
            output4,
            output5,
            output6,
            output7,
            output8,
            output9,
            output10,
            output11,
            output_offsets,
            rank,
        );
    }
}
