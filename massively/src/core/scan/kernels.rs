//! Device kernels for scan operations.

use super::*;

#[cubecl::cube(launch_unchecked, explicit_define)]
pub(super) fn u32_block_inclusive_scan_kernel(input: &[u32], len: &[u32], output: &mut [u32]) {
    let unit = UNIT_POS as usize;
    let cube_dim = BLOCK_SIZE as usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    let logical_len = len[0] as usize;
    let value = if global < logical_len {
        input[global]
    } else {
        0u32
    };
    let (prefix, _) = crate::core::collective::block_exclusive_sum(value);
    if global < logical_len {
        output[global] = prefix + value;
    }
}

/// Selects the final valid item of every fixed-size block.  This turns a
/// block-local inclusive scan into the reduction input for the next level.
#[cubecl::cube(launch_unchecked, explicit_define)]
pub(super) fn u32_block_tails_kernel(input: &[u32], len: &[u32], block_tails: &mut [u32]) {
    let block = ABSOLUTE_POS as usize;
    if block < block_tails.len() {
        let start = block * BLOCK_SIZE as usize;
        let logical_len = len[0] as usize;
        block_tails[block] = if start < logical_len {
            let end = start + usize::min(BLOCK_SIZE as usize, logical_len - start);
            input[end - 1usize]
        } else {
            0u32
        };
    }
}

#[cubecl::cube(launch_unchecked, explicit_define)]
pub(super) fn u32_add_block_prefix_kernel(block_prefixes: &[u32], len: &[u32], output: &mut [u32]) {
    let block = CUBE_POS as usize;
    let global = block * BLOCK_SIZE as usize + UNIT_POS as usize;
    if block > 0usize && global < len[0] as usize {
        output[global] += block_prefixes[block - 1usize];
    }
}

#[cubecl::cube(launch_unchecked)]
pub(super) fn copy_last_kernel(input: &[u32], len: &[u32], output: &mut [u32]) {
    if ABSOLUTE_POS == 0 {
        output[0] = if len[0] == 0u32 {
            0u32
        } else {
            input[len[0] as usize - 1usize]
        };
    }
}

/// Computes an order-preserving inclusive scan of a contiguous block.
/// Invalid tail lanes may hold any value: prefixes only depend on earlier
/// lanes, and callers consume only the valid prefix of each block.
#[cubecl::cube]
fn block_inclusive_scan<Item, Leaves, Layout, Op>(
    value: Item,
    #[comptime] plane_capacity: usize,
) -> Leaves
where
    Item: CubeType + Send + Sync + 'static,
    Leaves: SharedLeaves + MutableLeaves + PlaneShuffleLeaves + Send + Sync + 'static,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
    Op: ReductionOp<Item>,
{
    let mut shared = Leaves::new_shared(plane_capacity);
    let cells = Leaves::into_cells(Layout::decompose(value));
    let offset = RuntimeCell::<u32>::new(1u32);
    while offset.read() < PLANE_DIM {
        let left = Leaves::shuffle_leaves_up(Leaves::read(&cells), offset.read());
        if UNIT_POS_PLANE >= offset.read() {
            Leaves::store(
                &cells,
                Layout::decompose(Op::apply(
                    Layout::recompose(left),
                    Layout::recompose(Leaves::read(&cells)),
                )),
            );
        }
        offset.store(offset.read() * 2u32);
    }
    if UNIT_POS_PLANE + 1u32 == PLANE_DIM {
        Leaves::store_shared(Leaves::read(&cells), &mut shared, PLANE_POS as usize);
    }
    sync_cube();
    if UNIT_POS == 0u32 {
        let planes = CUBE_DIM.div_ceil(PLANE_DIM);
        let prefix = Leaves::into_cells(Leaves::load_shared(&shared, 0usize));
        let plane = RuntimeCell::<u32>::new(1u32);
        while plane.read() < planes {
            Leaves::store(
                &prefix,
                Layout::decompose(Op::apply(
                    Layout::recompose(Leaves::read(&prefix)),
                    Layout::recompose(Leaves::load_shared(&shared, plane.read() as usize)),
                )),
            );
            Leaves::store_shared(Leaves::read(&prefix), &mut shared, plane.read() as usize);
            plane.store(plane.read() + 1u32);
        }
    }
    sync_cube();
    if PLANE_POS > 0u32 {
        Leaves::store(
            &cells,
            Layout::decompose(Op::apply(
                Layout::recompose(Leaves::load_shared(&shared, PLANE_POS as usize - 1usize)),
                Layout::recompose(Leaves::read(&cells)),
            )),
        );
    }
    Leaves::read(&cells)
}

macro_rules! define_padded_scan_kernel {
    ($name:ident,$eval:ident,$method:ident; [$( $leaf:ident:$slot:ident ),+]) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        pub(super) fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $leaf: CubePrimitive + cubecl::frontend::Scalar, )+
            O0: CubePrimitive + cubecl::frontend::Scalar, O1: CubePrimitive + cubecl::frontend::Scalar, O2: CubePrimitive + cubecl::frontend::Scalar, O3: CubePrimitive + cubecl::frontend::Scalar,
            O4: CubePrimitive + cubecl::frontend::Scalar, O5: CubePrimitive + cubecl::frontend::Scalar, O6: CubePrimitive + cubecl::frontend::Scalar, O7: CubePrimitive + cubecl::frontend::Scalar,
            O8: CubePrimitive + cubecl::frontend::Scalar, O9: CubePrimitive + cubecl::frontend::Scalar, O10: CubePrimitive + cubecl::frontend::Scalar, O11: CubePrimitive + cubecl::frontend::Scalar,
            Leaves: SharedLeaves
                + MutableLeaves
                + PlaneShuffleLeaves
                + LoadMutPadded12<
                    O0 = O0, O1 = O1, O2 = O2, O3 = O3, O4 = O4, O5 = O5,
                    O6 = O6, O7 = O7, O8 = O8, O9 = O9, O10 = O10, O11 = O11,
                >
                + StorePadded12<
                    O0 = O0, O1 = O1, O2 = O2, O3 = O3, O4 = O4, O5 = O5,
                    O6 = O6, O7 = O7, O8 = O8, O9 = O9, O10 = O10, O11 = O11,
                >
                + Send + Sync + 'static,
            Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
            Expr: $eval<Item, $( $leaf ),+>,
            Op: ReductionOp<Item>,
        >(
            $( $slot: &[$leaf], )+
            read_offsets: &[u32],
            len: &[u32],
            output_offsets: &[u32],
            out0: &mut [O0], out1: &mut [O1], out2: &mut [O2], out3: &mut [O3],
            out4: &mut [O4], out5: &mut [O5], out6: &mut [O6], out7: &mut [O7],
            out8: &mut [O8], out9: &mut [O9], out10: &mut [O10], out11: &mut [O11],
            #[comptime] exclusive: bool,
            #[comptime] plane_capacity: usize,
        ) {
            let unit = UNIT_POS as usize;
            let block = CUBE_POS as usize;
            let global = block * BLOCK_SIZE as usize + unit;
            let logical_len = len[0] as usize;
            if block * BLOCK_SIZE as usize >= logical_len {
                terminate!();
            }
            let safe_global = usize::min(global, logical_len - 1usize);
            let scanned = block_inclusive_scan::<Item, Leaves, Layout, Op>(
                Expr::$method($( $slot, )+ read_offsets, safe_global), plane_capacity,
            );
            if global < logical_len {
                // A rotated inclusive scan holds the block total at its
                // first row and the exclusive prefix at every later row.
                // Summary extraction reads that total before finalization
                // replaces it with the preceding blocks' prefix.
                let start = block * BLOCK_SIZE as usize;
                let end = start + usize::min(BLOCK_SIZE as usize, logical_len - start);
                let destination = if exclusive {
                    if global + 1usize < end { global + 1usize } else { block * BLOCK_SIZE as usize }
                } else { global };
                scanned.store_padded(
                    out0, out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11,
                    output_offsets, destination,
                );
            }
        }
    };
}

define_padded_scan_kernel!(padded_scan_a13,Eval13,eval13; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7,L8:slot8,L9:slot9,L10:slot10,L11:slot11,L12:slot12]);

/// Extracts one aggregate per block from a block-local inclusive scan.
///
/// This is intentionally a separate stage from the local scan.  Unlike a
/// second block reduction over the original expression, it reads only one row
/// per block, so the stage boundary does not imply another full input pass.
#[cubecl::cube(launch_unchecked, explicit_define)]
#[allow(clippy::too_many_arguments)]
pub(super) fn padded_block_tails12<
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
    len: &[u32],
    #[comptime] exclusive: bool,
    partial0: &mut [O0],
    partial1: &mut [O1],
    partial2: &mut [O2],
    partial3: &mut [O3],
    partial4: &mut [O4],
    partial5: &mut [O5],
    partial6: &mut [O6],
    partial7: &mut [O7],
    partial8: &mut [O8],
    partial9: &mut [O9],
    partial10: &mut [O10],
    partial11: &mut [O11],
    partial_offsets: &[u32],
) {
    let block = ABSOLUTE_POS as usize;
    let logical_len = len[0] as usize;
    let blocks = crate::core::launch::logical_block_count(logical_len, BLOCK_SIZE as usize);
    let destination = block + if exclusive { 1usize } else { 0usize };
    if destination < blocks {
        let start = block * BLOCK_SIZE as usize;
        let end = start + usize::min(BLOCK_SIZE as usize, logical_len - start);
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
            if exclusive {
                block * BLOCK_SIZE as usize
            } else {
                end - 1usize
            },
        )
        .store_padded(
            partial0,
            partial1,
            partial2,
            partial3,
            partial4,
            partial5,
            partial6,
            partial7,
            partial8,
            partial9,
            partial10,
            partial11,
            partial_offsets,
            destination,
        );
    }
}

#[cubecl::cube(launch_unchecked, explicit_define)]
#[allow(clippy::too_many_arguments)]
pub(super) fn add_block_prefix_padded12<
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
        > + LoadMutPadded12<
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
        > + MutableLeaves
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
    len: &[u32],
    #[comptime] exclusive: bool,
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
    let start = block * BLOCK_SIZE as usize;
    let logical_len = len[0] as usize;
    if start >= logical_len || (!exclusive && block == 0usize) {
        terminate!();
    }
    let index = start + UNIT_POS as usize;
    let safe_index = usize::min(index, logical_len - 1usize);
    let value = Leaves::into_cells(Leaves::load_mut_padded(
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
        safe_index,
    ));
    if index < logical_len {
        let prefix = Leaves::into_cells(Leaves::load_padded(
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
            if exclusive { block } else { block - 1usize },
        ));
        if exclusive && UNIT_POS == 0u32 {
            Leaves::store(&value, Leaves::read(&prefix));
        } else {
            Leaves::store(
                &value,
                Layout::decompose(Op::apply(
                    Layout::recompose(Leaves::read(&prefix)),
                    Layout::recompose(Leaves::read(&value)),
                )),
            );
        }
        Leaves::read(&value).store_padded(
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
