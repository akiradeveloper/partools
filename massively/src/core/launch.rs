use cubecl::prelude::*;

const MAX_AXIS: u32 = 65_535;

/// Maximum number of explicit storage buffers a Massively kernel may bind.
/// CubeCL's kernel metadata binding is outside this budget.
#[cfg(test)]
pub(crate) const MAX_EXPLICIT_STORAGE_BINDINGS: usize = 28;

#[cfg(test)]
pub(crate) fn assert_binding_budget<K: cubecl::prelude::CubeKernel>(name: &str, kernel: &K) {
    let definition = kernel.define();
    let scope = &definition.body;
    let entry = scope.state().entry_func.get_entry_block(scope.ctx());
    let bindings = entry.deref(scope.ctx()).get_num_arguments();
    assert!(
        bindings <= MAX_EXPLICIT_STORAGE_BINDINGS,
        "{name} uses {bindings} explicit storage bindings; Massively permits {MAX_EXPLICIT_STORAGE_BINDINGS}",
    );
}

/// Constructs a generated test kernel whose ABI exactly fills the explicit
/// storage-binding budget. Keeping the repeated registration here makes each
/// module-level regression test describe the kernel type rather than copy the
/// identical arguments.
#[cfg(test)]
macro_rules! kernel_with_max_explicit_storage_bindings {
    ($kernel:ty, $settings:expr, $client:expr, $arg:expr $(, $comptime:expr)* $(,)?) => {{
        <$kernel>::new(
            $settings,
            $client,
            $arg.clone(),
            $arg.clone(),
            $arg.clone(),
            $arg.clone(),
            $arg.clone(),
            $arg.clone(),
            $arg.clone(),
            $arg.clone(),
            $arg.clone(),
            $arg.clone(),
            $arg.clone(),
            $arg.clone(),
            $arg.clone(),
            $arg.clone(),
            $arg.clone(),
            $arg.clone(),
            $arg.clone(),
            $arg.clone(),
            $arg.clone(),
            $arg.clone(),
            $arg.clone(),
            $arg.clone(),
            $arg.clone(),
            $arg.clone(),
            $arg.clone(),
            $arg.clone(),
            $arg.clone(),
            $arg,
            $( $comptime, )*
        )
    }};
}

#[cfg(test)]
pub(crate) use kernel_with_max_explicit_storage_bindings;

pub(crate) fn cube_count_1d(blocks: usize) -> Result<CubeCount, crate::Error> {
    let blocks = u32::try_from(blocks).map_err(|_| crate::Error::LengthTooLarge { len: blocks })?;
    if blocks <= MAX_AXIS {
        return Ok(CubeCount::Static(blocks, 1, 1));
    }
    let mut y = blocks.div_ceil(MAX_AXIS);
    while y <= MAX_AXIS {
        if blocks.is_multiple_of(y) {
            return Ok(CubeCount::Static(blocks / y, y, 1));
        }
        y += 1;
    }
    // Power-of-two packing keeps padded grids below the next power of two.
    // In particular, kernels with power-of-two tiles cannot wrap their u32
    // row index when the logical length approaches MIndex::MAX.
    let x = 32_768;
    let y = blocks.div_ceil(x);
    if y <= MAX_AXIS {
        Ok(CubeCount::Static(x, y, 1))
    } else {
        Ok(CubeCount::Static(x, x, y.div_ceil(x)))
    }
}

/// Unlike CubeCL's addition-based div_ceil, this also covers MIndex::MAX.
#[cubecl::cube]
pub(crate) fn logical_block_count(len: usize, #[comptime] tile_size: usize) -> usize {
    if len == 0usize {
        0usize
    } else {
        (len - 1usize) / tile_size + 1usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_padding_stays_within_the_index_domain() {
        for blocks in [1, 65_535, 65_537, 524_287, 524_288, 16_777_213, u32::MAX] {
            let CubeCount::Static(x, y, z) = cube_count_1d(blocks as usize).unwrap() else {
                panic!("expected a static dispatch");
            };
            assert!([x, y, z].iter().all(|&axis| axis <= MAX_AXIS));
            let actual = u64::from(x) * u64::from(y) * u64::from(z);
            assert!(actual >= u64::from(blocks));
            assert!(actual <= u64::from(blocks).next_power_of_two());
        }
    }
}

/// Upper bound on the number of subgroups in a workgroup. Shared row storage
/// needs one entry per subgroup, rather than one entry per invocation.
pub(crate) fn plane_count_bound<R: cubecl::prelude::Runtime>(
    exec: &crate::Executor<R>,
    block_size: u32,
) -> usize {
    block_size.div_ceil(exec.client().properties().hardware.plane_size_min.max(1)) as usize
}
