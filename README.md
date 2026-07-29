<div align="center">
<img src="assets/logo.svg" alt="massively" width="400">

[![Crates.io](https://img.shields.io/crates/v/massively.svg)](https://crates.io/crates/massively)
[![API doc](https://docs.rs/massively/badge.svg)](https://docs.rs/massively)
![CI](https://github.com/akiradeveloper/massively/actions/workflows/ci.yml/badge.svg)

----

**Composable GPU parallel primitives for Rust.**

</div>

## Overview

`massively` provides optimized, Thrust-style parallel primitives for
device-resident data on top of
[CubeCL](https://github.com/tracel-ai/cubecl). The same API can run on WGPU,
CUDA, or HIP through the corresponding CubeCL runtime.

Massively is designed for building complex GPU applications by composing a
small set of general-purpose building blocks. It does not try to provide a
separate, domain-specific algorithm for every application. Instead,
application logic is expressed by combining vector algorithms, segmented
algorithms, lazy iterators, and user-defined operations. Optimizing these shared
primitives benefits every application assembled from them.

The algorithms are organized into two complementary families:

- vector algorithms for map, scan, reduction, sorting, selection, and
  indexed movement
- segment algorithms that apply map, scan, reduction, ordering, and selection
  independently to offset-delimited regions

Memory movement is explicit, vector algorithms return owned device storage,
and user-defined operations are compiled into GPU kernels. Lazy maps, permutations, reversed
views, repetitions, tiling, and adjacent differences can be consumed without
first materializing an intermediate buffer.

The public API is built around a few ideas:

- explicit host/device transfer through `Executor`
- owning device storage through `DeviceVec` and zero-copy views through
  `DeviceSlice` and `DeviceSliceMut`
- logical row values assembled with `zip2` through `zip12`
- CubeCL-backed operations under `massively::op`, such as `UnaryOp`,
  `ExpandOp`, `PredicateOp`, and `ReductionOp`
- parallel algorithms under `massively::vector`, such as `map`, `reduce`,
  `flat_map`, `inclusive_scan`, `sort`, `gather`, `copy_where`, and by-key
  variants
- offset-delimited algorithms and the reusable `Segmentation` abstraction
  under `massively::seg`

## Composition First

The unit of reuse in Massively is an optimized primitive, not an end-to-end
domain algorithm. Applications construct pipelines from a few orthogonal
pieces:

- `lazy` iterators describe transformations and indexed views without
  materializing intermediate buffers
- `vector` algorithms provide global movement, selection, ordering, scans, and
  reductions
- `seg` applies the same kinds of operations independently to variable-length
  regions
- operations defined with CubeCL carry application-specific logic into the
  generated GPU kernels

These pieces compose across single- and multi-column data. Intermediate values
can remain lazy where possible, while explicit storage and synchronization
boundaries keep data movement visible. More specialized behavior should be
assembled from these parts so that the public API stays compact and
improvements to a primitive carry through to the larger applications that use
it.

## Setup

`massively` is runtime-agnostic: algorithms are generic over
`R: cubecl::Runtime`, and the application selects a backend through its direct
`cubecl` dependency. `cubecl` is also needed because its runtime types and
`#[cubecl::cube]` macro are part of user-defined operations. Use the same CubeCL
release as the one selected by `massively`.

```sh
cargo add massively
cargo add cubecl --no-default-features --features std,stdlib,wgpu
```

For another backend, enable `cuda` or `hip` on `cubecl` instead of `wgpu` and
construct `Executor` with the corresponding CubeCL runtime.

## Quick Example

This example doubles a device vector and returns owned device storage.

```rust
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op::UnaryOp, vector::map};

struct Double;

#[cubecl::cube]
impl UnaryOp<u32> for Double {
    type Output = u32;

    fn apply(value: u32) -> u32 {
        value * 2
    }
}

fn main() -> Result<(), massively::Error> {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[1_u32, 2, 3, 4]);
    let output = map(&exec, input.slice(..), Double)?;

    assert_eq!(exec.to_host(&output)?, vec![2, 4, 6, 8]);
    Ok(())
}
```

## Core Model

### Runtime And Memory

`massively` uses CubeCL runtimes directly. Pick a CubeCL runtime type and pass
one of its devices to `Executor::new`.

```rust
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::Executor;

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
```

The same `Executor<R>` drives device allocation, transfers, synchronization,
and algorithms for that runtime. An attempt to use storage with a different
executor is rejected with `Error::ForeignExecutor`.

Algorithms are ordinary functions. Their public constraints describe logical
iterators and operations; kernel lowering and dispatch remain private
implementation details. Algorithms that naturally produce a new sequence return
an `MVec`. Algorithms whose semantics require an existing destination, such as
`scatter`, take that destination as an argument.

```rust
let output = map(&exec, input, op)?;
let sum = reduce(&exec, input, zero, sum_op)?;
let continued = reduce(&exec, more_input, sum, sum_op)?;
```

Vector host/device movement remains explicit. Single-value inputs are ordinary
host values and are staged internally when a GPU kernel needs them. Algorithms
take an `Executor`, so transfers, launches, and ownership checks happen at a
visible execution boundary.

### Value And Length Boundaries

Single-value parameters use their ordinary item type. Single-value results
synchronize and return ordinary host values. Optional-index queries such as
`find_if` and `min_element` return `Option<MIndex>` directly. Device-side
single-value storage remains an implementation detail rather than a public
composable value abstraction.
Truth-valued algorithms use `MFlag`, while indices and lengths use `MIndex`.
Both aliases use `u32`.

Owned device allocations are also bounded by `MIndex`: `Executor::alloc` and
the storage allocation contract take `MIndex`, not `usize`. Consequently every
allocated row has a representable index, and a full-length index or permutation
vector can cover the same domain. A length originating as host `usize` must be
checked and converted before allocation; the conversion to backend byte counts
remains an internal implementation detail.

Algorithms with data-dependent output lengths, such as `copy_where` and
`reduce_by_key`, retain their physical allocation bound and device-resident
logical extent internally. Neither is exposed as a public length or capacity
query. The logical extent remains attached to each resulting device column and
propagates through later algorithms without host synchronization.
`Executor::to_host` remains the explicit boundary for copying vector data.

`MFlag` is the truth representation throughout Massively: zero is false and
nonzero is true. Predicate producers should use `flag::from_bool` to return a
canonical flag, while stencils accept any `MFlag` value. Consumers should use
`flag::is_set`; comparing an arbitrary flag to `1` is incorrect because other
nonzero values are also true. Summary algorithms synchronize and return
`MFlag`.

### Device Storage And Slices

`DeviceVec<R, T>` owns one contiguous device allocation. Algorithms read a
`DeviceSlice<R, T>` returned by `DeviceVec::slice`, and write a
`DeviceSliceMut<R, T>` returned by `DeviceVec::slice_mut`. Slices are zero-copy
views, retain their runtime type, and can be sliced again. Slice bounds use
`MIndex`.

### Multi-column Values

The `zip2` through `zip12` helpers combine slices or lazy iterators into one
logical row stream. For example, `zip3(a, b, c)` has item type
`(A, B, C)`. `zip` is associative at the schema level: both
`zip2(zip2(a, b), c)` and
`zip2(a, zip2(b, c))` expose the same flat item type. The internal storage tree
is not part of the public contract.

For an owned multi-column `MVec`, `MStorage::into_columns` returns a native flat
tuple of owning `DeviceVec` columns without copying or reallocating device data.
Tuple types, literals, and destructuring use Rust's native tuple syntax directly,
including inside a user-defined CubeCL operation.

Conceptually:

```text
DeviceSlice<T>                         = MIter<Item = T>
zip2(a, b)                             = MIter<Item = (A, B)>
zip3(a, b, c)                          = MIter<Item = (A, B, C)>
zip2(zip2(a, b), c)                    = MIter<Item = (A, B, C)>
zip2(a, zip2(b, c))                    = MIter<Item = (A, B, C)>
zip2(out_a, out_b)                     = MIterMut<Item = (A, B)>
MStorage::into_columns(output3)        = (DeviceVec<A>, DeviceVec<B>, DeviceVec<C>)
lazy::map(input, op)                   = fused lazy computation
lazy::permute(values, indices)         = lazy indexed view
lazy::reverse(input)                    = lazy reversed view
lazy::repeat_each(input, count)         = lazy per-item repetition
lazy::tile(input, count)                = lazy whole-input repetition
lazy::adjacent_difference(input, op)    = lazy neighboring computation
```

Input and output items support up to twelve columns. Keys passed to by-key
algorithms are limited to three columns; their value items retain the full
twelve-column limit. An operation that intentionally changes a row schema
expresses that conversion explicitly with `map`.

### Segmentations

`seg::Segmentation` represents one ordered partition of a flat range. Segment
lengths, zero-based segment IDs, and CSR-style offsets are interchangeable:

```text
lengths      [1, 2, 3]
segment IDs  [0, 1, 1, 2, 2, 2]
offsets      [0, 1, 3, 6]
```

Offsets are the canonical form. They are compact, preserve empty segments, and
give direct segment bounds. `Segmentation::from_lengths`,
`Segmentation::from_segment_ids`, and `Segmentation::from_offsets` validate and
privately materialize this form. `lengths()` and `segment_ids()` derive owned
device columns when needed, `local_indices()` derives each entry's zero-based
position inside its segment, and `offsets()` is a read-only zero-copy view.

IDs alone cannot encode trailing or all-empty segments, so
`from_segment_ids` also takes the segment count. For example,
`[0, 2, 2]` with four segments is equivalent to lengths `[1, 0, 2, 0]`
and offsets `[0, 1, 1, 3, 3]`.

The same segmentation can be applied to any equally long single- or
multi-column value iterator with `segments(values)`. The resulting
`SegmentIterator` retains the validated `Segmentation`; length-preserving
algorithms preserve it and length-changing owned results rebuild one. It also
supports segment-wise context broadcast without a special algorithm:

```rust,ignore
let ids = segmentation.segment_ids(&exec)?;
let entry_context = lazy::permute(contexts, ids.slice(..));
```

`contexts` must contain one row per segment before this unchecked indexed view
is consumed.

A uniform context uses
`lazy::constant(context).take(segmentation.value_count())` instead.
Whole-segment code can first zip values with the broadcast contexts and then
call `segmentation.segments(...)`, producing `Segment<(Value, Context)>` for
the existing segmented reduce, scan, filter, and other algorithms.
Empty segments have no entry to receive a broadcast; handle their context
contribution with a parallel map over
`zip2(segmentation.lengths(&exec)?, contexts)` and combine the two result
streams with another `zip2`/`map`.

For variable output from empty segments, zip lengths with contexts and use
`lazy::counting(0).take(segment_count + 1)` as singleton offsets. Applying the
ordinary `ForEachSegment(FlatMap(op))` then preserves one output segment per
input context without a special adapter.

Massively intentionally does not add a context-specific segmented adapter for
these cases. A small set of orthogonal primitives is preferred even when the
composition needs an extra GPU pass. Performance work may fuse that
composition internally without adding another public semantic abstraction.

Massively does not expose a generic iteration runner. A normal host `for` or
`while` remains the control flow, and applications may retain their workspace
between rounds. Recording or fusing that composition is an executor/compiler
optimization; it does not require another public iteration abstraction.

### Lazy Iterators

`lazy::constant`, `lazy::counting`, `lazy::stride`, `lazy::map`,
`lazy::permute`, `lazy::reverse`, `lazy::repeat_each`, `lazy::tile`, and
`lazy::adjacent_difference` produce `MIter` values without allocating result
storage. Their expressions are evaluated by the consuming algorithm, allowing
operations to be composed while keeping intermediate values off device memory.

Segment operations are selected with `ForEachSegment`. In addition to mapping,
scanning, reduction, sorting, and filtering, `Take(n)` keeps a bounded prefix
of every segment. `FindIf(pred)` and `AdjacentFind(equal)` return a local index
per segment, using the segment length when no match exists.

### Operations

User-defined operations are CubeCL cube traits. `massively` intentionally keeps
that connection visible: CubeCL is the kernel DSL, while `massively` supplies
the algorithm and iterator layer.

- `UnaryOp<Input>` maps one item to another.
- `ExpandOp<Input>` expands one item into zero or more items.
- `PredicateOp<Item>` tests one item and returns `MFlag`.
- `BinaryPredicateOp<Item>` compares two items and returns `MFlag`.
- `ReductionOp<Item>` combines two items.

Use `flag::from_bool` to turn a CubeCL comparison into canonical 0/1, and
`flag::is_set` when an `MFlag` must drive CubeCL control flow.

## Design Notes

The implementation favors reusable primitives such as scan, selection,
permutation, and segmented control over application-specific kernels. Complex
behavior belongs in compositions of these building blocks. This keeps the API
surface compact, avoids duplicating optimization work, and lets improvements to
core primitives benefit complete applications.

Multi-column support is a first-class requirement. The code avoids
single-column-only shortcuts and avoids arity explosion by separating control
generation from payload movement where possible, especially in by-key
algorithms.

## Further Reading

The [0.97 release validation](RELEASE_0.97.md) records the architecture changes,
Lampshade comparison, and remaining performance gaps. The
[API performance table](PERFORMANCE.md) covers the dedicated vector benchmarks.

### Correctness Examples

Every public algorithm has a runnable, single-column example in the
[API documentation](https://docs.rs/massively). Integration tests are grouped
under `massively/tests/vector` and `massively/tests/seg`. The independent,
non-publishable [`oracle`](oracle/) crate contains CPU AoS references and
property tests that compare public functions against those references, including
the full map input/output arity matrix and boundary sizes. Run `just test` for
all tests, or `just test-oracle` for the oracle tests alone.
