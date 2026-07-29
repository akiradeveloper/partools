//! Full CPU reference implementations and property comparisons for `massively`.
//!
//! Reference algorithms are classified as [`vector`] and [`seg`]. Operation
//! traits live in [`op`] and intentionally mirror the public GPU operation traits
//! without CubeCL runtime constraints.
//!
//! The comparison tests live in this crate as `tests/vector/mod.rs`,
//! `tests/vector_scale/mod.rs`, and `tests/seg.rs`. This keeps the dependency
//! one-way: the oracle test package depends on `massively`, while the
//! implementation does not depend on its reference implementation or on
//! `proptest`.

pub mod op;
pub mod seg;
pub mod vector;
