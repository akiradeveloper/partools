#![allow(private_bounds)]

use cubecl::prelude::{CubeType, Runtime};

use crate::api::iter::MStorageExtent;
use crate::op::BinaryPredicateOp;
use crate::{Error, Executor, MAlloc, MIndex, MIter, MIterMut, MStorage, MVec};

struct SetOperation<'a, R: Runtime, Left, Right, Less, const MODE: u8> {
    exec: &'a Executor<R>,
    left: Left,
    right: Right,
    less: Less,
}

impl<R, Item, Left, Right, Less, const MODE: u8> crate::api::iter::OutputOperation<R, Item>
    for SetOperation<'_, R, Left, Right, Less, MODE>
where
    R: Runtime,
    Item: CubeType + Send + Sync + 'static,
    Left: MIter<R, Item = Item>,
    Right: MIter<R, Item = Item>,
    Less: BinaryPredicateOp<Item>,
{
    type Result = Result<crate::DeviceVec<R, u32>, Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: crate::api::iter::KernelRow + crate::core::allocation::ScratchStorage<R>,
        Output: crate::api::iter::ConcreteOutput<R, Item>,
    {
        crate::core::set::set(
            self.exec,
            crate::api::iter::lower_fixed::<R, _>(self.left),
            crate::api::iter::lower_fixed::<R, _>(self.right),
            self.less,
            output,
            MODE,
        )
    }
}

macro_rules! set_api {
    ($name:ident, $into_name:ident, $mode:literal, $capacity:expr, $doc:literal) => {
        #[doc = $doc]
        pub fn $name<R, Left, Right, Item, Less>(
            exec: &Executor<R>,
            left: Left,
            right: Right,
            less: Less,
        ) -> Result<MVec<R, Item>, Error>
        where
            R: Runtime,
            Left: MIter<R, Item = Item>,
            Right: MIter<R, Item = Item>,
            Item: MAlloc<R>,
            Less: BinaryPredicateOp<Item>,
        {
            let left_len = left.capacity()?;
            let right_len = right.capacity()?;
            let capacity = ($capacity)(left_len, right_len)?;
            let mut output = exec.alloc::<Item>(capacity);
            let len = $into_name(exec, left, right, less, output.slice_mut(..))?;
            output.set_logical_extent(crate::core::extent::LogicalExtent::from_device(
                &len,
                capacity as usize,
            ));
            Ok(output)
        }

        #[doc = concat!("Caller-provided output variant of [`", stringify!($name), "`].")]
        #[doc(hidden)]
        pub(crate) fn $into_name<R, Left, Right, Less, Output>(
            exec: &Executor<R>,
            left: Left,
            right: Right,
            less: Less,
            output: Output,
        ) -> Result<MVec<R, MIndex>, Error>
        where
            R: Runtime,
            Left: MIter<R, Item = Output::Item>,
            Right: MIter<R, Item = Output::Item>,
            Less: BinaryPredicateOp<Left::Item>,
            Output: MIterMut<R>,
        {
            output.run_output_operation(SetOperation::<_, _, _, _, $mode> {
                exec,
                left,
                right,
                less,
            })
        }
    };
}

set_api!(
    set_union,
    set_union_into,
    0,
    |left: MIndex, right: MIndex| left.checked_add(right).ok_or(Error::LengthTooLarge {
        len: left as usize + right as usize,
    }),
    r#"Computes the multiset union of two sorted ranges.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{op, Executor, vector::set_union};

struct Less;

#[cubecl::cube]
impl op::BinaryPredicateOp<u32> for Less {
    fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
        massively::flag::from_bool(lhs < rhs)
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let left = exec.to_device(&[1_u32, 2, 2, 4]);
let right = exec.to_device(&[2_u32, 3, 4]);
let output = set_union(&exec, left.slice(..), right.slice(..), Less).unwrap();

assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2, 2, 3, 4]);
```
"#
);
set_api!(
    set_intersection,
    set_intersection_into,
    1,
    |left: MIndex, _right: MIndex| Ok(left),
    r#"Computes the multiset intersection of two sorted ranges.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{op, Executor, vector::set_intersection};

struct Less;

#[cubecl::cube]
impl op::BinaryPredicateOp<u32> for Less {
    fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
        massively::flag::from_bool(lhs < rhs)
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let left = exec.to_device(&[1_u32, 2, 2, 4]);
let right = exec.to_device(&[2_u32, 3, 4]);
let output = set_intersection(&exec, left.slice(..), right.slice(..), Less).unwrap();

assert_eq!(exec.to_host(&output).unwrap(), vec![2, 4]);
```
"#
);
set_api!(
    set_difference,
    set_difference_into,
    2,
    |left: MIndex, _right: MIndex| Ok(left),
    r#"Computes the multiset difference of two sorted ranges.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{op, Executor, vector::set_difference};

struct Less;

#[cubecl::cube]
impl op::BinaryPredicateOp<u32> for Less {
    fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
        massively::flag::from_bool(lhs < rhs)
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let left = exec.to_device(&[1_u32, 2, 2, 4]);
let right = exec.to_device(&[2_u32, 3, 4]);
let output = set_difference(&exec, left.slice(..), right.slice(..), Less).unwrap();

assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2]);
```
"#
);
