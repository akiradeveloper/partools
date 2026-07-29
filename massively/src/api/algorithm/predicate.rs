use cubecl::prelude::*;

use crate::op::PredicateOp;
use crate::{Error, Executor, MFlag, MIndex, MIter};

macro_rules! predicate_api {
    (
        $name:ident,
        $core_name:ident,
        $output:ty,
        |$exec:ident, $value:ident, $len:ident| $map:block,
        $doc:literal
    ) => {
        #[doc = $doc]
        pub fn $name<R, Input, Pred>(
            exec: &Executor<R>,
            input: Input,
            pred: Pred,
        ) -> Result<$output, Error>
        where
            R: Runtime,
            Input: MIter<R>,
            Pred: PredicateOp<Input::Item>,
        {
            let len = input.capacity()?;
            let value = crate::core::predicate::$core_name(
                exec,
                crate::api::iter::lower_fixed::<R, _>(input),
                pred,
            )?;
            let value = crate::api::value::read::<R, MIndex>(exec, &value)?;
            let $exec = exec;
            let $value = value;
            let $len = len;
            $map
        }
    };
}

/// Returns the logical number of items in the input as a device-resident value.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, vector::count};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3, 4]);
///
/// assert_eq!(count(&exec, input.slice(..)).unwrap(), 4);
/// ```
pub fn count<R, Input>(exec: &Executor<R>, input: Input) -> Result<MIndex, Error>
where
    R: Runtime,
    Input: MIter<R>,
{
    let extent = input.logical_extent()?;
    let storage = extent.materialize(exec)?;
    crate::api::value::read::<R, MIndex>(exec, &storage)
}

predicate_api!(
    count_if,
    count_if,
    MIndex,
    |_exec, value, _len| { Ok(value) },
    r#"Counts items satisfying a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, vector::count_if};

struct Even;

#[cubecl::cube]
impl op::PredicateOp<u32> for Even {
    fn apply(value: u32) -> massively::MFlag {
        massively::flag::from_bool(value % 2 == 0)
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[1_u32, 2, 3, 4]);

assert_eq!(count_if(&exec, input.slice(..), Even).unwrap(), 2);
```
"#
);
predicate_api!(
    all_of,
    count_if,
    MFlag,
    |_exec, value, len| { Ok(crate::flag::from_bool(value == len)) },
    r#"Returns whether every item satisfies a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, vector::all_of};

struct Even;

#[cubecl::cube]
impl op::PredicateOp<u32> for Even {
    fn apply(value: u32) -> massively::MFlag {
        massively::flag::from_bool(value % 2 == 0)
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[2_u32, 4, 6]);

assert!(massively::flag::is_set(
    all_of(&exec, input.slice(..), Even).unwrap()
));
```
"#
);
predicate_api!(
    any_of,
    count_if,
    MFlag,
    |_exec, value, _len| { Ok(crate::flag::from_bool(value != 0)) },
    r#"Returns whether any item satisfies a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, vector::any_of};

struct Even;

#[cubecl::cube]
impl op::PredicateOp<u32> for Even {
    fn apply(value: u32) -> massively::MFlag {
        massively::flag::from_bool(value % 2 == 0)
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[1_u32, 3, 4]);

assert!(massively::flag::is_set(
    any_of(&exec, input.slice(..), Even).unwrap()
));
```
"#
);
predicate_api!(
    none_of,
    count_if,
    MFlag,
    |_exec, value, _len| { Ok(crate::flag::from_bool(value == 0)) },
    r#"Returns whether no item satisfies a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, vector::none_of};

struct Even;

#[cubecl::cube]
impl op::PredicateOp<u32> for Even {
    fn apply(value: u32) -> massively::MFlag {
        massively::flag::from_bool(value % 2 == 0)
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[1_u32, 3, 5]);

assert!(massively::flag::is_set(
    none_of(&exec, input.slice(..), Even).unwrap()
));
```
"#
);
#[doc = r#"Returns the first index satisfying a predicate.

This operation synchronizes to resolve the optional index on the host.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, vector::find_if};

struct Even;

#[cubecl::cube]
impl op::PredicateOp<u32> for Even {
    fn apply(value: u32) -> massively::MFlag {
        massively::flag::from_bool(value % 2 == 0)
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[1_u32, 3, 4, 6]);

assert_eq!(
    find_if(&exec, input.slice(..), Even).unwrap(),
    Some(2)
);
```
"#]
pub fn find_if<R, Input, Pred>(
    exec: &Executor<R>,
    input: Input,
    pred: Pred,
) -> Result<Option<MIndex>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Pred: PredicateOp<Input::Item>,
{
    let value =
        crate::core::predicate::find_if(exec, crate::api::iter::lower_fixed::<R, _>(input), pred)?;
    crate::api::value::read_optional_index(exec, &value)
}
predicate_api!(
    is_partitioned,
    is_partitioned,
    MFlag,
    |_exec, value, _len| { Ok(crate::flag::from_bool(value == MIndex::MAX)) },
    r#"Returns whether passing items precede failing items.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, vector::is_partitioned};

struct Even;

#[cubecl::cube]
impl op::PredicateOp<u32> for Even {
    fn apply(value: u32) -> massively::MFlag {
        massively::flag::from_bool(value % 2 == 0)
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[2_u32, 4, 1, 3]);

assert!(massively::flag::is_set(
    is_partitioned(&exec, input.slice(..), Even).unwrap()
));
```
"#
);
