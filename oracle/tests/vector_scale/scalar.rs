use oracle::vector as reference;

use super::common::*;

#[test]
fn reduce() {
    let input = scale_input();
    let exec = exec();
    let device = exec.to_device(&input);
    assert_eq!(
        read_value(
            &exec,
            massively::vector::reduce(&exec, lazify(device.slice(..)), 0, MaxU32,).unwrap(),
        ),
        reference::reduce(&input, 0, MaxU32),
    );
}

#[test]
fn count_if() {
    let input = scale_input();
    let exec = exec();
    let device = exec.to_device(&input);
    assert_eq!(
        read_value(
            &exec,
            massively::vector::count_if(&exec, lazify(device.slice(..)), NonZero).unwrap(),
        ) as usize,
        reference::count_if(&input, NonZero),
    );
}

#[test]
fn inclusive_scan() {
    let input = scale_input();
    let exec = exec();
    let device = exec.to_device(&input);
    let output =
        massively::vector::inclusive_scan(&exec, lazify(device.slice(..)), MaxU32).unwrap();
    assert_eq!(
        exec.to_host(&output).unwrap(),
        reference::inclusive_scan(&input, MaxU32),
    );
}
