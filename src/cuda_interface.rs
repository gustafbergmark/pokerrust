use crate::vector::Vector;
extern "C" {
    fn prefix_sum_cuda(v: *mut f32);
    fn evaluate_showdown_cuda(
        opponent_range: *const f32,
        communal_cards: u64,
        card_order: *const u64,
        eval: *const u16,
        groups: *const u16,
        groups_size: u32,
        result: *mut f32,
    );
}

pub fn prefix_sum(v: &mut Vector) {
    unsafe { prefix_sum_cuda(v.values.as_mut_ptr()) }
}

pub fn evaluate_showdown_gpu(
    opponent_range: &Vector,
    communal_cards: u64,
    card_order: &Vec<u64>,
    eval: &Vec<u16>,
    groups: &Vec<u16>,
) -> Vector {
    let mut result = Vector::default();
    unsafe {
        evaluate_showdown_cuda(
            opponent_range.values.as_ptr(),
            communal_cards,
            card_order.as_ptr(),
            eval.as_ptr(),
            groups.as_ptr(),
            groups.len() as u32,
            result.values.as_mut_ptr(),
        );
    }
    result
}
