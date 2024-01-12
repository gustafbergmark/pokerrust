use crate::vector::Vector;
extern "C" {
    fn prefix_sum_cuda(v: *mut f32);
    fn evaluate_showdown_cuda(
        opponent_range: *const f32,
        communal_cards: u64,
        card_order: *const u64,
        eval: *const u16,
        coll_vec: *const u16,
        bet: f32,
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
    coll_vec: &Vec<u16>,
    bet: f32,
) -> Vector {
    let mut result = Vector::default();
    unsafe {
        evaluate_showdown_cuda(
            opponent_range.values.as_ptr(),
            communal_cards,
            card_order.as_ptr(),
            eval.as_ptr(),
            coll_vec.as_ptr(),
            bet,
            result.values.as_mut_ptr(),
        );
    }
    result
}
