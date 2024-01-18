use crate::vector::Vector;
extern "C" {

    fn init();
    fn evaluate_showdown_cuda(
        opponent_range: *const f32,
        communal_cards: u64,
        card_order: *const u64,
        eval: *const u16,
        coll_vec: *const u16,
        bet: f32,
        result: *mut f32,
    );

    fn evaluate_fold_cuda(
        opponent_range: *const f32,
        communal_cards: u64,
        card_order: *const u64,
        card_indexes: *const u16,
        updating_player: u16,
        folding_player: u16,
        bet: f32,
        result: *mut f32,
    );

    fn build_post_river_cuda(cards: u64, bet: f32) -> *const std::ffi::c_void;
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

pub fn evaluate_fold_gpu(
    opponent_range: &Vector,
    communal_cards: u64,
    card_order: &Vec<u64>,
    card_indexes: &Vec<u16>,
    updating_player: u16,
    folding_player: u16,
    bet: f32,
) -> Vector {
    let mut result = Vector::default();
    unsafe {
        evaluate_fold_cuda(
            opponent_range.values.as_ptr(),
            communal_cards,
            card_order.as_ptr(),
            card_indexes.as_ptr(),
            updating_player,
            folding_player,
            bet,
            result.values.as_mut_ptr(),
        );
    }
    result
}

pub fn build_post_river(cards: u64, bet: f32) -> *const std::ffi::c_void {
    let ptr = unsafe { build_post_river_cuda(cards, bet) };
    ptr
}

pub fn init_gpu() {
    unsafe { init() };
}
