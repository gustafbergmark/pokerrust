use crate::combination_map::CombinationMap;
use crate::enums::Player;
use crate::evaluator::Evaluator;
use crate::vector::{Float, Vector};
extern "C" {

    fn init();
    fn evaluate_showdown_cuda(
        opponent_range: *const Float,
        communal_cards: u64,
        card_order: *const u64,
        eval: *const u16,
        coll_vec: *const u16,
        bet: Float,
        result: *mut Float,
    );

    fn evaluate_fold_cuda(
        opponent_range: *const Float,
        communal_cards: u64,
        card_order: *const u64,
        card_indexes: *const u16,
        updating_player: u16,
        folding_player: u16,
        bet: Float,
        result: *mut Float,
    );

    fn build_river_cuda(cards: u64, bet: Float) -> *const std::ffi::c_void;
    fn transfer_flop_eval_cuda(
        card_order: *const u64,
        card_indexes: *const u16,
        eval: *const u16,
        coll_vec: *const u16,
    ) -> *const std::ffi::c_void;
    fn free_eval_cuda(ptr: *const std::ffi::c_void);

    fn evaluate_post_river_cuda(
        opponent_range: *const Float,
        state: *const std::ffi::c_void,
        evaluator: *const std::ffi::c_void,
        updating_player: u16,
        result: *mut Float,
    );
}

pub fn evaluate_showdown_gpu(
    opponent_range: &Vector,
    communal_cards: u64,
    card_order: &Vec<u64>,
    eval: &Vec<u16>,
    coll_vec: &Vec<u16>,
    bet: Float,
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
    bet: Float,
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

pub fn build_river(cards: u64, bet: Float) -> *const std::ffi::c_void {
    let ptr = unsafe { build_river_cuda(cards, bet) };
    ptr
}

pub fn init_gpu() {
    unsafe { init() };
}

pub fn transfer_flop_eval(eval: &Evaluator, communal_cards: u64) -> *const std::ffi::c_void {
    assert_eq!(communal_cards.count_ones(), 3);
    let mut evals = vec![];
    let mut collisions = vec![];
    let mut cards = 3;
    for i in 0..1326 {
        if (communal_cards & cards) > 0 {
            let mut e = vec![0; 1326 + 128 * 2];
            evals.append(&mut e);
            let mut c = vec![0; 52 * 51];
            collisions.append(&mut c);
        } else {
            assert_eq!((communal_cards | cards).count_ones(), 5);
            let mut e = eval.vectorized_eval(communal_cards | cards).clone();
            evals.append(&mut e);
            let mut c = eval.collisions(communal_cards | cards).clone();
            collisions.append(&mut c);
        }

        cards = CombinationMap::<(), 52, 2>::next(cards);
    }
    assert_eq!(evals.len(), 1326 * (1326 + 128 * 2));
    assert_eq!(collisions.len(), 1326 * 52 * 51);
    let res = unsafe {
        transfer_flop_eval_cuda(
            eval.card_order().as_ptr(),
            eval.card_indexes().as_ptr(),
            evals.as_ptr(),
            collisions.as_ptr(),
        )
    };
    return res;
}

pub fn free_eval(ptr: *const std::ffi::c_void) {
    unsafe {
        free_eval_cuda(ptr);
    }
}

pub fn evaluate_post_river_gpu(
    state: *const std::ffi::c_void,
    evaluator: *const std::ffi::c_void,
    opponent_range: &Vector,
    updating_player: Player,
) -> Vector {
    let mut result = Vector::default();
    let updating_player = match updating_player {
        Player::Small => 0,
        Player::Big => 1,
    };
    unsafe {
        evaluate_post_river_cuda(
            opponent_range.values.as_ptr(),
            state,
            evaluator,
            updating_player,
            result.values.as_mut_ptr(),
        );
    }
    result
}
