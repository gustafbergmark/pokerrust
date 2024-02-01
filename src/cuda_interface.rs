use crate::combination_map::CombinationMap;
use crate::enums::Player;
use crate::evaluator::Evaluator;
use crate::state::Pointer;
use crate::vector::{Float, Vector};

extern "C" {

    fn build_turn_cuda(cards: u64, bet: Float) -> *const std::ffi::c_void;
    fn transfer_flop_eval_cuda(
        flop: u64,
        card_order: *const u64,
        card_indexes: *const u16,
        eval: *const u16,
        coll_vec: *const u16,
    ) -> *const std::ffi::c_void;
    fn free_eval_cuda(ptr: *const std::ffi::c_void);

    fn evaluate_turn_cuda(
        opponent_range: *const Float,
        states: *const std::ffi::c_void,
        evaluator: *const std::ffi::c_void,
        updating_player: u16,
        calc_exploit: bool,
        result: *mut Float,
    );

}

pub fn build_turn(cards: u64, bet: Float) -> *const std::ffi::c_void {
    unsafe { build_turn_cuda(cards, bet) }
}
pub fn transfer_flop_eval<const M: usize>(
    eval: &Evaluator<M>,
    communal_cards: u64,
) -> *const std::ffi::c_void {
    assert_eq!(communal_cards.count_ones(), 3);
    let mut evals = vec![];
    let mut collisions = vec![];
    let mut cards = 3;
    for _ in 0..1326 {
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
    for cards in eval.card_order() {
        if *cards & communal_cards > 0 {
            continue;
        }
        let index = CombinationMap::<(), 52, 2>::get_ordering_index(*cards);
        assert_eq!(
            eval.vectorized_eval(cards | communal_cards)[..],
            evals[index * (1326 + 128 * 2)..(index + 1) * (1326 + 128 * 2)]
        )
    }
    assert_eq!(evals.len(), 1326 * (1326 + 128 * 2));
    assert_eq!(collisions.len(), 1326 * 52 * 51);
    let res = unsafe {
        transfer_flop_eval_cuda(
            communal_cards,
            eval.card_order().as_ptr(),
            eval.card_indexes().as_ptr(),
            evals.as_ptr(),
            collisions.as_ptr(),
        )
    };
    return res;
}

pub fn free_eval(ptr: Pointer) {
    unsafe {
        free_eval_cuda(ptr.0);
    }
}

pub fn evaluate_turn_gpu(
    states: Pointer,
    evaluator: Pointer,
    opponent_range: &Vector,
    updating_player: Player,
    calc_exploit: bool,
) -> Vector {
    let mut result = Vector::default();
    let updating_player = match updating_player {
        Player::Small => 0,
        Player::Big => 1,
    };
    unsafe {
        evaluate_turn_cuda(
            opponent_range.values.as_ptr(),
            states.0,
            evaluator.0,
            updating_player,
            calc_exploit,
            result.values.as_mut_ptr(),
        );
    }
    result
}
