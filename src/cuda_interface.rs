use crate::combination_map::CombinationMap;
use crate::enums::Player;
use crate::evaluator::Evaluator;
use crate::state::Pointer;
use crate::vector::{Float, Vector};

extern "C" {
    fn init_builder() -> *const std::ffi::c_void;

    fn upload_c(builder: *const std::ffi::c_void, index: i32, v: *const Float);
    fn download_c(builder: *const std::ffi::c_void, index: i32, v: *mut Float);
    fn build_turn_cuda(cards: u64, bet: Float, builder: *const std::ffi::c_void) -> i32;
    fn transfer_flop_eval_cuda(
        flop: u64,
        card_order: *const u64,
        card_indexes: *const u16,
        eval: *const u16,
        coll_vec: *const u16,
        abstraction: *const u16,
    ) -> *const std::ffi::c_void;
    fn free_eval_cuda(ptr: *const std::ffi::c_void);

    fn evaluate_cuda(
        builder: *const std::ffi::c_void,
        evaluator: *const std::ffi::c_void,
        updating_player: u16,
        calc_exploit: bool,
    );

}

pub fn upload_gpu(builder: Pointer, index: i32, v: &Vector) {
    unsafe { upload_c(builder.0, index, v.values.as_ptr()) }
}

pub fn download_gpu(builder: Pointer, index: i32) -> Vector {
    let mut result = Vector::default();
    unsafe { download_c(builder.0, index, result.values.as_mut_ptr()) };
    result
}
pub fn initialize_builder() -> Pointer {
    Pointer(unsafe { init_builder() })
}
pub fn build_turn(cards: u64, bet: Float, builder: Pointer) -> i32 {
    unsafe { build_turn_cuda(cards, bet, builder.0) }
}
pub fn transfer_flop_eval<const M: usize>(
    eval: &Evaluator<M>,
    communal_cards: u64,
) -> *const std::ffi::c_void {
    assert_eq!(communal_cards.count_ones(), 3);
    let mut evals = vec![];
    let mut collisions = vec![];
    let mut abstractions = vec![];
    let mut cards = 3;
    for _ in 0..1326 {
        if (communal_cards & cards) > 0 {
            let mut e = vec![0; 1326 + 128 * 2];
            evals.append(&mut e);
            let mut c = vec![0; 52 * 51];
            collisions.append(&mut c);
            let mut a = vec![0; 1326];
            abstractions.append(&mut a);
        } else {
            assert_eq!((communal_cards | cards).count_ones(), 5);
            let mut e = eval.vectorized_eval(communal_cards | cards).clone();
            evals.append(&mut e);
            let mut c = eval.collisions(communal_cards | cards).clone();
            collisions.append(&mut c);
            let mut a = eval.abstractions(communal_cards | cards).clone();
            abstractions.append(&mut a);
        }

        cards = CombinationMap::<(), 52, 2>::next(cards);
    }
    // for cards in eval.card_order() {
    //     if *cards & communal_cards > 0 {
    //         continue;
    //     }
    //     let index = CombinationMap::<(), 52, 2>::get_ordering_index(*cards);
    //     assert_eq!(
    //         eval.vectorized_eval(cards | communal_cards)[..],
    //         evals[index * (1326 + 128 * 2)..(index + 1) * (1326 + 128 * 2)]
    //     )
    // }
    assert_eq!(evals.len(), 1326 * (1326 + 128 * 2));
    assert_eq!(collisions.len(), 1326 * 52 * 51);
    assert_eq!(abstractions.len(), 1326 * 1326);
    let res = unsafe {
        transfer_flop_eval_cuda(
            communal_cards,
            eval.card_order().as_ptr(),
            eval.card_indexes().as_ptr(),
            evals.as_ptr(),
            collisions.as_ptr(),
            abstractions.as_ptr(),
        )
    };
    return res;
}

pub fn free_eval(ptr: Pointer) {
    unsafe {
        free_eval_cuda(ptr.0);
    }
}

pub fn evaluate_gpu(
    builder: Pointer,
    evaluator: Pointer,
    updating_player: Player,
    calc_exploit: bool,
) {
    let updating_player = match updating_player {
        Player::Small => 0,
        Player::Big => 1,
    };
    unsafe {
        evaluate_cuda(builder.0, evaluator.0, updating_player, calc_exploit);
    }
}
