use crate::combination_map::CombinationMap;
use crate::enums::Player;
use crate::evaluator::Evaluator;
use crate::game::{RIVERS, TURNS};
use crate::state::Pointer;
use crate::vector::{Float, Vector};
use itertools::Itertools;
use poker::Card;
use std::time::Instant;

extern "C" {
    fn init_builder() -> *const std::ffi::c_void;

    // pub fn load_blob(builder: *const std::ffi::c_void);
    // pub fn save_blob(builder: *const std::ffi::c_void);
    // pub fn hash(builder: *const std::ffi::c_void);

    fn upload_c(builder: *const std::ffi::c_void, index: i32, v: *const Float);
    fn download_c(builder: *const std::ffi::c_void, index: i32, v: *mut Float);

    fn upload_strategy_c(builder: *const std::ffi::c_void, v: *const Float);
    fn download_strategy_c(builder: *const std::ffi::c_void, v: *mut Float);
    fn set_memory_c(builder: *const std::ffi::c_void, v: *const Float);

    fn build_river_cuda(cards: u64, bet: Float, builder: *const std::ffi::c_void) -> i32;
    fn transfer_flop_eval_cuda(
        flop: u64,
        card_order: *const u64,
        card_indexes: *const u16,
        eval: *const u16,
        coll_vec: *const u16,
        abstraction: *const u16,
        turns: *const u64,
        rivers: *const u64,
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

pub fn set_builder_memory(builder: Pointer, v: &mut Vec<Float>) {
    unsafe { set_memory_c(builder.0, v.as_mut_ptr()) }
}
pub fn upload_strategy_gpu(builder: Pointer, ptr: &[Float]) {
    unsafe { upload_strategy_c(builder.0, ptr.as_ptr()) }
}
pub fn download_strategy_gpu(builder: Pointer, ptr: &mut [Float]) {
    unsafe { download_strategy_c(builder.0, ptr.as_mut_ptr()) }
}
pub fn initialize_builder() -> Pointer {
    Pointer(unsafe { init_builder() })
}
pub fn build_river(cards: u64, bet: Float, builder: Pointer) -> i32 {
    unsafe { build_river_cuda(cards, bet, builder.0) }
}
pub fn transfer_flop_eval<const M: usize>(
    eval: &Evaluator<M>,
    communal_cards: u64,
    turns: Vec<u64>,
    rivers: Vec<u64>,
    calc_exploit: bool,
) -> *const std::ffi::c_void {
    let _start = Instant::now();
    assert_eq!(communal_cards.count_ones(), 3);
    let mut evals2 = vec![0; 1326 * (1326 + 256 * 2)];
    let mut collisions2 = vec![0; 1326 * 52 * 51];
    let mut abstractions2 = vec![0; 1326 * 1326];
    let runouts = if calc_exploit {
        Card::generate_deck()
            .filter(|&card| eval.cards_to_u64(&[card]) & communal_cards == 0)
            .combinations(2)
            .collect::<Vec<_>>()
    } else {
        let mut runouts = vec![];
        for i in 0..TURNS {
            for j in 0..RIVERS {
                runouts.push(eval.u64_to_cards(turns[i] | rivers[j + i * RIVERS]))
            }
        }
        runouts
    };
    for runout in runouts {
        let cards = eval.cards_to_u64(&runout);
        let card_index = CombinationMap::<(), 52, 2>::get_ordering_index(cards);
        assert_eq!(communal_cards & cards, 0);
        assert_eq!((communal_cards | cards).count_ones(), 5);
        // Inverse eval for GPU memory coalescing
        let e = eval.vectorized_eval(communal_cards | cards).clone();
        let mut inverse = e.clone();
        for (i, &val) in e[..1326].into_iter().enumerate() {
            inverse[(val & 2047) as usize] &= 2048;
            inverse[(val & 2047) as usize] |= i as u16;
        }
        evals2[card_index * (1326 + 256 * 2)..(card_index + 1) * (1326 + 256 * 2)]
            .copy_from_slice(&inverse);
        let c = eval.collisions(communal_cards | cards).clone();
        collisions2[card_index * 52 * 51..(card_index + 1) * 52 * 51].copy_from_slice(&c);
        let a = eval.abstractions(communal_cards | cards).clone();
        abstractions2[card_index * 1326..(card_index + 1) * 1326].copy_from_slice(&a);
    }

    let mut evals = vec![];
    let mut collisions = vec![];
    let mut abstractions = vec![];
    let mut cards = 3;
    for _ in 0..1326 {
        if (communal_cards & cards) > 0 {
            let mut e = vec![0; 1326 + 256 * 2];
            evals.append(&mut e);
            let mut c = vec![0; 52 * 51];
            collisions.append(&mut c);
            let mut a = vec![0; 1326];
            abstractions.append(&mut a);
        } else {
            assert_eq!((communal_cards | cards).count_ones(), 5);
            // Inverse eval for GPU memory coalescing
            let e = eval.vectorized_eval(communal_cards | cards).clone();
            let mut inverse = e.clone();
            for (i, &val) in e[..1326].into_iter().enumerate() {
                inverse[(val & 2047) as usize] &= 2048;
                inverse[(val & 2047) as usize] |= i as u16;
            }
            evals.append(&mut inverse);
            //evals.extend_from_slice(&e);

            let mut c = eval.collisions(communal_cards | cards).clone();
            collisions.append(&mut c);
            let mut a = eval.abstractions(communal_cards | cards).clone();
            abstractions.append(&mut a);
        }

        cards = CombinationMap::<(), 52, 2>::next(cards);
    }
    assert_eq!(evals, evals2);
    assert_eq!(collisions, collisions2);
    assert_eq!(abstractions, abstractions2);

    assert_eq!(evals2.len(), 1326 * (1326 + 256 * 2));
    assert_eq!(collisions2.len(), 1326 * 52 * 51);
    assert_eq!(abstractions2.len(), 1326 * 1326);
    let res = unsafe {
        transfer_flop_eval_cuda(
            communal_cards,
            eval.card_order().as_ptr(),
            eval.card_indexes().as_ptr(),
            evals2.as_ptr(),
            collisions2.as_ptr(),
            abstractions2.as_ptr(),
            turns.as_ptr(),
            rivers.as_ptr(),
        )
    };
    //println!("Transfer eval time: {}", _start.elapsed().as_secs_f32());
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
