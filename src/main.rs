use crate::cuda_interface::init_gpu;
use std::time::Instant;

mod builder;
mod combination_map;
mod cuda_interface;
mod enums;
mod evaluator;
mod game;
mod permutation_handler;
mod state;
mod strategy;
mod vector;

fn main() {
    // Set heap size of gpu
    init_gpu();
    let mut game = builder::fixed_flop_poker();
    let start = Instant::now();
    for i in 1..=1 {
        game.perform_iter(i);
    }
    //dbg!(&game);
    dbg!(start.elapsed().as_millis());
}
