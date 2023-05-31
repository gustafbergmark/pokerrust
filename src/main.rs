#![feature(slice_group_by)]
use std::time::Instant;

mod builder;
mod enums;
mod evaluator;
mod game;
mod permutation_handler;
mod simd;
mod state;
mod strategy;

fn main() {
    let start = Instant::now();
    //let eval = evaluator::Evaluator::new();
    let mut game = builder::flop_poker();
    dbg!("Game created", start.elapsed().as_millis());
    let start = Instant::now();
    for i in 1..=100 {
        let _res = game.perform_iter(i as f32);
        //dbg!(&_res);
    }
    //dbg!(&game);
    dbg!(start.elapsed().as_millis());
}
