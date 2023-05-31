#![feature(slice_group_by)]
use std::time::Instant;

mod builder;
mod enums;
mod evaluator;
mod game;
mod permutation_handler;
mod state;
mod strategy;
mod vector;

fn main() {
    let start = Instant::now();
    //let eval = evaluator::Evaluator::new();
    let mut game = builder::flop_poker();
    dbg!("Game created", start.elapsed().as_millis());
    let start = Instant::now();
    for i in 1..=1000 {
        let _res = game.perform_iter(i as f32, i % 10 == 0);
        //dbg!(&_res);
    }
    //dbg!(&game);
    dbg!(start.elapsed().as_millis());
}
