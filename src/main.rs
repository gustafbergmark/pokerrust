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
    let mut game = builder::flop_poker();
    println!("Game created in {} seconds", start.elapsed().as_secs_f32());
    let start = Instant::now();
    for i in 1..=1_000_000 {
        game.perform_iter(i);
        //dbg!(&_res);
    }
    //dbg!(&game);
    dbg!(start.elapsed().as_millis());
}
