use crate::game::Game;
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
    let start = Instant::now();
    let mut game: Game<256> = builder::fixed_flop_poker();
    game.load();
    for i in 1..=0 {
        game.perform_iter(i);
    }
    game.save();
    //dbg!(&game);
    dbg!(start.elapsed().as_millis());
}
