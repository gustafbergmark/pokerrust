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
    let mut game: Game<256> = builder::fixed_flop_poker();
    game.load();
    for i in 0..=100 {
        game.perform_iter(i);
    }
    game.save();
}
