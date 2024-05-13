use crate::game::Game;
use crate::vector::Float;
use std::ffi::c_float;
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
    let mut times = vec![];
    for i in 1..=10_000 {
        game.perform_iter(i, &mut times);
    }
    println!(
        "Average: {}\n\
        Standard deviation {}",
        times.iter().sum::<f32>() / times.len() as f32,
        stdev(&times),
    );
}

fn stdev(values: &[Float]) -> Float {
    let avg = values.iter().sum::<Float>() / values.len() as Float;
    let mut stdev = 0.0;
    for value in values {
        stdev += (value - avg).powi(2);
    }
    return (stdev / values.len() as Float).sqrt();
}
