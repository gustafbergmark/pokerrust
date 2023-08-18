#![feature(slice_group_by)]
//#![feature(is_sorted)]
use crate::abstraction::{abstract_flop, abstract_turn, Abstraction};
use std::time::Instant;

mod abstraction;
mod builder;
mod combination_map;
mod enums;
mod evaluator;
mod game;
mod permutation_handler;
mod state;
mod strategy;
mod vector;

fn main() {
    let mut game = builder::flop_poker();
    dbg!("Game built");
    /*let abstraction = abstract_flop(&game.evaluator, &game.card_order);
    dbg!(abstraction.len());
    let start = Instant::now();
    let mut best_clustering: Abstraction<10_000, 3> = Abstraction::new(&abstraction[..]);
    println!(
        "Clustering created in {} seconds with {} variance",
        start.elapsed().as_secs_f32(),
        best_clustering.variance
    );
    best_clustering.save();
    for _ in 0..100 {
        println!("starting new search");
        let start = Instant::now();
        let clusters: Abstraction<10_000, 3> = Abstraction::new(&abstraction[..]);
        println!(
            "Clustering created in {} seconds with {} variance",
            start.elapsed().as_secs_f32(),
            clusters.variance
        );
        if clusters.variance < best_clustering.variance {
            best_clustering = clusters;
            best_clustering.save();
        }
    }*/
    let start = Instant::now();
    for i in 1..=1_000_000 {
        game.perform_iter(i);
        //dbg!(&_res);
    }
    //dbg!(&game);
    dbg!(start.elapsed().as_millis());
}
