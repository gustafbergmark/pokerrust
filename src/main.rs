use std::time::Instant;

mod builder;
mod enums;
mod evaluator;
mod game;
mod state;
mod strategy;

fn main() {
    let start = Instant::now();
    //let eval = evaluator::Evaluator::new();
    let mut game = builder::flop_poker();
    dbg!("Game created");
    for i in 1..=1 {
        let _res = game.perform_iter(i as f32);
    }
    //dbg!(&game);
    dbg!(start.elapsed().as_millis());
}
