use std::time::Instant;

mod builder;
mod enums;
mod game;
mod state;
mod strategy;

fn main() {
    let mut game = builder::flop_poker();
    dbg!("Game created");
    let start = Instant::now();
    for i in 1..=1 {
        let _res = game.perform_iter(i as f32);
    }
    //dbg!(&game);
    dbg!(start.elapsed().as_millis());
}
