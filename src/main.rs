use std::time::Instant;

mod builder;
mod game;

fn main() {
    let mut game = builder::kuhn();
    let start = Instant::now();
    for i in 1..=1_000_000 {
        let _res = game.perform_iter();
        if i % 10_000 == 0 {
            dbg!(i, _res);
        }
    }
    dbg!(start.elapsed().as_millis());
}
