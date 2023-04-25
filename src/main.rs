mod builder;
mod game;

fn main() {
    let mut game = builder::kuhn();
    for i in 1..=100_000 {
        let res = game.perform_iter();
        if i % 10_000 == 0 {
            dbg!(i, res, &game);
        }
    }
}
