use crate::evaluator;
use crate::evaluator::Evaluator;
use crate::state::State;
use itertools::Itertools;
use poker::{card, Card};
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::time::Instant;

pub(crate) struct Game {
    root: State,
    card_order: HashMap<u64, usize>,
    evaluator: evaluator::Evaluator,
}

impl Debug for Game {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.root.fmt(f)
    }
}

impl Game {
    pub fn new(root: State, evaluator: Evaluator) -> Self {
        let _test_pairs: Vec<([Card; 2], usize)> = vec![
            ([card!(Jack, Hearts), card!(Jack, Clubs)], 0),
            ([card!(Queen, Hearts), card!(Queen, Clubs)], 1),
            ([card!(King, Hearts), card!(King, Clubs)], 2),
        ];
        let all_pairs: Vec<(u64, usize)> = Card::generate_deck()
            .combinations(2)
            .enumerate()
            .map(|(pos, cards)| (evaluator.cards_to_u64(&cards), pos))
            .collect();
        let card_order: HashMap<u64, usize> = all_pairs.into_iter().collect();
        Game {
            root,
            card_order,
            evaluator,
        }
    }

    pub fn perform_iter(&mut self, iteration_weight: f32) -> f32 {
        let hands = self.card_order.keys().tuple_combinations();
        let mut count = 0;
        let mut ev = 0.0;
        let start = Instant::now();
        for (&c1, &c2) in hands {
            if c1 & c2 != 0 {
                continue;
            }
            ev += self.root.evaluate_state(
                c1,
                c2,
                1.0,
                1.0,
                &self.evaluator,
                iteration_weight,
                &self.card_order,
            );
            ev += self.root.evaluate_state(
                c2,
                c1,
                1.0,
                1.0,
                &self.evaluator,
                iteration_weight,
                &self.card_order,
            );
            count += 1;
            if count % 10 == 0 {
                println!(
                    "Finished {} hands, {}% done, time per hand: {} ms",
                    count,
                    count as f64 / 878475.0 * 100.0,
                    start.elapsed().as_millis() / count
                )
            }
        }
        ev
    }
}
