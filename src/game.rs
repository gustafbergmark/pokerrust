use crate::state::State;
use itertools::Itertools;
use poker::{card, Card, Evaluator};
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};

pub(crate) struct Game<const M: usize> {
    root: State<M>,
    card_order: HashMap<[Card; 2], usize>,
    evaluator: Evaluator,
}

impl<const M: usize> Debug for Game<M> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.root.fmt(f)
    }
}

impl<const M: usize> Game<M> {
    pub fn new(root: State<M>) -> Self {
        let test_pairs: Vec<([Card; 2], usize)> = vec![
            ([card!(Jack, Hearts), card!(Jack, Clubs)], 0),
            ([card!(Queen, Hearts), card!(Queen, Clubs)], 1),
            ([card!(King, Hearts), card!(King, Clubs)], 2),
        ];
        let all_pairs: Vec<([Card; 2], usize)> = Card::generate_deck()
            .combinations(2)
            .enumerate()
            .map(|(pos, cards)| (cards.try_into().unwrap(), pos))
            .collect();
        let card_order: HashMap<[Card; 2], usize> = all_pairs.into_iter().collect();
        Game {
            root,
            card_order,
            evaluator: Evaluator::new(),
        }
    }

    pub fn perform_iter(&mut self, iteration_weight: f32) -> f32 {
        let hands = self.card_order.keys().tuple_combinations();
        let mut count = 0;
        let mut ev = 0.0;
        for (&c1, &c2) in hands {
            ev += self.root.evaluate_state(
                &c1,
                &c2,
                1.0,
                1.0,
                &self.evaluator,
                iteration_weight,
                &self.card_order,
            );
            ev += self.root.evaluate_state(
                &c2,
                &c1,
                1.0,
                1.0,
                &self.evaluator,
                iteration_weight,
                &self.card_order,
            );
            count += 1;
            if count % 100 == 0 {
                println!(
                    "Finished {} hands, {}% done",
                    count,
                    count as f64 / 878475.0 * 100.0
                )
            }
        }
        ev
    }
}
