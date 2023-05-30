use crate::enums::Player::{Big, Small};
use crate::evaluator;
use crate::evaluator::Evaluator;
use crate::state::State;
use itertools::Itertools;
use poker::{card, Card};
use std::fmt::{Debug, Formatter};

pub(crate) struct Game {
    root: State,
    card_order: Vec<u64>,
    evaluator: evaluator::Evaluator,
}

impl Debug for Game {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.root.fmt(f)
    }
}

impl Game {
    pub fn new(root: State, evaluator: Evaluator) -> Self {
        let _kuhn_hands: Vec<u64> = vec![
            evaluator.cards_to_u64(&[card!(Jack, Hearts)]),
            evaluator.cards_to_u64(&[card!(Queen, Hearts)]),
            evaluator.cards_to_u64(&[card!(King, Hearts)]),
        ];
        let _all_hands: Vec<u64> = Card::generate_deck()
            .combinations(2)
            .map(|cards| evaluator.cards_to_u64(&cards))
            .collect();
        let card_order = _kuhn_hands;
        //let card_order = all_hands;
        Game {
            root,
            card_order,
            evaluator,
        }
    }

    pub fn perform_iter(&mut self, iteration_weight: f32) {
        self.root.evaluate_state(
            [1.0; 3],
            [1.0; 3],
            &self.evaluator,
            iteration_weight,
            &self.card_order,
            Small,
        );

        self.root.evaluate_state(
            [1.0; 3],
            [1.0; 3],
            &self.evaluator,
            iteration_weight,
            &self.card_order,
            Big,
        );
    }
}
