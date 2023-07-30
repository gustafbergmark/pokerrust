use crate::enums::Player::{Big, Small};
use crate::evaluator::Evaluator;
use crate::state::State;
use crate::vector::Vector;
use poker::{card, Card};
use std::fmt::{Debug, Formatter};
use std::time::Instant;

pub(crate) struct Game {
    root: State,
    card_order: Vec<u64>,
    evaluator: Evaluator,
}

impl Debug for Game {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.root.fmt(f)
    }
}

impl Game {
    pub fn new(root: State, evaluator: Evaluator, card_order: Vec<[Card; 2]>) -> Self {
        let _kuhn_hands: Vec<u64> = vec![
            evaluator.cards_to_u64(&[card!(Jack, Hearts)]),
            evaluator.cards_to_u64(&[card!(Queen, Hearts)]),
            evaluator.cards_to_u64(&[card!(King, Hearts)]),
        ];

        let card_order_nums = card_order
            .iter()
            .map(|hand| evaluator.cards_to_u64(hand))
            .collect();
        Game {
            root,
            card_order: card_order_nums,
            evaluator,
        }
    }

    pub fn perform_iter(&mut self, iter: usize) {
        let start = Instant::now();

        let _ = self.root.evaluate_state(
            &Vector::ones(),
            &Vector::ones(),
            &self.evaluator,
            &self.card_order,
            Small,
        );

        let _ = self.root.evaluate_state(
            &Vector::ones(),
            &Vector::ones(),
            &self.evaluator,
            &self.card_order,
            Big,
        );
        let iter_time = start.elapsed().as_secs_f32();
        if iter % 1 == 0 {
            let [exp_sb, exp_bb] = self.root.calc_exploit(
                &Vector::ones(),
                &Vector::ones(),
                &self.evaluator,
                &self.card_order,
            );
            println!(
                "Iteration {} done \n\
                  Exploitability: {} mb/h \n\
                  Iter time: {} \n\
                  Exploit calc time: {} \n",
                iter,
                (exp_sb.sum() + exp_bb.sum()) * 1000.0 / 1326.0 / 1255.0,
                iter_time,
                start.elapsed().as_secs_f32() - iter_time,
            );
        }
    }
}
