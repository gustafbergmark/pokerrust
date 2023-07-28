use crate::enums::Player::{Big, Small};
use crate::evaluator::Evaluator;
use crate::permutation_handler::PermutationHandler;
use crate::state::State;
use crate::vector::Vector;
use poker::{card, Card};
use std::fmt::{Debug, Formatter};
use std::time::Instant;

pub(crate) struct Game {
    root: State,
    card_order: Vec<u64>,
    evaluator: Evaluator,
    permuter: PermutationHandler,
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
        let permuter = PermutationHandler::new(&card_order);
        Game {
            root,
            card_order: card_order_nums,
            evaluator,
            permuter,
        }
    }

    pub fn perform_iter(&mut self, iter: usize) {
        let start = Instant::now();

        let utilsb = self.root.evaluate_state(
            &Vector::ones(),
            &Vector::ones(),
            &self.evaluator,
            &self.card_order,
            Small,
            &self.permuter,
        );

        let utilbb = self.root.evaluate_state(
            &Vector::ones(),
            &Vector::ones(),
            &self.evaluator,
            &self.card_order,
            Big,
            &self.permuter,
        );
        if iter % 1 == 0 {
            let [exp_sb, exp_bb] = self.root.calc_exploit(
                &Vector::ones(),
                &Vector::ones(),
                &self.evaluator,
                &self.card_order,
                &self.permuter,
            );
            println!(
                "Iteration {} done \n\
                  Exploitability: {} mb/h \n\
                  Time: {} \n",
                iter,
                (exp_sb.sum() + exp_bb.sum()) * 1000.0 / 1326.0,
                start.elapsed().as_secs_f32(),
            );
        }
    }
}
