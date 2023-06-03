use crate::enums::Player::{Big, Small};
use crate::evaluator::Evaluator;
use crate::permutation_handler::PermutationHandler;
use crate::state::State;
use crate::vector::Vector;
use itertools::Itertools;
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
        let card_order = _all_hands;
        //let card_order = all_hands;
        Game {
            root,
            card_order,
            evaluator,
            permuter: PermutationHandler::new(),
        }
    }

    pub fn perform_iter(&mut self, iteration_weight: f32, calc_exploit: bool) -> [Vector; 2] {
        let start = Instant::now();
        let [_, _, exp_sb] = self.root.evaluate_state(
            &Vector::ones(),
            &Vector::ones(),
            &self.evaluator,
            iteration_weight,
            &self.card_order,
            Small,
            &self.permuter,
            calc_exploit,
        );

        let [util_sb, util_bb, exp_bb] = self.root.evaluate_state(
            &Vector::ones(),
            &Vector::ones(),
            &self.evaluator,
            iteration_weight,
            &self.card_order,
            Big,
            &self.permuter,
            calc_exploit,
        );
        //dbg!(&self.root.card_strategies);
        let sb_avg = exp_sb.values.iter().sum::<f32>() / 1326.0 / 1225.0; // 1225 = 50 choose 2, the number of hands each hand play against
        let bb_avg = exp_bb.values.iter().sum::<f32>() / 1326.0 / 1225.0;
        if calc_exploit || true {
            println!(
                "Iteration {} done \n\
                  Exploitability: {} mb/h \n\
                  Time: {} \n",
                iteration_weight as i32,
                (sb_avg + bb_avg) * 1000.0,
                start.elapsed().as_secs_f32(),
            );
        }

        [util_sb, util_bb]
    }
}
