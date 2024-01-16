use crate::enums::Player::{Big, Small};
use crate::evaluator::Evaluator;
use crate::state::State;
use crate::vector::Vector;
use poker::card;
use std::fmt::{Debug, Formatter};
use std::time::Instant;

pub(crate) struct Game<'a> {
    root: State,
    evaluator: Evaluator<'a>,
}

impl Debug for Game<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.root.fmt(f)
    }
}

impl<'a> Game<'a> {
    pub fn new(root: State, evaluator: Evaluator<'a>) -> Self {
        let _kuhn_hands: Vec<u64> = vec![
            evaluator.cards_to_u64(&[card!(Jack, Hearts)]),
            evaluator.cards_to_u64(&[card!(Queen, Hearts)]),
            evaluator.cards_to_u64(&[card!(King, Hearts)]),
        ];
        Game { root, evaluator }
    }

    pub fn perform_iter(&mut self, iter: usize) {
        let start = Instant::now();

        let _ = self
            .root
            .evaluate_state(&Vector::ones(), &self.evaluator, Small, false);

        let _ = self
            .root
            .evaluate_state(&Vector::ones(), &self.evaluator, Big, false);
        let iter_time = start.elapsed().as_secs_f32();
        if iter % 10 == 0 {
            let exp_sb = self
                .root
                .evaluate_state(&Vector::ones(), &self.evaluator, Small, true);
            let exp_bb = self
                .root
                .evaluate_state(&Vector::ones(), &self.evaluator, Big, true);
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
