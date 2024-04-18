use crate::cuda_interface::{
    download_strategy_gpu, evaluate_gpu, free_eval, transfer_flop_eval, upload_strategy_gpu,
};
use crate::enums::Player::{Big, Small};
use crate::evaluator::Evaluator;
use crate::state::{Pointer, State};
use crate::vector::Vector;
use poker::Card;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::fmt::{Debug, Formatter};
use std::time::Instant;

pub const TURNS: usize = 6;
pub const RIVERS: usize = 6;

pub(crate) struct Game<const M: usize> {
    root: State<M>,
    evaluator: Evaluator<M>,
    builder: Pointer,
}

impl<const M: usize> Debug for Game<M> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.root.fmt(f)
    }
}

impl<const M: usize> Game<M> {
    pub fn new(root: State<M>, evaluator: Evaluator<M>, builder: Pointer) -> Self {
        Game {
            root,
            evaluator,
            builder,
        }
    }

    pub fn perform_iter(&mut self, iter: usize) {
        let mut flops = self
            .evaluator
            .flops
            .clone()
            .into_iter()
            .enumerate()
            .collect::<Vec<_>>();
        flops.shuffle(&mut thread_rng());
        for (index, flop) in flops {
            println!(
                "Starting iteration on fixed flop {:?}",
                self.evaluator.u64_to_cards(flop)
            );
            let _start = Instant::now();
            let mut turns = vec![];
            let mut rivers = vec![];
            let mut gputime = 0;
            for _ in 0..TURNS {
                let deck = Card::generate_shuffled_deck()
                    .into_iter()
                    .filter(|&&elem| self.evaluator.cards_to_u64(&[elem]) & flop == 0)
                    .cloned()
                    .collect::<Vec<_>>();
                turns.push(self.evaluator.cards_to_u64(&[deck[0]]));
                for i in 1..=RIVERS {
                    rivers.push(self.evaluator.cards_to_u64(&[deck[i]]))
                }
            }
            assert_eq!(turns.len(), TURNS);
            assert_eq!(rivers.len(), TURNS * RIVERS);
            self.evaluator.get_flop_eval(flop);
            println!(
                "Flop eval created in in {}s",
                _start.elapsed().as_secs_f32()
            );
            let eval_ptr = Pointer(transfer_flop_eval(
                &self.evaluator,
                flop,
                turns.clone(),
                rivers.clone(),
            ));
            upload_strategy_gpu(self.builder, index as i32);

            if cfg!(feature = "GPU") {
                let _ = self.root.evaluate_state(
                    &Vector::ones(),
                    &Vector::ones(),
                    &self.evaluator,
                    Small,
                    false,
                    0,
                    self.builder,
                    true,
                    0,
                    flop,
                    &turns,
                );
                let s = Instant::now();
                evaluate_gpu(self.builder, eval_ptr, Small, false);
                gputime += s.elapsed().as_millis();
            }
            let _ = self.root.evaluate_state(
                &Vector::ones(),
                &Vector::ones(),
                &self.evaluator,
                Small,
                false,
                0,
                self.builder,
                false,
                0,
                flop,
                &turns,
            );

            if cfg!(feature = "GPU") {
                let _ = self.root.evaluate_state(
                    &Vector::ones(),
                    &Vector::ones(),
                    &self.evaluator,
                    Big,
                    false,
                    0,
                    self.builder,
                    true,
                    0,
                    flop,
                    &turns,
                );
                let s = Instant::now();
                evaluate_gpu(self.builder, eval_ptr, Big, false);
                gputime += s.elapsed().as_millis();
            }

            let _ = self.root.evaluate_state(
                &Vector::ones(),
                &Vector::ones(),
                &self.evaluator,
                Big,
                false,
                0,
                self.builder,
                false,
                0,
                flop,
                &turns,
            );
            println!(
                "Iteration time: {}s, GPU time {}ms",
                _start.elapsed().as_secs_f32(),
                gputime
            );
            // Exploitability calculation must be redone for public chance sampling
            // let iter_time = _start.elapsed().as_secs_f32();
            // if iter % 10 == 0 {
            //     if cfg!(feature = "GPU") {
            //         let _ = self.root.evaluate_state(
            //             &Vector::ones(),
            //             &Vector::ones(),
            //             &self.evaluator,
            //             Small,
            //             true,
            //             0,
            //             self.builder,
            //             true,
            //             0,
            //             flop,
            //         );
            //         evaluate_gpu(self.builder, eval_ptr, Small, true);
            //     }
            //     let exp_sb = self.root.evaluate_state(
            //         &Vector::ones(),
            //         &Vector::ones(),
            //         &self.evaluator,
            //         Small,
            //         true,
            //         0,
            //         self.builder,
            //         false,
            //         0,
            //         flop,
            //     );
            //     if cfg!(feature = "GPU") {
            //         let _ = self.root.evaluate_state(
            //             &Vector::ones(),
            //             &Vector::ones(),
            //             &self.evaluator,
            //             Big,
            //             true,
            //             0,
            //             self.builder,
            //             true,
            //             0,
            //             flop,
            //         );
            //         evaluate_gpu(self.builder, eval_ptr, Big, true);
            //     }
            //     let exp_bb = self.root.evaluate_state(
            //         &Vector::ones(),
            //         &Vector::ones(),
            //         &self.evaluator,
            //         Big,
            //         true,
            //         0,
            //         self.builder,
            //         false,
            //         0,
            //         flop,
            //     );
            //     println!(
            //         "Iteration {} done \n\
            //       Exploitability: {} mb/h \n\
            //       Iter time: {} \n\
            //       Exploit calc time: {} \n",
            //         iter,
            //         (exp_sb.sum() + exp_bb.sum()) * 1000.0 / 1326.0 / 1255.0 / 2.0, // 1000 for milli, 1326 for own hands, 1255 for opponent, 2 for two strategies
            //         iter_time,
            //         _start.elapsed().as_secs_f32() - iter_time,
            //     );
            // }
            free_eval(eval_ptr);
            download_strategy_gpu(self.builder, index as i32);
        }
    }
}
