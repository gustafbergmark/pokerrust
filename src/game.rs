use crate::cuda_interface::{evaluate_gpu, free_eval, transfer_flop_eval};
use crate::enums::Player::{Big, Small};
use crate::evaluator::Evaluator;
use crate::state::{Pointer, State};
use crate::vector::Vector;
use std::fmt::{Debug, Formatter};
use std::time::Instant;

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
        let start = Instant::now();
        // 7 is the fixed flop
        let eval_ptr = Pointer(transfer_flop_eval(&self.evaluator, 7));

        if cfg!(feature = "GPU") {
            let _ = self.root.evaluate_state(
                &Vector::ones(),
                &Vector::ones(),
                &self.evaluator,
                Small,
                false,
                0,
                self.builder,
                false,
            );
            evaluate_gpu(self.builder, eval_ptr, Small, false);
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
        );
        println!("Iteration time: {}s", start.elapsed().as_secs_f32());

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
            );
            evaluate_gpu(self.builder, eval_ptr, Big, false);
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
        );
        let iter_time = start.elapsed().as_secs_f32();
        if iter % 10 == 0 {
            if cfg!(feature = "GPU") {
                let _ = self.root.evaluate_state(
                    &Vector::ones(),
                    &Vector::ones(),
                    &self.evaluator,
                    Small,
                    true,
                    0,
                    self.builder,
                    true,
                );
                evaluate_gpu(self.builder, eval_ptr, Small, true);
            }
            let exp_sb = self.root.evaluate_state(
                &Vector::ones(),
                &Vector::ones(),
                &self.evaluator,
                Small,
                true,
                0,
                self.builder,
                false,
            );
            if cfg!(feature = "GPU") {
                let _ = self.root.evaluate_state(
                    &Vector::ones(),
                    &Vector::ones(),
                    &self.evaluator,
                    Big,
                    true,
                    0,
                    self.builder,
                    true,
                );
                evaluate_gpu(self.builder, eval_ptr, Big, true);
            }
            let exp_bb = self.root.evaluate_state(
                &Vector::ones(),
                &Vector::ones(),
                &self.evaluator,
                Big,
                true,
                0,
                self.builder,
                false,
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
        free_eval(eval_ptr);
    }
}
