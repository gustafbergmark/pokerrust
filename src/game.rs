use crate::cuda_interface::{
    download_strategy_gpu, evaluate_gpu, free_eval, initialize_builder, set_builder_memory,
    transfer_flop_eval, upload_strategy_gpu,
};
use crate::enums::Player::{Big, Small};
use crate::evaluator::Evaluator;
use crate::state::{Pointer, State};
use crate::vector::{Float, Vector};
use poker::Card;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::hash_map::DefaultHasher;
use std::collections::VecDeque;
use std::fmt::{Debug, Formatter};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufReader, BufWriter};
use std::time::Instant;

pub const TURNS: usize = 49;
pub const RIVERS: usize = 48;

pub const FLOP_STRATEGY_SIZE: usize = 63 * 9 * 26 * 256;

pub(crate) struct Game<const M: usize> {
    root: State<M>,
    evaluator: Evaluator<M>,
    builder: Pointer,
    blob: Vec<Float>,
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
            blob: vec![0.0; 63 * 9 * 26 * 1755 * 256],
        }
    }

    pub fn save(&self) {
        println!("Saving game");
        let mut buf = VecDeque::new();
        self.root.save(&mut buf);
        buf.make_contiguous();
        let mut hasher = DefaultHasher::new();
        assert_eq!(buf.len(), buf.as_slices().0.len());
        as_bytes(buf.as_slices().0).hash(&mut hasher);
        dbg!(buf.len(), hasher.finish());
        match std::fs::write(
            "./files/game.bin",
            bincode::serialize(&buf).expect("Failed to serialize"),
        ) {
            Ok(_) => println!("Saved game"),
            Err(e) => panic!("{}", e),
        }
        let blob = as_bytes(&self.blob);
        hasher = DefaultHasher::new();
        blob.hash(&mut hasher);
        println!("Saving blob hash: {}", hasher.finish());

        match bincode::serialize_into(
            BufWriter::new(File::create("./files/blob.bin").expect("Failed to open file")),
            &self.blob,
        ) {
            Ok(_) => println!("Saved blob"),
            Err(e) => panic!("{}", e),
        }
    }

    pub fn load(&mut self) {
        let start = Instant::now();
        let mut buf = match std::fs::read("./files/game.bin") {
            Ok(eval) => {
                let mut buf: VecDeque<Float> =
                    bincode::deserialize(&eval).expect("Failed to deserialize");
                buf.make_contiguous();
                let mut hasher = DefaultHasher::new();
                assert_eq!(buf.len(), buf.as_slices().0.len());
                as_bytes(buf.as_slices().0).hash(&mut hasher);
                dbg!(buf.len(), hasher.finish());
                buf
            }
            Err(e) => panic!("{}", e),
        };
        self.root.load(&mut buf);
        assert_eq!(buf.len(), 0);
        self.blob = bincode::deserialize_from(BufReader::new(
            File::open("./files/blob.bin").expect("Failed to open blob.bin"),
        ))
        .expect("Failed to deserialize blob");
        let mut hasher = DefaultHasher::new();
        let bytes = as_bytes(&self.blob);
        bytes.hash(&mut hasher);
        println!("Deserialized blob with hash {}", hasher.finish());
        println!("Loaded root in {}ms", start.elapsed().as_millis());
    }

    pub fn perform_iter(&mut self, iter: usize) {
        // let mut flops = self
        //     .evaluator
        //     .flops
        //     .clone()
        //     .into_iter()
        //     .enumerate()
        //     .collect::<Vec<_>>();
        // flops.shuffle(&mut thread_rng());
        let flops = vec![(0, 7)]; // Test with first possible flop
        let mut count = 0;
        for &(index, flop) in &flops {
            count += 1;
            let _start = Instant::now();
            let mut turns = vec![];
            let mut rivers = vec![];
            let mut gputime = 0;
            // for _ in 0..TURNS {
            //     let deck = Card::generate_shuffled_deck()
            //         .into_iter()
            //         .filter(|&&elem| self.evaluator.cards_to_u64(&[elem]) & flop == 0)
            //         .cloned()
            //         .collect::<Vec<_>>();
            //     turns.push(self.evaluator.cards_to_u64(&[deck[0]]));
            //     for i in 1..=RIVERS {
            //         rivers.push(self.evaluator.cards_to_u64(&[deck[i]]))
            //     }
            // }
            for i in 0..52 {
                let turn = 1_u64 << i;
                if turn & flop > 0 {
                    continue;
                }
                turns.push(turn);
                for j in 0..52 {
                    let river = 1_u64 << j;
                    if (river & (turn | flop)) > 0 {
                        continue;
                    }
                    rivers.push(river);
                }
            }
            assert_eq!(turns.len(), TURNS);
            assert_eq!(rivers.len(), TURNS * RIVERS);
            self.evaluator.get_flop_eval(flop, &turns, &rivers, true);
            let eval_ptr = Pointer(transfer_flop_eval(
                &self.evaluator,
                flop,
                turns.clone(),
                rivers.clone(),
                true,
            ));
            upload_strategy_gpu(
                self.builder,
                &self.blob[FLOP_STRATEGY_SIZE * index..FLOP_STRATEGY_SIZE * (index + 1)],
            );
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
                    flop,
                    &turns,
                    &mut 0,
                );
                let s = Instant::now();
                evaluate_gpu(self.builder, eval_ptr, Small, false);
                gputime += s.elapsed().as_millis();
            }
            let val1 = self.root.evaluate_state(
                &Vector::ones(),
                &Vector::ones(),
                &self.evaluator,
                Small,
                false,
                0,
                self.builder,
                false,
                flop,
                &turns,
                &mut 0,
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
                    flop,
                    &turns,
                    &mut 0,
                );
                let s = Instant::now();
                evaluate_gpu(self.builder, eval_ptr, Big, false);
                gputime += s.elapsed().as_millis();
            }

            let val2 = self.root.evaluate_state(
                &Vector::ones(),
                &Vector::ones(),
                &self.evaluator,
                Big,
                false,
                0,
                self.builder,
                false,
                flop,
                &turns,
                &mut 0,
            );
            // println!(
            //     "Iteration time: {}s, GPU time {}ms, sum of values: {} {}",
            //     _start.elapsed().as_secs_f32(),
            //     gputime,
            //     val1.sum() * 1000.0 / 1326.0 / 1255.0,
            //     val2.sum() * 1000.0 / 1326.0 / 1255.0
            // );
            let iter_time = _start.elapsed().as_secs_f32();
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
                        flop,
                        &vec![],
                        &mut 0,
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
                    flop,
                    &vec![],
                    &mut 0,
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
                        flop,
                        &vec![],
                        &mut 0,
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
                    flop,
                    &vec![],
                    &mut 0,
                );
                println!(
                    "Iteration {} done \n\
                          Exploitability: {} mb/h \n\
                          Iter time: {} \n\
                          Exploit calc time: {} \n",
                    iter,
                    (exp_sb.sum() + exp_bb.sum()) * 1000.0 / 1326.0 / 1255.0 / 2.0, // 1000 for milli, 1326 for own hands, 1255 for opponent, 2 for two strategies
                    iter_time,
                    _start.elapsed().as_secs_f32() - iter_time,
                );
            }

            free_eval(eval_ptr);
            download_strategy_gpu(
                self.builder,
                &mut self.blob[FLOP_STRATEGY_SIZE * index..FLOP_STRATEGY_SIZE * (index + 1)],
            );
        }
    }
}

fn as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const _, v.len() * 4) }
}

fn from_bytes(v: Vec<u8>) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const f32, v.len() / 4).to_vec() }
}
