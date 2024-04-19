use crate::evaluator::Evaluator;
use crate::vector::Float;
use crate::vector::Vector;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt::{Debug, Formatter};
use std::iter::zip;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) enum Strategy<const M: usize> {
    Regular(RegularStrategy),
    Abstract(AbstractStrategy<M>),
}

impl<const M: usize> Strategy<M> {
    pub fn add_strategy(&mut self) {
        match self {
            Strategy::Regular(strategy) => strategy.add_strategy(),
            Strategy::Abstract(strategy) => strategy.add_strategy(),
        }
    }

    pub fn save(&self, buf: &mut VecDeque<Float>) {
        match self {
            Strategy::Regular(strategy) => strategy.save(buf),
            Strategy::Abstract(strategy) => strategy.save(buf),
        }
    }

    pub fn load(&mut self, buf: &mut VecDeque<Float>) {
        match self {
            Strategy::Regular(strategy) => strategy.load(buf),
            Strategy::Abstract(strategy) => strategy.load(buf),
        }
    }

    pub fn update_add(&mut self, updates: &Vec<Vector>, evaluator: &Evaluator<M>, cards: u64) {
        match self {
            Strategy::Regular(strategy) => strategy.update_add(updates),
            Strategy::Abstract(strategy) => strategy.update_add(updates, evaluator, cards),
        }
    }

    pub fn get_strategy(&self, evaluator: &Evaluator<M>, cards: u64) -> Vec<Vector> {
        match self {
            Strategy::Regular(strategy) => strategy.get_strategy(),
            Strategy::Abstract(strategy) => strategy.get_strategy(evaluator, cards),
        }
    }
}

// holds historic winnings of each move and hand
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct RegularStrategy {
    regrets: Vec<Vector>,
}

impl Debug for RegularStrategy {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.get_strategy())
    }
}

impl RegularStrategy {
    pub fn new() -> Self {
        Self { regrets: vec![] }
    }

    pub fn add_strategy(&mut self) {
        self.regrets.push(Vector::default());
    }

    pub fn save(&self, buf: &mut VecDeque<Float>) {
        for action in &self.regrets {
            for &val in &action.values {
                buf.push_back(val);
            }
        }
    }

    pub fn load(&mut self, buf: &mut VecDeque<Float>) {
        for action in self.regrets.iter_mut() {
            for i in 0..1326 {
                action[i] = buf.pop_front().unwrap();
            }
        }
    }

    pub fn update_add(&mut self, updates: &Vec<Vector>) {
        assert_eq!(self.regrets.len(), updates.len());
        for (regret, update) in zip(self.regrets.iter_mut(), updates.iter()) {
            *regret += *update;
            regret
                .values
                .iter_mut()
                .for_each(|elem| *elem = elem.max(0.0));
        }
    }

    pub fn get_strategy(&self) -> Vec<Vector> {
        let mut regret_match = self.regrets.clone();
        let sum: Vector = regret_match.iter().cloned().sum();
        for k in 0..self.regrets.len() {
            for i in 0..1326 {
                if sum[i] <= 1e-4 {
                    regret_match[k][i] = 1.0 / (self.regrets.len() as Float);
                } else {
                    regret_match[k][i] /= sum[i];
                }
            }
        }
        regret_match
    }
}

// holds historic winnings of each move and hand
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct AbstractStrategy<const M: usize> {
    regrets: Vec<Vec<Float>>,
    updates: Vec<Vec<Float>>,
}

impl<const M: usize> Debug for AbstractStrategy<M> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "AbstractStrategy")
    }
}

impl<const M: usize> AbstractStrategy<M> {
    pub fn new() -> Self {
        AbstractStrategy {
            regrets: vec![],
            updates: vec![],
        }
    }

    pub fn save(&self, buf: &mut VecDeque<Float>) {
        for action in &self.regrets {
            for &val in action {
                buf.push_back(val);
            }
        }
    }

    pub fn load(&mut self, buf: &mut VecDeque<Float>) {
        for action in self.regrets.iter_mut() {
            for i in 0..M {
                action[i] = buf.pop_front().unwrap();
            }
        }
    }

    pub fn add_strategy(&mut self) {
        self.regrets.push(vec![0.0; M]);
    }

    pub fn update_add(&mut self, updates: &Vec<Vector>, evaluator: &Evaluator<M>, cards: u64) {
        // Allocate only when this subgame is in focus
        if self.updates.is_empty() {
            self.updates = vec![vec![0.0; M]; self.regrets.len()]
        }
        let abstraction = evaluator.abstractions(cards);
        for i in 0..1326 {
            let abstract_index = abstraction[i] as usize;
            assert!(abstract_index < M);
            for k in 0..self.regrets.len() {
                self.updates[k][abstract_index] += updates[k][i]
            }
        }
    }

    pub fn get_strategy(&self, evaluator: &Evaluator<M>, cards: u64) -> Vec<Vector> {
        let abstraction = evaluator.abstractions(cards);
        let mut regret_match = vec![Vector::default(); self.regrets.len()];
        let mut sum = [0.0; M];
        for i in 0..M {
            for j in 0..self.regrets.len() {
                sum[i] += self.regrets[j][i];
            }
        }
        for i in 0..1326 {
            let abstract_index = abstraction[i] as usize;
            if abstract_index >= M {
                dbg!(abstract_index);
            }
            assert!(abstract_index < M);
            for k in 0..self.regrets.len() {
                if sum[abstract_index] <= 1e-4 {
                    regret_match[k][i] = 1.0 / (self.regrets.len() as Float);
                } else {
                    regret_match[k][i] = self.regrets[k][abstract_index] / sum[abstract_index];
                }
            }
        }
        regret_match
    }

    pub fn apply_updates(&mut self) {
        for k in 0..self.regrets.len() {
            for i in 0..M {
                self.regrets[k][i] += self.updates[k][i];
                self.regrets[k][i] = self.regrets[k][i].max(0.0);
                self.updates[k][i] = 0.0;
            }
        }
        // Deallocate update memory
        self.updates = vec![];
    }
}
