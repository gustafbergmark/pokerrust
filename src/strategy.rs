use crate::vector::Float;
use crate::vector::Vector;
use std::fmt::{Debug, Formatter};
use std::iter::zip;

// holds historic winnings of each move and hand
#[derive(Clone, PartialEq)]
pub(crate) struct Strategy {
    regrets: Vec<Vector>,
}

impl Debug for Strategy {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.get_strategy())
    }
}

impl Strategy {
    pub fn new() -> Self {
        Strategy { regrets: vec![] }
    }

    pub fn add_strategy(&mut self) {
        self.regrets.push(Vector::default());
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
