use crate::vector::Vector;
use std::fmt::{Debug, Formatter};

// holds historic winnings of each move and hand
#[derive(Clone)]
pub(crate) struct Strategy {
    regrets: [Vector; 2],
    strategy_sum: [Vector; 2],
}

impl Debug for Strategy {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.get_average_strategy())
    }
}

impl Strategy {
    pub fn new() -> Self {
        Strategy {
            regrets: [Vector::default(); 2],
            strategy_sum: [Vector::default(); 2],
        }
    }

    pub fn update_add(&mut self, update: &[Vector; 2], calc_expoit: bool) {
        if !calc_expoit {
            for i in 0..2 {
                self.regrets[i] += update[i];
                self.regrets[i]
                    .values
                    .iter_mut()
                    .for_each(|elem| *elem = elem.max(0.0));
            }
        }
    }

    pub fn get_strategy(&mut self, iteration_weight: f32, calc_exploit: bool) -> [Vector; 2] {
        if !calc_exploit {
            let regret_match: [Vector; 2] = self.regrets.clone();
            let normalized = Self::normalize(&regret_match);
            for i in 0..2 {
                let discount = iteration_weight / (iteration_weight + 1.0);
                self.strategy_sum[i] *= discount;
                self.strategy_sum[i] += normalized[i];
            }
            normalized
        } else {
            self.get_average_strategy()
        }
    }

    pub fn get_average_strategy(&self) -> [Vector; 2] {
        Self::normalize(&self.strategy_sum)
    }

    fn normalize(v: &[Vector; 2]) -> [Vector; 2] {
        let mut res = v.clone();
        for i in 0..1326 {
            let mut norm = 0.0;
            for j in 0..2 {
                norm += v[j][i];
            }
            for j in 0..2 {
                if norm != 0.0 {
                    res[j][i] /= norm;
                } else {
                    res[j][i] = 0.5;
                }
            }
        }
        res
    }
}
