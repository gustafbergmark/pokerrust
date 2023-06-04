use crate::vector::Vector;
use std::fmt::{Debug, Formatter};

// holds historic winnings of each move and hand
#[derive(Clone)]
pub(crate) struct Strategy {
    regrets: [Vector; 2],
}

impl Debug for Strategy {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.get_strategy())
    }
}

impl Strategy {
    pub fn new() -> Self {
        Strategy {
            regrets: [Vector::default(); 2],
        }
    }

    pub fn update_add(&mut self, update: &[Vector; 2]) {
        for i in 0..2 {
            self.regrets[i] += update[i];
            self.regrets[i]
                .values
                .iter_mut()
                .for_each(|elem| *elem = elem.max(0.0));
        }
    }

    pub fn get_strategy(&self) -> [Vector; 2] {
        let regret_match: [Vector; 2] = self.regrets.clone();
        let normalized = Self::normalize(&regret_match);
        normalized
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
