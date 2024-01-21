use crate::vector::Float;
use crate::vector::Vector;
use std::fmt::{Debug, Formatter};

// holds historic winnings of each move and hand
#[derive(Clone, PartialEq)]
pub(crate) struct Strategy {
    regrets: [Vector; 3],
}

impl Debug for Strategy {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.get_strategy(3))
    }
}

impl Strategy {
    pub fn new() -> Self {
        Strategy {
            regrets: [Vector::default(); 3],
        }
    }

    pub fn update_add(&mut self, update: &[Vector; 3]) {
        for i in 0..3 {
            self.regrets[i] += update[i];
            self.regrets[i]
                .values
                .iter_mut()
                .for_each(|elem| *elem = elem.max(0.0));
        }
    }

    pub fn get_strategy(&self, actions: usize) -> [Vector; 3] {
        let mut regret_match: [Vector; 3] = self.regrets;
        Self::normalize(&mut regret_match);
        for k in 0..actions {
            for i in 0..1326 {
                if regret_match[k][i].is_nan() {
                    regret_match[k][i] = 1.0 / (actions as Float);
                }
            }
        }
        for k in actions..3 {
            for i in 0..1326 {
                if regret_match[k][i].is_nan() {
                    regret_match[k][i] = 0.0;
                } else {
                    // Double check that the unused are all zeroes
                    assert_eq!(regret_match[k][i], 0.0);
                };
            }
        }

        regret_match
    }

    fn normalize(v: &mut [Vector; 3]) {
        let norm = v[0] + v[1] + v[2];
        v[0] /= norm;
        v[1] /= norm;
        v[2] /= norm;
    }
}
