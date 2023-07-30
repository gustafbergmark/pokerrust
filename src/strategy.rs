use crate::vector::Vector;
use std::fmt::{Debug, Formatter};

// holds historic winnings of each move and hand
#[derive(Clone, PartialEq)]
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
        let mut regret_match: [Vector; 2] = self.regrets;
        Self::normalize(&mut regret_match);
        regret_match
    }

    fn normalize(v: &mut [Vector; 2]) {
        let norm = v[0] + v[1];
        v[0] /= norm;
        v[1] /= norm;
    }
}
