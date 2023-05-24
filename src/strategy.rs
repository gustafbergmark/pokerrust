use std::fmt::{Debug, Formatter};

// holds historic winnings of each move and hand
#[derive(Clone)]
pub(crate) struct Strategy<const M: usize> {
    regrets: [f32; M],
    strategy_sum: [f32; M],
}

impl<const M: usize> Debug for Strategy<M> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.get_average_strategy())
    }
}

impl<const M: usize> Strategy<M> {
    pub fn new() -> Self {
        Strategy {
            regrets: [0.0; M],
            strategy_sum: [0.0; M],
        }
    }

    pub fn update_add(&mut self, update: &[f32]) {
        for i in 0..self.regrets.len() {
            self.regrets[i] += update[i];
        }
    }

    pub fn get_strategy(&mut self, realization_weight: f32, iteration_weight: f32) -> [f32; M] {
        let mut regret_match: [f32; M] = self.regrets.clone();
        regret_match
            .iter_mut()
            .for_each(|elem| *elem = if *elem > 0.0 { *elem } else { 0.0 });
        let normalized = Self::normalize(&regret_match);
        for i in 0..self.strategy_sum.len() {
            self.strategy_sum[i] += normalized[i] * realization_weight * iteration_weight;
        }
        normalized
    }

    pub fn get_average_strategy(&self) -> [f32; M] {
        Self::normalize(&self.strategy_sum)
    }

    fn normalize(v: &[f32; M]) -> [f32; M] {
        let norm: f32 = v.iter().sum();
        if norm != 0.0 {
            let mut res = v.clone();
            res.iter_mut().for_each(|elem| *elem /= norm);
            res
        } else {
            [1.0 / (v.len() as f32); M]
        }
    }
}
