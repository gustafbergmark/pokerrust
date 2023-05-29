use std::fmt::{Debug, Formatter};

// holds historic winnings of each move and hand
#[derive(Clone)]
pub(crate) struct Strategy {
    regrets: u16,
    strategy_sum: u16,
}

impl Debug for Strategy {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.get_average_strategy())
    }
}

impl Strategy {
    pub fn new() -> Self {
        Strategy {
            regrets: 0,
            strategy_sum: 0,
        }
    }

    pub fn update_add(&mut self, update: &[f32]) {
        let mut regrets = Self::from_discrete(self.regrets);
        for i in 0..2 {
            regrets[i] += update[i];
        }
        self.regrets = Self::to_discrete(&regrets);
    }

    pub fn get_strategy(&mut self, realization_weight: f32, iteration_weight: f32) -> [f32; 2] {
        let regret_match: [f32; 2] = Self::from_discrete(self.regrets);
        let mut strat_sum = Self::from_discrete(self.strategy_sum);
        for i in 0..2 {
            strat_sum[i] = strat_sum[i] * (1.0 - 1.0 / iteration_weight)
                + regret_match[i] * realization_weight * (1.0 / iteration_weight);
        }
        self.strategy_sum = Self::to_discrete(&strat_sum);
        regret_match
    }

    pub fn get_average_strategy(&self) -> [f32; 2] {
        Self::from_discrete(self.strategy_sum)
    }

    fn normalize(v: &[f32; 2]) -> [f32; 2] {
        let mut res = v.clone();
        res.iter_mut()
            .for_each(|elem| *elem = if *elem > 0.0 { *elem } else { 0.0 });
        let norm: f32 = res.iter().sum();
        if norm != 0.0 {
            res.iter_mut().for_each(|elem| *elem /= norm);
            res
        } else {
            [0.5; 2]
        }
    }

    fn to_discrete(v: &[f32; 2]) -> u16 {
        let v = Self::normalize(v);
        (v[0] * u16::MAX as f32).round() as u16
    }

    fn from_discrete(v: u16) -> [f32; 2] {
        let u = u16::MAX - v;
        let ret = [v as f32, u as f32];
        Self::normalize(&ret)
    }
}
