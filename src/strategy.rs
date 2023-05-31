use std::fmt::{Debug, Formatter};

// holds historic winnings of each move and hand
#[derive(Clone)]
pub(crate) struct Strategy {
    regrets: [[f32; 1326]; 2],
    strategy_sum: [[f32; 1326]; 2],
}

impl Debug for Strategy {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.get_average_strategy())
    }
}

impl Strategy {
    pub fn new() -> Self {
        Strategy {
            regrets: [[0.0; 1326]; 2],
            strategy_sum: [[0.0; 1326]; 2],
        }
    }

    pub fn update_add(&mut self, update: &[[f32; 1326]; 2]) {
        for i in 0..2 {
            for j in 0..1326 {
                self.regrets[i][j] += update[i][j];
                if self.regrets[i][j] < 0.0 {
                    self.regrets[i][j] = 0.0
                }
            }
        }
    }

    pub fn get_strategy(&mut self, iteration_weight: f32) -> [[f32; 1326]; 2] {
        let regret_match: [[f32; 1326]; 2] = self.regrets.clone();
        let normalized = Self::normalize(&regret_match);
        for i in 0..2 {
            for j in 0..1326 {
                self.strategy_sum[i][j] *= iteration_weight / (iteration_weight + 1.0);
                self.strategy_sum[i][j] += normalized[i][j];
            }
        }
        normalized
    }

    pub fn get_average_strategy(&self) -> [[f32; 1326]; 2] {
        Self::normalize(&self.strategy_sum)
    }

    fn normalize(v: &[[f32; 1326]; 2]) -> [[f32; 1326]; 2] {
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
