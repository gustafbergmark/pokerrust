use crate::evaluator::Evaluator;
use crate::vector::Float;
use crate::vector::Vector;
use std::fmt::{Debug, Formatter};
use std::iter::zip;

#[derive(Debug, Clone, PartialEq)]
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

    pub fn update_add(
        &mut self,
        updates: &Vec<Vector>,
        opponent_range: &Vector,
        evaluator: &Evaluator,
        cards: u64,
    ) {
        match self {
            Strategy::Regular(strategy) => strategy.update_add(updates),
            Strategy::Abstract(strategy) => {
                strategy.update_add(updates, opponent_range, evaluator, cards)
            }
        }
    }

    pub fn get_strategy(
        &self,
        opponent_range: &Vector,
        evaluator: &Evaluator,
        cards: u64,
    ) -> Vec<Vector> {
        match self {
            Strategy::Regular(strategy) => strategy.get_strategy(),
            Strategy::Abstract(strategy) => strategy.get_strategy(opponent_range, evaluator, cards),
        }
    }
}

// holds historic winnings of each move and hand
#[derive(Clone, PartialEq)]
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
#[derive(Clone, PartialEq)]
pub(crate) struct AbstractStrategy<const M: usize> {
    regrets: Vec<[Float; M]>,
}

impl<const M: usize> Debug for AbstractStrategy<M> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "AbstractStrategy")
    }
}

impl<const M: usize> AbstractStrategy<M> {
    pub fn new() -> Self {
        AbstractStrategy { regrets: vec![] }
    }

    pub fn add_strategy(&mut self) {
        self.regrets.push([Float::default(); M]);
    }

    pub fn update_add(
        &mut self,
        updates: &Vec<Vector>,
        opponent_range: &Vector,
        evaluator: &Evaluator,
        cards: u64,
    ) {
        let phs = phs(opponent_range, evaluator, cards);
        for i in 0..1326 {
            let abstract_index = (phs[i] * M as Float).floor() as usize;
            for k in 0..self.regrets.len() {
                self.regrets[k][abstract_index] += updates[k][i]
            }
        }
        for k in 0..self.regrets.len() {
            for i in 0..M {
                self.regrets[k][i] = self.regrets[k][i].max(0.0);
            }
        }
    }

    pub fn get_strategy(
        &self,
        opponent_range: &Vector,
        evaluator: &Evaluator,
        cards: u64,
    ) -> Vec<Vector> {
        let phs = phs(opponent_range, evaluator, cards);
        let mut regret_match = vec![Vector::default(); self.regrets.len()];
        let mut sum = [0.0; M];
        for i in 0..M {
            for j in 0..self.regrets.len() {
                sum[i] += self.regrets[j][i];
            }
        }
        for i in 0..1326 {
            let abstract_index = (phs[i] * M as Float).floor() as usize;
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
}

fn phs(opponent_range: &Vector, evaluator: &Evaluator, cards: u64) -> Vector {
    let card_order = evaluator.card_order();
    let mut total = Vector::default();
    let mut range_sum = 0.0;
    let mut collisions = [0.0; 52];
    for (index, &cards) in card_order.iter().enumerate() {
        range_sum += opponent_range[index];
        let card = Evaluator::separate_cards(cards);
        for c in card {
            collisions[c] += opponent_range[index];
        }
    }
    for index in 0..1326 {
        total[index] = range_sum + opponent_range[index];
        let cards = Evaluator::separate_cards(card_order[index]);
        for card in cards {
            total[index] -= collisions[card];
        }
    }

    let sorted = &evaluator.vectorized_eval(cards);
    let mut groups = vec![];
    let mut current = vec![sorted[0] & 2047];
    for &eval in sorted[1..1326].iter() {
        if eval & 2048 > 0 {
            groups.push(current);
            current = vec![];
        }
        current.push(eval & 2047);
    }
    assert!(!current.is_empty());
    groups.push(current);

    let mut collisions = [0.0; 52];

    let mut cumulative = 0.0;
    let mut worse = Vector::default();

    for group in groups.iter() {
        let mut current_cumulative = 0.0;

        let mut current_collisions = [0.0; 52];
        for &index in group {
            let index = index as usize;
            let cards = card_order[index];
            let card = Evaluator::separate_cards(cards);
            worse[index] += cumulative;
            current_cumulative += opponent_range[index];
            for c in card {
                worse[index] -= collisions[c];
                current_collisions[c] += opponent_range[index];
            }
        }
        cumulative += current_cumulative;
        for i in 0..52 {
            collisions[i] += current_collisions[i];
        }
    }

    let mut collisions = [0.0; 52];

    let mut cumulative = 0.0;
    let mut better = Vector::default();

    for group in groups.iter().rev() {
        let mut current_cumulative = 0.0;

        let mut current_collisions = [0.0; 52];
        for &index in group {
            let index = index as usize;
            let cards = card_order[index];
            let card = Evaluator::separate_cards(cards);
            better[index] -= cumulative;
            current_cumulative += opponent_range[index];
            for c in card {
                better[index] += collisions[c];
                current_collisions[c] += opponent_range[index];
            }
        }
        cumulative += current_cumulative;
        for i in 0..52 {
            collisions[i] += current_collisions[i];
        }
    }
    let mut res = Vector::default();
    for i in 0..1326 {
        // prob of worse + half prob of equal is phs
        if total[i] < 1e-5 {
            res[i] = 0.5;
        } else {
            res[i] = (worse[i] - better[i]) / (2.0 * total[i]) + 0.5;
        }
        res[i] = res[i].clamp(0.0, 1.0 - 1e-6);
        // println!("res[i]: {}", res[i]);
        if !(res[i] < 1.0) {
            println!("{} {} {} {}", res[i], worse[i], better[i], total[i]);
        }
        assert!(res[i] < 1.0);
        assert!(res[i] >= 0.0);
    }
    res
}
