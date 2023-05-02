use crate::game::Player::{Big, Small};
use crate::game::TerminalState::{BBWins, NonTerminal, SBWins, Showdown};
use itertools::Itertools;
use poker::{card, Card};
use std::fmt::{Debug, Formatter};
use std::iter::zip;

static CARD_ORDER: [Card; 3] = [
    card!(Jack, Hearts),
    card!(Queen, Hearts),
    card!(King, Hearts),
];

fn card_index(card: Card) -> usize {
    CARD_ORDER.iter().position(|elem| *elem == card).unwrap()
}

#[derive(Debug)]
pub(crate) struct Game<const M: usize> {
    root: State<M>,
}

impl<const M: usize> Game<M> {
    pub fn new(root: State<M>) -> Self {
        Game { root }
    }

    pub fn perform_iter(&mut self) -> f32 {
        let hands = CARD_ORDER.clone().into_iter().tuple_combinations();
        let mut ev = 0.0;
        for (c1, c2) in hands {
            ev += self.root.evaluate_state(c1, c2, 1.0, 1.0);
            ev += self.root.evaluate_state(c2, c1, 1.0, 1.0);
        }
        ev
    }
}

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

    pub fn get_strategy(&mut self, realization_weight: f32) -> [f32; M] {
        let mut regret_match: [f32; M] = self.regrets.clone();
        regret_match
            .iter_mut()
            .for_each(|elem| *elem = if *elem > 0.0 { *elem } else { 0.0 });
        let normalized = Self::normalize(&regret_match);
        for i in 0..self.strategy_sum.len() {
            self.strategy_sum[i] += normalized[i] * realization_weight;
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

#[derive(Clone, Debug)]
pub(crate) struct State<const M: usize> {
    terminal: TerminalState,
    sbbet: f32,
    bbbet: f32,
    next_to_act: Player,
    card_strategies: Vec<Strategy<M>>,
    actions: Vec<State<M>>,
}

impl<const M: usize> State<M> {
    pub fn new(terminal: TerminalState, sbbet: f32, bbbet: f32, next_to_act: Player) -> Self {
        State {
            terminal,
            sbbet,
            bbbet,
            next_to_act,
            card_strategies: vec![Strategy::new(); 3],
            actions: vec![],
        }
    }

    pub fn add_action(&mut self, state: State<M>) {
        self.actions.push(state);
    }

    pub fn get_action(&self, action: Action) -> State<M> {
        match self.next_to_act {
            Small => match action {
                Action::Fold => State {
                    terminal: BBWins,
                    sbbet: self.sbbet,
                    bbbet: self.bbbet,
                    next_to_act: Big,
                    card_strategies: vec![],
                    actions: vec![],
                },
                Action::Check => State {
                    terminal: NonTerminal,
                    sbbet: self.sbbet,
                    bbbet: self.bbbet,
                    next_to_act: Big,
                    card_strategies: vec![Strategy::new(); 3],
                    actions: vec![],
                },
                Action::Call => State {
                    terminal: Showdown,
                    sbbet: self.bbbet,
                    bbbet: self.bbbet,
                    next_to_act: Big,
                    card_strategies: vec![],
                    actions: vec![],
                },

                Action::Raise => State {
                    terminal: NonTerminal,
                    sbbet: self.bbbet + 1.0,
                    bbbet: self.bbbet,
                    next_to_act: Big,
                    card_strategies: vec![Strategy::new(); 3],
                    actions: vec![],
                },
            },
            Big => match action {
                Action::Fold => State {
                    terminal: SBWins,
                    sbbet: self.sbbet,
                    bbbet: self.bbbet,
                    next_to_act: Small,
                    card_strategies: vec![],
                    actions: vec![],
                },
                Action::Check => State {
                    terminal: NonTerminal,
                    sbbet: self.sbbet,
                    bbbet: self.bbbet,
                    next_to_act: Small,
                    card_strategies: vec![Strategy::new(); 3],
                    actions: vec![],
                },
                Action::Call => State {
                    terminal: Showdown,
                    sbbet: self.sbbet,
                    bbbet: self.sbbet,
                    next_to_act: Small,
                    card_strategies: vec![],
                    actions: vec![],
                },

                Action::Raise => State {
                    terminal: NonTerminal,
                    sbbet: self.sbbet,
                    bbbet: self.sbbet + 1.0,
                    next_to_act: Small,
                    card_strategies: vec![Strategy::new(); 3],
                    actions: vec![],
                },
            },
        }
    }

    pub fn get_card_strategy(&mut self, card: Card) -> &mut Strategy<M> {
        let i = card_index(card);
        &mut self.card_strategies[i]
    }

    pub fn evaluate_state(
        &mut self,
        card_sb: Card,
        card_bb: Card,
        prob_sb: f32,
        prob_bb: f32,
    ) -> f32 {
        match self.terminal {
            NonTerminal => {
                let (card, _) = match self.next_to_act {
                    Small => (card_sb, card_bb),
                    Big => (card_bb, card_sb),
                };
                let (prob, other_prob) = match self.next_to_act {
                    Small => (prob_sb, prob_bb),
                    Big => (prob_bb, prob_sb),
                };
                // get avg strategy and individual payoffs
                let mut avgstrat = 0.0;
                let mut payoffs = [0.0; 2];

                let strategy = self.get_card_strategy(card).get_strategy(other_prob);

                let mut i = 0;
                for (next, action_prob) in zip(self.actions.iter_mut(), strategy) {
                    let new_prob = prob * action_prob;

                    let util = match self.next_to_act {
                        Small => -next.evaluate_state(card_sb, card_bb, new_prob, prob_bb),
                        Big => -next.evaluate_state(card_sb, card_bb, prob_sb, new_prob),
                    };

                    avgstrat += util * action_prob;
                    payoffs[i] = util;
                    i += 1;
                }
                // update strategy
                let mut update = [0.0; 2];
                for (i, util) in payoffs.iter().enumerate() {
                    let diff = util - avgstrat;
                    let regret = diff * other_prob;
                    update[i] = regret;
                }
                self.get_card_strategy(card).update_add(&update);
                return avgstrat;
            }

            _ => {
                let (card, other_card) = match self.next_to_act {
                    Small => (card_sb, card_bb),
                    Big => (card_bb, card_sb),
                };
                let other_fold = match self.terminal {
                    NonTerminal => false,
                    SBWins => true,
                    BBWins => true,
                    Showdown => false,
                };
                return if card > other_card || other_fold {
                    match self.next_to_act {
                        Small => self.bbbet,
                        Big => self.sbbet,
                    }
                } else {
                    match self.next_to_act {
                        Small => -self.sbbet,
                        Big => -self.bbbet,
                    }
                };
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Player {
    Small,
    Big,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Action {
    Fold,
    Check,
    Call,
    Raise,
}

#[derive(Copy, Clone, Debug)]
pub(crate) enum TerminalState {
    NonTerminal,
    SBWins,
    BBWins,
    Showdown,
}
