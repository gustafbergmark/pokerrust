use crate::game::Player::{Big, Small};
use crate::game::TerminalState::{BBWins, NonTerminal, SBWins, Showdown};
use itertools::Itertools;
use poker::{card, Card};
use std::fmt::{Debug, Formatter};
use std::iter::zip;

fn normalize(v: &Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().sum();
    if norm != 0.0 {
        v.iter().map(|&elem| elem / norm).collect()
    } else {
        vec![1.0 / (v.len() as f32); v.len()]
    }
}

static CARD_ORDER: [Card; 3] = [
    card!(Jack, Hearts),
    card!(Queen, Hearts),
    card!(King, Hearts),
];

fn card_index(card: Card) -> usize {
    CARD_ORDER.iter().position(|elem| *elem == card).unwrap()
}

#[derive(Debug)]
pub(crate) struct Game {
    root: State,
}

impl Game {
    pub fn new(root: State) -> Self {
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
pub(crate) struct Strategy {
    regrets: Vec<f32>,
    strategy_sum: Vec<f32>,
}

impl Debug for Strategy {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.get_average_strategy())
    }
}

impl Strategy {
    pub fn new(num_actions: usize) -> Self {
        Strategy {
            regrets: vec![0.0; num_actions],
            strategy_sum: vec![0.0; num_actions],
        }
    }

    pub fn update_add(&mut self, update: Vec<f32>) {
        for i in 0..self.regrets.len() {
            self.regrets[i] += update[i];
        }
    }

    pub fn get_strategy(&mut self, realization_weight: f32) -> Vec<f32> {
        let regret_match = self
            .regrets
            .clone()
            .into_iter()
            .map(|elem| if elem < 0.0 { 0.0 } else { elem })
            .collect();
        let normalized = normalize(&regret_match);
        for i in 0..self.strategy_sum.len() {
            self.strategy_sum[i] += normalized[i] * realization_weight;
        }
        normalized
    }

    pub fn get_average_strategy(&self) -> Vec<f32> {
        normalize(&self.strategy_sum)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct State {
    terminal: TerminalState,
    sbbet: f32,
    bbbet: f32,
    next_to_act: Player,
    card_strategies: Vec<Strategy>,
    actions: Vec<State>,
}

impl State {
    pub fn new(terminal: TerminalState, sbbet: f32, bbbet: f32, next_to_act: Player) -> Self {
        State {
            terminal,
            sbbet,
            bbbet,
            next_to_act,
            card_strategies: vec![Strategy::new(2); 3],
            actions: vec![],
        }
    }

    pub fn add_action(&mut self, state: State) {
        self.actions.push(state);
    }

    pub fn get_action(&self, action: Action) -> State {
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
                    card_strategies: vec![Strategy::new(2); 3],
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
                    card_strategies: vec![Strategy::new(2); 3],
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
                    card_strategies: vec![Strategy::new(2); 3],
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
                    card_strategies: vec![Strategy::new(2); 3],
                    actions: vec![],
                },
            },
        }
    }

    pub fn get_card_strategy(&mut self, card: Card) -> &mut Strategy {
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
                let mut payoffs = Vec::new();

                let strategy = self.get_card_strategy(card).get_strategy(other_prob);

                for (next, action_prob) in zip(self.actions.iter_mut(), strategy) {
                    let new_prob = prob * action_prob;

                    let util = match self.next_to_act {
                        Small => -next.evaluate_state(card_sb, card_bb, new_prob, prob_bb),
                        Big => -next.evaluate_state(card_sb, card_bb, prob_sb, new_prob),
                    };

                    avgstrat += util * action_prob;
                    payoffs.push(util);
                }
                // update strategy
                let mut update = Vec::new();
                for util in payoffs {
                    let diff = util - avgstrat;
                    let regret = diff * other_prob;
                    update.push(regret);
                }
                self.get_card_strategy(card).update_add(update);
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
