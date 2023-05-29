use crate::enums::Action::*;
use crate::enums::Player::*;
use crate::enums::TerminalState::*;
use crate::enums::*;
use crate::evaluator::Evaluator;
use crate::strategy::Strategy;
use itertools::Itertools;
use poker::Card;
use std::collections::HashMap;
use std::iter::zip;

#[derive(Clone, Debug)]
pub(crate) struct State {
    terminal: TerminalState,
    pub action: Action,
    pub cards: u64,
    sbbet: f32,
    bbbet: f32,
    next_to_act: Player,
    card_strategies: Vec<Strategy>,
    next_states: Vec<State>,
}

impl State {
    pub fn new(
        terminal: TerminalState,
        action: Action,
        sbbet: f32,
        bbbet: f32,
        next_to_act: Player,
    ) -> Self {
        State {
            terminal,
            action,
            cards: 0,
            sbbet,
            bbbet,
            next_to_act,
            card_strategies: vec![Strategy::new(); 1326],
            next_states: vec![],
        }
    }

    pub fn add_action(&mut self, state: State) {
        self.next_states.push(state);
    }

    pub fn get_action(&self, action: Action, evaluator: &Evaluator) -> Vec<State> {
        match self.next_to_act {
            Small => match action {
                Fold => vec![State {
                    terminal: BBWins,
                    action,
                    cards: self.cards.clone(),
                    sbbet: self.sbbet,
                    bbbet: self.bbbet,
                    next_to_act: Big,
                    card_strategies: vec![],
                    next_states: vec![],
                }],
                Check => vec![State {
                    terminal: NonTerminal,
                    action,
                    cards: self.cards.clone(),
                    sbbet: self.sbbet,
                    bbbet: self.bbbet,
                    next_to_act: Big,
                    card_strategies: vec![Strategy::new(); 1326],
                    next_states: vec![],
                }],
                Call => {
                    if self.cards == 0 {
                        vec![State {
                            terminal: Flop,
                            action,
                            cards: self.cards.clone(),
                            sbbet: self.bbbet,
                            bbbet: self.bbbet,
                            next_to_act: Big,
                            card_strategies: vec![],
                            next_states: vec![],
                        }]
                    } else {
                        vec![State {
                            terminal: Showdown,
                            action,
                            cards: self.cards.clone(),
                            sbbet: self.bbbet,
                            bbbet: self.bbbet,
                            next_to_act: Big,
                            card_strategies: vec![],
                            next_states: vec![],
                        }]
                    }
                }

                Raise => vec![State {
                    terminal: NonTerminal,
                    action,
                    cards: self.cards.clone(),
                    sbbet: self.bbbet + 1.0,
                    bbbet: self.bbbet,
                    next_to_act: Big,
                    card_strategies: vec![Strategy::new(); 1326],
                    next_states: vec![],
                }],
                Deal => {
                    let deck = Card::generate_deck();
                    let flops = deck.combinations(3);
                    let mut next_states = Vec::new();
                    for flop in flops {
                        let next_state = State {
                            terminal: NonTerminal,
                            action: Deal,
                            cards: evaluator.cards_to_u64(&flop),
                            sbbet: self.sbbet,
                            bbbet: self.bbbet,
                            next_to_act: Small,
                            card_strategies: vec![Strategy::new(); 1326],
                            next_states: vec![],
                        };
                        next_states.push(next_state)
                    }
                    assert_eq!(next_states.len(), 22100);
                    next_states
                }
            },
            Big => match action {
                Fold => vec![State {
                    terminal: SBWins,
                    action,
                    cards: self.cards.clone(),
                    sbbet: self.sbbet,
                    bbbet: self.bbbet,
                    next_to_act: Small,
                    card_strategies: vec![],
                    next_states: vec![],
                }],
                Check => vec![State {
                    terminal: NonTerminal,
                    action,
                    cards: self.cards.clone(),
                    sbbet: self.sbbet,
                    bbbet: self.bbbet,
                    next_to_act: Small,
                    card_strategies: vec![Strategy::new(); 1326],
                    next_states: vec![],
                }],
                Call => {
                    if self.cards == 0 {
                        vec![State {
                            terminal: Flop,
                            action,
                            cards: self.cards.clone(),
                            sbbet: self.bbbet,
                            bbbet: self.bbbet,
                            next_to_act: Small,
                            card_strategies: vec![],
                            next_states: vec![],
                        }]
                    } else {
                        vec![State {
                            terminal: Showdown,
                            action,
                            cards: self.cards.clone(),
                            sbbet: self.bbbet,
                            bbbet: self.bbbet,
                            next_to_act: Small,
                            card_strategies: vec![],
                            next_states: vec![],
                        }]
                    }
                }

                Raise => vec![State {
                    terminal: NonTerminal,
                    action,
                    cards: self.cards.clone(),
                    sbbet: self.sbbet,
                    bbbet: self.sbbet + 1.0,
                    next_to_act: Small,
                    card_strategies: vec![Strategy::new(); 1326],
                    next_states: vec![],
                }],
                Deal => {
                    let deck = Card::generate_deck();
                    let flops = deck.combinations(3);
                    let mut next_states = Vec::new();
                    for flop in flops {
                        let next_state = State {
                            terminal: NonTerminal,
                            action: Deal,
                            cards: evaluator.cards_to_u64(&flop),
                            sbbet: self.sbbet,
                            bbbet: self.bbbet,
                            next_to_act: Small,
                            card_strategies: vec![Strategy::new(); 1326],
                            next_states: vec![],
                        };
                        next_states.push(next_state)
                    }
                    next_states
                }
            },
        }
    }

    pub fn get_card_strategy(&mut self, card: u64, map: &HashMap<u64, usize>) -> &mut Strategy {
        let i = map.get(&card).unwrap();
        &mut self.card_strategies[*i]
    }

    pub fn evaluate_state(
        &mut self,
        card_sb: u64,
        card_bb: u64,
        prob_sb: f32,
        prob_bb: f32,
        evaluator: &Evaluator,
        iteration_weight: f32,
        card_order: &HashMap<u64, usize>,
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

                let strategy = self
                    .get_card_strategy(card, card_order)
                    .get_strategy(other_prob, iteration_weight);

                let mut i = 0;
                for (next, action_prob) in zip(self.next_states.iter_mut(), strategy) {
                    let new_prob = prob * action_prob;

                    let util = match self.next_to_act {
                        Small => -next.evaluate_state(
                            card_sb,
                            card_bb,
                            new_prob,
                            prob_bb,
                            evaluator,
                            iteration_weight,
                            card_order,
                        ),
                        Big => -next.evaluate_state(
                            card_sb,
                            card_bb,
                            prob_sb,
                            new_prob,
                            evaluator,
                            iteration_weight,
                            card_order,
                        ),
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
                self.get_card_strategy(card, card_order).update_add(&update);
                return avgstrat;
            }

            Showdown => {
                let (cards, other_cards) = match self.next_to_act {
                    Small => (self.cards | card_sb, self.cards | card_bb),
                    Big => (self.cards | card_bb, self.cards | card_sb),
                };
                return if evaluator.evaluate(cards) > evaluator.evaluate(other_cards) {
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
            Flop => {
                let mut count = 0;
                let mut total = 0.0;
                let hole_cards: u64 = card_sb | card_bb;
                for next_state in self.next_states.iter_mut() {
                    let intersect = next_state.cards & hole_cards;
                    if intersect > 0 {
                        continue;
                    } else {
                        count += 1;
                        let res = next_state.evaluate_state(
                            card_sb,
                            card_bb,
                            prob_sb,
                            prob_bb,
                            evaluator,
                            iteration_weight,
                            card_order,
                        );
                        total += res;
                    }
                }
                if count != 17296 {
                    //dbg!(set, self.next_states.len(), &self);
                }
                assert_eq!(count, 17296); // 17296 = 48 choose 3
                total / count as f32
            }
            SBWins => self.bbbet,
            BBWins => self.sbbet,
        }
    }
}
