use crate::enums::Action::*;
use crate::enums::Player::*;
use crate::enums::TerminalState::*;
use crate::enums::*;
use crate::evaluator::Evaluator;
use crate::permutation_handler::permute;
use crate::strategy::Strategy;
use crate::vector::Vector;
use itertools::Itertools;
use poker::Suit::{Clubs, Diamonds, Hearts, Spades};
use poker::{Card, Suit};
use std::collections::HashSet;
use std::iter::zip;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct State {
    terminal: TerminalState,
    pub action: Action,
    pub cards: u64,
    sbbet: f64,
    bbbet: f64,
    next_to_act: Player,
    pub card_strategies: Option<Strategy>,
    next_states: Vec<State>,
    permutations: Vec<[Suit; 4]>,
}

impl State {
    pub fn new(
        terminal: TerminalState,
        action: Action,
        sbbet: f64,
        bbbet: f64,
        next_to_act: Player,
    ) -> Self {
        State {
            terminal,
            action,
            cards: 0,
            sbbet,
            bbbet,
            next_to_act,
            card_strategies: Some(Strategy::new()),
            next_states: vec![],
            permutations: vec![],
        }
    }

    pub fn add_action(&mut self, state: State) {
        self.next_states.push(state);
    }

    pub fn get_action(&self, action: Action, evaluator: &Evaluator) -> Vec<State> {
        let opponent = match self.next_to_act {
            Small => Big,
            Big => Small,
        };
        let fold_winner = match self.next_to_act {
            Small => BBWins,
            Big => SBWins,
        };
        let other_bet = match self.next_to_act {
            Small => self.bbbet,
            Big => self.sbbet,
        };
        match action {
            Fold => vec![State {
                terminal: fold_winner,
                action,
                cards: self.cards,
                sbbet: self.sbbet,
                bbbet: self.bbbet,
                next_to_act: opponent,
                card_strategies: None,
                next_states: vec![],
                permutations: vec![],
            }],
            Check => vec![State {
                terminal: NonTerminal,
                action,
                cards: self.cards,
                sbbet: self.sbbet,
                bbbet: self.bbbet,
                next_to_act: opponent,
                card_strategies: Some(Strategy::new()),
                next_states: vec![],
                permutations: vec![],
            }],
            Call => {
                if self.cards == 0 {
                    vec![State {
                        terminal: Flop,
                        action,
                        cards: self.cards,
                        sbbet: other_bet,
                        bbbet: other_bet,
                        next_to_act: opponent,
                        card_strategies: None,
                        next_states: vec![],
                        permutations: vec![],
                    }]
                } else {
                    vec![State {
                        terminal: Showdown,
                        action,
                        cards: self.cards,
                        sbbet: other_bet,
                        bbbet: other_bet,
                        next_to_act: opponent,
                        card_strategies: None,
                        next_states: vec![],
                        permutations: vec![],
                    }]
                }
            }

            Raise => vec![State {
                terminal: NonTerminal,
                action,
                cards: self.cards,
                sbbet: self.sbbet
                    + match self.next_to_act {
                        Small => 1.0,
                        Big => 0.0,
                    },
                bbbet: self.bbbet
                    + match self.next_to_act {
                        Small => 0.0,
                        Big => 1.0,
                    },
                next_to_act: opponent,
                card_strategies: Some(Strategy::new()),
                next_states: vec![],
                permutations: vec![],
            }],
            Deal => {
                let deck = Card::generate_deck();
                let flops = deck.combinations(3);
                let mut set: HashSet<u64> = HashSet::new();
                let mut next_states = Vec::new();
                for flop in flops {
                    let num_flop = evaluator.cards_to_u64(&flop);
                    if set.contains(&num_flop) {
                        continue;
                    }
                    let mut possible_permutations: Vec<[Suit; 4]> = vec![];
                    let permutations = [Clubs, Hearts, Spades, Diamonds]
                        .into_iter()
                        .permutations(4);
                    for permutation in permutations {
                        let mut perm_flop = flop.clone();
                        for card in perm_flop.iter_mut() {
                            let new_suit = match card.suit() {
                                Clubs => permutation[0],
                                Hearts => permutation[1],
                                Spades => permutation[2],
                                Diamonds => permutation[3],
                            };
                            *card = Card::new(card.rank(), new_suit);
                        }
                        let hand = evaluator.cards_to_u64(&perm_flop);
                        if set.insert(hand) {
                            possible_permutations.push(permutation.try_into().unwrap())
                        }
                    }

                    let next_state = State {
                        terminal: NonTerminal,
                        action: Deal,
                        cards: num_flop,
                        sbbet: self.sbbet,
                        bbbet: self.bbbet,
                        next_to_act: Small,
                        card_strategies: Some(Strategy::new()),
                        next_states: vec![],
                        permutations: possible_permutations,
                    };
                    next_states.push(next_state)
                }
                next_states
            }
        }
    }

    pub fn evaluate_state(
        &mut self,
        opponent_range: &Vector,
        evaluator: &Evaluator,
        card_order: &Vec<u64>,
        updating_player: Player,
        calc_exploit: bool,
    ) -> Vector {
        //(util of sb, util of bb, exploitability of updating player)
        match self.terminal {
            NonTerminal => {
                // Observe: This vector is also used when calculating the exploitability
                let mut average_strategy = if updating_player == self.next_to_act && calc_exploit {
                    Vector::from(&[f64::NEG_INFINITY; 1326])
                } else {
                    Vector::default()
                };
                let mut results = [Vector::default(); 2];

                let strategy = self.card_strategies.as_ref().unwrap().get_strategy();

                for (a, (next, action_prob)) in
                    zip(self.next_states.iter_mut(), strategy).enumerate()
                {
                    let utility = if self.next_to_act == updating_player {
                        next.evaluate_state(
                            opponent_range,
                            evaluator,
                            card_order,
                            updating_player,
                            calc_exploit,
                        )
                    } else {
                        next.evaluate_state(
                            &(*opponent_range * action_prob),
                            evaluator,
                            card_order,
                            updating_player,
                            calc_exploit,
                        )
                    };

                    if updating_player == self.next_to_act {
                        if !calc_exploit {
                            results[a] = utility;
                            average_strategy += utility * action_prob;
                        } else {
                            for i in 0..1326 {
                                average_strategy[i] = average_strategy[i].max(utility[i]);
                            }
                        }
                    } else {
                        average_strategy += utility;
                    }
                }
                // update strategy
                if self.next_to_act == updating_player && !calc_exploit {
                    let mut update = [Vector::default(); 2];
                    for (i, &util) in results.iter().enumerate() {
                        update[i] = util - average_strategy;
                    }
                    self.card_strategies.as_mut().unwrap().update_add(&update);
                }

                average_strategy
            }

            Showdown => {
                let mut result = Vector::default();
                let (self_bet, opponent_bet) = match updating_player {
                    Small => (self.sbbet, self.bbbet),
                    Big => (self.bbbet, self.sbbet),
                };

                let sorted = evaluator.vectorized_eval(self.cards);
                let groups = sorted.group_by(|&(a, _), &(b, _)| a == b);

                let mut collisions = [0.0; 52];

                let mut cumulative = 0.0;

                for group in groups {
                    let mut current_cumumulative = 0.0;

                    let mut current_collisions = [0.0; 52];
                    // forward pass
                    for &(_, index) in group {
                        let index = index as usize;
                        let cards = card_order[index];
                        if card_order[index] & self.cards > 0 {
                            continue;
                        }
                        let card = Evaluator::separate_cards(cards);
                        result[index] += cumulative;
                        current_cumumulative += opponent_range[index];
                        for c in card {
                            result[index] -= collisions[c];
                            current_collisions[c] += opponent_range[index];
                        }
                    }
                    cumulative += current_cumumulative * opponent_bet;
                    for i in 0..52 {
                        collisions[i] += current_collisions[i] * opponent_bet;
                    }
                }

                let mut collisions = [0.0; 52];

                let mut cumulative = 0.0;

                let groups = sorted.group_by(|&(a, _), &(b, _)| a == b);

                for group in groups.rev() {
                    let mut current_cumulative = 0.0;

                    let mut current_collisions = [0.0; 52];
                    // forward pass
                    for &(_, index) in group {
                        let index = index as usize;
                        let cards = card_order[index];
                        if card_order[index] & self.cards > 0 {
                            continue;
                        }
                        let card = Evaluator::separate_cards(cards);
                        result[index] -= cumulative;
                        current_cumulative += opponent_range[index];
                        for c in card {
                            result[index] += collisions[c];
                            current_collisions[c] += opponent_range[index];
                        }
                    }
                    cumulative += current_cumulative * self_bet;
                    for i in 0..52 {
                        collisions[i] += current_collisions[i] * self_bet;
                    }
                }
                result
            }
            SBWins => {
                let mut result = Vector::default();
                let mut range_sum = 0.0;
                let mut collisions = [0.0; 52];
                for (index, &cards) in card_order.iter().enumerate() {
                    if cards & self.cards > 0 {
                        continue;
                    }
                    range_sum += opponent_range[index];
                    let card = Evaluator::separate_cards(cards);
                    for c in card {
                        collisions[c] += opponent_range[index];
                    }
                }
                for index in 0..1326 {
                    if card_order[index] & self.cards > 0 {
                        continue;
                    }
                    result[index] = range_sum + opponent_range[index];
                    let cards = Evaluator::separate_cards(card_order[index]);
                    for card in cards {
                        result[index] -= collisions[card];
                    }
                }
                result *= self.bbbet * if updating_player == Small { 1.0 } else { -1.0 };
                result
            }
            BBWins => {
                let mut result = Vector::default();
                let mut range_sum = 0.0;
                let mut collisions = [0.0; 52];
                for (index, &cards) in card_order.iter().enumerate() {
                    if cards & self.cards > 0 {
                        continue;
                    }
                    range_sum += opponent_range[index];
                    let card = Evaluator::separate_cards(cards);
                    for c in card {
                        collisions[c] += opponent_range[index];
                    }
                }
                for index in 0..1326 {
                    if card_order[index] & self.cards > 0 {
                        continue;
                    }
                    result[index] = range_sum + opponent_range[index];
                    let cards = Evaluator::separate_cards(card_order[index]);
                    for card in cards {
                        result[index] -= collisions[card];
                    }
                }
                result *= self.sbbet * if updating_player == Small { -1.0 } else { 1.0 };
                result
            }
            Flop => {
                let mut total = Vector::default();
                for next_state in self.next_states.iter_mut() {
                    let res = next_state.evaluate_state(
                        &(*opponent_range * (1.0 / 22100.0)),
                        evaluator,
                        card_order,
                        updating_player,
                        calc_exploit,
                    );
                    for &permutation in &next_state.permutations {
                        total += permute(permutation, res)
                    }
                }
                total
            }
        }
    }
}
