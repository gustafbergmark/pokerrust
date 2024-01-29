use crate::cuda_interface::{build_turn, evaluate_turn_gpu, free_eval, transfer_flop_eval};
use crate::enums::Action::*;
use crate::enums::Player::*;
use crate::enums::TerminalState::*;
use crate::enums::*;
use crate::evaluator::Evaluator;
use crate::permutation_handler::permute;
use crate::strategy::Strategy;
use crate::vector::{Float, Vector};
use itertools::Itertools;
use poker::Suit::*;
use poker::{Card, Suit};
use std::collections::HashSet;
use std::iter::zip;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct State {
    pub terminal: TerminalState,
    pub action: Action,
    pub cards: u64,
    pub sbbet: Float,
    pub bbbet: Float,
    next_to_act: Player,
    card_strategies: Strategy,
    next_states: Vec<State>,
    permutations: Vec<[Suit; 4]>,
    gpu_pointer: Option<*const std::ffi::c_void>,
}

impl State {
    pub fn new(
        terminal: TerminalState,
        action: Action,
        sbbet: Float,
        bbbet: Float,
        next_to_act: Player,
    ) -> Self {
        State {
            terminal,
            action,
            cards: 0,
            sbbet,
            bbbet,
            next_to_act,
            card_strategies: Strategy::new(),
            next_states: vec![],
            permutations: vec![],
            gpu_pointer: None,
        }
    }

    pub fn add_action(&mut self, state: State) {
        self.next_states.push(state);
        if self.terminal == NonTerminal {
            self.card_strategies.add_strategy();
        }
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
        let mut state = State {
            terminal: self.terminal,
            action,
            cards: self.cards,
            sbbet: self.sbbet,
            bbbet: self.bbbet,
            next_to_act: opponent,
            card_strategies: Strategy::new(),
            next_states: vec![],
            permutations: vec![],
            gpu_pointer: None,
        };

        match action {
            Fold => state.terminal = fold_winner,
            Check => {
                state.terminal = NonTerminal;
                state.sbbet = self.bbbet;
            }
            Call => {
                state.sbbet = other_bet;
                state.bbbet = other_bet;
                match self.cards.count_ones() {
                    0 => state.terminal = Flop,
                    3 => {
                        let gpu_pointer = if cfg!(feature = "GPU") {
                            Some(build_turn(self.cards, other_bet))
                        } else {
                            None
                        };
                        state.terminal = Turn;
                        state.gpu_pointer = gpu_pointer;
                    }
                    4 => state.terminal = River,
                    5 => state.terminal = Showdown,
                    _ => panic!("Wrong number of communal cards"),
                }
            }
            Raise => {
                let raise_amount = if self.cards.count_ones() < 4 {
                    1.0
                } else {
                    2.0
                };
                state.terminal = NonTerminal;
                state.sbbet = match self.next_to_act {
                    Small => self.bbbet + raise_amount,
                    Big => self.sbbet,
                };
                state.bbbet = match self.next_to_act {
                    Small => self.bbbet,
                    Big => self.sbbet + raise_amount,
                };
            }
            DealFlop => {
                let deck = Card::generate_deck();
                //let flops = deck.combinations(3); // Full game
                let flops = deck.take(3).combinations(3); // Fixed flop game
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
                    let mut next_state = state.clone();
                    next_state.terminal = NonTerminal;
                    next_state.cards = num_flop;
                    next_state.next_to_act = Small;
                    next_state.permutations = possible_permutations;
                    next_states.push(next_state);
                }
                return next_states;
            }
            DealTurn => {
                let deck = Card::generate_deck();
                let mut next_states = Vec::new();
                for turn in deck {
                    let num_turn = evaluator.cards_to_u64(&[turn]);
                    if num_turn & self.cards > 0 {
                        continue;
                    }
                    let mut next_state = state.clone();
                    next_state.terminal = NonTerminal;
                    next_state.cards = self.cards | num_turn;
                    next_state.next_to_act = Small;
                    next_states.push(next_state);
                }
                return next_states;
            }
            DealRiver => {
                let deck = Card::generate_deck();
                let mut next_states = Vec::new();
                for river in deck {
                    let num_river = evaluator.cards_to_u64(&[river]);
                    if (num_river & self.cards) > 0 {
                        continue;
                    }
                    let mut next_state = state.clone();
                    next_state.terminal = NonTerminal;
                    next_state.cards = self.cards | num_river;
                    next_state.next_to_act = Small;
                    next_states.push(next_state);
                }
                return next_states;
            }
            DealHole => panic!("DealHole should only be used in root of game"),
        }
        return vec![state];
    }

    pub fn evaluate_state(
        &mut self,
        opponent_range: &Vector,
        evaluator: &Evaluator,
        updating_player: Player,
        calc_exploit: bool,
        gpu_eval_ptr: Option<*const std::ffi::c_void>,
    ) -> Vector {
        //(util of sb, util of bb, exploitability of updating player)
        match self.terminal {
            NonTerminal => {
                // Observe: This vector is also used when calculating the exploitability
                let mut average_strategy = if updating_player == self.next_to_act && calc_exploit {
                    Vector::from(&[Float::NEG_INFINITY; 1326])
                } else {
                    Vector::default()
                };
                let mut results = vec![];

                let strategy = self.card_strategies.get_strategy();
                assert_eq!(self.next_states.len(), strategy.len());

                for (next, action_prob) in zip(self.next_states.iter_mut(), strategy) {
                    let utility = if self.next_to_act == updating_player {
                        next.evaluate_state(
                            opponent_range,
                            evaluator,
                            updating_player,
                            calc_exploit,
                            gpu_eval_ptr,
                        )
                    } else {
                        next.evaluate_state(
                            &(*opponent_range * action_prob),
                            evaluator,
                            updating_player,
                            calc_exploit,
                            gpu_eval_ptr,
                        )
                    };

                    if updating_player == self.next_to_act {
                        if !calc_exploit {
                            results.push(utility);
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
                    let mut update = vec![];
                    for util in results {
                        update.push(util - average_strategy);
                    }
                    self.card_strategies.update_add(&update);
                }

                average_strategy
            }

            Showdown => self.evaluate_showdown(opponent_range, evaluator, updating_player),
            SBWins => self.evaluate_fold(opponent_range, evaluator, updating_player, Big),
            BBWins => self.evaluate_fold(opponent_range, evaluator, updating_player, Small),
            Flop => {
                let mut total = Vector::default();
                let mut count = 0.0;
                for next_state in self.next_states.iter_mut() {
                    let eval_ptr = transfer_flop_eval(evaluator, next_state.cards);
                    let mut new_range = *opponent_range;
                    // It is impossible to have hands which contains flop cards
                    for i in 0..1326 {
                        if (evaluator.card_order()[i] & next_state.cards) > 0 {
                            new_range[i] = 0.0;
                        }
                    }
                    let mut res = next_state.evaluate_state(
                        &new_range,
                        evaluator,
                        updating_player,
                        calc_exploit,
                        Some(eval_ptr),
                    );
                    // For safety for the future
                    for i in 0..1326 {
                        if (evaluator.card_order()[i] & next_state.cards) > 0 {
                            res[i] = 0.0;
                        }
                    }
                    for &permutation in &next_state.permutations {
                        count += 1.0;
                        total += permute(permutation, res, evaluator)
                    }
                    free_eval(eval_ptr);
                }
                total * (1.0 / count)
            }
            Turn => {
                if cfg!(feature = "GPU") {
                    let gpu = evaluate_turn_gpu(
                        self.gpu_pointer.expect("GPU pointer missing"),
                        gpu_eval_ptr.expect("GPU eval missing"),
                        opponent_range,
                        updating_player,
                        calc_exploit,
                    );
                    gpu
                } else {
                    let mut total = Vector::default();
                    for next_state in self.next_states.iter_mut() {
                        let mut new_range = *opponent_range;
                        // It is impossible to have hands which contains flop cards
                        for i in 0..1326 {
                            if (evaluator.card_order()[i] & next_state.cards) > 0 {
                                new_range[i] = 0.0;
                            }
                        }
                        let mut res = next_state.evaluate_state(
                            &new_range,
                            evaluator,
                            updating_player,
                            calc_exploit,
                            gpu_eval_ptr,
                        );
                        // For safety for the future
                        for i in 0..1326 {
                            if (evaluator.card_order()[i] & next_state.cards) > 0 {
                                res[i] = 0.0;
                            }
                        }
                        total += res;
                    }
                    total *= 1.0 / (self.next_states.len() as Float);
                    total
                }
            }
            River => {
                let mut total = Vector::default();
                let res = self
                    .next_states
                    .iter_mut()
                    .map(|next_state| {
                        next_state.evaluate_state(
                            opponent_range,
                            evaluator,
                            updating_player,
                            calc_exploit,
                            gpu_eval_ptr,
                        )
                    })
                    .collect::<Vec<_>>();
                for val in res {
                    total += val;
                }

                let res = total * (1.0 / (self.next_states.len() as Float));
                res
            }
        }
    }

    fn evaluate_fold(
        &self,
        opponent_range: &Vector,
        evaluator: &Evaluator,
        updating_player: Player,
        folding_player: Player,
    ) -> Vector {
        let card_order = evaluator.card_order();
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
        let bet = match folding_player {
            Small => self.sbbet,
            Big => self.bbbet,
        };
        result *= bet
            * if updating_player == folding_player {
                -1.0
            } else {
                1.0
            };
        result
    }

    fn evaluate_showdown(
        &self,
        opponent_range: &Vector,
        evaluator: &Evaluator,
        updating_player: Player,
    ) -> Vector {
        let mut result = Vector::default();
        let card_order = evaluator.card_order();
        let (self_bet, opponent_bet) = match updating_player {
            Small => (self.sbbet, self.bbbet),
            Big => (self.bbbet, self.sbbet),
        };

        let sorted = &evaluator.vectorized_eval(self.cards);
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

        for group in groups.iter() {
            let mut current_cumulative = 0.0;

            let mut current_collisions = [0.0; 52];
            for &index in group {
                let index = index as usize;
                let cards = card_order[index];
                if cards & self.cards > 0 {
                    continue;
                }
                let card = Evaluator::separate_cards(cards);
                result[index] += cumulative;
                current_cumulative += opponent_range[index];
                for c in card {
                    result[index] -= collisions[c];
                    current_collisions[c] += opponent_range[index];
                }
            }
            cumulative += current_cumulative * opponent_bet;
            for i in 0..52 {
                collisions[i] += current_collisions[i] * opponent_bet;
            }
        }

        let mut collisions = [0.0; 52];

        let mut cumulative = 0.0;

        for group in groups.iter().rev() {
            let mut current_cumulative = 0.0;

            let mut current_collisions = [0.0; 52];
            for &index in group {
                let index = index as usize;
                let cards = card_order[index];
                if cards & self.cards > 0 {
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
}
