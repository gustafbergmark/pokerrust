use crate::cuda_interface::evaluate_showdown_gpu;
use crate::enums::Action::*;
use crate::enums::Player::*;
use crate::enums::TerminalState::*;
use crate::enums::*;
use crate::evaluator::Evaluator;
use crate::permutation_handler::permute;
use crate::strategy::Strategy;
use crate::vector::Vector;
use approx::{assert_abs_diff_eq, relative_eq};
use assert_approx_eq::assert_approx_eq;
use itertools::Itertools;
use poker::Suit::*;
use poker::{Card, Suit};
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use std::collections::HashSet;
use std::iter::zip;
use std::time::Instant;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct State {
    terminal: TerminalState,
    pub action: Action,
    pub cards: u64,
    sbbet: f32,
    bbbet: f32,
    next_to_act: Player,
    pub card_strategies: Option<Strategy>,
    next_states: Vec<State>,
    permutations: Vec<[Suit; 4]>,
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
                sbbet: self.bbbet, // Semi-ugly solution for first round
                bbbet: self.bbbet,
                next_to_act: opponent,
                card_strategies: Some(Strategy::new()),
                next_states: vec![],
                permutations: vec![],
            }],
            Call => match self.cards.count_ones() {
                0 => vec![State {
                    terminal: Flop,
                    action,
                    cards: self.cards,
                    sbbet: other_bet,
                    bbbet: other_bet,
                    next_to_act: opponent,
                    card_strategies: None,
                    next_states: vec![],
                    permutations: vec![],
                }],
                3 => vec![State {
                    terminal: Turn,
                    action,
                    cards: self.cards,
                    sbbet: other_bet,
                    bbbet: other_bet,
                    next_to_act: opponent,
                    card_strategies: None,
                    next_states: vec![],
                    permutations: vec![],
                }],
                4 => vec![State {
                    terminal: River,
                    action,
                    cards: self.cards,
                    sbbet: other_bet,
                    bbbet: other_bet,
                    next_to_act: opponent,
                    card_strategies: None,
                    next_states: vec![],
                    permutations: vec![],
                }],
                5 => vec![State {
                    terminal: Showdown,
                    action,
                    cards: self.cards,
                    sbbet: other_bet,
                    bbbet: other_bet,
                    next_to_act: opponent,
                    card_strategies: None,
                    next_states: vec![],
                    permutations: vec![],
                }],
                _ => panic!("Wrong numher of communal cards"),
            },

            Raise => vec![State {
                terminal: NonTerminal,
                action,
                cards: self.cards,
                sbbet: match self.next_to_act {
                    Small => self.bbbet + 1.0,
                    Big => self.sbbet,
                },
                bbbet: match self.next_to_act {
                    Small => self.bbbet,
                    Big => self.sbbet + 1.0,
                },
                next_to_act: opponent,
                card_strategies: Some(Strategy::new()),
                next_states: vec![],
                permutations: vec![],
            }],
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

                    let next_state = State {
                        terminal: NonTerminal,
                        action: DealFlop,
                        cards: num_flop,
                        sbbet: self.sbbet,
                        bbbet: self.bbbet,
                        next_to_act: Small,
                        card_strategies: Some(Strategy::new()),
                        next_states: vec![],
                        permutations: possible_permutations,
                    };
                    next_states.push(next_state);
                }
                next_states
            }
            DealTurn => {
                let deck = Card::generate_deck();
                let mut next_states = Vec::new();
                for turn in deck {
                    let num_turn = evaluator.cards_to_u64(&[turn]);
                    if num_turn & self.cards > 0 {
                        continue;
                    }
                    let next_state = State {
                        terminal: NonTerminal,
                        action: DealTurn,
                        cards: self.cards | num_turn,
                        sbbet: self.sbbet,
                        bbbet: self.bbbet,
                        next_to_act: Small,
                        card_strategies: Some(Strategy::new()),
                        next_states: vec![],
                        permutations: vec![],
                    };
                    next_states.push(next_state);
                }
                next_states
            }
            DealRiver => {
                let deck = Card::generate_deck();
                let mut next_states = Vec::new();
                for river in deck {
                    let num_river = evaluator.cards_to_u64(&[river]);
                    if num_river & self.cards > 0 {
                        continue;
                    }
                    let next_state = State {
                        terminal: NonTerminal,
                        action: DealRiver,
                        cards: self.cards | num_river,
                        sbbet: self.sbbet,
                        bbbet: self.bbbet,
                        next_to_act: Small,
                        card_strategies: Some(Strategy::new()),
                        next_states: vec![],
                        permutations: vec![],
                    };
                    next_states.push(next_state);
                }
                next_states
            }
            DealHole => panic!("DealHole should only be used in root of game"),
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
                    Vector::from(&[f32::NEG_INFINITY; 1326])
                } else {
                    Vector::default()
                };
                let mut results = [Vector::default(); 3];

                let strategy = self
                    .card_strategies
                    .as_ref()
                    .unwrap()
                    .get_strategy(self.next_states.len());

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
                    let mut update = [Vector::default(); 3];
                    for (i, &util) in results.iter().enumerate() {
                        if i < self.next_states.len() {
                            update[i] = util - average_strategy;
                        }
                    }
                    self.card_strategies.as_mut().unwrap().update_add(&update);
                }
                average_strategy
            }

            Showdown => {
                self.evaluate_showdown2(opponent_range, evaluator, card_order, updating_player)
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
                let mut count = 0.0;
                for next_state in self.next_states.iter_mut() {
                    let res = next_state.evaluate_state(
                        opponent_range,
                        evaluator,
                        card_order,
                        updating_player,
                        calc_exploit,
                    );
                    for &permutation in &next_state.permutations {
                        count += 1.0;
                        total += permute(permutation, res)
                    }
                }
                total * (1.0 / count)
            }
            Turn => {
                let mut total = Vector::default();
                let mut count = 0.0;
                for next_state in self.next_states.iter_mut() {
                    let res = next_state.evaluate_state(
                        opponent_range,
                        evaluator,
                        card_order,
                        updating_player,
                        calc_exploit,
                    );
                    count += 1.0;
                    total += res;
                }
                total * (1.0 / count)
            }
            River => {
                // Some parallelization
                let mut total = Vector::default();
                let res = self
                    .next_states
                    //.par_iter_mut()
                    .iter_mut()
                    .map(|next_state| {
                        next_state.evaluate_state(
                            opponent_range,
                            evaluator,
                            card_order,
                            updating_player,
                            calc_exploit,
                        )
                    })
                    .collect::<Vec<_>>();
                for val in res {
                    total += val;
                }

                total * (1.0 / self.next_states.len() as f32)
            }
        }
    }

    fn evaluate_showdown2(
        &self,
        opponent_range: &Vector,
        evaluator: &Evaluator,
        card_order: &Vec<u64>,
        updating_player: Player,
    ) -> Vector {
        let eval = evaluator.vectorized_eval(self.cards);
        let coll = evaluator.collisions(self.cards);
        let gpu = Instant::now();
        let result_gpu =
            evaluate_showdown_gpu(&opponent_range.clone(), self.cards, card_order, eval, coll);
        let gpu = gpu.elapsed().as_micros();

        let cpu = Instant::now();
        let result = self.evaluate_showdown(opponent_range, evaluator, card_order, updating_player);
        let cpu = cpu.elapsed().as_micros();
        for i in 0..1326 {
            //println!("{} {} {}", i, result_gpu[i], result[i]);
            assert_approx_eq!(result_gpu[i], result[i], 1e-2);
        }
        panic!("Run once");
        //println!("COMPLETE {} cpu {} gpu {}", self.cards, cpu, gpu);
        result
    }

    fn evaluate_showdown(
        &self,
        opponent_range: &Vector,
        evaluator: &Evaluator,
        card_order: &Vec<u64>,
        updating_player: Player,
    ) -> Vector {
        let mut result = Vector::default();
        let (self_bet, opponent_bet) = match updating_player {
            Small => (self.sbbet, self.bbbet),
            Big => (self.bbbet, self.sbbet),
        };

        let sorted = &evaluator.vectorized_eval(self.cards);
        let mut groups = vec![];
        let mut current = vec![sorted[0] & 2047];
        for (index, &eval) in sorted[1..1326].iter().enumerate() {
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
            cumulative += current_cumulative; // * opponent_bet; //TODO
            for i in 0..52 {
                collisions[i] += current_collisions[i]; // * opponent_bet; //TODO
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
            cumulative += current_cumulative; // * self_bet; //TODO
            for i in 0..52 {
                collisions[i] += current_collisions[i]; // * self_bet; //TODO
            }
        }

        // // Naive
        // let peval = poker::Evaluator::new();
        // let mut res = Vector::default();
        // for (i1, &hand1) in card_order.iter().enumerate() {
        //     for (i2, &hand2) in card_order.iter().enumerate() {
        //         if hand1 & hand2 > 0 {
        //             continue;
        //         }
        //         if (hand1 & self.cards > 0) || (hand2 & self.cards > 0) {
        //             continue;
        //         }
        //         let hand1 = evaluator.u64_to_cards(hand1 | self.cards);
        //         let hand2 = evaluator.u64_to_cards(hand2 | self.cards);
        //         if (peval.evaluate(&hand1).unwrap() > peval.evaluate(&hand2).unwrap()) {
        //             res[i1] += opponent_range[i2]; // * opponent_bet; //TODO
        //         } else if (peval.evaluate(&hand1).unwrap() < peval.evaluate(&hand2).unwrap()) {
        //             res[i1] -= opponent_range[i2]; // * self_bet; //TODO
        //         };
        //     }
        // }
        // for i in 0..1326 {
        //     assert_approx_eq!(res[i], result[i], 1e-3);
        // }
        // println!("Naive and CPU same");
        result
    }
}
