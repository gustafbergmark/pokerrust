use crate::cuda_interface::{build_river, download_gpu, upload_gpu};
use crate::enums::Action::*;
use crate::enums::Player::*;
use crate::enums::TerminalState::*;
use crate::enums::*;
use crate::evaluator::{separate_cards, Evaluator};
use crate::game::TURNS;
use crate::permutation_handler::permute;
use crate::strategy::{AbstractStrategy, RegularStrategy, Strategy};
use crate::vector::{Float, Vector};
use itertools::Itertools;
use poker::Suit::*;
use poker::{Card, Suit};
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};
use std::ffi::c_void;
use std::iter::zip;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Pointer(pub(crate) *const c_void);
unsafe impl Send for Pointer {}
unsafe impl Sync for Pointer {}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct State<const M: usize> {
    pub terminal: TerminalState,
    pub action: Action,
    pub cards: u64,
    pub sbbet: Float,
    pub bbbet: Float,
    next_to_act: Player,
    card_strategies: Strategy<M>,
    next_states: Vec<State<M>>,
    permutations: Vec<[Suit; 4]>,
    gpu_pointer: Option<i32>,
}

impl<const M: usize> State<M> {
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
            card_strategies: Strategy::Regular(RegularStrategy::new()),
            next_states: vec![],
            permutations: vec![],
            gpu_pointer: None,
        }
    }

    pub fn add_action(&mut self, state: State<M>) {
        self.next_states.push(state);
        if self.terminal == NonTerminal {
            self.card_strategies.add_strategy();
        }
    }

    pub fn get_action(
        &self,
        action: Action,
        evaluator: &Evaluator<M>,
        builder: Pointer,
    ) -> Vec<State<M>> {
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
        let strategy = if self.cards.count_ones() < 4 {
            Strategy::Regular(RegularStrategy::new())
        } else {
            Strategy::Abstract(AbstractStrategy::new())
        };
        let mut state = State {
            terminal: self.terminal,
            action,
            cards: self.cards,
            sbbet: self.sbbet,
            bbbet: self.bbbet,
            next_to_act: opponent,
            card_strategies: strategy,
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
                    3 => state.terminal = Turn,
                    4 => {
                        let gpu_pointer = if cfg!(feature = "GPU") {
                            Some(build_river(self.cards, other_bet, builder))
                        } else {
                            None
                        };
                        state.terminal = River;
                        state.gpu_pointer = gpu_pointer;
                    }
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
                //let flops = deck.take(3).combinations(3); // Fixed flop game
                let flops = deck.combinations(3); // Fixed flop game
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
                for turn in deck {
                    let num_turn = evaluator.cards_to_u64(&[turn]);
                    if num_turn & self.cards > 0 {
                        continue;
                    }
                    state.terminal = NonTerminal;
                    state.cards = self.cards | num_turn;
                    state.card_strategies = Strategy::Abstract(AbstractStrategy::new());
                    state.next_to_act = Small;
                    break;
                }
            }
            DealRiver => {
                // Do not create river subgames on CPU if they are created on GPU
                if cfg!(feature = "GPU") {
                    return vec![];
                }
                let deck = Card::generate_deck();
                // A bit of a hack, need 5 cards for building, but the last card added is symbolic only
                for river in deck {
                    let num_river = evaluator.cards_to_u64(&[river]);
                    if (num_river & self.cards) > 0 {
                        continue;
                    }
                    state.terminal = NonTerminal;
                    state.cards = self.cards | num_river;
                    state.card_strategies = Strategy::Abstract(AbstractStrategy::new());
                    state.next_to_act = Small;
                    break;
                }
            }
            DealHole => panic!("DealHole should only be used in root of game"),
        }
        return vec![state];
    }

    pub fn evaluate_state(
        &mut self,
        sb_range: &Vector,
        bb_range: &Vector,
        evaluator: &Evaluator<M>,
        updating_player: Player,
        calc_exploit: bool,
        communal_cards: u64,
        builder: Pointer,
        upload: bool,
        turn_index: i32,
        fixed_flop: u64,
        turns: &Vec<u64>,
    ) -> Vector {
        //(util of sb, util of bb, exploitability of updating player)
        let opponent_range = match updating_player {
            Small => bb_range,
            Big => sb_range,
        };
        match self.terminal {
            NonTerminal => {
                // Observe: This vector is also used when calculating the exploitability
                let mut average_strategy = if updating_player == self.next_to_act && calc_exploit {
                    Vector::from(&[Float::NEG_INFINITY; 1326])
                } else {
                    Vector::default()
                };
                let mut results = vec![];

                let strategy = self.card_strategies.get_strategy(evaluator, communal_cards);
                assert_eq!(self.next_states.len(), strategy.len());

                for (next, action_prob) in zip(self.next_states.iter_mut(), strategy) {
                    let utility = match self.next_to_act {
                        Small => next.evaluate_state(
                            &(*sb_range * action_prob),
                            bb_range,
                            evaluator,
                            updating_player,
                            calc_exploit,
                            communal_cards,
                            builder,
                            upload,
                            turn_index,
                            fixed_flop,
                            turns,
                        ),

                        Big => next.evaluate_state(
                            sb_range,
                            &(*bb_range * action_prob),
                            evaluator,
                            updating_player,
                            calc_exploit,
                            communal_cards,
                            builder,
                            upload,
                            turn_index,
                            fixed_flop,
                            turns,
                        ),
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
                if self.next_to_act == updating_player && !calc_exploit && !upload {
                    let mut update = vec![];
                    for util in results {
                        update.push(util - average_strategy);
                    }
                    self.card_strategies
                        .update_add(&update, evaluator, communal_cards);
                }
                average_strategy
            }

            Showdown => self.evaluate_showdown(opponent_range, evaluator, communal_cards),
            SBWins => self.evaluate_fold(opponent_range, evaluator, updating_player, Big),
            BBWins => self.evaluate_fold(opponent_range, evaluator, updating_player, Small),
            Flop => {
                let mut total = Vector::default();
                let mut count = 0.0;
                for next_state in self.next_states.iter_mut() {
                    if next_state.cards == fixed_flop || fixed_flop == 0 {
                        let mut new_sb_range = *sb_range;
                        let mut new_bb_range = *bb_range;
                        // It is impossible to have hands which contains flop cards
                        for i in 0..1326 {
                            if (evaluator.card_order()[i] & next_state.cards) > 0 {
                                new_sb_range[i] = 0.0;
                                new_bb_range[i] = 0.0;
                            }
                        }
                        let mut res = next_state.evaluate_state(
                            &new_sb_range,
                            &new_bb_range,
                            evaluator,
                            updating_player,
                            calc_exploit,
                            next_state.cards,
                            builder,
                            upload,
                            turn_index,
                            fixed_flop,
                            turns,
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
                    }
                }
                total * (1.0 / count)
            }
            Turn => {
                let mut total = Vector::default();
                assert_eq!(self.next_states.len(), 1);
                let next_state = &mut self.next_states[0];
                let mut count = 0;
                for &num_turn in turns {
                    if num_turn & communal_cards > 0 {
                        continue;
                    }
                    let new_cards = communal_cards | num_turn;
                    let mut new_sb_range = *sb_range;
                    let mut new_bb_range = *bb_range;
                    // It is impossible to have hands which contains flop cards
                    for i in 0..1326 {
                        if (evaluator.card_order()[i] & new_cards) > 0 {
                            new_sb_range[i] = 0.0;
                            new_bb_range[i] = 0.0;
                        }
                    }
                    let mut res = next_state.evaluate_state(
                        &new_sb_range,
                        &new_bb_range,
                        evaluator,
                        updating_player,
                        calc_exploit,
                        new_cards,
                        builder,
                        upload,
                        count,
                        fixed_flop,
                        turns,
                    );
                    for i in 0..1326 {
                        if (evaluator.card_order()[i] & new_cards) > 0 {
                            res[i] = 0.0;
                        }
                    }
                    total += res;
                    count += 1;
                }
                // Apply the aggregated updates from all iteration on the abstract strategy
                if !upload && !calc_exploit {
                    next_state.apply_updates(updating_player);
                }
                assert_eq!(count, TURNS as i32);
                total * (1.0 / TURNS as Float)
            }
            River => {
                if cfg!(feature = "GPU") {
                    if upload {
                        upload_gpu(
                            builder,
                            self.gpu_pointer.expect("Missing GPU index") * TURNS as i32
                                + turn_index,
                            opponent_range,
                        );
                        // No updates during upload, return does not matter
                        Vector::default()
                    } else {
                        download_gpu(
                            builder,
                            self.gpu_pointer.expect("Missing GPU index") * TURNS as i32
                                + turn_index,
                        )
                    }
                } else {
                    let mut total = Vector::default();
                    assert_eq!(self.next_states.len(), 1);
                    let next_state = &mut self.next_states[0];
                    let mut count = 0;
                    for river_card in 0..52 {
                        let num_river = 1 << river_card;
                        if num_river & communal_cards > 0 {
                            continue;
                        }
                        count += 1;
                        let new_cards = communal_cards | num_river;
                        let mut new_sb_range = *sb_range;
                        let mut new_bb_range = *bb_range;
                        // It is impossible to have hands which contains flop cards
                        for i in 0..1326 {
                            if (evaluator.card_order()[i] & new_cards) > 0 {
                                new_sb_range[i] = 0.0;
                                new_bb_range[i] = 0.0;
                            }
                        }
                        let mut res = next_state.evaluate_state(
                            &new_sb_range,
                            &new_bb_range,
                            evaluator,
                            updating_player,
                            calc_exploit,
                            new_cards,
                            builder,
                            upload,
                            turn_index,
                            fixed_flop,
                            turns,
                        );
                        for i in 0..1326 {
                            if (evaluator.card_order()[i] & new_cards) > 0 {
                                res[i] = 0.0;
                            }
                        }
                        total += res;
                    }
                    // Apply the aggregated updates from all iteration on the abstract strategy
                    // next_state.apply_updates(updating_player);
                    assert_eq!(count, 48);
                    total * (1.0 / 48.0)
                }
            }
        }
    }
    pub fn apply_updates(&mut self, updating_player: Player) {
        match &mut self.card_strategies {
            Strategy::Regular(_) => {}
            Strategy::Abstract(strat) => {
                if self.next_to_act == updating_player {
                    strat.apply_updates();
                }
            }
        }
        for next in self.next_states.iter_mut() {
            next.apply_updates(updating_player);
        }
    }

    pub fn save(&self, buf: &mut VecDeque<Float>) {
        self.card_strategies.save(buf);
        for next in &self.next_states {
            next.save(buf);
        }
    }

    pub fn load(&mut self, buf: &mut VecDeque<Float>) {
        self.card_strategies.load(buf);
        for next in self.next_states.iter_mut() {
            next.load(buf);
        }
    }

    fn evaluate_fold(
        &self,
        opponent_range: &Vector,
        evaluator: &Evaluator<M>,
        updating_player: Player,
        folding_player: Player,
    ) -> Vector {
        let card_order = evaluator.card_order();
        let mut result = Vector::default();
        let mut range_sum = 0.0;
        let mut collisions = [0.0; 52];
        for (index, &cards) in card_order.iter().enumerate() {
            range_sum += opponent_range[index];
            let card = separate_cards(cards);
            for c in card {
                collisions[c] += opponent_range[index];
            }
        }
        for index in 0..1326 {
            result[index] = range_sum + opponent_range[index];
            let cards = separate_cards(card_order[index]);
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
        evaluator: &Evaluator<M>,
        communal_cards: u64,
    ) -> Vector {
        let mut result = Vector::default();
        //return result;
        let card_order = evaluator.card_order();
        assert_eq!(self.sbbet, self.bbbet);
        let bet = self.sbbet;

        let sorted = &evaluator.vectorized_eval(communal_cards);
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
                let card = separate_cards(cards);
                result[index] += cumulative;
                current_cumulative += opponent_range[index];
                for c in card {
                    result[index] -= collisions[c];
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

        for group in groups.iter().rev() {
            let mut current_cumulative = 0.0;

            let mut current_collisions = [0.0; 52];
            for &index in group {
                let index = index as usize;
                let cards = card_order[index];
                let card = separate_cards(cards);
                result[index] -= cumulative;
                current_cumulative += opponent_range[index];
                for c in card {
                    result[index] += collisions[c];
                    current_collisions[c] += opponent_range[index];
                }
            }
            cumulative += current_cumulative;
            for i in 0..52 {
                collisions[i] += current_collisions[i];
            }
        }
        result * bet
    }
}
