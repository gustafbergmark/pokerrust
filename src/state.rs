use crate::enums::Action::*;
use crate::enums::Player::*;
use crate::enums::TerminalState::*;
use crate::enums::*;
use crate::evaluator::Evaluator;
use crate::permutation_handler::PermutationHandler;
use crate::strategy::Strategy;
use crate::vector::Vector;
use itertools::Itertools;
use poker::Suit::{Clubs, Diamonds, Hearts, Spades};
use poker::{Card, Suit};
//use rayon::iter::IndexedParallelIterator;
//use rayon::iter::IntoParallelRefMutIterator;
//use rayon::iter::ParallelIterator;
use std::collections::HashSet;
use std::iter::zip;

#[derive(Clone, Debug)]
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
        match self.next_to_act {
            Small => match action {
                Fold => vec![State {
                    terminal: BBWins,
                    action,
                    cards: self.cards.clone(),
                    sbbet: self.sbbet,
                    bbbet: self.bbbet,
                    next_to_act: Big,
                    card_strategies: None,
                    next_states: vec![],
                    permutations: vec![],
                }],
                Check => vec![State {
                    terminal: NonTerminal,
                    action,
                    cards: self.cards.clone(),
                    sbbet: self.sbbet,
                    bbbet: self.bbbet,
                    next_to_act: Big,
                    card_strategies: Some(Strategy::new()),
                    next_states: vec![],
                    permutations: vec![],
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
                            card_strategies: None,
                            next_states: vec![],
                            permutations: vec![],
                        }]
                    } else {
                        vec![State {
                            terminal: Showdown,
                            action,
                            cards: self.cards.clone(),
                            sbbet: self.bbbet,
                            bbbet: self.bbbet,
                            next_to_act: Big,
                            card_strategies: None,
                            next_states: vec![],
                            permutations: vec![],
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
                        let permutations = [Spades, Hearts, Diamonds, Clubs]
                            .into_iter()
                            .permutations(4);
                        for permutation in permutations {
                            let mut perm_flop = flop.clone();
                            for card in perm_flop.iter_mut() {
                                let new_suit = match card.suit() {
                                    Spades => permutation[0],
                                    Hearts => permutation[1],
                                    Diamonds => permutation[2],
                                    Clubs => permutation[3],
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
                    assert_eq!(set.len(), 22100); // 52 choose 3
                    assert_eq!(next_states.len(), 1755); // the number of strategically different flops
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
                    card_strategies: None,
                    next_states: vec![],
                    permutations: vec![],
                }],
                Check => vec![State {
                    terminal: NonTerminal,
                    action,
                    cards: self.cards.clone(),
                    sbbet: self.sbbet,
                    bbbet: self.bbbet,
                    next_to_act: Small,
                    card_strategies: Some(Strategy::new()),
                    next_states: vec![],
                    permutations: vec![],
                }],
                Call => {
                    if self.cards == 0 {
                        vec![State {
                            terminal: Flop,
                            action,
                            cards: self.cards.clone(),
                            sbbet: self.sbbet,
                            bbbet: self.sbbet,
                            next_to_act: Small,
                            card_strategies: None,
                            next_states: vec![],
                            permutations: vec![],
                        }]
                    } else {
                        vec![State {
                            terminal: Showdown,
                            action,
                            cards: self.cards.clone(),
                            sbbet: self.sbbet,
                            bbbet: self.sbbet,
                            next_to_act: Small,
                            card_strategies: None,
                            next_states: vec![],
                            permutations: vec![],
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
                    assert_eq!(set.len(), 22100);
                    assert_eq!(next_states.len(), 1755);
                    next_states
                }
            },
        }
    }

    pub fn evaluate_state(
        &mut self,
        range_sb: &Vector,
        range_bb: &Vector,
        evaluator: &Evaluator,
        iteration_weight: f32,
        card_order: &Vec<u64>,
        updating_player: Player,
        permuter: &PermutationHandler,
    ) -> [Vector; 2] {
        //(util of sb, util of bb, exploitability of updating player)
        match self.terminal {
            NonTerminal => {
                let (range, other_range) = match self.next_to_act {
                    Small => (range_sb, range_bb),
                    Big => (range_bb, range_sb),
                };
                // get avg strategy and individual payoffs
                let mut avgstrat = Vector::default();
                let mut exploitability = if self.next_to_act == updating_player {
                    Vector::default()
                } else {
                    Vector {
                        values: [f32::NEG_INFINITY; 1326],
                    }
                };
                let mut payoffs = [Vector::default(); 2];

                let strategy = self.card_strategies.as_mut().unwrap().get_strategy();

                for (a, (next, action_prob)) in
                    zip(self.next_states.iter_mut(), strategy).enumerate()
                {
                    let new_range = *range * action_prob;

                    let [util, exploit] = match self.next_to_act {
                        Small => next.evaluate_state(
                            &new_range,
                            other_range,
                            evaluator,
                            iteration_weight,
                            card_order,
                            updating_player,
                            permuter,
                        ),
                        Big => next.evaluate_state(
                            other_range,
                            &new_range,
                            evaluator,
                            iteration_weight,
                            card_order,
                            updating_player,
                            permuter,
                        ),
                    };
                    if updating_player == self.next_to_act {
                        payoffs[a].values[..].copy_from_slice(&util.values);
                        avgstrat += util * action_prob;
                    } else {
                        avgstrat += util;
                    }
                    if self.next_to_act == updating_player {
                        exploitability += exploit // something needs to be done here...
                    } else {
                        for i in 0..1326 {
                            exploitability.values[i] =
                                exploitability.values[i].max(exploit.values[i]);
                        }
                    }
                }
                // update strategy
                if self.next_to_act == updating_player {
                    let mut update = [Vector::default(); 2];
                    for (i, &util) in payoffs.iter().enumerate() {
                        update[i] = util - avgstrat;
                    }
                    self.card_strategies.as_mut().unwrap().update_add(&update);
                }

                match self.next_to_act {
                    Small => [avgstrat, exploitability],
                    Big => [avgstrat, exploitability],
                }
            }

            Showdown | BBWins | SBWins => {
                let mut sb_res = Vector::default();
                let mut bb_res = Vector::default();

                match self.terminal {
                    Showdown => {
                        /*let sorted: Vec<(u16, u16)> = card_order
                        .clone()
                        .into_iter()
                        .enumerate()
                        .map(|(i, elem)| {
                            (evaluator.evaluate(elem | self.cards).unwrap_or(0), i as u16)
                        })
                        .sorted() // could be done quicker, saves max 1 sec
                        .collect();*/

                        let sorted = evaluator.vectorized_eval(self.cards);
                        //assert_eq!(new_version.clone(), sorted.clone());

                        let groups = sorted.group_by(|&(a, _), &(b, _)| a == b);

                        let mut collisions_sb = [0.0; 52];
                        let mut collisions_bb = [0.0; 52];

                        let mut cum_sb = 0.0;
                        let mut cum_bb = 0.0;

                        for group in groups {
                            let mut current_cum_sb = 0.0;
                            let mut current_cum_bb = 0.0;

                            let mut current_collisions_sb = [0.0; 52];
                            let mut current_collisions_bb = [0.0; 52];
                            // forward pass
                            for &(_, index) in group {
                                let index = index as usize;
                                let cards = card_order[index];
                                if card_order[index] & self.cards > 0 {
                                    continue;
                                }
                                let card = Evaluator::separate_cards(cards);
                                sb_res[index] += cum_sb;
                                bb_res[index] += cum_bb;
                                current_cum_sb += self.bbbet * range_bb[index];
                                current_cum_bb += self.sbbet * range_sb[index];
                                for c in card {
                                    sb_res[index] -= collisions_sb[c];
                                    bb_res[index] -= collisions_bb[c];
                                    current_collisions_sb[c] += self.bbbet * range_bb[index];
                                    current_collisions_bb[c] += self.sbbet * range_sb[index];
                                }
                            }
                            cum_sb += current_cum_sb;
                            cum_bb += current_cum_bb;
                            for i in 0..52 {
                                collisions_sb[i] += current_collisions_sb[i];
                                collisions_bb[i] += current_collisions_bb[i];
                            }
                        }

                        let mut collisions_sb = [0.0; 52];
                        let mut collisions_bb = [0.0; 52];

                        let mut cum_sb = 0.0;
                        let mut cum_bb = 0.0;

                        let groups = sorted.group_by(|&(a, _), &(b, _)| a == b);

                        for group in groups.rev() {
                            let mut current_cum_sb = 0.0;
                            let mut current_cum_bb = 0.0;

                            let mut current_collisions_sb = [0.0; 52];
                            let mut current_collisions_bb = [0.0; 52];
                            // forward pass
                            for &(_, index) in group {
                                let index = index as usize;
                                let cards = card_order[index];
                                if card_order[index] & self.cards > 0 {
                                    continue;
                                }
                                let card = Evaluator::separate_cards(cards);
                                sb_res[index] -= cum_sb;
                                bb_res[index] -= cum_bb;
                                current_cum_sb += self.sbbet * range_bb[index];
                                current_cum_bb += self.bbbet * range_sb[index];
                                for c in card {
                                    sb_res[index] += collisions_sb[c];
                                    bb_res[index] += collisions_bb[c];
                                    current_collisions_sb[c] += self.sbbet * range_bb[index];
                                    current_collisions_bb[c] += self.bbbet * range_sb[index];
                                }
                            }
                            cum_sb += current_cum_sb;
                            cum_bb += current_cum_bb;
                            for i in 0..52 {
                                collisions_sb[i] += current_collisions_sb[i];
                                collisions_bb[i] += current_collisions_bb[i];
                            }
                        }
                    }
                    SBWins => {
                        let mut sb_sum = 0.0;
                        let mut bb_sum = 0.0;
                        let mut collisions_sb = [0.0; 52];
                        let mut collisions_bb = [0.0; 52];
                        for (index, &cards) in card_order.iter().enumerate() {
                            if cards & self.cards > 0 {
                                continue;
                            }
                            sb_sum += range_sb[index];
                            bb_sum += range_bb[index];
                            let card = Evaluator::separate_cards(cards);
                            for c in card {
                                collisions_sb[c] += range_bb[index];
                                collisions_bb[c] += range_sb[index];
                            }
                        }
                        for index in 0..1326 {
                            if card_order[index] & self.cards > 0 {
                                continue;
                            }
                            sb_res[index] = bb_sum + range_bb[index]; // inclusion exclusion
                            bb_res[index] = sb_sum + range_sb[index];
                            let cards = Evaluator::separate_cards(card_order[index]);
                            for card in cards {
                                sb_res[index] -= collisions_sb[card];
                                bb_res[index] -= collisions_bb[card];
                            }
                        }
                        sb_res
                            .values
                            .iter_mut()
                            .for_each(|elem| *elem *= self.bbbet);
                        bb_res
                            .values
                            .iter_mut()
                            .for_each(|elem| *elem *= -self.bbbet);
                    }
                    BBWins => {
                        let mut sb_sum = 0.0;
                        let mut bb_sum = 0.0;
                        let mut collisions_sb = [0.0; 52];
                        let mut collisions_bb = [0.0; 52];
                        for (index, &cards) in card_order.iter().enumerate() {
                            if cards & self.cards > 0 {
                                continue;
                            }
                            sb_sum += range_sb[index];
                            bb_sum += range_bb[index];
                            let card = Evaluator::separate_cards(cards);
                            for c in card {
                                collisions_sb[c] += range_bb[index];
                                collisions_bb[c] += range_sb[index];
                            }
                        }
                        for index in 0..1326 {
                            if card_order[index] & self.cards > 0 {
                                continue;
                            }
                            sb_res[index] = bb_sum + range_bb[index]; // inclusion exclusion
                            bb_res[index] = sb_sum + range_sb[index];
                            let cards = Evaluator::separate_cards(card_order[index]);
                            for card in cards {
                                sb_res[index] -= collisions_sb[card];
                                bb_res[index] -= collisions_bb[card];
                            }
                        }
                        sb_res
                            .values
                            .iter_mut()
                            .for_each(|elem| *elem *= -self.sbbet);
                        bb_res
                            .values
                            .iter_mut()
                            .for_each(|elem| *elem *= self.sbbet);
                    }
                    _ => todo!(),
                }

                let (util, exploitability) = match updating_player {
                    Small => (sb_res, bb_res),
                    Big => (bb_res, sb_res),
                };
                [util, exploitability]
            }
            Flop => {
                let mut total = [Vector::default(); 2];
                for next_state in self.next_states.iter_mut() {
                    let res = next_state.evaluate_state(
                        range_sb,
                        range_bb,
                        evaluator,
                        iteration_weight,
                        card_order,
                        updating_player,
                        permuter,
                    );
                    for &permutation in &next_state.permutations {
                        let permuted_res = res.map(|v| permuter.permute(permutation, v));
                        for i in 0..2 {
                            total[i] += permuted_res[i];
                        }
                    }
                }
                for t in total.iter_mut() {
                    *t *= 1.0 / 22100.0;
                }
                total
            }
        }
    }
}
