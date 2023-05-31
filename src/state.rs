use crate::enums::Action::*;
use crate::enums::Player::*;
use crate::enums::TerminalState::*;
use crate::enums::*;
use crate::evaluator::Evaluator;
use crate::permutation_handler::PermutationHandler;
use crate::strategy::Strategy;
use itertools::Itertools;
use poker::Suit::{Clubs, Diamonds, Hearts, Spades};
use poker::{Card, Suit};
use std::collections::HashSet;
use std::iter::zip;
use std::time::Instant;

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
        range_sb: &[f32; 1326],
        range_bb: &[f32; 1326],
        evaluator: &Evaluator,
        iteration_weight: f32,
        card_order: &Vec<u64>,
        updating_player: Player,
        permuter: &PermutationHandler,
    ) -> ([f32; 1326], [f32; 1326], [f32; 1326]) {
        //(util of sb, util of bb, exploitability of updating player)
        match self.terminal {
            NonTerminal => {
                let (range, other_range) = match self.next_to_act {
                    Small => (range_sb, range_bb),
                    Big => (range_bb, range_sb),
                };
                // get avg strategy and individual payoffs
                let mut avgstrat = [0.0; 1326];
                let mut other_util = [0.0; 1326];
                let mut exploitability = if self.next_to_act == updating_player {
                    [0.0; 1326]
                } else {
                    [f32::NEG_INFINITY; 1326]
                };
                let mut payoffs = [[0.0; 1326]; 2];

                let strategy = self
                    .card_strategies
                    .as_mut()
                    .unwrap()
                    .get_strategy(iteration_weight);

                for (a, (next, action_prob)) in
                    zip(self.next_states.iter_mut(), strategy).enumerate()
                {
                    let mut new_range = *range;
                    for i in 0..1326 {
                        new_range[i] *= action_prob[i];
                    }

                    let (util_sb, util_bb, exploit) = match self.next_to_act {
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
                    let (util, other) = match self.next_to_act {
                        Small => (util_sb, util_bb),
                        Big => (util_bb, util_sb),
                    };

                    for i in 0..1326 {
                        avgstrat[i] += util[i] * action_prob[i];
                        other_util[i] += other[i];
                    }
                    if self.next_to_act == updating_player {
                        for i in 0..1326 {
                            exploitability[i] += exploit[i];
                        }
                    } else {
                        for i in 0..1326 {
                            exploitability[i] = exploitability[i].max(exploit[i]);
                        }
                    }
                    payoffs[a][..].copy_from_slice(&util);
                }
                // update strategy
                if self.next_to_act == updating_player {
                    let mut update = [[0.0; 1326]; 2];
                    for (i, &util) in payoffs.iter().enumerate() {
                        for card in 0..1326 {
                            let regret = util[card] - avgstrat[card];
                            update[i][card] = regret;
                        }
                    }
                    self.card_strategies.as_mut().unwrap().update_add(&update);
                }

                match self.next_to_act {
                    Small => (avgstrat, other_util, exploitability),
                    Big => (other_util, avgstrat, exploitability),
                }
            }

            Showdown | BBWins | SBWins => {
                /*let _start1 = Instant::now();
                let mut sb_res = [0.0; 1326];
                let mut bb_res = [0.0; 1326];
                for (i1, &hand1) in card_order.iter().enumerate() {
                    for (i2, &hand2) in card_order.iter().enumerate() {
                        if hand1 & hand2 > 0 {
                            continue;
                        }
                        if hand1 & self.cards > 0 || hand2 & self.cards > 0 {
                            continue;
                        }
                        if ((evaluator.evaluate(hand1 | self.cards).unwrap()
                            > evaluator.evaluate(hand2 | self.cards).unwrap())
                            && self.terminal != BBWins)
                            || self.terminal == SBWins
                        {
                            sb_res[i1] += self.bbbet * range_bb[i2];
                            bb_res[i2] -= self.bbbet * range_sb[i1];
                        } else if (evaluator.evaluate(hand1 | self.cards).unwrap()
                            < evaluator.evaluate(hand2 | self.cards).unwrap())
                            || self.terminal == BBWins
                        {
                            sb_res[i1] -= self.sbbet * range_bb[i2];
                            bb_res[i2] += self.sbbet * range_sb[i1];
                        };
                    }
                }
                //dbg!(_start1.elapsed().as_micros());*/

                let _start2 = Instant::now();
                let mut sb_res2 = [0.0; 1326];
                let mut bb_res2 = [0.0; 1326];

                match self.terminal {
                    Showdown => {
                        let sorted: Vec<(u16, usize)> = card_order
                            .clone()
                            .into_iter()
                            .enumerate()
                            .map(|(i, elem)| {
                                (evaluator.evaluate(elem | self.cards).unwrap_or(0), i)
                            })
                            .sorted() // could be done quicker, saves max 1 sec
                            .collect();
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
                                let cards = card_order[index];
                                if card_order[index] & self.cards > 0 {
                                    continue;
                                }
                                let card = Evaluator::separate_cards(cards);
                                sb_res2[index] += cum_sb;
                                bb_res2[index] += cum_bb;
                                current_cum_sb += self.bbbet * range_bb[index];
                                current_cum_bb += self.sbbet * range_sb[index];
                                for c in card {
                                    sb_res2[index] -= collisions_sb[c];
                                    bb_res2[index] -= collisions_bb[c];
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
                                let cards = card_order[index];
                                if card_order[index] & self.cards > 0 {
                                    continue;
                                }
                                let card = Evaluator::separate_cards(cards);
                                sb_res2[index] -= cum_sb;
                                bb_res2[index] -= cum_bb;
                                current_cum_sb += self.sbbet * range_bb[index];
                                current_cum_bb += self.bbbet * range_sb[index];
                                for c in card {
                                    sb_res2[index] += collisions_sb[c];
                                    bb_res2[index] += collisions_bb[c];
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
                        for index in 0..sb_res2.len() {
                            if card_order[index] & self.cards > 0 {
                                continue;
                            }
                            sb_res2[index] = bb_sum + range_bb[index]; // inclusion exclusion
                            bb_res2[index] = sb_sum + range_sb[index];
                            let cards = Evaluator::separate_cards(card_order[index]);
                            for card in cards {
                                sb_res2[index] -= collisions_sb[card];
                                bb_res2[index] -= collisions_bb[card];
                            }
                        }
                        sb_res2.iter_mut().for_each(|elem| *elem *= self.bbbet);
                        bb_res2.iter_mut().for_each(|elem| *elem *= -self.bbbet);
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
                        for index in 0..sb_res2.len() {
                            if card_order[index] & self.cards > 0 {
                                continue;
                            }
                            sb_res2[index] = bb_sum + range_bb[index]; // inclusion exclusion
                            bb_res2[index] = sb_sum + range_sb[index];
                            let cards = Evaluator::separate_cards(card_order[index]);
                            for card in cards {
                                sb_res2[index] -= collisions_sb[card];
                                bb_res2[index] -= collisions_bb[card];
                            }
                        }
                        sb_res2.iter_mut().for_each(|elem| *elem *= -self.sbbet);
                        bb_res2.iter_mut().for_each(|elem| *elem *= self.sbbet);
                    }
                    _ => todo!(),
                }
                //dbg!(_start2.elapsed().as_micros());

                // FIX COLLISIONS FOR TWO PAIR HANDS
                /*assert!(zip(sb_res, sb_res2)
                    .map(|(a, b)| (a - b).abs())
                    .all(|x| x <= 1e-6));
                assert!(zip(bb_res, bb_res2)
                    .map(|(a, b)| (a - b).abs())
                    .all(|x| x <= 1e-6));*/
                let exploitability = match updating_player {
                    Small => bb_res2,
                    Big => sb_res2,
                };
                (sb_res2, bb_res2, exploitability)
            }
            Flop => {
                let mut total = ([0.0; 1326], [0.0; 1326], [0.0; 1326]);
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
                    assert!(!next_state.permutations.is_empty());
                    for &permutation in &next_state.permutations {
                        let permuted_res = (
                            permuter.permute(permutation, res.0),
                            permuter.permute(permutation, res.1),
                            permuter.permute(permutation, res.2),
                        );
                        for i in 0..total.0.len() {
                            total.0[i] += permuted_res.0[i];
                            total.1[i] += permuted_res.1[i];
                            total.2[i] += permuted_res.2[i];
                        }
                    }
                }
                for i in 0..total.0.len() {
                    total.0[i] /= 22100.0; // 52 choose 3
                    total.1[i] /= 22100.0;
                    total.2[i] /= 22100.0;
                }
                total
            }
        }
    }
}
