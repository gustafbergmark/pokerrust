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
                    assert_eq!(self.sbbet, self.bbbet);
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
                    assert_eq!(self.sbbet, self.bbbet);
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
        card_order: &Vec<u64>,
        updating_player: Player,
        permuter: &PermutationHandler,
    ) -> Vector {
        //(util of sb, util of bb, exploitability of updating player)
        match self.terminal {
            NonTerminal => {
                let (range, other_range) = match self.next_to_act {
                    Small => (range_sb, range_bb),
                    Big => (range_bb, range_sb),
                };
                // get avg strategy and individual payoffs
                let mut avgstrat = Vector::default();
                let mut payoffs = [Vector::default(); 2];

                let strategy = self.card_strategies.as_ref().unwrap().get_strategy();

                for (a, (next, action_prob)) in
                    zip(self.next_states.iter_mut(), strategy).enumerate()
                {
                    let new_range = *range * action_prob;

                    let util = match self.next_to_act {
                        Small => next.evaluate_state(
                            &new_range,
                            other_range,
                            evaluator,
                            card_order,
                            updating_player,
                            permuter,
                        ),
                        Big => next.evaluate_state(
                            other_range,
                            &new_range,
                            evaluator,
                            card_order,
                            updating_player,
                            permuter,
                        ),
                    };
                    if updating_player == self.next_to_act {
                        //payoffs[a].values[..].copy_from_slice(&util.values);
                        payoffs[a] = util;
                        avgstrat += util * action_prob;
                    } else {
                        avgstrat += util;
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

                avgstrat
            }

            Showdown | BBWins | SBWins => {
                let mut sb_res = Vector::default();
                let mut bb_res = Vector::default();

                /*let mut sb_res2 = [0.0; 1326];
                let mut bb_res2 = [0.0; 1326];
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
                            sb_res2[i1] += self.bbbet * range_bb[i2];
                            bb_res2[i2] -= self.bbbet * range_sb[i1];
                        } else if (evaluator.evaluate(hand1 | self.cards).unwrap()
                            < evaluator.evaluate(hand2 | self.cards).unwrap())
                            || self.terminal == BBWins
                        {
                            sb_res2[i1] -= self.sbbet * range_bb[i2];
                            bb_res2[i2] += self.sbbet * range_sb[i1];
                        };
                    }
                }*/

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

                        if updating_player == Small {
                            let groups = sorted.group_by(|&(a, _), &(b, _)| a == b);

                            let mut collisions_sb = [0.0; 52];

                            let mut cum_sb = 0.0;

                            for group in groups {
                                let mut current_cum_sb = 0.0;

                                let mut current_collisions_sb = [0.0; 52];
                                // forward pass
                                for &(_, index) in group {
                                    let index = index as usize;
                                    let cards = card_order[index];
                                    if card_order[index] & self.cards > 0 {
                                        continue;
                                    }
                                    let card = Evaluator::separate_cards(cards);
                                    sb_res[index] += cum_sb;
                                    current_cum_sb += range_bb[index];
                                    for c in card {
                                        sb_res[index] -= collisions_sb[c];
                                        current_collisions_sb[c] += range_bb[index];
                                    }
                                }
                                cum_sb += current_cum_sb * self.bbbet;
                                for i in 0..52 {
                                    collisions_sb[i] += current_collisions_sb[i] * self.bbbet;
                                }
                            }

                            let mut collisions_sb = [0.0; 52];

                            let mut cum_sb = 0.0;

                            let groups = sorted.group_by(|&(a, _), &(b, _)| a == b);

                            for group in groups.rev() {
                                let mut current_cum_sb = 0.0;

                                let mut current_collisions_sb = [0.0; 52];
                                // forward pass
                                for &(_, index) in group {
                                    let index = index as usize;
                                    let cards = card_order[index];
                                    if card_order[index] & self.cards > 0 {
                                        continue;
                                    }
                                    let card = Evaluator::separate_cards(cards);
                                    sb_res[index] -= cum_sb;
                                    current_cum_sb += range_bb[index];
                                    for c in card {
                                        sb_res[index] += collisions_sb[c];
                                        current_collisions_sb[c] += range_bb[index];
                                    }
                                }
                                cum_sb += current_cum_sb * self.sbbet;
                                for i in 0..52 {
                                    collisions_sb[i] += current_collisions_sb[i] * self.sbbet;
                                }
                            }
                        }
                        if updating_player == Big {
                            let groups = sorted.group_by(|&(a, _), &(b, _)| a == b);

                            let mut collisions_bb = [0.0; 52];

                            let mut cum_bb = 0.0;

                            for group in groups {
                                let mut current_cum_bb = 0.0;

                                let mut current_collisions_bb = [0.0; 52];
                                // forward pass
                                for &(_, index) in group {
                                    let index = index as usize;
                                    let cards = card_order[index];
                                    if card_order[index] & self.cards > 0 {
                                        continue;
                                    }
                                    let card = Evaluator::separate_cards(cards);
                                    bb_res[index] += cum_bb;
                                    current_cum_bb += range_sb[index];
                                    for c in card {
                                        bb_res[index] -= collisions_bb[c];
                                        current_collisions_bb[c] += range_sb[index];
                                    }
                                }
                                cum_bb += current_cum_bb * self.sbbet;
                                for i in 0..52 {
                                    collisions_bb[i] += current_collisions_bb[i] * self.sbbet;
                                }
                            }

                            let mut collisions_bb = [0.0; 52];

                            let mut cum_bb = 0.0;

                            let groups = sorted.group_by(|&(a, _), &(b, _)| a == b);

                            for group in groups.rev() {
                                let mut current_cum_bb = 0.0;

                                let mut current_collisions_bb = [0.0; 52];
                                // forward pass
                                for &(_, index) in group {
                                    let index = index as usize;
                                    let cards = card_order[index];
                                    if card_order[index] & self.cards > 0 {
                                        continue;
                                    }
                                    let card = Evaluator::separate_cards(cards);
                                    bb_res[index] -= cum_bb;
                                    current_cum_bb += range_sb[index];
                                    for c in card {
                                        bb_res[index] += collisions_bb[c];
                                        current_collisions_bb[c] += range_sb[index];
                                    }
                                }
                                cum_bb += current_cum_bb * self.bbbet;
                                for i in 0..52 {
                                    collisions_bb[i] += current_collisions_bb[i] * self.bbbet;
                                }
                            }
                        }
                    }
                    SBWins => {
                        if updating_player == Small {
                            let mut bb_sum = 0.0;
                            let mut collisions_sb = [0.0; 52];
                            for (index, &cards) in card_order.iter().enumerate() {
                                if cards & self.cards > 0 {
                                    continue;
                                }
                                bb_sum += range_bb[index];
                                let card = Evaluator::separate_cards(cards);
                                for c in card {
                                    collisions_sb[c] += range_bb[index];
                                }
                            }
                            for index in 0..1326 {
                                if card_order[index] & self.cards > 0 {
                                    continue;
                                }
                                sb_res[index] = bb_sum + range_bb[index]; // inclusion exclusion
                                let cards = Evaluator::separate_cards(card_order[index]);
                                for card in cards {
                                    sb_res[index] -= collisions_sb[card];
                                }
                            }
                            sb_res *= self.bbbet;
                        }
                        if updating_player == Big {
                            let mut sb_sum = 0.0;
                            let mut collisions_bb = [0.0; 52];
                            for (index, &cards) in card_order.iter().enumerate() {
                                if cards & self.cards > 0 {
                                    continue;
                                }
                                sb_sum += range_sb[index];
                                let card = Evaluator::separate_cards(cards);
                                for c in card {
                                    collisions_bb[c] += range_sb[index];
                                }
                            }
                            for index in 0..1326 {
                                if card_order[index] & self.cards > 0 {
                                    continue;
                                }
                                bb_res[index] = sb_sum + range_sb[index];
                                let cards = Evaluator::separate_cards(card_order[index]);
                                for card in cards {
                                    bb_res[index] -= collisions_bb[card];
                                }
                            }
                            bb_res *= -self.bbbet;
                        }
                    }
                    BBWins => {
                        if updating_player == Small {
                            let mut bb_sum = 0.0;
                            let mut collisions_sb = [0.0; 52];
                            for (index, &cards) in card_order.iter().enumerate() {
                                if cards & self.cards > 0 {
                                    continue;
                                }
                                bb_sum += range_bb[index];
                                let card = Evaluator::separate_cards(cards);
                                for c in card {
                                    collisions_sb[c] += range_bb[index];
                                }
                            }
                            for index in 0..1326 {
                                if card_order[index] & self.cards > 0 {
                                    continue;
                                }
                                sb_res[index] = bb_sum + range_bb[index]; // inclusion exclusion
                                let cards = Evaluator::separate_cards(card_order[index]);
                                for card in cards {
                                    sb_res[index] -= collisions_sb[card];
                                }
                            }
                            sb_res *= -self.sbbet;
                        }
                        if updating_player == Big {
                            let mut sb_sum = 0.0;
                            let mut collisions_bb = [0.0; 52];
                            for (index, &cards) in card_order.iter().enumerate() {
                                if cards & self.cards > 0 {
                                    continue;
                                }
                                sb_sum += range_sb[index];
                                let card = Evaluator::separate_cards(cards);
                                for c in card {
                                    collisions_bb[c] += range_sb[index];
                                }
                            }
                            for index in 0..1326 {
                                if card_order[index] & self.cards > 0 {
                                    continue;
                                }
                                bb_res[index] = sb_sum + range_sb[index];
                                let cards = Evaluator::separate_cards(card_order[index]);
                                for card in cards {
                                    bb_res[index] -= collisions_bb[card];
                                }
                            }
                            bb_res *= self.sbbet;
                        }
                    }
                    _ => todo!(),
                }

                /*assert!(zip(sb_res.values, sb_res2)
                    .map(|(a, b)| (a - b).abs())
                    .all(|x| x <= 1e-6));
                assert!(zip(bb_res.values, bb_res2)
                    .map(|(a, b)| (a - b).abs())
                    .all(|x| x <= 1e-6));*/
                let (util, _exploitability) = match updating_player {
                    Small => (sb_res, bb_res),
                    Big => (bb_res, sb_res),
                };
                util
            }
            Flop => {
                let mut total = Vector::default();
                let range_sb_new = *range_sb * (1.0 / 22100.0);
                let range_bb_new = *range_bb * (1.0 / 22100.0);
                for next_state in self.next_states.iter_mut() {
                    let res = next_state.evaluate_state(
                        &range_sb_new,
                        &range_bb_new,
                        evaluator,
                        card_order,
                        updating_player,
                        permuter,
                    );
                    for &permutation in &next_state.permutations {
                        total += permuter.permute(permutation, res)
                    }
                }
                /*total[0] *= 1.0 / 19600.0; // 50 choose 3 possible flops for each hand
                if calc_exploit {
                    total[1] *= 1.0 / 19600.0;
                }*/
                total
            }
        }
    }

    pub fn calc_exploit(
        &mut self,
        range_sb: &Vector,
        range_bb: &Vector,
        evaluator: &Evaluator,
        card_order: &Vec<u64>,
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
                let strategy = self.card_strategies.as_ref().unwrap().get_strategy();

                for (a, (next, action_prob)) in
                    zip(self.next_states.iter_mut(), strategy).enumerate()
                {
                    let new_range = *range * action_prob;

                    let util = match self.next_to_act {
                        Small => next.calc_exploit(
                            &new_range,
                            other_range,
                            evaluator,
                            card_order,
                            permuter,
                        ),
                        Big => next.calc_exploit(
                            other_range,
                            &new_range,
                            evaluator,
                            card_order,
                            permuter,
                        ),
                    };
                }
                [Vector::default(); 2]
            }

            Showdown | BBWins | SBWins => {
                let mut sb_res = Vector::default();
                let mut bb_res = Vector::default();

                let mut norm_sb = Vector::default();
                let mut norm_bb = Vector::default();

                match self.terminal {
                    Showdown => {
                        assert_eq!(self.sbbet, self.bbbet);
                        let bet = self.sbbet;

                        let sorted = evaluator.vectorized_eval(self.cards);
                        //Small Blind
                        let groups = sorted.group_by(|&(a, _), &(b, _)| a == b);

                        let mut collisions_sb = [0.0; 52];

                        let mut cum_sb = 0.0;

                        for group in groups {
                            let mut current_cum_sb = 0.0;

                            let mut current_collisions_sb = [0.0; 52];
                            // forward pass
                            for &(_, index) in group {
                                let index = index as usize;
                                let cards = card_order[index];
                                if card_order[index] & self.cards > 0 {
                                    assert_eq!(range_sb[index], 0.0);
                                    assert_eq!(range_bb[index], 0.0);
                                    continue;
                                }
                                let card = Evaluator::separate_cards(cards);
                                sb_res[index] += cum_sb;
                                current_cum_sb += range_bb[index];
                                for c in card {
                                    sb_res[index] -= collisions_sb[c];
                                    current_collisions_sb[c] += range_bb[index];
                                }
                            }
                            cum_sb += current_cum_sb;
                            for i in 0..52 {
                                collisions_sb[i] += current_collisions_sb[i];
                            }
                        }

                        let mut collisions_sb = [0.0; 52];

                        let mut cum_sb = 0.0;

                        let groups = sorted.group_by(|&(a, _), &(b, _)| a == b);

                        for group in groups.rev() {
                            let mut current_cum_sb = 0.0;

                            let mut current_collisions_sb = [0.0; 52];
                            // forward pass
                            for &(_, index) in group {
                                let index = index as usize;
                                let cards = card_order[index];
                                if card_order[index] & self.cards > 0 {
                                    continue;
                                }
                                let card = Evaluator::separate_cards(cards);
                                sb_res[index] -= cum_sb;
                                current_cum_sb += range_bb[index];
                                for c in card {
                                    sb_res[index] += collisions_sb[c];
                                    current_collisions_sb[c] += range_bb[index];
                                }
                            }
                            cum_sb += current_cum_sb;
                            for i in 0..52 {
                                collisions_sb[i] += current_collisions_sb[i];
                            }
                        }
                        sb_res *= bet;
                        let range_bb_sum = range_bb.sum();
                        assert!((range_bb_sum - cum_sb).abs() < 1e-6);
                        for i in 0..1326 {
                            if card_order[i] & self.cards > 0 {
                                continue;
                            }
                            let cards = Evaluator::separate_cards(card_order[i]);
                            let norm =
                                range_bb_sum - collisions_sb[cards[0]] - collisions_sb[cards[1]]
                                    + range_bb[i];
                            if norm < 1e-8 {
                                norm_sb[i] = 0.0;
                            } else {
                                norm_sb[i] = sb_res[i] / norm;
                            }
                        }

                        // Big Blind
                        let groups = sorted.group_by(|&(a, _), &(b, _)| a == b);
                        let mut collisions_bb = [0.0; 52];

                        let mut cum_bb = 0.0;

                        for group in groups {
                            let mut current_cum_bb = 0.0;

                            let mut current_collisions_bb = [0.0; 52];
                            // forward pass
                            for &(_, index) in group {
                                let index = index as usize;
                                let cards = card_order[index];
                                if card_order[index] & self.cards > 0 {
                                    continue;
                                }
                                let card = Evaluator::separate_cards(cards);
                                bb_res[index] += cum_bb;
                                current_cum_bb += range_sb[index];
                                for c in card {
                                    bb_res[index] -= collisions_bb[c];
                                    current_collisions_bb[c] += range_sb[index];
                                }
                            }
                            cum_bb += current_cum_bb;
                            for i in 0..52 {
                                collisions_bb[i] += current_collisions_bb[i];
                            }
                        }

                        let mut collisions_bb = [0.0; 52];

                        let mut cum_bb = 0.0;

                        let groups = sorted.group_by(|&(a, _), &(b, _)| a == b);

                        for group in groups.rev() {
                            let mut current_cum_bb = 0.0;

                            let mut current_collisions_bb = [0.0; 52];
                            // forward pass
                            for &(_, index) in group {
                                let index = index as usize;
                                let cards = card_order[index];
                                if card_order[index] & self.cards > 0 {
                                    continue;
                                }
                                let card = Evaluator::separate_cards(cards);
                                bb_res[index] -= cum_bb;
                                current_cum_bb += range_sb[index];
                                for c in card {
                                    bb_res[index] += collisions_bb[c];
                                    current_collisions_bb[c] += range_sb[index];
                                }
                            }
                            cum_bb += current_cum_bb;
                            for i in 0..52 {
                                collisions_bb[i] += current_collisions_bb[i];
                            }
                        }
                        bb_res *= bet;
                        let range_sb_sum = range_sb.sum();
                        assert!((range_sb_sum - cum_bb).abs() < 1e-6);

                        for i in 0..1326 {
                            if card_order[i] & self.cards > 0 {
                                continue;
                            }
                            let cards = Evaluator::separate_cards(card_order[i]);
                            let norm =
                                range_sb_sum - collisions_bb[cards[0]] - collisions_bb[cards[1]]
                                    + range_sb[i];
                            if norm < 1e-8 {
                                norm_bb[i] = 0.0;
                            } else {
                                norm_bb[i] = bb_res[i] / norm;
                            }
                        }

                        //assert!((bb_res2.sum() - bb_res.sum()).abs() < 1e-6);
                        //dbg!("correct");

                        let res1 = (norm_sb * range_sb.norm()).sum();
                        let res2 = (norm_bb * range_bb.norm()).sum();
                        if (res1 + res2).abs() > 1e-8 {
                            dbg!((res1 + res2).abs(), res1, res2);
                            let mut sb_res2 = Vector::default();
                            for (i1, &hand1) in card_order.iter().enumerate() {
                                let mut norm = 0.0;
                                for (i2, &hand2) in card_order.iter().enumerate() {
                                    if hand1 & hand2 > 0 {
                                        continue;
                                    }
                                    if hand1 & self.cards > 0 || hand2 & self.cards > 0 {
                                        continue;
                                    }
                                    norm += range_bb[i2];
                                    if evaluator.evaluate(hand1 | self.cards).unwrap()
                                        > evaluator.evaluate(hand2 | self.cards).unwrap()
                                    {
                                        sb_res2[i1] += bet * range_bb[i2];
                                    } else if evaluator.evaluate(hand1 | self.cards).unwrap()
                                        < evaluator.evaluate(hand2 | self.cards).unwrap()
                                    {
                                        sb_res2[i1] -= bet * range_bb[i2];
                                    };
                                }
                                if norm == 0.0 {
                                    sb_res2[i1] = 0.0;
                                } else {
                                    sb_res2[i1] /= norm;
                                }
                            }
                            let mut bb_res2 = Vector::default();
                            for (i1, &hand1) in card_order.iter().enumerate() {
                                let mut norm = 0.0;
                                for (i2, &hand2) in card_order.iter().enumerate() {
                                    if hand1 & hand2 > 0 {
                                        continue;
                                    }
                                    if hand1 & self.cards > 0 || hand2 & self.cards > 0 {
                                        continue;
                                    }
                                    norm += range_sb[i2];
                                    if evaluator.evaluate(hand1 | self.cards).unwrap()
                                        > evaluator.evaluate(hand2 | self.cards).unwrap()
                                    {
                                        bb_res2[i1] += bet * range_sb[i2];
                                    } else if evaluator.evaluate(hand1 | self.cards).unwrap()
                                        < evaluator.evaluate(hand2 | self.cards).unwrap()
                                    {
                                        bb_res2[i1] -= bet * range_sb[i2];
                                    };
                                }
                                if norm == 0.0 {
                                    bb_res2[i1] = 0.0;
                                } else {
                                    bb_res2[i1] /= norm;
                                }
                            }

                            let res1 = (sb_res2 * range_sb.norm()).sum();
                            let res2 = (bb_res2 * range_bb.norm()).sum();
                            if (res1 + res2).abs() > 1e-8 {
                                dbg!("slow", (res1 + res2).abs(), res1, res2);
                            }
                            todo!()
                        }
                    }
                    SBWins => {
                        let mut bb_sum = 0.0;
                        let mut collisions_sb = [0.0; 52];
                        for (index, &cards) in card_order.iter().enumerate() {
                            if cards & self.cards > 0 {
                                continue;
                            }
                            bb_sum += range_bb[index];
                            let card = Evaluator::separate_cards(cards);
                            for c in card {
                                collisions_sb[c] += range_bb[index];
                            }
                        }
                        for index in 0..1326 {
                            if card_order[index] & self.cards > 0 {
                                continue;
                            }
                            sb_res[index] = bb_sum + range_bb[index]; // inclusion exclusion
                            let cards = Evaluator::separate_cards(card_order[index]);
                            for card in cards {
                                sb_res[index] -= collisions_sb[card];
                            }
                        }
                        sb_res *= self.bbbet;
                        let mut sb_sum = 0.0;
                        let mut collisions_bb = [0.0; 52];
                        for (index, &cards) in card_order.iter().enumerate() {
                            if cards & self.cards > 0 {
                                continue;
                            }
                            sb_sum += range_sb[index];
                            let card = Evaluator::separate_cards(cards);
                            for c in card {
                                collisions_bb[c] += range_sb[index];
                            }
                        }
                        for index in 0..1326 {
                            if card_order[index] & self.cards > 0 {
                                continue;
                            }
                            bb_res[index] = sb_sum + range_sb[index];
                            let cards = Evaluator::separate_cards(card_order[index]);
                            for card in cards {
                                bb_res[index] -= collisions_bb[card];
                            }
                        }
                        bb_res *= -self.bbbet;
                    }
                    BBWins => {
                        let mut bb_sum = 0.0;
                        let mut collisions_sb = [0.0; 52];
                        for (index, &cards) in card_order.iter().enumerate() {
                            if cards & self.cards > 0 {
                                continue;
                            }
                            bb_sum += range_bb[index];
                            let card = Evaluator::separate_cards(cards);
                            for c in card {
                                collisions_sb[c] += range_bb[index];
                            }
                        }
                        for index in 0..1326 {
                            if card_order[index] & self.cards > 0 {
                                continue;
                            }
                            sb_res[index] = bb_sum + range_bb[index]; // inclusion exclusion
                            let cards = Evaluator::separate_cards(card_order[index]);
                            for card in cards {
                                sb_res[index] -= collisions_sb[card];
                            }
                        }
                        sb_res *= -self.sbbet;

                        let mut sb_sum = 0.0;
                        let mut collisions_bb = [0.0; 52];
                        for (index, &cards) in card_order.iter().enumerate() {
                            if cards & self.cards > 0 {
                                continue;
                            }
                            sb_sum += range_sb[index];
                            let card = Evaluator::separate_cards(cards);
                            for c in card {
                                collisions_bb[c] += range_sb[index];
                            }
                        }
                        for index in 0..1326 {
                            if card_order[index] & self.cards > 0 {
                                continue;
                            }
                            bb_res[index] = sb_sum + range_sb[index];
                            let cards = Evaluator::separate_cards(card_order[index]);
                            for card in cards {
                                bb_res[index] -= collisions_bb[card];
                            }
                        }
                        bb_res *= self.sbbet;
                    }
                    _ => todo!(),
                }
                [norm_sb, norm_bb]
            }
            Flop => {
                let mut total = [Vector::default(); 2];
                let range_sb_new = *range_sb * (1.0 / 22100.0);
                let range_bb_new = *range_bb * (1.0 / 22100.0);
                for next_state in self.next_states.iter_mut() {
                    let mut possible = Vector::ones();
                    for i in 0..1326 {
                        if next_state.cards & card_order[i] > 0 {
                            possible[i] = 0.0;
                        }
                    }
                    let res = next_state.calc_exploit(
                        &(range_sb_new * possible),
                        &(range_bb_new * possible),
                        evaluator,
                        card_order,
                        permuter,
                    );
                    for &permutation in &next_state.permutations {
                        total[0] += permuter.permute(permutation, res[0]);
                        total[1] += permuter.permute(permutation, res[1]);
                    }
                }
                total
            }
        }
    }
}
