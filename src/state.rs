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
        range_sb: &Vector,
        range_bb: &Vector,
        evaluator: &Evaluator,
        card_order: &Vec<u64>,
        updating_player: Player,
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
                        ),
                        Big => next.evaluate_state(
                            other_range,
                            &new_range,
                            evaluator,
                            card_order,
                            updating_player,
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

                match self.terminal {
                    Showdown => {
                        let sorted = evaluator.vectorized_eval(self.cards);

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
                    );
                    for &permutation in &next_state.permutations {
                        total += permute(permutation, res)
                    }
                }
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
    ) -> [Vector; 2] {
        match self.terminal {
            NonTerminal => {
                // get avg strategy and individual payoffs
                let (mut sb_avg, mut bb_avg) = match self.next_to_act {
                    Small => (
                        Vector {
                            values: [f64::NEG_INFINITY; 1326],
                        },
                        Vector::default(),
                    ),
                    Big => (
                        Vector::default(),
                        Vector {
                            values: [f64::NEG_INFINITY; 1326],
                        },
                    ),
                };

                let strategy = self.card_strategies.as_ref().unwrap().get_strategy();

                for (next, action_prob) in zip(self.next_states.iter_mut(), strategy) {
                    let [sb_res, bb_res] = match self.next_to_act {
                        Small => next.calc_exploit(
                            &(*range_sb * action_prob),
                            range_bb,
                            evaluator,
                            card_order,
                        ),
                        Big => next.calc_exploit(
                            range_sb,
                            &(*range_bb * action_prob),
                            evaluator,
                            card_order,
                        ),
                    };
                    match self.next_to_act {
                        Small => {
                            for i in 0..1326 {
                                sb_avg[i] = sb_avg[i].max(sb_res[i]);
                            }
                            bb_avg += bb_res;
                        }
                        Big => {
                            sb_avg += sb_res;
                            for i in 0..1326 {
                                bb_avg[i] = bb_avg[i].max(bb_res[i]);
                            }
                        }
                    }
                }

                [sb_avg, bb_avg]
            }

            Showdown | BBWins | SBWins => {
                let mut sb_res = Vector::default();
                let mut bb_res = Vector::default();
                match self.terminal {
                    Showdown => {
                        let sorted = evaluator.vectorized_eval(self.cards);
                        //assert_eq!(new_version.clone(), sorted.clone());

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
                [sb_res, bb_res]
            }
            Flop => {
                let mut total = [Vector::default(); 2];
                let range_sb_new = *range_sb * (1.0 / 22100.0);
                let range_bb_new = *range_bb * (1.0 / 22100.0);
                for next_state in self.next_states.iter_mut() {
                    let [sb_res, bb_res] = next_state.calc_exploit(
                        &range_sb_new,
                        &range_bb_new,
                        evaluator,
                        card_order,
                    );
                    for &permutation in &next_state.permutations {
                        total[0] += permute(permutation, sb_res);
                        total[1] += permute(permutation, bb_res);
                    }
                }
                total
            }
        }
    }
}
