use crate::enums::Action::*;
use crate::enums::Player::*;
use crate::enums::TerminalState::*;
use crate::enums::*;
use crate::evaluator::Evaluator;
use crate::strategy::Strategy;
use itertools::Itertools;
use poker::Card;
use std::iter::zip;

#[derive(Clone, Debug)]
pub(crate) struct State {
    terminal: TerminalState,
    pub action: Action,
    pub cards: u64,
    sbbet: f32,
    bbbet: f32,
    next_to_act: Player,
    card_strategies: Option<Strategy>,
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
            card_strategies: Some(Strategy::new()),
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
                    card_strategies: None,
                    next_states: vec![],
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
                }],
                Call => {
                    if self.cards == 0 && false {
                        vec![State {
                            terminal: Flop,
                            action,
                            cards: self.cards.clone(),
                            sbbet: self.bbbet,
                            bbbet: self.bbbet,
                            next_to_act: Big,
                            card_strategies: None,
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
                            card_strategies: None,
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
                    card_strategies: Some(Strategy::new()),
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
                            card_strategies: Some(Strategy::new()),
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
                    card_strategies: None,
                    next_states: vec![],
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
                }],
                Call => {
                    if self.cards == 0 && false {
                        vec![State {
                            terminal: Flop,
                            action,
                            cards: self.cards.clone(),
                            sbbet: self.sbbet,
                            bbbet: self.sbbet,
                            next_to_act: Small,
                            card_strategies: None,
                            next_states: vec![],
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
                            card_strategies: Some(Strategy::new()),
                            next_states: vec![],
                        };
                        next_states.push(next_state)
                    }
                    next_states
                }
            },
        }
    }

    pub fn evaluate_state(
        &mut self,
        range_sb: [f32; 3],
        range_bb: [f32; 3],
        evaluator: &Evaluator,
        iteration_weight: f32,
        card_order: &Vec<u64>,
        updating_player: Player,
    ) -> ([f32; 3], [f32; 3]) {
        match self.terminal {
            NonTerminal => {
                let (range, other_range) = match self.next_to_act {
                    Small => (range_sb, range_bb),
                    Big => (range_bb, range_sb),
                };
                // get avg strategy and individual payoffs
                let mut avgstrat = [0.0; 3];
                let mut other_util = [0.0; 3];
                let mut payoffs = [[0.0; 3]; 2];

                let strategy = self
                    .card_strategies
                    .as_mut()
                    .unwrap()
                    .get_strategy(1.0 /*???*/, iteration_weight);

                let mut a = 0;
                for (next, action_prob) in zip(self.next_states.iter_mut(), strategy) {
                    let mut new_range = range.clone();
                    for i in 0..3 {
                        new_range[i] *= action_prob[i];
                    }

                    let (util_sb, util_bb) = match self.next_to_act {
                        Small => next.evaluate_state(
                            new_range,
                            other_range,
                            evaluator,
                            iteration_weight,
                            card_order,
                            updating_player,
                        ),
                        Big => next.evaluate_state(
                            other_range,
                            new_range,
                            evaluator,
                            iteration_weight,
                            card_order,
                            updating_player,
                        ),
                    };
                    let (util, other) = match self.next_to_act {
                        Small => (util_sb, util_bb),
                        Big => (util_bb, util_sb),
                    };

                    for i in 0..3 {
                        avgstrat[i] += util[i] * action_prob[i];
                        payoffs[a][i] = util[i];

                        other_util[i] += other[i];
                    }
                    a += 1;
                }
                // update strategy
                if self.next_to_act == updating_player {
                    let mut update = [[0.0; 3]; 2];
                    for (i, &util) in payoffs.iter().enumerate() {
                        for card in 0..3 {
                            let diff = util[card] - avgstrat[card];
                            let regret = diff;
                            update[i][card] = regret;
                        }
                    }
                    self.card_strategies.as_mut().unwrap().update_add(&update);
                }

                return match self.next_to_act {
                    Small => (avgstrat, other_util),
                    Big => (other_util, avgstrat),
                };
            }

            Showdown | BBWins | SBWins => {
                let mut sb_res = [0.0; 3];
                let mut bb_res = [0.0; 3];
                for (i1, &hand1) in card_order.iter().enumerate() {
                    for (i2, &hand2) in card_order.iter().enumerate() {
                        if hand1 & hand2 > 0 {
                            continue;
                        }
                        if hand1 & self.cards > 0 || hand2 & self.cards > 0 {
                            continue;
                        }
                        if (evaluator.evaluate(hand1 | self.cards)
                            > evaluator.evaluate(hand2 | self.cards)
                            && self.terminal != BBWins)
                            || self.terminal == SBWins
                        {
                            sb_res[i1] += self.bbbet * range_bb[i2];
                            bb_res[i2] -= self.bbbet * range_sb[i1];
                        } else {
                            sb_res[i1] -= self.sbbet * range_bb[i2];
                            bb_res[i2] += self.sbbet * range_sb[i1];
                        };
                    }
                }

                let mut sb_res2 = [0.0; 3];
                let mut bb_res2 = [0.0; 3];

                match self.terminal {
                    Showdown => {
                        let sorted: Vec<usize> = card_order
                            .clone()
                            .into_iter()
                            .enumerate()
                            .map(|(i, elem)| (evaluator.evaluate(elem), i))
                            .sorted()
                            .map(|x| x.1)
                            .collect();

                        let mut cum_sb = 0.0;
                        let mut cum_bb = 0.0;
                        // forward pass
                        for &index in &sorted {
                            sb_res2[index] += cum_sb;
                            bb_res2[index] += cum_bb;
                            cum_sb += self.bbbet * range_bb[index];
                            cum_bb += self.sbbet * range_sb[index];
                        }

                        //backward pass;
                        cum_sb = 0.0;
                        cum_bb = 0.0;

                        for &index in sorted.iter().rev() {
                            sb_res2[index] -= cum_sb;
                            bb_res2[index] -= cum_bb;
                            cum_sb += self.sbbet * range_bb[index];
                            cum_bb += self.bbbet * range_sb[index];
                        }
                    }
                    SBWins => {
                        let sb_sum = range_sb.iter().sum::<f32>();
                        let bb_sum = range_bb.iter().sum::<f32>();
                        for index in 0..sb_res2.len() {
                            sb_res2[index] = self.bbbet * (bb_sum - range_bb[index]);
                            bb_res2[index] = -self.bbbet * (sb_sum - range_sb[index]);
                        }
                    }
                    BBWins => {
                        let sb_sum = range_sb.iter().sum::<f32>();
                        let bb_sum = range_bb.iter().sum::<f32>();
                        for index in 0..sb_res2.len() {
                            sb_res2[index] = -self.sbbet * (bb_sum - range_bb[index]);
                            bb_res2[index] = self.sbbet * (sb_sum - range_sb[index]);
                        }
                    }
                    _ => todo!(),
                }

                // FIX COLLISIONS FOR TWO PAIR HANDS

                assert!(zip(sb_res, sb_res2)
                    .map(|(a, b)| (a - b).abs())
                    .all(|x| x <= 1e-6));
                assert!(zip(bb_res, bb_res2)
                    .map(|(a, b)| (a - b).abs())
                    .all(|x| x <= 1e-6));

                (sb_res, bb_res)
            }
            Flop => {
                todo!();
                /*let mut count = 0;
                let mut total = 0.0;
                for next_state in self.next_states.iter_mut() {
                    count += 1;
                    let res = next_state.evaluate_state(
                        range_sb,
                        range_bb,
                        evaluator,
                        iteration_weight,
                        card_order,
                    );
                    total += res;
                }
                if count != 22100 {
                    //dbg!(set, self.next_states.len(), &self);
                }
                assert_eq!(count, 17296); // 17296 = 48 choose 3
                total / count as f32*/
            }
        }
    }
}
