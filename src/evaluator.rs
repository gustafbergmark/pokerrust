use itertools::Itertools;
use poker::Card;
use serde::{Deserialize, Serialize};
use serde_json_any_key::*;
use std::collections::HashMap;
use std::fs;

#[derive(Serialize, Deserialize, Debug)]
pub struct Evaluator {
    #[serde(with = "any_key_map")]
    vectorized_eval: HashMap<u64, Vec<(u16, u16)>>,
    #[serde(with = "any_key_map")]
    card_nums: HashMap<Card, u64>,
}
#[allow(unused)]
impl Evaluator {
    pub fn new(card_order: &Vec<[Card; 2]>) -> Self {
        match fs::read_to_string("./files/evaluator.json") {
            Ok(eval_json) => serde_json::from_str(&eval_json).unwrap(),
            Err(_) => {
                let evaluator = poker::Evaluator::new();
                let deck = Card::generate_deck();
                let mut card_nums = HashMap::new();
                for (i, card) in Card::generate_deck().enumerate() {
                    card_nums.insert(card, 1 << i);
                }

                let mut hands = Vec::new();
                for hand in deck.combinations(5) {
                    let mut num_hand = 0;
                    for &card in &hand {
                        num_hand |= card_nums.get(&card).unwrap();
                    }
                    hands.push((evaluator.evaluate(hand).unwrap(), num_hand));
                }
                hands.sort();

                let mut count: u16 = 0;
                let mut evals = HashMap::new();
                let mut prev_hand = hands[0].0;
                for (eval, num) in hands {
                    if eval > prev_hand {
                        count += 1;
                        prev_hand = eval;
                    }
                    evals.insert(num, count);
                }

                let card_order: Vec<u64> = card_order
                    .iter()
                    .map(|cards| {
                        card_nums.get(&cards[0]).unwrap() | card_nums.get(&cards[1]).unwrap()
                    })
                    .collect();

                let deck = Card::generate_deck();
                let mut vectorized_flop = Vec::new();
                for flop in deck.combinations(3) {
                    let mut num_hand = 0;
                    for &card in &flop {
                        num_hand |= card_nums.get(&card).unwrap();
                    }

                    let sorted: Vec<(u16, u16)> = card_order
                        .clone()
                        .into_iter()
                        .enumerate()
                        .map(|(i, elem)| (*evals.get(&(elem | num_hand)).unwrap_or(&0), i as u16))
                        .sorted() // could be done quicker, saves max 1 sec
                        .collect();
                    vectorized_flop.push((num_hand, sorted));
                }

                let eval = Evaluator {
                    card_nums,
                    vectorized_eval: vectorized_flop.into_iter().collect(),
                };
                let serialized = serde_json::to_string(&eval).unwrap();
                fs::write("./files/evaluator.json", serialized);
                eval
            }
        }
    }

    pub fn vectorized_eval(&self, cards: u64) -> &Vec<(u16, u16)> {
        self.vectorized_eval.get(&cards).unwrap()
    }

    pub fn cards_to_u64(&self, cards: &[Card]) -> u64 {
        let mut res = 0;
        for card in cards {
            res |= self.card_nums.get(card).expect("Non-existent card");
        }
        res
    }

    pub fn separate_cards(mut cards: u64) -> [usize; 2] {
        let mut res = [0, 0];
        let i = cards.trailing_zeros();
        let card = 1 << i;
        res[0] = i as usize;
        cards -= card;
        res[1] = cards.trailing_zeros() as usize;
        res
    }
}
