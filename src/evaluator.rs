use crate::combination_map::CombinationMap;
use itertools::Itertools;
use poker::Card;
use serde::{Deserialize, Serialize};
use serde_json_any_key::*;
use std::collections::HashMap;
use std::fs;

#[derive(Serialize, Deserialize, Debug)]
pub struct Evaluator {
    #[serde(with = "any_key_map")]
    card_nums: HashMap<Card, u64>,
    evals: CombinationMap<u16, 52, 7>,
}

impl Evaluator {
    pub fn new() -> Self {
        match fs::read_to_string("./files/evaluator.json") {
            Ok(eval_json) => serde_json::from_str(&eval_json).unwrap(),
            Err(_) => {
                let evaluator = poker::Evaluator::new();
                let deck = Card::generate_deck();
                let mut card_nums = HashMap::new();
                for (i, card) in Card::generate_deck().enumerate() {
                    card_nums.insert(card, 1 << i);
                }
                let mut map: CombinationMap<u16, 52, 7> = CombinationMap::new();
                let mut hands = Vec::new();
                for hand in deck.combinations(7) {
                    let mut num_hand = 0;
                    for &card in &hand {
                        num_hand |= card_nums.get(&card).unwrap();
                    }
                    hands.push((evaluator.evaluate(hand).unwrap(), num_hand));
                }
                hands.sort();

                let mut count: u16 = 0;
                let mut prev_hand = hands[0].0;
                for (eval, num) in hands {
                    if eval > prev_hand {
                        count += 1;
                        prev_hand = eval;
                    }
                    map.insert(num, count);
                }

                let eval = Evaluator {
                    card_nums,
                    evals: map,
                };
                let serialized = serde_json::to_string(&eval).unwrap();
                let _ = fs::write("./files/evaluator.json", serialized);
                eval
            }
        }
    }

    pub fn get(&self, hand: u64) -> Option<&u16> {
        self.evals.get(hand)
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
