use crate::combination_map::CombinationMap;
use itertools::Itertools;
use poker::{box_cards, Card};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json_any_key::*;
use std::collections::HashMap;
use std::env::current_dir;

#[derive(Debug)]
pub struct Evaluator<'a> {
    // First 1326 are card orders, rest 128 is start group indexes for each gpu thread
    vectorized_eval: CombinationMap<Vec<u16>, 52, 5>,
    collisions: CombinationMap<Vec<u16>, 52, 5>,
    gpu_eval: CombinationMap<CombinationMap<(&'a Vec<u16>, &'a Vec<u16>), 52, 2>, 52, 3>,
    card_nums: HashMap<Card, u64>,
    num_cards: HashMap<u64, Card>,
}
#[allow(unused)]
impl Evaluator<'_> {
    pub fn new(card_order: &Vec<[Card; 2]>) -> Self {
        let mut card_nums = HashMap::new();
        let mut num_cards = HashMap::new();
        for (i, card) in Card::generate_deck().enumerate() {
            card_nums.insert(card, 1 << i);
            num_cards.insert(1 << i, card);
        }
        let (vectorized_eval, collisions) = match std::fs::read("./files/eval_small.bin") {
            Ok(eval) => bincode::deserialize(&eval).expect("Failed to deserialize"),
            Err(_) => {
                let evaluator = poker::Evaluator::new();
                // For full game
                //let deck = Card::generate_deck();

                // For fixed flop game
                let deck = Card::generate_deck().collect::<Vec<_>>();
                let fixed_flop = &deck[..3];
                let deck = deck[3..].to_vec().into_iter();

                // create card_order as u64
                let card_order_num: Vec<u64> = card_order
                    .clone()
                    .iter()
                    .map(|cards| {
                        card_nums.get(&cards[0]).unwrap() | card_nums.get(&cards[1]).unwrap()
                    })
                    .collect();

                let result = deck
                    //.combinations(5) // Full game
                    .combinations(2) // Fixed Flop
                    .par_bridge()
                    .into_par_iter()
                    .map(|hand| {
                        let hand = box_cards!(hand, fixed_flop); // Fixed Flop
                        let num_hand = Self::cards_to_u64_inner(&hand, &card_nums);
                        let mut result: Vec<u16> = vec![0; 1326 + 128 * 2];
                        let mut coll_vec = vec![0; 52 * 51];
                        let mut evals = vec![];
                        for (i, cards) in card_order.iter().enumerate() {
                            let num_cards = Self::cards_to_u64_inner(cards, &card_nums);
                            if num_hand & num_cards > 0 {
                                evals.push((poker::Eval::WORST, i));
                            } else {
                                let combined = box_cards!(cards, hand);
                                evals.push((
                                    evaluator.evaluate(combined).expect("Failed to evaluate"),
                                    i,
                                ));
                            }
                        }
                        evals.sort();
                        let mut prev_eval = poker::Eval::WORST;
                        let mut last_group = 0;
                        for (sorted_index, &(eval, mut index)) in evals.iter().enumerate() {
                            if eval > prev_eval || sorted_index == 0 {
                                prev_eval = eval;
                                last_group = sorted_index;
                                // 2048 bit set indicates start of new group
                                index |= 2048;
                            }
                            if sorted_index % 11 == 0 {
                                result[1326 + sorted_index / 11] = last_group as u16;
                            }
                            result[sorted_index] = index as u16;
                        }
                        // do next group for last element as well:
                        let mut next_group = 1326;
                        let mut prev_eval = poker::Eval::BEST;
                        for (sorted_index, (eval, index)) in evals.into_iter().enumerate().rev() {
                            if eval < prev_eval {
                                prev_eval = eval;
                                next_group = sorted_index + 1;
                            }
                            if (sorted_index % 11 == 10) || (sorted_index == 1325) {
                                result[1326 + 128 + sorted_index / 11] = next_group as u16;
                            }
                        }
                        // Calculate collisions for GPU
                        let mut indexes = [0; 52];
                        let mut card_group = [0; 52];
                        let mut current_group = 0;
                        for (i, &sorted_index) in result[..1326].iter().enumerate() {
                            if sorted_index & 2048 > 0 {
                                current_group += 1;
                            }
                            let sorted_index = sorted_index & 2047;
                            let hand = card_order_num[sorted_index as usize];
                            let cards = Self::separate_cards(hand);
                            for c in cards {
                                let mut val = i as u16;
                                // Include new group information in 2048 bit
                                if current_group > card_group[c] {
                                    card_group[c] = current_group;
                                    val |= 2048;
                                }
                                let index = indexes[c];
                                indexes[c] += 1;
                                coll_vec[c * 51 + index] = val;
                            }
                        }

                        (num_hand, result, coll_vec)
                    })
                    .collect::<Vec<_>>();

                let mut order_map: CombinationMap<Vec<u16>, 52, 5> = CombinationMap::new();
                let mut collision_map: CombinationMap<Vec<u16>, 52, 5> = CombinationMap::new();
                for (key, order, collisions) in result {
                    order_map.insert(key, order);
                    collision_map.insert(key, collisions);
                }
                let result = (order_map, collision_map);
                match std::fs::write(
                    "./files/eval_small.bin",
                    bincode::serialize(&result).expect("Failed to serialize"),
                ) {
                    Ok(_) => println!("Created vectorized_eval"),
                    Err(e) => panic!("{}", e),
                }
                result
            }
        };

        Evaluator {
            vectorized_eval,
            collisions,
            gpu_eval: CombinationMap::new(),
            card_nums,
            num_cards,
        }
    }

    pub fn vectorized_eval(&self, communal_cards: u64) -> &Vec<u16> {
        self.vectorized_eval.get(communal_cards).unwrap()
    }

    pub fn collisions(&self, communal_cards: u64) -> &Vec<u16> {
        self.collisions.get(communal_cards).unwrap()
    }

    pub fn gpu_eval(&self, communal_cards: u64) -> &Vec<u16> {
        self.vectorized_eval.get(communal_cards).unwrap()
    }

    pub fn cards_to_u64(&self, cards: &[Card]) -> u64 {
        Self::cards_to_u64_inner(cards, &self.card_nums)
    }

    pub fn u64_to_cards(&self, cards: u64) -> Vec<Card> {
        let mut res = Vec::new();
        for i in 0..52 {
            if (1 << i) & cards > 0 {
                res.push(self.num_cards.get(&(1 << i)).unwrap().clone())
            }
        }
        res
    }

    fn cards_to_u64_inner(cards: &[Card], card_nums: &HashMap<Card, u64>) -> u64 {
        let mut res = 0;
        for card in cards {
            res |= card_nums.get(card).expect("Non-existent card");
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
