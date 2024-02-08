use crate::combination_map::CombinationMap;
use itertools::Itertools;
use poker::{box_cards, Card};
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Debug)]
pub struct Evaluator<const M: usize> {
    // First 1326 are card orders, following 256 is start group indexes
    // for each gpu thread and last 256 is end next group index for each gpu thread
    card_order: Vec<u64>,
    card_indexes: Vec<u16>,
    vectorized_eval: CombinationMap<Vec<u16>, 52, 5>,
    collisions: CombinationMap<Vec<u16>, 52, 5>,
    abstractions: CombinationMap<Vec<u16>, 52, 5>,
    //gpu_eval: CombinationMap<CombinationMap<(&'a Vec<u16>, &'a Vec<u16>), 52, 2>, 52, 3>,
    card_nums: HashMap<Card, u64>,
    num_cards: HashMap<u64, Card>,
}
#[allow(unused)]
impl<const M: usize> Evaluator<M> {
    pub fn new() -> Self {
        let mut card_nums = HashMap::new();
        let mut num_cards = HashMap::new();
        for (i, card) in Card::generate_deck().enumerate() {
            card_nums.insert(card, 1 << i);
            num_cards.insert(1 << i, card);
        }
        let mut card_order: Vec<[Card; 2]> = Card::generate_deck()
            .combinations(2)
            .map(|e| e.try_into().unwrap())
            .collect();

        // create card_order as u64
        let card_order: Vec<u64> = card_order
            .clone()
            .iter()
            .map(|cards| card_nums.get(&cards[0]).unwrap() | card_nums.get(&cards[1]).unwrap())
            .sorted()
            .collect();

        let (vectorized_eval, collisions, abstractions) =
            /*match std::fs::read("./files/eval_small.bin") {
                Ok(eval) => bincode::deserialize(&eval).expect("Failed to deserialize"),
                Err(_) =>*/ {
                    let evaluator = poker::Evaluator::new();
                    // For full game
                    //let deck = Card::generate_deck();

                    // For fixed flop game
                    let deck = Card::generate_deck().collect::<Vec<_>>();
                    let fixed_flop = &deck[..3];
                    let deck = deck[3..].to_vec().into_iter();

                    let result = deck
                        //.combinations(5) // Full game
                        .combinations(2) // Fixed Flop
                        .par_bridge()
                        .into_par_iter()
                        .map(|hand| {
                            let hand = box_cards!(hand, fixed_flop); // Fixed Flop
                            let num_hand = Self::cards_to_u64_inner(&hand, &card_nums);
                            let mut result: Vec<u16> = vec![0; 1326 + 256 * 2];
                            let mut coll_vec = vec![0; 52 * 51];
                            let mut evals = vec![];
                            for (i, &cards) in card_order.iter().enumerate() {
                                if num_hand & cards > 0 {
                                    evals.push((poker::Eval::WORST, i));
                                } else {
                                    let combined = box_cards!(
                                        Self::u64_to_cards_inner(cards, &num_cards),
                                        hand
                                    );
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
                                if sorted_index % 6 == 0 {
                                    result[1326 + sorted_index / 6] = last_group as u16;
                                }
                                result[sorted_index] = index as u16;
                            }
                            // do next group for last element as well:
                            let mut next_group = 1326;
                            let mut prev_eval = poker::Eval::BEST;
                            for (sorted_index, (eval, index)) in evals.into_iter().enumerate().rev()
                            {
                                if eval < prev_eval {
                                    prev_eval = eval;
                                    next_group = sorted_index + 1;
                                }
                                if (sorted_index % 6 == 10) || (sorted_index == 1325) {
                                    result[1326 + 256 + sorted_index / 6] = next_group as u16;
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
                                let hand = card_order[sorted_index as usize];
                                let cards = separate_cards(hand);
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

                            let abstractions = phs(M, &result, &card_order, num_hand);

                            (num_hand, result, coll_vec, abstractions)
                        })
                        .collect::<Vec<_>>();

                    let mut order_map: CombinationMap<Vec<u16>, 52, 5> = CombinationMap::new();
                    let mut collision_map: CombinationMap<Vec<u16>, 52, 5> = CombinationMap::new();
                    let mut abstraction_map: CombinationMap<Vec<u16>, 52, 5> =
                        CombinationMap::new();
                    for (key, order, collisions, abstraction) in result {
                        order_map.insert(key, order);
                        collision_map.insert(key, collisions);
                        abstraction_map.insert(key, abstraction);
                    }
                    let result = (order_map, collision_map, abstraction_map);
                    // match std::fs::write(
                    //     "./files/eval_small.bin",
                    //     bincode::serialize(&result).expect("Failed to serialize"),
                    // ) {
                    //     Ok(_) => println!("Created vectorized_eval"),
                    //     Err(e) => panic!("{}", e),
                    // }
                    result
                };
        //};

        let mut card_indexes_builder = vec![vec![]; 52];
        for (i, cards) in card_order.iter().enumerate() {
            for card in separate_cards(*cards) {
                card_indexes_builder[card].push(i as u16);
            }
        }
        let mut card_indexes = vec![];
        for i in 0..52 {
            assert_eq!(card_indexes_builder[i].len(), 51);
            for j in 0..51 {
                card_indexes.push(card_indexes_builder[i][j]);
            }
        }
        assert_eq!(card_indexes.len(), 51 * 52);

        Evaluator {
            card_order,
            card_indexes,
            vectorized_eval,
            collisions,
            abstractions,
            //gpu_eval: CombinationMap::new(),
            card_nums,
            num_cards,
        }
    }

    pub fn card_order(&self) -> &Vec<u64> {
        &self.card_order
    }

    pub fn card_indexes(&self) -> &Vec<u16> {
        &self.card_indexes
    }
    pub fn vectorized_eval(&self, communal_cards: u64) -> &Vec<u16> {
        self.vectorized_eval.get(communal_cards).unwrap()
    }

    pub fn abstractions(&self, communal_cards: u64) -> &Vec<u16> {
        self.abstractions.get(communal_cards).unwrap()
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

    fn u64_to_cards_inner(cards: u64, num_cards: &HashMap<u64, Card>) -> Vec<Card> {
        let mut res = Vec::new();
        for i in 0..52 {
            if (1 << i) & cards > 0 {
                res.push(num_cards.get(&(1 << i)).unwrap().clone())
            }
        }
        res
    }
    pub fn u64_to_cards(&self, cards: u64) -> Vec<Card> {
        Self::u64_to_cards_inner(cards, &self.num_cards)
    }

    fn cards_to_u64_inner(cards: &[Card], card_nums: &HashMap<Card, u64>) -> u64 {
        let mut res = 0;
        for card in cards {
            res |= card_nums.get(card).expect("Non-existent card");
        }
        res
    }
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
fn phs(
    abstractions: usize,
    evaluation: &Vec<u16>,
    card_order: &Vec<u64>,
    communal_cards: u64,
) -> Vec<u16> {
    let sorted = evaluation;
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

    let mut collisions = [0; 52];

    let mut cumulative = 0;
    let mut result: Vec<u32> = vec![0; 1326];
    // Doubled integers to not have to divide by 2 in current_collisions
    for group in groups.iter() {
        let mut current_cumulative = 0;
        let mut current_collisions = [0; 52];
        for &index in group {
            let index = index as usize;
            let cards = card_order[index];
            if cards & communal_cards > 0 {
                continue;
            }
            let card = separate_cards(cards);
            result[index] += cumulative;
            current_cumulative += 2;
            for c in card {
                result[index] -= collisions[c];
                current_collisions[c] += 2;
            }
        }
        for &index in group {
            let index = index as usize;
            let cards = card_order[index];
            if cards & communal_cards > 0 {
                continue;
            }
            let card = separate_cards(cards);
            // +1 because inclusion exclusion
            result[index] += current_cumulative + 1;
            for c in card {
                result[index] -= current_collisions[c];
            }
        }
        cumulative += current_cumulative;
        for i in 0..52 {
            collisions[i] += current_collisions[i];
        }
    }
    // 5 communal cards, 2 on own hand leads to 45 choose 2 opponent hands
    let mut res = vec![];
    for elem in result {
        res.push(((elem * abstractions as u32) / (990 * 2)) as u16);
    }
    res
}
