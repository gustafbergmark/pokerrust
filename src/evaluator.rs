use crate::combination_map::CombinationMap;
use crate::game::{RIVERS, TURNS};
use itertools::Itertools;
use poker::Suit::{Clubs, Diamonds, Hearts, Spades};
use poker::{box_cards, Card, Suit};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
pub struct Evaluator<const M: usize> {
    // First 1326 are card orders, following 256 is start group indexes
    // for each gpu thread and last 256 is end next group index for each gpu thread
    card_order: Vec<u64>,
    card_indexes: Vec<u16>,
    pub flops: Vec<u64>,
    vectorized_eval: CombinationMap<Vec<u16>, 52, 5>,
    collisions: CombinationMap<Vec<u16>, 52, 5>,
    river_abstractions: CombinationMap<Vec<u16>, 52, 5>,
    turn_abstractions: CombinationMap<Vec<u16>, 52, 4>,
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

        let mut flops = vec![];
        let mut set: HashSet<u64> = HashSet::new();
        for flop in Card::generate_deck().combinations(3) {
            let num_flop = Self::cards_to_u64_inner(&flop, &card_nums);
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
                let hand = Self::cards_to_u64_inner(&perm_flop, &card_nums);
                if set.insert(hand) {
                    possible_permutations.push(permutation.try_into().unwrap())
                }
            }
            flops.push(Self::cards_to_u64_inner(&flop, &card_nums));
        }
        let turn_abstractions = match std::fs::read("./files/turn_abstractions.bin") {
            Ok(eval) => bincode::deserialize(&eval).expect("Failed to deserialize"),
            Err(_) => {
                let mut turn_abstractions = CombinationMap::new();
                for (i, &flop) in flops.iter().enumerate() {
                    println!("Finished abstraction for flop number {i}");
                    for turn in Card::generate_deck() {
                        let turn = Evaluator::<M>::cards_to_u64_inner(&[turn], &card_nums);
                        if flop & turn > 0 {
                            continue;
                        }
                        if let Some(_) = turn_abstractions.get(flop | turn) {
                            continue;
                        }
                        let mut ehs = vec![0.0; 1326];
                        let mut count = 0;
                        for river in Card::generate_deck() {
                            let river = Evaluator::<M>::cards_to_u64_inner(&[river], &card_nums);
                            if river & (flop | turn) > 0 {
                                continue;
                            }
                            count += 1;
                            let cards = flop | turn | river;
                            assert_eq!(cards.count_ones(), 5);
                            let eval = Self::calculate_eval(cards, &num_cards, &card_order);
                            let phs = phs(&eval, &card_order, cards);
                            for i in 0..1326 {
                                ehs[i] += phs[i] / 48.0;
                            }
                        }
                        assert_eq!(count, 48);
                        let ehs = ehs
                            .into_iter()
                            .map(|elem| (elem * M as f32) as u16)
                            .collect();
                        turn_abstractions.insert(flop | turn, ehs);
                    }
                }
                match std::fs::write(
                    "./files/turn_abstractions.bin",
                    bincode::serialize(&turn_abstractions).expect("Failed to serialize"),
                ) {
                    Ok(_) => println!("Created turn abstractions"),
                    Err(e) => panic!("{}", e),
                }
                turn_abstractions
            }
        };

        Evaluator {
            card_order,
            card_indexes,
            flops,
            vectorized_eval: CombinationMap::new(),
            collisions: CombinationMap::new(),
            river_abstractions: CombinationMap::new(),
            turn_abstractions,
            card_nums,
            num_cards,
        }
    }

    fn calculate_eval(
        communal_cards: u64,
        num_cards: &HashMap<u64, Card>,
        card_order: &Vec<u64>,
    ) -> Vec<u16> {
        assert_eq!(communal_cards.count_ones(), 5);
        let evaluator = poker::Evaluator::new();
        let hand = Evaluator::<M>::u64_to_cards_inner(communal_cards, num_cards);

        let mut result: Vec<u16> = vec![0; 1326];
        let mut evals = vec![];
        for (i, &cards) in card_order.iter().enumerate() {
            if communal_cards & cards > 0 {
                evals.push((poker::Eval::WORST, i));
            } else {
                let combined = box_cards!(Self::u64_to_cards_inner(cards, &num_cards), hand);
                evals.push((evaluator.evaluate(combined).expect("Failed to evaluate"), i));
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
            result[sorted_index] = index as u16;
        }
        result
    }

    pub fn get_flop_eval(
        &mut self,
        flop: u64,
        turns: &Vec<u64>,
        rivers: &Vec<u64>,
        calc_exploit: bool,
    ) {
        let evaluator = poker::Evaluator::new();

        let runouts = if calc_exploit {
            Card::generate_deck()
                .filter(|&card| self.cards_to_u64(&[card]) & flop == 0)
                .combinations(2)
                .collect::<Vec<_>>()
        } else {
            let mut runouts = vec![];
            for i in 0..TURNS {
                for j in 0..RIVERS {
                    runouts.push(self.u64_to_cards(turns[i] | rivers[j + i * RIVERS]))
                }
            }
            runouts
        };

        let result = runouts
            .into_par_iter()
            .map(|hand| {
                let hand = box_cards!(hand, self.u64_to_cards(flop));

                let num_hand = self.cards_to_u64(&hand);
                let mut result: Vec<u16> = vec![0; 1326 + 256 * 2];
                let mut coll_vec = vec![0; 52 * 51];
                let mut evals = vec![];
                for (i, &cards) in self.card_order.iter().enumerate() {
                    if num_hand & cards > 0 {
                        evals.push((poker::Eval::WORST, i));
                    } else {
                        let combined =
                            box_cards!(Self::u64_to_cards_inner(cards, &self.num_cards), hand);
                        evals.push((evaluator.evaluate(combined).expect("Failed to evaluate"), i));
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
                for (sorted_index, (eval, index)) in evals.into_iter().enumerate().rev() {
                    if eval < prev_eval {
                        prev_eval = eval;
                        next_group = sorted_index + 1;
                    }
                    if (sorted_index % 6 == 5) || (sorted_index == 1325) {
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
                    let hand = self.card_order[sorted_index as usize];
                    let cards = separate_cards(hand);
                    for c in cards {
                        let mut val = sorted_index;
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

                let phs = phs(&result, &self.card_order, num_hand);
                let abstractions: Vec<u16> = phs
                    .into_iter()
                    .map(|elem| (elem * (M - 1) as f32) as u16)
                    .collect();
                assert!(*abstractions.iter().max().unwrap() < M as u16);
                (num_hand, result, coll_vec, abstractions)
            })
            .collect::<Vec<_>>();

        for (key, order, collisions, abstraction) in result {
            self.vectorized_eval.insert(key, order);
            self.collisions.insert(key, collisions);
            self.river_abstractions.insert(key, abstraction);
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
        match communal_cards.count_ones() {
            4 => self.turn_abstractions.get(communal_cards).unwrap(),
            5 => self.river_abstractions.get(communal_cards).unwrap(),
            _ => panic!("Wrong number of communal cards"),
        }
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
fn phs(evaluation: &Vec<u16>, card_order: &Vec<u64>, communal_cards: u64) -> Vec<f32> {
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
    let mut result: Vec<i32> = vec![0; 1326];
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
            result[index] += current_cumulative / 2 + 1;
            for c in card {
                result[index] -= current_collisions[c] / 2;
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
        assert!(elem >= 0 && elem <= 990 * 2);
        res.push((elem as f32) / (990.0 * 2.0));
    }
    res
}
