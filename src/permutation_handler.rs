use itertools::Itertools;
use poker::Suit::*;
use poker::{Card, Suit};
use std::collections::HashMap;
use std::iter::zip;

pub struct PermutationHandler {
    map: HashMap<[Suit; 4], [usize; 1326]>,
}

impl PermutationHandler {
    pub fn new() -> Self {
        let card_order: Vec<[Card; 2]> = Card::generate_deck()
            .combinations(2)
            .map(|e| e.try_into().unwrap())
            .collect();

        let index_map: HashMap<[Card; 2], usize> = card_order
            .iter()
            .to_owned()
            .enumerate()
            .map(|(i, &cards)| (cards, i))
            .collect();
        let mut permutation_map = HashMap::new();
        let permutations = [Spades, Hearts, Diamonds, Clubs]
            .into_iter()
            .permutations(4);
        for permutation in permutations {
            let permutation: [Suit; 4] = permutation.try_into().unwrap();
            let p = Self::get_permutation(&card_order, permutation, &index_map);
            permutation_map.insert(permutation, p);
        }
        assert_eq!(permutation_map.len(), 24);
        PermutationHandler {
            map: permutation_map,
        }
    }

    fn get_permutation(
        card_order: &Vec<[Card; 2]>,
        permutation: [Suit; 4],
        index_map: &HashMap<[Card; 2], usize>,
    ) -> [usize; 1326] {
        let mut card_order = card_order.clone();
        let mut result = [0; 1326];
        for (index, cards) in card_order.iter_mut().enumerate() {
            for card in cards.iter_mut() {
                let new_suit = match card.suit() {
                    Spades => permutation[0],
                    Hearts => permutation[1],
                    Diamonds => permutation[2],
                    Clubs => permutation[3],
                };
                *card = Card::new(card.rank(), new_suit)
            }
            let mut rev_cards = *cards;
            rev_cards.reverse();
            let new_index = if let Some(i) = index_map.get(cards) {
                *i
            } else {
                *index_map.get(&rev_cards).unwrap()
            };
            result[index] = new_index
        }
        //dbg!(&result);
        result
    }

    pub fn permute(&self, permutation: [Suit; 4], v: [f32; 1326]) -> [f32; 1326] {
        let p = *self.map.get(&permutation).unwrap();
        let mut result = [0.0; 1326];
        for (i, value) in zip(p, v) {
            result[i] = value;
        }
        result
    }
}
