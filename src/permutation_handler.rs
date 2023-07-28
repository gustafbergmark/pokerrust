use crate::vector::Vector;
use itertools::Itertools;
use poker::Suit::*;
use poker::{Card, Suit};
use std::collections::HashMap;
//use std::iter::zip;
use std::ops::Range;

pub struct PermutationHandler {
    map: HashMap<[Suit; 4], [usize; 1326]>,
}

impl PermutationHandler {
    pub fn new(card_order: &Vec<[Card; 2]>) -> Self {
        //dbg!(&card_order.iter().enumerate().collect::<Vec<_>>());
        let index_map: HashMap<[Card; 2], usize> = card_order
            .iter()
            .to_owned()
            .enumerate()
            .map(|(i, &cards)| (cards, i))
            .collect();
        let mut permutation_map = HashMap::new();
        let permutations = [Clubs, Hearts, Spades, Diamonds]
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
                    Clubs => permutation[0],
                    Hearts => permutation[1],
                    Spades => permutation[2],
                    Diamonds => permutation[3],
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

    pub fn permute(&self, permutation: [Suit; 4], v: Vector) -> Vector {
        /*let p = *self.map.get(&permutation).unwrap();
        let mut result = [0.0; 1326];
        for (i, value) in zip(p, v.values) {
            result[i] = value;
        }*/

        let mut result = [0.0; 1326];
        let values = v.values;
        let possible_suit_combinations = [
            [Clubs, Clubs],
            [Clubs, Hearts],
            [Clubs, Spades],
            [Clubs, Diamonds],
            [Hearts, Hearts],
            [Hearts, Spades],
            [Hearts, Diamonds],
            [Spades, Spades],
            [Spades, Diamonds],
            [Diamonds, Diamonds],
        ];
        for [s1, s2] in possible_suit_combinations {
            let p1 = match s1 {
                Clubs => permutation[0],
                Hearts => permutation[1],
                Spades => permutation[2],
                Diamonds => permutation[3],
            };
            let p2 = match s2 {
                Clubs => permutation[0],
                Hearts => permutation[1],
                Spades => permutation[2],
                Diamonds => permutation[3],
            };
            let (r1, _) = Self::get_color_position([s1, s2]);
            let (r2, f2) = Self::get_color_position([p1, p2]);
            result[r2.clone()].copy_from_slice(&values[r1]);
            if f2 {
                Self::transpose(&mut result[r2])
            }
        }
        /*assert_eq!(
            &result2.iter().enumerate().collect::<Vec<_>>(),
            &result.iter().enumerate().collect::<Vec<_>>()
        );*/
        Vector { values: result }
    }

    fn get_color_position(suits: [Suit; 2]) -> (Range<usize>, bool) {
        match suits {
            [Clubs, Clubs] => (0..78, false),
            [Clubs, Hearts] => (78..247, false),
            [Clubs, Spades] => (247..416, false),
            [Clubs, Diamonds] => (416..585, false),
            [Hearts, Hearts] => (585..663, false),
            [Hearts, Spades] => (663..832, false),
            [Hearts, Diamonds] => (832..1001, false),
            [Spades, Spades] => (1001..1079, false),
            [Spades, Diamonds] => (1079..1248, false),
            [Diamonds, Diamonds] => (1248..1326, false),
            _ => (Self::get_color_position([suits[1], suits[0]]).0, true),
        }
    }

    fn transpose(v: &mut [f64]) {
        let mut res = [0.0; 169];
        for (i, chunk) in v.chunks_exact(13).enumerate() {
            for (j, val) in chunk.iter().enumerate() {
                res[i + 13 * j] = *val;
            }
        }
        v.copy_from_slice(&res[..])
    }
}
