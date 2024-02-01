use crate::combination_map::CombinationMap;
use crate::evaluator::Evaluator;
use crate::vector::Vector;
use poker::Suit::*;
use poker::{Card, Suit};

pub fn permute<const M: usize>(
    permutation: [Suit; 4],
    v: Vector,
    evaluator: &Evaluator<M>,
) -> Vector {
    let mut result = Vector::default();
    let card_order: Vec<[Card; 2]> = evaluator
        .card_order()
        .into_iter()
        .map(|&c| evaluator.u64_to_cards(c).try_into().unwrap())
        .collect();
    for (prev_index, mut cards) in card_order.into_iter().enumerate() {
        for card in cards.iter_mut() {
            let new_suit = match card.suit() {
                Clubs => permutation[0],
                Hearts => permutation[1],
                Spades => permutation[2],
                Diamonds => permutation[3],
            };
            *card = Card::new(card.rank(), new_suit);
        }
        let new_num = evaluator.cards_to_u64(&cards);
        let new_index = CombinationMap::<(), 52, 2>::get_ordering_index(new_num);
        result[new_index] = v[prev_index];
    }

    result
}
