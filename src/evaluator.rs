use itertools::Itertools;
use poker::Card;
use std::collections::HashMap;

pub struct Evaluator {
    evals: HashMap<u64, u16>,
    card_nums: HashMap<Card, u64>,
}

impl Evaluator {
    pub fn new() -> Self {
        //let evaluator = poker::Evaluator::new();
        let deck = Card::generate_deck();

        let mut card_nums = HashMap::new();
        for (i, card) in Card::generate_deck().enumerate() {
            card_nums.insert(card, 1 << i);
        }

        let mut hands = Vec::new();
        for hand in deck.combinations(1) {
            let mut num_hand = 0;
            for &card in &hand {
                num_hand |= card_nums.get(&card).unwrap();
            }
            hands.push((num_hand, num_hand));
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

        Evaluator { evals, card_nums }
    }

    pub fn evaluate(&self, cards: u64) -> u16 {
        *self.evals.get(&cards).expect("Non-existent hand")
    }

    pub fn cards_to_u64(&self, cards: &[Card]) -> u64 {
        let mut res = 0;
        for card in cards {
            res |= self.card_nums.get(card).expect("Non-existent card");
        }
        res
    }
}
