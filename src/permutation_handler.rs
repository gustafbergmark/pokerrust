use crate::vector::Vector;
use poker::Suit;
use poker::Suit::*;
use std::ops::Range;

pub fn permute(permutation: [Suit; 4], v: Vector) -> Vector {
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
        let (r1, _) = get_color_position([s1, s2]);
        let (r2, f2) = get_color_position([p1, p2]);
        result[r2.clone()].copy_from_slice(&values[r1]);
        if f2 {
            transpose(&mut result[r2])
        }
    }
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
        _ => (get_color_position([suits[1], suits[0]]).0, true),
    }
}

fn transpose(v: &mut [f32]) {
    let mut res = [0.0; 169];
    for (i, chunk) in v.chunks_exact(13).enumerate() {
        for (j, val) in chunk.iter().enumerate() {
            res[i + 13 * j] = *val;
        }
    }
    v.copy_from_slice(&res[..])
}
