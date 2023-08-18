use crate::combination_map::{choose, CombinationMap};
use crate::evaluator::Evaluator;
use crate::permutation_handler::permute_u64;
use crate::vector::Vector;
use itertools::Itertools;
use poker::Card;
use poker::Suit::*;
use rand::distributions::{Distribution, WeightedIndex};
use rand::prelude::IteratorRandom;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::iter::zip;
use std::time::Instant;

pub fn abstract_flop(
    evaluator: &Evaluator,
    card_order: &[u64; 1326],
) -> Vec<([u64; 2], f32, Vector<50>)> {
    match fs::read_to_string("./files/abstract_flop.json") {
        Ok(eval_json) => serde_json::from_str(&eval_json).unwrap(),
        Err(_) => {
            let mut result = Vec::new();
            let deck = Card::generate_deck();
            let flops = deck.combinations(3);
            let mut set: HashSet<u64> = HashSet::new();
            let mut count = 0;
            for flop in flops {
                let num_flop = evaluator.cards_to_u64(&flop);
                if set.contains(&num_flop) {
                    continue;
                }
                let mut weight = 0.0;
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
                    let hand = evaluator.cards_to_u64(&perm_flop);
                    if set.insert(hand) {
                        weight += 1.0;
                    }
                }
                count += weight as i32;
                if count % 100 == 0 {
                    println!("{} flops done", count);
                }
                let distribution = abstract_flop_subgame(evaluator, card_order, num_flop);
                for i in 0..1326 {
                    if card_order[i] & num_flop > 0 {
                        continue;
                    }
                    assert_eq!(card_order[i].count_ones(), 2);
                    result.push(([num_flop, card_order[i]], weight, distribution[i]));
                }
            }
            dbg!(count);
            let serialized = serde_json::to_string(&result).unwrap();
            let _ = fs::write("./files/abstract_flop.json", serialized);
            result
        }
    }
}

pub fn abstract_flop_subgame(
    evaluator: &Evaluator,
    card_order: &[u64; 1326],
    communal_cards: u64,
) -> [Vector<50>; 1326] {
    let mut result = [Vector::default(); 1326];
    let deck = Card::generate_deck();
    for cards in deck.combinations(2) {
        let num_cards = evaluator.cards_to_u64(&cards);
        if num_cards & communal_cards > 0 {
            continue;
        }
        let ehs = abstract_river(evaluator, card_order, communal_cards | num_cards);
        for i in 0..1326 {
            if ehs[i] < 0.0 {
                assert!(card_order[i] & (communal_cards | num_cards) > 0);
                continue;
            }
            assert!(ehs[i] <= 1.0);
            let bucket = (ehs[i] * 49.0).floor() as usize;
            result[i][bucket] += 1.0;
        }
    }
    result
}

pub fn abstract_turn(
    evaluator: &Evaluator,
    card_order: &[u64; 1326],
) -> Vec<([u64; 2], f32, Vector<50>)> {
    match fs::read_to_string("./files/abstract_turn.json") {
        Ok(eval_json) => serde_json::from_str(&eval_json).unwrap(),
        Err(_) => {
            let mut result = Vec::new();
            let deck = Card::generate_deck();
            let flops = deck.combinations(4);
            let mut set: HashSet<u64> = HashSet::new();
            let mut count = 0;
            for flop in flops {
                let num_flop = evaluator.cards_to_u64(&flop);
                if set.contains(&num_flop) {
                    continue;
                }
                let mut weight = 0.0;
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
                    let hand = evaluator.cards_to_u64(&perm_flop);
                    if set.insert(hand) {
                        weight += 1.0;
                    }
                }
                count += weight as i32;
                if count % 100 == 0 {
                    println!("{} turns done", count);
                }
                let distribution = abstract_turn_subgame(evaluator, card_order, num_flop);
                for i in 0..1326 {
                    if card_order[i] & num_flop > 0 {
                        continue;
                    }
                    assert_eq!(card_order[i].count_ones(), 2);
                    result.push(([num_flop, card_order[i]], weight, distribution[i]));
                }
            }
            dbg!(count);
            let serialized = serde_json::to_string(&result).unwrap();
            let _ = fs::write("./files/abstract_turn.json", serialized);
            result
        }
    }
}

pub fn abstract_turn_subgame(
    evaluator: &Evaluator,
    card_order: &[u64; 1326],
    communal_cards: u64,
) -> [Vector<50>; 1326] {
    let mut result = [Vector::default(); 1326];
    let deck = Card::generate_deck();
    for cards in deck {
        let num_cards = evaluator.cards_to_u64(&[cards]);
        if num_cards & communal_cards > 0 {
            continue;
        }
        let ehs = abstract_river(evaluator, card_order, communal_cards | num_cards);
        for i in 0..1326 {
            if ehs[i] < 0.0 {
                assert!(card_order[i] & (communal_cards | num_cards) > 0);
                continue;
            }
            assert!(ehs[i] <= 1.0);
            let bucket = (ehs[i] * 49.0).floor() as usize;
            result[i][bucket] += 1.0;
        }
    }
    result
}

// returns the Percentile Hand Strength for all hole cards given a set of 5 communal cards.
// Impossible hands get a negative value returned.
pub fn abstract_river(
    evaluator: &Evaluator,
    card_order: &[u64; 1326],
    communal_cards: u64,
) -> Vector<1326> {
    let mut result: Vector<1326> = Vector::from(&[-1.0; 1326]);
    let mut evals = [(0, 0); 1326];
    for i in 0..1326 {
        evals[i] = (
            *evaluator
                .get(card_order[i] | communal_cards)
                .unwrap_or(&u16::MAX),
            i,
        );
    }
    evals.sort();

    let groups = evals.group_by(|&(a, _), &(b, _)| a == b);

    let mut collisions = [0.0; 52];

    let mut cumulative = 0.0;
    for group in groups {
        if group[0].0 == u16::MAX {
            break;
        }

        let mut current_collisions = [0.0; 52];

        for &(_, index) in group {
            let card = Evaluator::separate_cards(card_order[index]);
            for c in card {
                current_collisions[c] += 1.0;
            }
        }

        // forward pass
        for &(_, index) in group {
            result[index] = 0.0;
            let cards = card_order[index];
            let card = Evaluator::separate_cards(cards);
            let mut current_group = group.len() as f32 - 1.0 + 2.0; // -1 to not count the cards themself, +2 since they are removed twice during collision counting
            for c in card {
                result[index] -= collisions[c];
                current_group -= current_collisions[c];
            }
            result[index] += cumulative + current_group / 2.0;
            assert!(result[index] >= 0.0);
        }

        cumulative += group.len() as f32;
        for i in 0..52 {
            collisions[i] += current_collisions[i];
        }
    }
    result * (1.0 / 990.0) // 990 = 45 choose 2
}

pub fn emd<const M: usize>(p: &Vector<M>, q: &Vector<M>) -> f32 {
    let mut emd = [0.0; M];
    for i in 0..M - 1 {
        emd[i + 1] = p[i] + emd[i] - q[i];
    }
    emd.iter().map(|&x| x.abs()).sum()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Abstraction<const M: usize, const C: usize> {
    pub variance: f32,
    map: CombinationMap<CombinationMap<u16, 52, 2>, 52, C>,
}

impl<const M: usize, const C: usize> Abstraction<M, C> {
    pub fn new(observations: &[([u64; 2], f32, Vector<50>)]) -> Self {
        let start = Instant::now();
        let mut rng = rand::thread_rng();
        let &(_, _, distribution) = observations.iter().choose(&mut rng).unwrap();
        let mut centers = vec![distribution; M];
        let mut center_dists = vec![vec![0.0; M]; M];
        let mut u: Vec<f32> = observations
            .iter()
            .map(|elem| emd(&elem.2, &centers[0]))
            .collect();
        let mut c: Vec<usize> = vec![0; observations.len()];

        for i in 1..M {
            if i % 100 == 0 {
                //dbg!(i, start.elapsed().as_secs_f32() / i as f32);
            }
            let (j, center) = (i - 1, &centers[i - 1]);
            for (x, (_, _, distribution)) in observations.iter().enumerate() {
                if center_dists[c[x]][j] < 2.0 * u[x] {
                    let dist = emd(distribution, &center);
                    if dist < u[x] {
                        u[x] = dist;
                        c[x] = j;
                    }
                }
            }

            // square distances
            let square_dist = u.iter().map(|&elem| elem * elem).collect::<Vec<_>>();
            let weighted = WeightedIndex::new(&square_dist).unwrap();
            let choice = weighted.sample(&mut rng);
            centers[i] = observations[choice].2;
            c[choice] = i;
            u[choice] = 0.0;
            for j in 0..i {
                let dist = emd(&centers[i], &centers[j]);
                center_dists[i][j] = dist;
                center_dists[j][i] = dist;
            }
        }
        // update c and u after last center is added
        let (j, center) = (M - 1, &centers[M - 1]);
        for (x, (_, _, distribution)) in observations.iter().enumerate() {
            if center_dists[c[x]][j] < 2.0 * u[x] {
                let dist = emd(distribution, &center);
                if dist < u[x] {
                    u[x] = dist;
                    c[x] = j;
                }
            }
        }
        let (results, centers) = Self::clustering(centers, observations, c, u);
        let mut map = CombinationMap::new();
        let mut set: HashSet<[u64; 2]> = HashSet::new();
        let mut avg_dist = 0.0;
        for (i, ([flop, hole_cards], cluster)) in results.into_iter().enumerate() {
            assert_eq!([flop, hole_cards], observations[i].0);
            let permutations = [Clubs, Hearts, Spades, Diamonds]
                .into_iter()
                .permutations(4);
            for permutation in permutations {
                let flop = permute_u64(permutation.clone().try_into().unwrap(), flop);
                let hole_cards = permute_u64(permutation.try_into().unwrap(), hole_cards);
                assert_eq!(hole_cards.count_ones(), 2);
                if set.contains(&[flop, hole_cards]) {
                    continue;
                }
                avg_dist += emd(&observations[i].2, &centers[cluster]);
                match map.get_mut(flop) {
                    None => {
                        let mut flop_map = CombinationMap::new();
                        let prev_inserted = flop_map.insert(hole_cards, cluster as u16);
                        map.insert(flop, flop_map);
                        assert_eq!(prev_inserted, None);
                    }
                    Some(flop_map) => {
                        let prev_inserted = flop_map.insert(hole_cards, cluster as u16);
                        assert_eq!(prev_inserted, None);
                    }
                }
                set.insert([flop, hole_cards]);
            }
        }
        Self {
            variance: avg_dist / (1326.0 * choose(50, C) as f32),
            /*52 choose 2 * 50 choose C*/
            map,
        }
    }

    pub fn clustering(
        mut centers: Vec<Vector<50>>,
        observations: &[([u64; 2], f32, Vector<50>)],
        mut c: Vec<usize>,
        mut u: Vec<f32>,
    ) -> (Vec<([u64; 2], usize)>, Vec<Vector<50>>) {
        let mut r = vec![true; observations.len()];
        let mut lc = 0;
        loop {
            lc += 1;
            let mut s = vec![f32::INFINITY; M];
            let mut center_dists = vec![vec![0.0; M]; M];
            for i in 0..M {
                for j in i + 1..M {
                    let dist = emd(&centers[i], &centers[j]);
                    center_dists[i][j] = dist;
                    center_dists[j][i] = dist;
                    s[i] = s[i].min(dist / 2.0);
                    s[j] = s[j].min(dist / 2.0);
                }
            }
            for (x, (_, _, distribution)) in observations.iter().enumerate() {
                for (j, center) in centers.iter().enumerate() {
                    if j == c[x] {
                        continue;
                    }
                    if u[x] > s[c[x]] {
                        if center_dists[c[x]][j] < 2.0 * u[x] {
                            if r[x] {
                                r[x] = false;
                                u[x] = emd(distribution, &centers[c[x]]);
                            }
                            if 2.0 * u[x] > center_dists[c[x]][j] {
                                let d = emd(distribution, center);
                                if d < u[x] {
                                    c[x] = j;
                                    u[x] = d;
                                }
                            }
                        }
                    }
                }
            }

            let mut m = vec![Vector::default(); M];
            let mut num_in_cluster: [f32; M] = [0.0; M];
            for (x, &(_, weight, distribution)) in observations.iter().enumerate() {
                m[c[x]] += distribution * weight;
                num_in_cluster[c[x]] += weight;
            }
            for i in 0..M {
                if num_in_cluster[i] > 0.0 {
                    m[i] *= 1.0 / num_in_cluster[i];
                } else {
                    m[i] = centers[i];
                }
            }
            let diffs: Vec<f32> = zip(centers.clone(), m.clone())
                .map(|(a, b)| emd(&a, &b))
                .collect();

            for x in 0..observations.len() {
                u[x] += diffs[c[x]];
                r[x] = true;
            }
            let update = diffs.iter().sum::<f32>();
            if update == 0.0 {
                let mut within_cluster_variance = 0.0;
                for x in 0..observations.len() {
                    within_cluster_variance += emd(&observations[x].2, &centers[c[x]]);
                }
                dbg!(within_cluster_variance / observations.len() as f32, lc,);
                break;
            }
            centers = m;
        }
        (
            zip(observations, c).map(|(&(a, _, _), b)| (a, b)).collect(),
            centers,
        )
    }
    pub fn save(&self) {
        let serialized = serde_json::to_string(self).unwrap();
        let res = fs::write("./files/flop_clusters.json", serialized);
        dbg!(res);
    }

    pub fn load(path: String) -> Self {
        serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
    }
}
