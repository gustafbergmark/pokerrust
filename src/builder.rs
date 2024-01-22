use crate::cuda_interface::build_river;
use crate::enums::Action;
use crate::enums::Action::*;
use crate::enums::Player::*;
use crate::enums::TerminalState::*;
use crate::evaluator::Evaluator;
use crate::game::Game;
use crate::state::State;
use itertools::Itertools;
use poker::Card;
use std::thread::sleep;
use std::time::{Duration, Instant};

// A variant where the flop is fixed, and no raising the first two rounds of betting.
// Should be about 1 GB in size
pub(crate) fn fixed_flop_poker() -> Game<'static> {
    let mut card_order: Vec<[Card; 2]> = Card::generate_deck()
        .combinations(2)
        .map(|e| e.try_into().unwrap())
        .collect();
    card_order.sort();
    let mut root = State::new(NonTerminal, DealHole, 0.5, 1.0, Small);
    let start = Instant::now();
    let evaluator = Evaluator::new(card_order);
    println!(
        "Evaluator created in {} seconds",
        start.elapsed().as_secs_f32()
    );
    let start = Instant::now();
    let _states = build(&mut root, &evaluator, 0);
    //dbg!(_states);
    println!("Game created in {} seconds", start.elapsed().as_secs_f32());
    panic!("only build");
    Game::new(root, evaluator)
}

fn build(state: &mut State, evaluator: &Evaluator, raises: u8) -> usize {
    let mut count = 1;
    for action in possible_actions(state, raises) {
        let new_raises = match action {
            Raise => raises + 1,
            _ => 0,
        };
        for mut next_state in state.get_action(action, evaluator) {
            count += build(&mut next_state, evaluator, new_raises);
            state.add_action(next_state);
        }
    }
    if state.terminal == River {
        let start = Instant::now();
        build_river(state.cards, state.sbbet);
        dbg!(start.elapsed().as_micros());
        panic!("Build once");
        //sleep(Duration::from_millis(100));
    }
    count
}

fn possible_actions(state: &State, raises: u8) -> Vec<Action> {
    match state.action {
        Fold => vec![],
        Check => {
            if state.cards.count_ones() <= 3 {
                vec![Call]
            } else {
                vec![Call, Raise]
            }
        }
        Call => match state.cards.count_ones() {
            0 => vec![DealFlop],
            3 => vec![DealTurn],
            4 => vec![DealRiver],
            5 => vec![],
            _ => panic!("Wrong number of communal cards"),
        },
        Raise => {
            if raises < 4 {
                vec![Fold, Call, Raise]
            } else {
                vec![Fold, Call]
            }
        }
        DealHole => vec![Fold, Check],
        DealFlop => vec![Check],
        DealTurn => vec![Check, Raise],
        DealRiver => vec![Check, Raise],
    }
}
