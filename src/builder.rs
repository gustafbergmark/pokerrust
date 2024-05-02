use crate::cuda_interface::initialize_builder;
use crate::enums::Action;
use crate::enums::Action::*;
use crate::enums::Player::*;
use crate::enums::TerminalState::*;
use crate::evaluator::Evaluator;
use crate::game::Game;
use crate::state::{Pointer, State};
use std::time::Instant;

// A variant where the flop is fixed, and no raising the first two rounds of betting.
// Should be about 1 GB in size
pub(crate) fn fixed_flop_poker<const M: usize>() -> Game<M> {
    let mut root = State::new(NonTerminal, DealHole, 0.5, 1.0, Small);
    let start = Instant::now();
    let evaluator = Evaluator::new();
    println!(
        "Evaluator created in {} seconds",
        start.elapsed().as_secs_f32()
    );
    let start = Instant::now();
    // Raises = 1 since original bb counts as raise
    let builder = initialize_builder();

    let _states = build(&mut root, &evaluator, 1, builder, &mut 0);
    println!(
        "Game created in {} seconds with {} states",
        start.elapsed().as_secs_f32(),
        _states
    );
    Game::new(root, evaluator, builder)
}

fn build<const M: usize>(
    state: &mut State<M>,
    evaluator: &Evaluator<M>,
    raises: u8,
    builder: Pointer,
    gpu_index: &mut usize,
) -> usize {
    let mut count = 1;
    for action in possible_actions(state, raises) {
        let new_raises = match action {
            Raise => raises + 1,
            DealHole | DealFlop | DealTurn | DealRiver => 0,
            _ => raises,
        };
        let new_gpu_index = &mut gpu_index.clone();

        let mut end_index = *gpu_index;
        for mut next_state in state.get_action(action, evaluator, builder, new_gpu_index) {
            count += build(
                &mut next_state,
                evaluator,
                new_raises,
                builder,
                new_gpu_index,
            );
            state.add_action(next_state);
            // Save the new gpu pointer and reset pointer for next possible flop
            end_index = *new_gpu_index;
            if action == DealFlop {
                *new_gpu_index = *gpu_index;
            }
        }
        *gpu_index = end_index;
    }
    count
}

fn possible_actions<const M: usize>(state: &State<M>, raises: u8) -> Vec<Action> {
    match state.action {
        Fold => vec![],
        Check => vec![Call, Raise],

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
        DealHole => vec![Fold, Check, Raise],
        DealFlop => vec![Check, Raise],
        DealTurn => vec![Check, Raise],
        DealRiver => vec![Check, Raise],
    }
}
