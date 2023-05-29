use crate::enums::Action;
use crate::enums::Action::*;
use crate::enums::Player::*;
use crate::enums::TerminalState::*;
use crate::evaluator::Evaluator;
use crate::game::Game;
use crate::state::State;

pub(crate) fn flop_poker() -> Game {
    let mut root = State::new(NonTerminal, Deal, 1.0, 1.0, Small);
    let evaluator = Evaluator::new();
    let states = build(&mut root, &evaluator);
    dbg!(states);
    Game::new(root, evaluator)
}

fn build(state: &mut State, evaluator: &Evaluator) -> usize {
    let mut count = 1;
    for action in possible_actions(state) {
        for mut next_state in state.get_action(action, evaluator) {
            count += build(&mut next_state, evaluator);
            state.add_action(next_state);
        }
    }
    count
}

fn possible_actions(state: &State) -> Vec<Action> {
    match state.action {
        Fold => vec![],
        Check => vec![Call, Raise],
        Call => {
            if state.cards == 0 {
                vec![Deal]
            } else {
                vec![]
            }
        }
        Raise => vec![Fold, Call],
        Deal => vec![Check, Raise],
    }
}
