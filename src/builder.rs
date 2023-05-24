use crate::enums::Action;
use crate::enums::Action::*;
use crate::enums::Player::*;
use crate::enums::TerminalState::*;
use crate::game::Game;
use crate::state::State;

pub(crate) fn flop_poker() -> Game<2> {
    let mut root = State::new(NonTerminal, Deal, 1.0, 1.0, Small);
    let states = build(&mut root);
    dbg!(states);
    Game::new(root)
}

fn build<const M: usize>(state: &mut State<M>) -> usize {
    let mut count = 1;
    for action in possible_actions(state) {
        for mut next_state in state.get_action(action) {
            count += build(&mut next_state);
            state.add_action(next_state);
        }
    }
    count
}

fn possible_actions<const M: usize>(state: &State<M>) -> Vec<Action> {
    match state.action {
        Fold => vec![],
        Check => vec![Call, Raise],
        Call => {
            if state.cards.is_empty() {
                vec![Deal]
            } else {
                vec![]
            }
        }
        Raise => vec![Fold, Call],
        Deal => vec![Check, Raise],
    }
}
