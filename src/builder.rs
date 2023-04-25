use crate::game::Action::*;
use crate::game::Player::*;
use crate::game::*;

pub(crate) fn kuhn() -> Game {
    let mut root = State::new(TerminalState::NonTerminal, 1.0, 1.0, Small);

    let mut check = root.get_action(Check);
    let mut raise = root.get_action(Raise);

    let checkcall = check.get_action(Call);
    let mut checkraise = check.get_action(Raise);

    let checkraisecall = checkraise.get_action(Call);
    let checkraisefold = checkraise.get_action(Fold);

    let raisefold = raise.get_action(Fold);
    let raisecall = raise.get_action(Call);

    checkraise.add_action(checkraisefold);
    checkraise.add_action(checkraisecall);

    check.add_action(checkcall);
    check.add_action(checkraise);

    raise.add_action(raisefold);
    raise.add_action(raisecall);

    root.add_action(check);
    root.add_action(raise);

    Game::new(root)
}
