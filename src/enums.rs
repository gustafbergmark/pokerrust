#[derive(Clone, Copy, Debug)]
pub(crate) enum Player {
    Small,
    Big,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Action {
    Fold,
    Check,
    Call,
    Raise,
    Deal,
}

#[derive(Copy, Clone, Debug)]
pub(crate) enum TerminalState {
    NonTerminal,
    SBWins,
    BBWins,
    Showdown,
    Flop,
}
