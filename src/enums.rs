use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Ord, PartialOrd, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) enum Player {
    Small,
    Big,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum Action {
    Fold,
    Check,
    Call,
    Raise,
    DealHole,
    DealFlop,
    DealTurn,
    DealRiver,
}

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) enum TerminalState {
    NonTerminal,
    SBWins,
    BBWins,
    Showdown,
    Flop,
    Turn,
    River,
}
