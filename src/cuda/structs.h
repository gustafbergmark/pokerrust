//
// Created by gustaf on 2024-01-18.
//

#ifndef POKERRUST_STRUCTS_H
#define POKERRUST_STRUCTS_H

#define DataType float

enum TerminalState {
    NonTerminal,
    SBWins,
    BBWins,
    Showdown,
    River,
};

enum Action {
    Fold,
    Check,
    Call,
    Raise,
    DealRiver,
    DealTurn
};

// 0 = Small, 1 = Big
enum Player {
    Small,
    Big,
};

struct State {
    TerminalState terminal;
    Action action;
    long cards;
    DataType sbbet;
    DataType bbbet;
    Player next_to_act;
    short transitions;
    DataType *card_strategies[48];
    State *next_states[48];
};

#endif //POKERRUST_STRUCTS_H
