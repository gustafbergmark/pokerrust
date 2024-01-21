//
// Created by gustaf on 2024-01-18.
//

#ifndef POKERRUST_STRUCTS_H
#define POKERRUST_STRUCTS_H

enum TerminalState {
    NonTerminal,
    SBWins,
    BBWins,
    Showdown,
};

enum Action {
    Fold,
    Check,
    Call,
    Raise,
    DealRiver,
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
    float sbbet;
    float bbbet;
    Player next_to_act;
    short transitions;
    float *card_strategies[3];
    State *next_states[3];
};

struct Evaluator {
    long card_order[1326];
    short card_indexes[52*51];
    short eval[1326 + 128*2];
    short coll_vec[52*51];
};
#endif //POKERRUST_STRUCTS_H
