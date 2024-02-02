//
// Created by gustaf on 2024-01-18.
//

#ifndef POKERRUST_STRUCTS_H
#define POKERRUST_STRUCTS_H

#define DataType float
#define TPB 128
#define ITERS 11
#define ABSTRACTIONS 256

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

struct __align__(512) Vector {
DataType values[1326];
};

struct __align__(512) AbstractVector {
DataType values[ABSTRACTIONS];
};

struct __align__(32) State {
    TerminalState terminal;
    Action action;
    long cards;
    DataType sbbet;
    DataType bbbet;
    Player next_to_act;
    short transitions;
    Vector *card_strategies[48];
    State *next_states[48];
};

struct Context {
    State *state;
    Vector *opponent_range;
    int transition;
};

struct Builder {
    int current_index;
    State *states;
    Vector *vectors;
    AbstractVector *abstract_vectors;
    Vector *communication;
    Vector *opponent_ranges;
    Vector *results;
};


#endif //POKERRUST_STRUCTS_H
