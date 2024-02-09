//
// Created by gustaf on 2024-01-18.
//

#ifndef POKERRUST_STRUCTS_H
#define POKERRUST_STRUCTS_H

#define DataType float
#define TPB 256
#define ITERS 6
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
};

// 0 = Small, 1 = Big
enum Player {
    Small,
    Big,
};

// 1327 values because prefix sum needs 1 extra and it doesn't matter
struct __align__(512) Vector {
DataType values[1327];
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
    AbstractVector *card_strategies[3];
    State *next_states[3];
    State *parent;
};

struct Builder {
    int current_index;
    State *states;
    AbstractVector *abstract_vectors;
    Vector *communication;
    Vector *opponent_ranges;
    Vector *results;
};


#endif //POKERRUST_STRUCTS_H
