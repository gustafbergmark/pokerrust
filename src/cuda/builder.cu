#include <stdio.h>
#include <bit>
#include <cuda_runtime.h>
#include "structs.h"
#include "evaluator.cuh"


int possible_actions(State *state, short raises, Action *result) {
    switch (state->action) {
        case Fold :
            return 0;
        case Check :
            result[0] = Call;
            result[1] = Raise;
            return 2;
        case Call :
            if (__builtin_popcountll(state->cards) == 4) {
                result[0] = DealRiver;
                return 1;
            } else {
                return 0;
            }
        case Raise :
            result[0] = Fold;
            result[1] = Call;
            if (raises < 4) {
                result[2] = Raise;
                return 3;
            } else {
                return 2;
            }
        case DealRiver :
            result[0] = Check;
            result[1] = Raise;
            return 2;
    }
    return 0;
}


int get_action(State *state, Action action, State *new_states) {
    Player opponent = state->next_to_act == Small ? Big : Small;
    TerminalState fold_winner = state->next_to_act == Small ? BBWins : SBWins;
    DataType other_bet = state->next_to_act == Small ? state->bbbet : state->sbbet;

    // Copy state and reset some values
    *new_states = *state;
    new_states->next_to_act = opponent;
    new_states->action = action;
    new_states->transitions = 0;

    switch (action) {
        case Fold :
            new_states->terminal = fold_winner;
            break;
        case Check:
            new_states->terminal = NonTerminal;
            break;
        case Call:
            new_states->terminal = __builtin_popcountll(state->cards) == 4 ? River : Showdown;
            new_states->sbbet = other_bet;
            new_states->bbbet = other_bet;
            break;
        case Raise :
            new_states->terminal = NonTerminal;
            new_states->sbbet = state->next_to_act == Small ? state->bbbet + 2.0 : state->sbbet;
            new_states->bbbet = state->next_to_act == Big ? state->sbbet + 2.0 : state->bbbet;
            break;
        case DealRiver:
            for (int c = 0; c < 52; c++) {
                long card = 1l << c;
                if (card & state->cards) continue;
                *new_states = *state;
                new_states->next_to_act = Small;
                new_states->action = action;
                new_states->transitions = 0;
                new_states->cards |= card;
                new_states->terminal = NonTerminal;
                break;
            }
    }
    return 1;
}

void
add_transition(State *parent, State *child, State *root, State *device_root,
               AbstractVector *abstract_vectors, int *abstract_vector_index) {
    if (parent->terminal == NonTerminal) {
        parent->card_strategies[parent->transitions] = (abstract_vectors + *abstract_vector_index);
        *abstract_vector_index += 1;
    }
    // Update pointers to work on gpu;
    parent->next_states[parent->transitions] = device_root + (child - root);
    child->parent = device_root + (parent - root);
    parent->transitions += 1;
}


int build(State *state, short raises, State *root, State *device_root, int *state_index,
          AbstractVector *abstract_vectors, int *abstract_vector_index) {
    Action actions[3] = {};
    int count = 1;
    int num_actions = possible_actions(state, raises, actions);
    for (int i = 0; i < num_actions; i++) {
        Action action = actions[i];
        int new_raises = action == Raise ? raises + 1 : 0;

        State *new_state = root + *state_index;
        get_action(state, action, new_state);
        *state_index += 1;
        count += build(new_state, new_raises, root, device_root, state_index,
                       abstract_vectors, abstract_vector_index);
        add_transition(state, new_state, root, device_root, abstract_vectors,
                       abstract_vector_index);


    }
    return count;
}

void
build_river(long cards, DataType bet, State *root, State *device_root, int *state_index,
            AbstractVector *abstract_vectors, int *abstract_vector_index) {
    *root = {.terminal = River,
            .action = Call,
            .cards = cards,
            .sbbet = bet,
            .bbbet = bet,
            .next_to_act = Small,
            .transitions = 0,
            .card_strategies = {},
            .next_states =  {}};
    *state_index += 1;
    int count = build(root, 0, root, device_root, state_index, abstract_vectors,
                      abstract_vector_index);
//    printf("count: %d\n",count); // 28
//    fflush(stdout);
}
