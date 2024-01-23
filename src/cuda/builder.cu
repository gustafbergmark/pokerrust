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
        case DealTurn :
            result[0] = Check;
            result[1] = Raise;
            return 2;
    }
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
                new_states += 1;
            }
            return 48;
    }
    return 1;
}

void
add_transition(State *parent, State *child, DataType *vectors, int *vector_index, State *root, State *device_root) {
    if (parent->terminal == NonTerminal) {
        parent->card_strategies[parent->transitions] = vectors + (*vector_index * 1326);
        *vector_index += 1;
    }
    // Update pointers to work on gpu;
    parent->next_states[parent->transitions] = device_root + (child - root);
    parent->transitions += 1;
}


int build(State *state, short raises, State *root, State *device_root, DataType *vectors, int *state_index,
          int *vector_index) {
    Action actions[3] = {};
    int count = 1;
    int num_actions = possible_actions(state, raises, actions);
    for (int i = 0; i < num_actions; i++) {
        Action action = actions[i];
        int new_raises = action == Raise ? raises + 1 : 0;
        State *new_states = root + *state_index;
        int num_states = get_action(state, action, new_states);
        *state_index += num_states;
        for (int j = 0; j < num_states; j++) {
            State *new_state = new_states + j;
            count += build(new_state, new_raises, root, device_root, vectors, state_index, vector_index);
            add_transition(state, new_state, vectors, vector_index, root, device_root);
        }
    }
    return count;
}

void
build_post_turn_kernel(long cards, DataType bet, State *root, State *device_root, DataType *vectors, int *state_index,
                       int *vector_index) {
    *root = {.terminal = NonTerminal,
            .action = DealTurn,
            .cards = cards,
            .sbbet = bet,
            .bbbet = bet,
            .next_to_act = Small,
            .transitions = 0,
            .card_strategies = {},
            .next_states =  {}};
    *state_index += 1;
    build(root, 0, root, device_root, vectors, state_index, vector_index);
}


extern "C" {
void init() {
    size_t *size = (size_t *) malloc(sizeof(size_t));
    cudaDeviceGetLimit(size, cudaLimitStackSize);
    printf("old stack size: %zu\n", *size);
    // Allocate 8 GiBi heap
    size_t heap_size = 8l * 1024l * 1024l * 1024l;
    cudaDeviceSetLimit(cudaLimitStackSize, 2 * 4096);
    cudaDeviceGetLimit(size, cudaLimitStackSize);
    printf("new stack size: %zu\n", *size);
    fflush(stdout);
}
State *build_post_turn_cuda(long cards, DataType bet) {
    cudaError_t err;
    int vector_index = 0;
    int state_index = 0;
    int state_size = sizeof(State) * (27 * 48 * 9 + 27);
    State *root = (State *) malloc(state_size);

    State *device_root;
    cudaMalloc(&device_root, state_size);

    DataType *vectors;
    int vectors_size = sizeof(DataType) * 1326 * (26 * 48 * 9 + 26);
    cudaMalloc(&vectors, vectors_size);
    cudaMemset(vectors, 0, vectors_size);

    build_post_turn_kernel(cards, bet, root, device_root, vectors, &state_index, &vector_index);
    cudaMemcpy(device_root, root, state_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Build error: %s\n", cudaGetErrorString(err));
        fflush(stdout);
    }
    return device_root;
}
}
