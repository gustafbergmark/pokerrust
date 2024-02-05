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
        case DealTurn :
            // will not happen
            return 0;
    }
    return 1;
}

void
add_transition(State *parent, State *child, Vector *vectors, int *vector_index, State *root, State *device_root,
               AbstractVector *abstract_vectors, int *abstract_vector_index) {
    if (parent->terminal == NonTerminal) {
        if( __builtin_popcountll(parent->cards) < 5) {
            parent->card_strategies[parent->transitions] = vectors + *vector_index;
            *vector_index += 1;
        } else {
            // Hideous but should work
            parent->card_strategies[parent->transitions] = (Vector*)(abstract_vectors + *abstract_vector_index);
            *abstract_vector_index += 1;
        }
    }
    // Update pointers to work on gpu;
    parent->next_states[parent->transitions] = device_root + (child - root);
    child->parent = device_root + (parent - root);
    parent->transitions += 1;
}


int build(State *state, short raises, State *root, State *device_root, Vector *vectors, int *state_index,
          int *vector_index, AbstractVector *abstract_vectors, int *abstract_vector_index) {
    Action actions[3] = {};
    int count = 1;
    int num_actions = possible_actions(state, raises, actions);
    for (int i = 0; i < num_actions; i++) {
        Action action = actions[i];
        int new_raises = action == Raise ? raises + 1 : 0;

        State *new_state = root + *state_index;
        get_action(state, action, new_state);
        *state_index += 1;
        count += build(new_state, new_raises, root, device_root, vectors, state_index, vector_index,
                       abstract_vectors, abstract_vector_index);
        add_transition(state, new_state, vectors, vector_index, root, device_root, abstract_vectors,
                       abstract_vector_index);


    }
    return count;
}

void
build_post_turn(long cards, DataType bet, State *root, State *device_root, Vector *vectors, int *state_index,
                int *vector_index, AbstractVector *abstract_vectors, int *abstract_vector_index) {
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
    int count = build(root, 0, root, device_root, vectors, state_index, vector_index, abstract_vectors,
                      abstract_vector_index);
    //printf("count: %d\n",count); // 270
    //fflush(stdout);
}


extern "C" {
Builder *init_builder() {
    Builder *builder = (Builder *) calloc(1, sizeof(Builder));
    builder->current_index = 0;
    cudaMalloc(&builder->states, 63 * 49 * 270 * sizeof(State));
    cudaMalloc(&builder->vectors, 63 * 49 * 26 * sizeof(Vector));
    cudaMemset(builder->vectors, 0, 63 * 49 * 26 * sizeof(Vector));
    cudaMalloc(&builder->abstract_vectors, 63 * 49 * 26 * 9 * sizeof(AbstractVector));
    cudaMemset(builder->abstract_vectors, 0, 63 * 49 * 26 * 9 * sizeof(AbstractVector));
    cudaMallocHost(&builder->communication, 63 * sizeof(Vector));
    cudaMalloc(&builder->opponent_ranges, 63 * sizeof(Vector));
    cudaMalloc(&builder->results, 63 * sizeof(Vector));
    printf("GPU builder created\n");
    fflush(stdout);
    return builder;
}

void upload_c(Builder *builder, int index, DataType *vector) {
    memcpy(builder->communication + index, vector, 1326 * sizeof(DataType));
}

void download_c(Builder *builder, int index, DataType *vector) {
    memcpy(vector, builder->communication + index, 1326 * sizeof(DataType));
}

int build_turn_cuda(long cards, DataType bet, Builder *builder) {
    cudaError_t err;
    int start = builder->current_index;
    for (int c = 0; c < 52; c++) {
        long turn = 1l << c;
        if (cards & turn) continue;
        long new_cards = cards | turn;
        int vector_index = 0;
        int abstract_vector_index = 0;
        int state_index = 0;
        int state_size = sizeof(State) * (27 * 9 + 27);
        State *root = (State *) malloc(state_size);

        State *device_root = builder->states + builder->current_index * 270;
        Vector *vectors = builder->vectors + builder->current_index * 26;
        AbstractVector *abstract_vectors = builder->abstract_vectors + builder->current_index * 26 * 9;
        builder->current_index += 1;

        build_post_turn(new_cards, bet, root, device_root, vectors, &state_index, &vector_index, abstract_vectors,
                        &abstract_vector_index);
        cudaMemcpy(device_root, root, state_size, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Build error: %s\n", cudaGetErrorString(err));
            fflush(stdout);
        }
    }
    return start / 49;
}
}
