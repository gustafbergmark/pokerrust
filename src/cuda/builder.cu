#include <stdio.h>
#include <cuda_runtime.h>
#include "structs.h"



__device__ int possible_actions(Action prev, short raises, Action *result) {
    switch (prev) {
        case Fold :
            return 0;
        case Check :
            result[0] = Call;
            result[1] = Raise;
            return 2;
        case Call :
            return 0;
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
}


__device__ void get_action(State *state, Action action, State *new_state) {
    Player opponent = state->next_to_act == Small ? Big : Small;
    TerminalState fold_winner = state->next_to_act == Small ? BBWins : SBWins;
    DataType other_bet = state->next_to_act == Small ? state->bbbet : state->sbbet;

    // Copy state and reset some values
    *new_state = *state;
    new_state->next_to_act = opponent;
    new_state->action = action;
    new_state->transitions = 0;

    switch (action) {
        case Fold :
            new_state->terminal = fold_winner;
            break;
        case Check:
            new_state->terminal = NonTerminal;
            break;
        case Call:
            new_state->terminal = Showdown;
            new_state->sbbet = other_bet;
            new_state->bbbet = other_bet;
            break;
        case Raise :
            new_state->terminal = NonTerminal;
            new_state->sbbet = state->next_to_act == Small ? state->bbbet + 2.0 : state->sbbet;
            new_state->bbbet = state->next_to_act == Big ? state->sbbet + 2.0 : state->bbbet;
            break;
    }
}

__device__ void add_transition(State *parent, State *child, DataType* vectors, int* vector_index) {
    parent->card_strategies[parent->transitions] = vectors + (*vector_index * 1326);
    *vector_index += 1;
    parent->next_states[parent->transitions] = child;
    parent->transitions += 1;
}


__device__ int build(State *state, short raises, State* root, DataType* vectors, int *state_index, int* vector_index) {
    Action actions[3] = {};
    int count = 1;
    int num_actions = possible_actions(state->action, raises, actions);
    for (int i = 0; i < num_actions; i++) {
        Action action = actions[i];
        int new_raises = raises + (action == Raise ? 1 : 0);
        State *new_state = root + *state_index;
        *state_index += 1;
        get_action(state, action, new_state);
        count += build(new_state, new_raises, root, vectors,  state_index, vector_index);
        add_transition(state, new_state, vectors, vector_index);
    }
    return count;
}

__global__ void build_post_river_kernel(long cards, DataType bet, State *root, DataType* vectors, int *state_index, int* vector_index) {
    *root = {.terminal = NonTerminal,
            .action = DealRiver,
            .cards = cards,
            .sbbet = bet,
            .bbbet = bet,
            .next_to_act = Small,
            .transitions = 0,
            .card_strategies = {},
            .next_states =  {}};
    *state_index += 1;
    build(root, 0, root, vectors, state_index, vector_index);
}


extern "C" {
void init() {
    size_t *size = (size_t*) malloc(sizeof(size_t));
    cudaDeviceGetLimit(size, cudaLimitMallocHeapSize);
    printf("old heap size: %zu\n", *size);
    cudaDeviceGetLimit(size, cudaLimitStackSize);
    printf("old stack size: %zu\n", *size);
    // Allocate 6 GiBi heap
    size_t heap_size = 8l * 1024l * 1024l * 1024l;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size);
    // Too small stack will result in: unspecified launch failure
    cudaDeviceSetLimit(cudaLimitStackSize, 2*4096);
    cudaDeviceGetLimit(size, cudaLimitMallocHeapSize);
    printf("new heap size: %zu\n", *size);
    cudaDeviceGetLimit(size, cudaLimitStackSize);
    printf("new stack size: %zu\n", *size);
    fflush(stdout);
}
State *build_post_river_cuda(long cards, DataType bet) {
    cudaError_t err;
    int vector_index = 0;
    int state_index = 0;
    int *device_state_index;
    cudaMalloc(&device_state_index, sizeof(int));
    cudaMemcpy(device_state_index, &state_index, sizeof(int), cudaMemcpyHostToDevice);

    int *device_vector_index;
    cudaMalloc(&device_vector_index, sizeof(int));
    cudaMemcpy(device_vector_index, &vector_index, sizeof(int), cudaMemcpyHostToDevice);

    State *root;
    cudaMalloc(&root, sizeof(State) * 27);

    DataType *vectors;
    int vectors_size = sizeof(DataType) * 26 * 1326;
    cudaMalloc(&vectors, vectors_size);
    cudaMemset(vectors, 0, vectors_size);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("pre Error: %s\n", cudaGetErrorString(err));
        fflush(stdout);
    }

    build_post_river_kernel<<<1, 1>>>(cards, bet, root, vectors, device_state_index, device_vector_index);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        fflush(stdout);
    }

//    cudaMemcpy(&state_index, device_state_index, sizeof(int), cudaMemcpyDeviceToHost);
//    cudaMemcpy(&vector_index, device_vector_index, sizeof(int), cudaMemcpyDeviceToHost);
//    printf("pointer: %p state_index: %d vector_index: %d\n", root, state_index, vector_index);
//    fflush(stdout);

    cudaFree(device_state_index);
    cudaFree(device_vector_index);
    return root;
}
Evaluator * transfer_post_river_eval_cuda(long *card_order, short *card_indexes, short *eval, short* coll_vec) {
    cudaError_t err;
    Evaluator* device_eval;
    cudaMalloc(&device_eval, sizeof (Evaluator));
    cudaMemcpy(&device_eval->card_order, card_order, 1326 * sizeof (long), cudaMemcpyHostToDevice);
    cudaMemcpy(&device_eval->card_indexes, card_indexes, 52 * 51 * sizeof (short), cudaMemcpyHostToDevice);
    cudaMemcpy(&device_eval->eval, eval, (1326 + 128 * 2) * sizeof (short), cudaMemcpyHostToDevice);
    cudaMemcpy(&device_eval->coll_vec, coll_vec, 52 * 51 * sizeof (short), cudaMemcpyHostToDevice);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        fflush(stdout);
    }

    return device_eval;
}
void free_eval_cuda(Evaluator * device_eval) {
    cudaFree(device_eval);
}
}
