//
// Created by gustaf on 2024-02-09.
//
#include "poker.cu"
#include "structs.h"
#include "evaluator.cuh"
#include "builder.cu"

extern "C" {

void evaluate_cuda(Builder *builder,
                   Evaluator *evaluator,
                   short updating_player,
                   bool calc_exploit) {
    cudaError_t err;
    Vector *device_scratch;
    size_t scratch_size = sizeof(Vector) * 63 * 49 * 9 * 10; // 10 scratch per kernel
    cudaMalloc(&device_scratch, scratch_size);
    cudaMemset(device_scratch, 0, scratch_size);
    cudaMemcpy(builder->opponent_ranges, builder->communication, 63 * 49 * 9 * sizeof(Vector), cudaMemcpyHostToDevice);
    cudaMemset(builder->results, 0, 63 * 49 * 9 * sizeof(Vector));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Setup error: %s\n", cudaGetErrorString(err));
        fflush(stdout);
    }
//    for(int i = 0; i < 1326; i++) {
//        printf("%d %f\n", i, builder->communication[0].values[i]);
//    }
//    fflush(stdout);
    evaluate_all<<< 63 * 49 * 9, TPB>>>(builder->opponent_ranges, builder->results, builder->states, evaluator,
                                        updating_player == 0 ? Small : Big, calc_exploit, device_scratch);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Main execution error: %s\n", cudaGetErrorString(err));
        fflush(stdout);
    }
    cudaMemcpy(builder->communication, builder->results, 63 * 49 * 9 * sizeof(Vector), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(device_scratch);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Aggregation error: %s\n", cudaGetErrorString(err));
        fflush(stdout);
    }
    fflush(stdout);

}
Evaluator *transfer_flop_eval_cuda(long flop, long *card_order, short *card_indexes, short *eval, short *coll_vec,
                                   short *abstractions) {
    cudaError_t err;
    Evaluator *device_eval;
    cudaMalloc(&device_eval, sizeof(Evaluator));
    cudaMemcpy(&device_eval->flop, &flop, sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(&device_eval->card_order, card_order, 1326 * sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(&device_eval->card_indexes, card_indexes, 52 * 51 * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(&device_eval->eval, eval, 1326 * (1326 + 256 * 2) * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(&device_eval->coll_vec, coll_vec, 1326 * 52 * 51 * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(&device_eval->abstractions, abstractions, 1326 * 1326 * sizeof(short), cudaMemcpyHostToDevice);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Evaluator error: %s\n", cudaGetErrorString(err));
        fflush(stdout);
    }
    return device_eval;
}
void free_eval_cuda(Evaluator *device_eval) {
    cudaFree(device_eval);
}
Builder *init_builder() {
    Builder *builder = (Builder *) calloc(1, sizeof(Builder));
    builder->current_index = 0;
    cudaMalloc(&builder->states, 63 * 49 * 9 * 28 * sizeof(State));
    cudaMalloc(&builder->abstract_vectors, 63 * 49 * 9 * 26 * sizeof(AbstractVector));
    cudaMalloc(&builder->updates, 63 * 49 * 9 * 26 * sizeof(AbstractVector));
    cudaMemset(builder->abstract_vectors, 0, 63 * 49 * 26 * 9 * sizeof(AbstractVector));
    cudaMallocHost(&builder->communication, 63 * 49 * 9 * sizeof(Vector));
    cudaMalloc(&builder->opponent_ranges, 63 * 49 * 9 * sizeof(Vector));
    cudaMalloc(&builder->results, 63 * 49 * 9 * sizeof(Vector));
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

int build_river_cuda(long cards, DataType bet, Builder *builder) {
    cudaError_t err;
    int start = builder->current_index;
    int abstract_vector_index = 0;
    int state_index = 0;
    int state_size = sizeof(State) * (28);
    State *root = (State *) malloc(state_size);

    State *device_root = builder->states + builder->current_index * 28;
    AbstractVector *abstract_vectors = builder->abstract_vectors + builder->current_index * 26;
    AbstractVector *updates = builder->updates + builder->current_index * 26;
    builder->current_index += 1;

    build_river(cards, bet, root, device_root, &state_index, abstract_vectors, updates,
                &abstract_vector_index);
    cudaMemcpy(device_root, root, state_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Build error: %s\n", cudaGetErrorString(err));
        fflush(stdout);
    }
//    printf("index: %d\n", start);
//    printf("vector index: %d\n", abstract_vector_index);
//    fflush(stdout);
    return start;
}
}