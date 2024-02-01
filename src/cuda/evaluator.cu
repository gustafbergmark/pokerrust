//
// Created by gustaf on 2024-01-22.
//
#include "evaluator.cuh"
#include <stdio.h>
#include <bit>
#include <cuda_runtime.h>



__device__ int choose2(int n) {
    if (n>1) {
        return (n*(n-1)) / 2;
    } else {
        return 0;
    }
}

__device__ int get_index(long set) {
    int res = 0;
    int i = __ffsll(set)-1;
    res += i;
    set ^= 1l << i;
    i = __ffsll(set)-1;
    res += choose2(i);
    return res;
}

__device__ long from_index(int index) {
    int limit = (int)(sqrtf((float)(2*index))) + 1;
    //if (limit*(limit-1)/2 > index) limit--;
    limit -= (limit*(limit-1)) > (2*index);
    index -= limit * (limit-1) / 2;
    return (1l<<(limit)) | ( 1l << index);
}



extern "C" {
Evaluator *transfer_flop_eval_cuda(long flop, long *card_order, short *card_indexes, short *eval, short *coll_vec, short *abstractions) {
    cudaError_t err;
    Evaluator *device_eval;
    cudaMalloc(&device_eval, sizeof(Evaluator));
    cudaMemcpy(&device_eval->flop, &flop, sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(&device_eval->card_order, card_order, 1326 * sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(&device_eval->card_indexes, card_indexes, 52 * 51 * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(&device_eval->eval, eval, 1326 * (1326 + 128 * 2) * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(&device_eval->coll_vec, coll_vec, 1326 * 52 * 51 * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(&device_eval->abstractions, abstractions, 1326 * 1326 * sizeof(short), cudaMemcpyHostToDevice);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        fflush(stdout);
    }
    return device_eval;
}
void free_eval_cuda(Evaluator *device_eval) {
    cudaFree(device_eval);
}
}