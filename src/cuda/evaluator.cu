//
// Created by gustaf on 2024-01-22.
//
#include "evaluator.cuh"
#include <stdio.h>
#include <bit>
#include <cuda_runtime.h>



__device__ int choose(int n, int k) {
    int res = 1;
    for (int i = n - k + 1; i <= n; i++) {
        res *= i;
    }
    int div = 1;
    for (int i = 1; i <= k; i++) {
        div *= i;
    }
    return res / div;
}

__device__ int get_index(long set) {
    int res = 0;
    int i = __ffsll(set)-1;
    res += choose(i, 1);
    set ^= 1 << i;
    i = __ffsll(set)-1;
    res += choose(i, 2);
    return res;
}



extern "C" {
Evaluator *transfer_flop_eval_cuda(long *card_order, short *card_indexes, short *eval, short *coll_vec) {
    cudaError_t err;
    Evaluator *device_eval;
    cudaMalloc(&device_eval, sizeof(Evaluator));
    cudaMemcpy(&device_eval->card_order, card_order, 1326 * sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(&device_eval->card_indexes, card_indexes, 52 * 51 * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(&device_eval->eval, eval, 1326 * (1326 + 128 * 2) * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(&device_eval->coll_vec, coll_vec, 1326 * 52 * 51 * sizeof(short), cudaMemcpyHostToDevice);

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