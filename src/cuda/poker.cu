#include <stdio.h>
#include <cuda_runtime.h>
#include "math.h"
#include "structs.h"
#include "evaluator.cuh"
// Everything expect a  dimension of 1x128, and vectors of size 1326 (most of the time)

__device__ void multiply(DataType *v1, DataType *v2, DataType *res) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int b = 0; b < 11; b++) {
        int index = i + 128 * b;
        if (index < 1326) {
            res[index] = v1[index] * v2[index];
        }
    }
}

__device__ void fma(DataType *v1, DataType *v2, DataType *res) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int b = 0; b < 11; b++) {
        int index = i + 128 * b;
        if (index < 1326) {
            res[index] += v1[index] * v2[index];
        }
    }
}

__device__ void add_assign(DataType *v1, DataType *v2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int b = 0; b < 11; b++) {
        int index = i + 128 * b;
        if (index < 1326) {
            v1[index] += v2[index];
        }
    }
}

__device__ void sub_assign(DataType *v1, DataType *v2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int b = 0; b < 11; b++) {
        int index = i + 128 * b;
        if (index < 1326) {
            v1[index] -= v2[index];
        }
    }
}


__device__ void zero(DataType *v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int b = 0; b < 11; b++) {
        int index = i + 128 * b;
        if (index < 1326) {
            v[index] = 0;
        }
    }
}

__device__ void p_sum(DataType *input, int i) {
    int offset = 1;
    for (int d = 64; d > 0; d >>= 1) {
        __syncthreads();
        if (i < d) {
            int ai = offset * (2 * i + 1) - 1;
            int bi = offset * (2 * i + 2) - 1;
            input[bi] += input[ai];
        }
        offset *= 2;
    }
    if (i == 0) {
        input[127] = 0;
    }
    for (int d = 1; d < 128; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (i < d) {
            int ai = offset * (2 * i + 1) - 1;
            int bi = offset * (2 * i + 2) - 1;
            DataType t = input[ai];
            input[ai] = input[bi];
            input[bi] += t;
        }
    }
    __syncthreads();
}

__device__ void cuda_prefix_sum(DataType *input, DataType *temp) {
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    temp[i] = 0;
    for (int b = 0; b < 11; b++) {
        int index = i * 11 + b;
        if (index < 1326 && i < 127) {
            temp[i] += input[index];
        }
    }
    p_sum(temp, i);

    DataType prefix = temp[i];
    for (int b = 0; b < 11; b++) {
        int index = i * 11+b;
        if (index < 1326) {
            DataType t = input[index];
            input[index] = prefix;
            prefix += t;
        }
    }
    __syncthreads();
}

__device__ void get_strategy(State *state, DataType *scratch, DataType *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    DataType *sum = scratch;
    zero(sum);
    for (int i = 0; i < state->transitions; i++) {
        add_assign(sum, state->card_strategies[i]);
    }
    for (int i = 0; i < state->transitions; i++) {
        for (int b = 0; b < 11; b++) {
            int index = tid + 128* b;
            if (index < 1326) {
                if (sum[index] <= 1e-4) {
                    result[index + i * 1326] = 1.0 / ((DataType) state->transitions);
                } else {
                    result[index + i * 1326] = state->card_strategies[i][index] / sum[index];
                }
            }
        }
    }
}

__device__ void update_strategy(State *state, DataType *update) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < state->transitions; i++) {
        add_assign(state->card_strategies[i], update + 1326 * i);
        for (int b = 0; b < 11; b++) {
            int index = tid + 128 * b;
            if (index < 1326) {
                state->card_strategies[i][index] = max(state->card_strategies[i][index], 0.0);
            }
        }
    }
}

__device__ void
handle_collisions(int i, long communal_cards, long *card_order, short *eval, short *coll_vec,
                  DataType *sorted_range, DataType *sorted_eval) {
    __syncthreads();
    // Handle collisions before prefix sum consumes sorted_range
    // First two warps handles forward direction
    if (i < 52) {
        int offset = i * 51;
        DataType sum = 0.0;
        DataType group_sum = 0.0;
        for (int c = 0; c < 51; c++) {
            int index = coll_vec[offset + c];
            // Skip impossible hands, unnecessary here, but consistent
            if ((communal_cards & card_order[eval[index & 2047] & 2047]) > 0) continue;
            // 2048 bit set => new group
            if (index & 2048) {
                sum += group_sum;
                group_sum = 0.0;
            }
            atomicAdd(&sorted_eval[index & 2047], -sum);
            group_sum += sorted_range[index & 2047];
        }
    }

    // Last two warps handles backwards direction
    if ((i >= 64) && (i < (52 + 64))) {
        int temp_i = i - 64;
        int offset = temp_i * 51;
        DataType sum = 0.0;
        DataType group_sum = 0.0;
        for (int c = 0; c < 51; c++) {
            // Go backwards
            int index = coll_vec[offset + 50 - c];
            // Skip impossible hands
            if ((communal_cards & card_order[eval[index & 2047] & 2047]) > 0) continue;
            // Reverse ordering, because reverse iteration
            atomicAdd(&sorted_eval[index & 2047], sum);
            group_sum += sorted_range[index & 2047];

            // 2048 bit set => new group
            if (index & 2048) {
                sum += group_sum;
                group_sum = 0.0;
            }
        }
    }
    __syncthreads();
}

__device__ void
evaluate_showdown_kernel_inner(DataType *opponent_range, long communal_cards, long *card_order, short *eval,
                               short *coll_vec, DataType bet, DataType *result, DataType *sorted_range,
                               DataType *sorted_eval,
                               DataType *temp) {
    __syncthreads();
    // Setup
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Sort hands by eval
    for (int b = 0; b < 11; b++) {
        int index = i + 128* b;
        if (index < 1326) {
            // reset values
            sorted_range[index] = 0;
            sorted_eval[index] = 0;
            result[index] = 0;
            // Impossible hand since overlap with communal cards
            if ((communal_cards & card_order[eval[index] & 2047]) > 0) continue;
            sorted_range[index] = opponent_range[eval[index] & 2047];
        }
        if (index == 1326) {
            sorted_range[index] = 0;
        }
    }

    // Handle card collisions
    handle_collisions(i, communal_cards, card_order, eval, coll_vec, sorted_range, sorted_eval);

    // Calculate prefix sum
    cuda_prefix_sum(sorted_range, temp);
    if (i == 0) {
        sorted_range[1326] = sorted_range[1325] + opponent_range[eval[1325] & 2047];
    }
    __syncthreads();

    // Calculate showdown value of all hands
    int prev_group = eval[1326 + i];
    for (int b = 0; b < 11; b++) {
        int index = i * 11 + b;
        if (index < 1326) {
            // Impossible hand since overlap with communal cards
            if ((communal_cards & card_order[eval[index] & 2047]) > 0) continue;
            if (eval[index] & 2048) { prev_group = index; }
            DataType worse = sorted_range[prev_group];
            sorted_eval[index] += worse;
        }
    }

    int next_group = eval[1326 + 128 + i];
    for (int b = 10; b >= 0; b--) {
        int index = i * 11 + b;
        if (index < 1326) {
            // Impossible hand since overlap with communal cards
            if ((communal_cards & card_order[eval[index] & 2047]) > 0) continue;
            DataType better = sorted_range[1326] - sorted_range[next_group];
            sorted_eval[index] -= better;
            // Observe reverse order because of reverse iteration
            if (eval[index] & 2048) { next_group = index; }
        }
    }

    // Write result
    __syncthreads();
    for (int b = 0; b < 11; b++) {
        int index = i + 128 * b;
        if (index < 1326) {
            result[eval[index] & 2047] = sorted_eval[index] * bet;
        }
    }
    __syncthreads();
}

__global__ void
evaluate_showdown_kernel(DataType *opponent_range, long communal_cards, long *card_order, short *eval,
                         short *coll_vec, DataType bet, DataType *result, Evaluator *evaluator) {
    __shared__ DataType sorted_range[1327];
    __shared__ DataType sorted_eval[1326];
    __shared__ DataType temp[128];
    evaluate_showdown_kernel_inner(opponent_range, communal_cards, card_order, eval, coll_vec, bet, result,
                                   sorted_range, sorted_eval, temp);
}

__device__ void
evaluate_fold_kernel_inner(DataType *opponent_range, long communal_cards, long *card_order, short *card_indexes,
                           short updating_player, short folding_player, DataType bet, DataType *result,
                           DataType *range_sum,
                           DataType *temp) {
    __syncthreads();
    // Setup
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int b = 0; b < 11; b++) {
        int index = i + 128* b;
        if (index < 1326) {
            // reset values
            range_sum[index] = 0;
            // Because of inclusion-exclusion, we need to add the
            // probability that the opponent got exactly the same hand
            result[index] = 0;
            // Impossible hand since overlap with communal cards
            if (communal_cards & card_order[index]) continue;
            range_sum[index] = opponent_range[index];
            result[index] = opponent_range[index];
        }
    }

    // Calculate prefix sum
    cuda_prefix_sum(range_sum, temp);

    // using result[1325] is a bit hacky, but correct
    DataType total = range_sum[1325] + result[1325];

    if (i < 52) {
        DataType card_sum = 0.0;
        for (int c = 0; c < 51; c++) {
            short index = card_indexes[i * 51 + c];
            if (communal_cards & card_order[index]) continue;
            card_sum += opponent_range[index];
        }
        for (int c = 0; c < 51; c++) {
            short index = card_indexes[i * 51 + c];
            if (communal_cards & card_order[index]) continue;
            atomicAdd(&result[index], -card_sum);
        }
    }
    __syncthreads();

    for (int b = 0; b < 11; b++) {
        int index = i + 128* b;
        if (index < 1326) {
            if (communal_cards & card_order[index]) continue;
            result[index] += total;
            if (updating_player == folding_player) {
                result[index] *= -bet;
            } else {
                result[index] *= bet;
            }
        }
    }
}

__global__ void
evaluate_fold_kernel(DataType *opponent_range, long communal_cards, long *card_order, short *card_indexes,
                     short updating_player, short folding_player, DataType bet, DataType *result) {
    __shared__ DataType range_sum[1326];
    __shared__ DataType temp[128];
    evaluate_fold_kernel_inner(opponent_range, communal_cards, card_order, card_indexes, updating_player,
                               folding_player, bet, result, range_sum, temp);
}

__device__ void evaluate_post_turn_kernel_inner(DataType *opponent_range,
                                                 State *state,
                                                 Evaluator *evaluator,
                                                 Player updating_player,
                                                 DataType *scratch,
                                                 DataType *result, DataType *sorted_range, DataType *sorted_eval,
                                                 DataType *temp) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    switch (state->terminal) {
        case Showdown : {
            long set = state->cards ^ evaluator->flop;
            int eval_index = get_index(set);
            short *eval = evaluator->eval + eval_index * (1326 + 128 * 2);
            short *coll_vec = evaluator->coll_vec + eval_index * 52 * 51;
            evaluate_showdown_kernel_inner(opponent_range, state->cards, evaluator->card_order, eval,
                                           coll_vec, state->sbbet, result, sorted_range, sorted_eval, temp);

        }
            break;
        case SBWins :
            evaluate_fold_kernel_inner(opponent_range, state->cards, evaluator->card_order, evaluator->card_indexes,
                                       updating_player, 1, state->bbbet, result, sorted_eval, temp);
            break;
        case BBWins :
            evaluate_fold_kernel_inner(opponent_range, state->cards, evaluator->card_order, evaluator->card_indexes,
                                       updating_player, 0, state->sbbet, result, sorted_eval, temp);
            break;
        case NonTerminal : {
            DataType *average_strategy = result;
            zero(average_strategy);
            DataType *action_probs = scratch;
            scratch += 1326 * state->transitions; // + state->transitions
            get_strategy(state, scratch, action_probs);
            DataType *results = scratch;
            scratch += 1326 * state->transitions; // + state-> transitions
            for (int i = 0; i < state->transitions; i++) {
                DataType *new_result = results + 1326 * i;
                DataType *action_prob = action_probs + 1326 * i;
                State *next = state->next_states[i];
                DataType *new_range;
                if (state->next_to_act == updating_player) {
                    new_range = opponent_range;
                } else {
                    new_range = scratch;
                    scratch += 1326; // + 1
                    multiply(opponent_range, action_prob, new_range);
                }
                evaluate_post_turn_kernel_inner(new_range, next, evaluator, updating_player, scratch, new_result,
                                                 sorted_range, sorted_eval, temp);
                if (updating_player == state->next_to_act) {
                    fma(new_result, action_prob, average_strategy);
                } else {
                    add_assign(average_strategy, new_result);
                }
            }
            if (state->next_to_act == updating_player) {
                for (int i = 0; i < state->transitions; i++) {
                    DataType *util = results + 1326 * i;
                    sub_assign(util, average_strategy);
                }
                update_strategy(state, results);
            }
        }
            break;
        case River:
            zero(result);
            DataType *new_result = scratch;
            scratch += 1326;
            for (int i = 0; i < state->transitions; i++) {
                zero(new_result);
                State *next = state->next_states[i];
                evaluate_post_turn_kernel_inner(opponent_range, next, evaluator, updating_player, scratch, new_result,
                                                 sorted_range, sorted_eval, temp);
                add_assign(result, new_result);
            }
            for (int b = 0; b < 11; b++) {
                int index = tid + 128 * b;
                if (index < 1326) {
                    result[index] /= ((DataType) state->transitions);
                }
            }
            break;
    }
}

__global__ void evaluate_post_turn_kernel(DataType *opponent_range,
                                      State *state,
                                      Evaluator *evaluator,
                                      Player updating_player,
                                      DataType *scratch,
                                      DataType *result) {
    __shared__ DataType sorted_range[1327];
    __shared__ DataType sorted_eval[1326];
    __shared__ DataType temp[128];
    evaluate_post_turn_kernel_inner(opponent_range, state, evaluator, updating_player, scratch, result, sorted_range,
                                     sorted_eval, temp);
}


extern "C" {
void evaluate_showdown_cuda(DataType *opponent_range, long communal_cards, long *card_order, short *eval,
                            short *coll_vec, DataType bet, DataType *result, Evaluator *evaluator) {
    DataType *device_opponent_range;
    cudaMalloc(&device_opponent_range, 1326 * sizeof(DataType));
    cudaMemcpy(device_opponent_range, opponent_range, 1326 * sizeof(DataType), cudaMemcpyHostToDevice);

    DataType *device_result;
    cudaMalloc(&device_result, 1326 * sizeof(DataType));

    long *device_card_order;
    cudaMalloc(&device_card_order, 1326 * sizeof(long));
    cudaMemcpy(device_card_order, card_order, 1326 * sizeof(long), cudaMemcpyHostToDevice);

    short *device_eval;
    cudaMalloc(&device_eval, (1326 + 128 * 2) * sizeof(short));
    cudaMemcpy(device_eval, eval, (1326 + 128 * 2) * sizeof(short), cudaMemcpyHostToDevice);

    short *device_coll_vec;
    cudaMalloc(&device_coll_vec, 52 * 51 * sizeof(short));
    cudaMemcpy(device_coll_vec, coll_vec, 52 * 51 * sizeof(short), cudaMemcpyHostToDevice);

    evaluate_showdown_kernel<<<1, 128>>>(device_opponent_range, communal_cards, device_card_order, device_eval,
                                         device_coll_vec, bet, device_result, evaluator);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
    fflush(stdout);
    cudaMemcpy(result, device_result, 1326 * sizeof(DataType), cudaMemcpyDeviceToHost);

    cudaFree(device_opponent_range);
    cudaFree(device_result);
    cudaFree(device_card_order);
    cudaFree(device_eval);
    cudaFree(device_coll_vec);
}

void evaluate_fold_cuda(DataType *opponent_range, long communal_cards, long *card_order, short *card_indexes,
                        short updating_player,
                        short folding_player, DataType bet, DataType *result) {
    DataType *device_opponent_range;
    cudaMalloc(&device_opponent_range, 1326 * sizeof(DataType));
    cudaMemcpy(device_opponent_range, opponent_range, 1326 * sizeof(DataType), cudaMemcpyHostToDevice);

    DataType *device_result;
    cudaMalloc(&device_result, 1326 * sizeof(DataType));

    long *device_card_order;
    cudaMalloc(&device_card_order, 1326 * sizeof(long));
    cudaMemcpy(device_card_order, card_order, 1326 * sizeof(long), cudaMemcpyHostToDevice);

    short *device_card_indexes;
    cudaMalloc(&device_card_indexes, 52 * 51 * sizeof(short));
    cudaMemcpy(device_card_indexes, card_indexes, 52 * 51 * sizeof(short), cudaMemcpyHostToDevice);

    evaluate_fold_kernel<<<1, 128>>>(device_opponent_range, communal_cards, device_card_order, device_card_indexes,
                                     updating_player, folding_player, bet, device_result);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    cudaDeviceSynchronize();
    cudaMemcpy(result, device_result, 1326 * sizeof(DataType), cudaMemcpyDeviceToHost);

    cudaFree(device_opponent_range);
    cudaFree(device_result);
    cudaFree(device_card_order);
    cudaFree(device_card_indexes);
}

void evaluate_post_turn_cuda(DataType *opponent_range,
                         State *state,
                         Evaluator *evaluator,
                         short updating_player,
                         DataType *result) {
    DataType *device_opponent_range;
    cudaMalloc(&device_opponent_range, 1326 * sizeof(DataType));
    cudaMemcpy(device_opponent_range, opponent_range, 1326 * sizeof(DataType), cudaMemcpyHostToDevice);

    DataType *device_result;
    cudaMalloc(&device_result, 1326 * sizeof(DataType));

    DataType *device_scratch;
    int scratch_size = 1326 * sizeof(DataType) * (7 *
                                                  6 +
                                                  10); // Max 7 ( 3 results, 3 action probs, 1 new_probs) vectors per level, max depth of 6 and 1 for river
    cudaMalloc(&device_scratch, scratch_size);
    cudaMemset(device_scratch, 0, scratch_size);

    evaluate_post_turn_kernel<<<1, 128>>>(device_opponent_range, state, evaluator, updating_player == 0 ? Small : Big,
                                      device_scratch,
                                      device_result);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        fflush(stdout);
    }
    fflush(stdout);
    cudaMemcpy(result, device_result, 1326 * sizeof(DataType), cudaMemcpyDeviceToHost);

    cudaFree(device_opponent_range);
    cudaFree(device_scratch);
    cudaFree(device_result);
}
}
