#include <stdio.h>
#include <cuda_runtime.h>
#include "math.h"
#include "structs.h"
#include "evaluator.cuh"
#include <sys/time.h>
#include <cmath>
#include <assert.h>

__device__ void multiply(Vector *__restrict__ v1, Vector *__restrict__ v2, Vector *__restrict__ res) {
    int i = threadIdx.x;
    for (int b = 0; b < ITERS; b++) {
        int index = i + TPB * b;
        if (index < 1326) {
            res->values[index] = v1->values[index] * v2->values[index];
        }
    }
}

__device__ void divide(Vector *v1, DataType val) {
    int i = threadIdx.x;
    val = 1.0f / val;
    for (int b = 0; b < ITERS; b++) {
        int index = i + TPB * b;
        if (index < 1326) {
            v1->values[index] *= val;
        }
    }
}

__device__ void fma(Vector *__restrict__ v1, Vector *__restrict__ v2, Vector *__restrict__ res) {
    int i = threadIdx.x;
    for (int b = 0; b < ITERS; b++) {
        int index = i + TPB * b;
        if (index < 1326) {
            res->values[index] += v1->values[index] * v2->values[index];
        }
    }
}

__device__ void add_assign(Vector *__restrict__ v1, Vector *__restrict__ v2) {
    int i = threadIdx.x;
    for (int b = 0; b < ITERS; b++) {
        int index = i + TPB * b;
        if (index < 1326) {
            v1->values[index] += v2->values[index];
        }
    }
}

__device__ void sub_assign(Vector *__restrict__ v1, Vector *__restrict__ v2) {
    int i = threadIdx.x;
    for (int b = 0; b < ITERS; b++) {
        int index = i + TPB * b;
        if (index < 1326) {
            v1->values[index] -= v2->values[index];
        }
    }
}

__device__ void copy(Vector *__restrict__ from, Vector *__restrict__ into) {
    int i = threadIdx.x;
    for (int b = 0; b < ITERS; b++) {
        int index = i + TPB * b;
        if (index < 1326) {
            into->values[index] = from->values[index];
        }
    }
}


__device__ void zero(Vector *v) {
    int i = threadIdx.x;
    for (int b = 0; b < ITERS; b++) {
        int index = i + TPB * b;
        if (index < 1326) {
            v->values[index] = 0.0f;
        }
    }
}

__device__ void p_sum(DataType *input, int i) {
    int offset = 1;
    for (int d = TPB / 2; d > 0; d >>= 1) {
        __syncthreads();
        if (i < d) {
            int ai = offset * (2 * i + 1) - 1;
            int bi = offset * (2 * i + 2) - 1;
            input[bi] += input[ai];
        }
        offset *= 2;
    }
    if (i == 0) {
        input[TPB - 1] = 0.0f;
    }
    for (int d = 1; d < TPB; d *= 2) {
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

__device__ void cuda_prefix_sum(Vector *input) {
    __syncthreads();
    __shared__ DataType temp[TPB];
    int i = threadIdx.x;
    temp[i] = 0;
    for (int b = 0; b < ITERS; b++) {
        int index = i * ITERS + b;
        if (index < 1326 && i < TPB - 1) {
            temp[i] += input->values[index];
        }
    }
    p_sum(temp, i);

    DataType prefix = temp[i];
    for (int b = 0; b < ITERS; b++) {
        int index = i * ITERS + b;
        if (index <= 1326) {
            DataType t = input->values[index];
            input->values[index] = prefix;
            prefix += t;
        }
    }
    __syncthreads();
}

__device__ DataType reduce_sum(DataType *vector) {
    int i = threadIdx.x;
    __shared__ DataType temp[TPB];
    temp[i] = 0;
    for (int b = 0; b < ITERS; b++) {
        int index = i + TPB * b;
        if (index < 1326) {
            temp[i] += vector[index];
        }
    }
    __syncthreads();
    for (int k = TPB / 2; k > 0; k >>= 1) {
        if (i < k) {
            temp[i] += temp[i + k];
        }
        __syncthreads();
    }
    return temp[0];
}

__device__ DataType get_strategy_sum(State *state) {
    DataType sum = 0.0f;
    for (int k = 0; k < state->transitions; k++) {
        sum += state->card_strategies[k]->values[threadIdx.x];
    }
    return sum;
}

__device__ void
get_strategy_abstract(State *state, Vector *scratch, Vector *result, short *abstractions) {
    int tid = threadIdx.x;
    Vector *sum = scratch;
    int transitions = state->transitions;
    DataType vals[2] = {0.0f, 0.0f};
#pragma unroll 3
    for (int k = 0; k < transitions; k++) {
        vals[0] += state->card_strategies[k]->values[tid];
        vals[1] += state->card_strategies[k]->values[tid + 128];
    }
    sum->values[tid] = vals[0];
    sum->values[tid + 128] = vals[1];
    __syncthreads();
    for (int b = 0; b < ITERS; b++) {
        int index = tid + TPB * b;
        if (index < 1326) {
            short abstract_index = abstractions[index];
            for (int k = 0; k < transitions; k++) {
                if (sum->values[abstract_index] <= 1e-4f) {
                    result[k].values[index] = 1.0f / ((DataType) transitions);
                } else {
                    result[k].values[index] =
                            state->card_strategies[k]->values[abstract_index] / sum->values[abstract_index];
                }
            }
        }
    }
    __syncthreads();
}

__device__ void update_strategy_abstract(State *__restrict__ state, Vector *__restrict__ update,
                                         short *abstractions) {
    int tid = threadIdx.x;
    __syncthreads();
    for (int b = 0; b < ITERS; b++) {
        int index = tid + TPB * b;
        if (index < 1326) {
            short abstract_index = abstractions[index];
            for (int k = 0; k < state->transitions; k++) {
                atomicAdd(&state->card_strategies[k]->values[abstract_index], update[k].values[index]);
            }
        }
    }
    __syncthreads();
    for (int k = 0; k < state->transitions; k++) {
        state->card_strategies[k]->values[tid] = max(state->card_strategies[k]->values[tid], 0.0f);
        state->card_strategies[k]->values[tid + 128] = max(state->card_strategies[k]->values[tid + 128], 0.0f);
    }
    __syncthreads();
}

__device__ void
handle_collisions(short *coll_vec,
                  Vector *sorted_range, Vector *result, short *eval) {
    int i = threadIdx.x;
    __syncthreads();
    // Handle collisions before prefix sum consumes sorted_range
    // First two warps handles forward direction
    if (i < 52) {
        int offset = i * 51;
        DataType sum = 0.0f;
        DataType group_sum = 0.0f;
        for (int c = 0; c < 51; c++) {
            int index = coll_vec[offset + c];
            // 2048 bit set => new group
            if (index & 2048) {
                sum += group_sum;
                group_sum = 0.0f;
            }
            atomicAdd(&result->values[index & 2047], -sum);
            group_sum += sorted_range->values[eval[index & 2047] & 2047];
        }
    }

    // Last two warps handles backwards direction
    if ((i >= 64) && (i < (52 + 64))) {
        int temp_i = i - 64;
        int offset = temp_i * 51;
        DataType sum = 0.0f;
        DataType group_sum = 0.0f;
        for (int c = 0; c < 51; c++) {
            // Go backwards
            int index = coll_vec[offset + 50 - c];
            // Reverse ordering, because reverse iteration
            atomicAdd(&result->values[index & 2047], sum);
            group_sum += sorted_range->values[eval[index & 2047] & 2047];

            // 2048 bit set => new group
            if (index & 2048) {
                sum += group_sum;
                group_sum = 0.0f;
            }
        }
    }
    __syncthreads();
}

__device__ DataType
evaluate_showdown(Vector *opponent_range, DataType bucket_reach, DataType player_reach, Vector *result,
                  short *eval, short *coll_vec, DataType bet, short *abstractions, bool calc_exploit,
                  DataType *reach_probs) {
    // Setup
    int tid = threadIdx.x;
    __shared__ Vector sorted_range[1];
    reach_probs[tid] = bucket_reach;
    zero(result);
    __syncthreads();
    //return 0;

    for (int b = 0; b < ITERS; b++) {
        int index = tid + b * TPB;
        if (index < 1326) {
            int sorted_index = eval[index] & 2047;
            sorted_range->values[sorted_index] =
                    opponent_range->values[index] * reach_probs[abstractions[index]];
        }
    }
    __syncthreads();

    // Handle card collisions
    handle_collisions(coll_vec, sorted_range, result, eval);

    // Calculate prefix sum in place
    cuda_prefix_sum(sorted_range);

    // Calculate showdown value of all hands
    int prev_group = eval[1326 + tid];
    DataType values[ITERS];
    for (int b = 0; b < ITERS; b++) {
        int index = tid * ITERS + b;
        if (index < 1326) {
            if (eval[index] & 2048) { prev_group = index; }
            // worse hands
            values[b] = sorted_range->values[prev_group];
        }
    }

    int next_group = eval[1326 + 256 + tid];
    for (int b = ITERS - 1; b >= 0; b--) {
        int index = tid * ITERS + b;
        if (index < 1326) {
            // better hands
            values[b] -= sorted_range->values[1326] - sorted_range->values[next_group];
            // Observe reverse order because of reverse iteration
            if (eval[index] & 2048) { next_group = index; }
        }
    }
    __syncthreads();
    // write registers to shared
    for (int b = 0; b < ITERS; b++) {
        int index = tid * ITERS + b;
        if (index < 1326) {
            sorted_range->values[index] = values[b];
        }
    }

    // Write result
    __syncthreads();
    for (int b = 0; b < ITERS; b++) {
        int index = tid + TPB * b;
        if (index < 1326) {
            result->values[index] =
                    (result->values[index] + sorted_range->values[eval[index] & 2047]) * bet;
        }
    }
    __syncthreads();
    reach_probs[tid] = 0.0f;
    __syncthreads();
    // Aggregate utilities for abstraction
    for (int b = 0; b < ITERS; b++) {
        int index = tid + TPB * b;
        if (index < 1326) {
            atomicAdd(&reach_probs[abstractions[index]], result->values[index]);
        }
    }
    __syncthreads();
    DataType res = reach_probs[tid];
    reach_probs[tid] = player_reach;
    __syncthreads();
    // Multiply with player reach prob right now
    if (!calc_exploit) {
        for (int b = 0; b < ITERS; b++) {
            int index = tid + TPB * b;
            if (index < 1326) {
                result->values[index] *= reach_probs[abstractions[index]];
            }
        }
    }
    __syncthreads();
    return res;
}


__device__ DataType
evaluate_fold(Vector *opponent_range, DataType opponent_reach, DataType player_reach, short *card_indexes, DataType bet,
              Vector *result, short *abstractions, bool calc_exploit, DataType *reach_probs) {
    __shared__ DataType card_collisions[52];
    // Setup
    int tid = threadIdx.x;
    reach_probs[tid] = opponent_reach;
    __syncthreads();
    for (int b = 0; b < ITERS; b++) {
        int index = tid + b * TPB;
        if (index < 1326) {
            DataType prob = opponent_range->values[index] * reach_probs[abstractions[index]];
            result->values[index] = prob;
        }
    }
    __syncthreads();
    DataType total = reduce_sum(result->values);

    __syncthreads();
    DataType card_sum = 0.0f;
    if (tid < 52) {
        for (int c = 0; c < 51; c++) {
            short index = card_indexes[tid * 51 + c];
            card_sum += result->values[index];
        }
        card_collisions[tid] = card_sum;
    }
    __syncthreads();
    for (int b = 0; b < ITERS; b++) {
        int index = tid + TPB * b;
        if (index < 1326) {
            long cards = from_index(index);
            int card1 = __ffsll(cards) - 1;
            cards -= 1l << card1;
            int card2 = __ffsll(cards) - 1;
            assert(cards - (1l << card2) == 0);
            result->values[index] += (total - card_collisions[card1] - card_collisions[card2]);
            result->values[index] *= bet;
        }
    }
    __syncthreads();
    reach_probs[tid] = 0.0f;
    __syncthreads();
    // Aggregate utilities for abstraction
    for (int b = 0; b < ITERS; b++) {
        int index = tid + TPB * b;
        if (index < 1326) {
            atomicAdd(&reach_probs[abstractions[index]], result->values[index]);
        }
    }
    __syncthreads();
    DataType res = reach_probs[tid];
    __syncthreads();
    reach_probs[tid] = player_reach;
    __syncthreads();
    // Multiply with player reach prob right now
    if (!calc_exploit) {
        for (int b = 0; b < ITERS; b++) {
            int index = tid + TPB * b;
            if (index < 1326) {
                result->values[index] *= reach_probs[abstractions[index]];
            }
        }
    }
    __syncthreads();
    return res;
}

__device__ void remove_collisions(Vector *vector, long cards) {
    int tid = threadIdx.x;
    for (int b = 0; b < ITERS; b++) {
        int index = tid + TPB * b;
        if (index < 1326) {
            if (from_index(index) & cards) vector->values[index] = 0.0f;
        }
    }
    __syncthreads();
}

__device__ void evaluate_river(Vector *opponent_range,
                               State *root_state,
                               Vector *scratch,
                               long communal_cards,
                               short *card_indexes,
                               short *eval,
                               short *coll_vec,
                               short *abstractions,
                               Player updating_player,
                               bool calc_exploit) {
    __shared__ DataType reach_probs[TPB];
    __syncthreads();
    State *state = root_state;
    int tid = threadIdx.x;
    // Every 2 bits signifies the transition count at a specific depth
    int transition_state = 0;
    int depth = 0;
    DataType opponent_reach[7] = {1.0f};
    DataType player_reach[7] = {1.0f};
    DataType strat_sums[7];
    DataType average_abstract[7];
    while (depth >= 0) {
        switch (state->terminal) {
            case Showdown :
                average_abstract[depth] = evaluate_showdown(opponent_range, opponent_reach[depth], player_reach[depth],
                                                            scratch + depth, eval,
                                                            coll_vec, state->sbbet, abstractions, calc_exploit,
                                                            reach_probs);
                state = state->parent;
                depth--;
                break;
            case SBWins :
                average_abstract[depth] = evaluate_fold(opponent_range, opponent_reach[depth], player_reach[depth],
                                                        card_indexes,
                                                        updating_player == 1 ? -state->bbbet : state->bbbet,
                                                        scratch + depth, abstractions,
                                                        calc_exploit, reach_probs);
                state = state->parent;
                depth--;
                break;
            case BBWins :
                average_abstract[depth] = evaluate_fold(opponent_range, opponent_reach[depth], player_reach[depth],
                                                        card_indexes,
                                                        updating_player == 0 ? -state->sbbet : state->sbbet,
                                                        scratch + depth, abstractions, calc_exploit, reach_probs);
                state = state->parent;
                depth--;
                break;
            case NonTerminal : {
                int transitions = state->transitions;
                Vector *average_strategy = scratch + depth;
                int transition = (transition_state >> (2 * depth)) & 3;
                if (transition == 0) {
                    if ((updating_player == state->next_to_act) && calc_exploit) {
                        for (int b = 0; b < ITERS; b++) {
                            int index = tid + TPB * b;
                            if (index < 1326) {
                                average_strategy->values[index] = -INFINITY;
                            }
                        }
                    } else {
                        zero(average_strategy);
                    }
                    strat_sums[depth] = get_strategy_sum(state);
                    average_abstract[depth] = 0.0f;
                } else {
                    Vector *new_result = average_strategy + 1;
                    if (updating_player == state->next_to_act) {
                        if (!calc_exploit) {
                            // Should work if we multiply by updating players reach prob at leaf nodes
                            add_assign(average_strategy, new_result);
                            DataType regret = state->card_strategies[transition - 1]->values[tid];
                            if (strat_sums[depth] <= CUTOFF) {
                                average_abstract[depth] += average_abstract[depth + 1] / ((DataType) transitions);
                            } else {
                                average_abstract[depth] += average_abstract[depth + 1] * regret / strat_sums[depth];
                            }
                            // We add the util to the regrets here, as it won't be read again this iteration, and we don't
                            // have to store the utility until later.
                            state->card_strategies[transition - 1]->values[tid] += average_abstract[depth + 1];
                        } else {
                            for (int b = 0; b < ITERS; b++) {
                                int index = tid + TPB * b;
                                if (index < 1326) {
                                    average_strategy->values[index] = max(average_strategy->values[index],
                                                                          new_result->values[index]);
                                }
                            }
                        }
                    } else {
                        add_assign(average_strategy, new_result);
                        average_abstract[depth] += average_abstract[depth + 1];
                    }
                }

                if (transition == transitions) {
                    if ((state->next_to_act == updating_player) && !calc_exploit) {
                        // update strategy and remove the utility of the average strategy
                        for (int k = 0; k < transitions; k++) {
                            DataType regret = state->card_strategies[k]->values[tid];
                            regret -= average_abstract[depth];
                            regret = max(regret, 0.0f);
                            state->card_strategies[k]->values[tid] = regret;
                        }
                    }
                    state = state->parent;
                    depth--;
                } else {
                    State *next = state->next_states[transition];
                    // Update reach probabilities of ranges
                    if (state->next_to_act == updating_player) {
                        if (strat_sums[depth] <= CUTOFF) {
                            player_reach[depth + 1] =
                                    player_reach[depth] / ((DataType) transitions);
                        } else {
                            player_reach[depth + 1] =
                                    player_reach[depth] * state->card_strategies[transition]->values[tid] /
                                    strat_sums[depth];
                        }
                        opponent_reach[depth + 1] = opponent_reach[depth];
                    } else {
                        if (strat_sums[depth] <= CUTOFF) {
                            opponent_reach[depth + 1] =
                                    opponent_reach[depth] / ((DataType) transitions);
                        } else {
                            opponent_reach[depth + 1] =
                                    opponent_reach[depth] * state->card_strategies[transition]->values[tid] /
                                    strat_sums[depth];
                        }
                        player_reach[depth + 1] = player_reach[depth];
                    }
                    state = next;
                    // Reset transition count for depth + 1
                    transition_state &= ~(3 << ((depth + 1) * 2));
                    transition_state += 1 << (depth * 2);
                    depth++;
                }
                break;
            }
        }
    }
//    if (!calc_exploit) {
//        __shared__ DataType test[TPB];
//        test[tid] = 0.0f;
//        __syncthreads();
//        for (int b = 0; b < ITERS; b++) {
//            int index = tid + TPB * b;
//            if (index < 1326) {
//                atomicAdd(&test[abstractions[index]], scratch->values[index]);
//            }
//        }
//        __syncthreads();
//        if (abs(test[tid] - average_abstract[0]) > 1e-4) {
//            printf("fail: %f %f\n", test[tid], average_abstract[0]);
//        }
//    }
}

__global__ void evaluate_all(Vector *opponent_ranges, Vector *results, State *root_states, Evaluator *evaluator,
                             Player updating_player, bool calc_exploit, Vector *scratches) {
    int block = blockIdx.x;
    Vector *opponent_range = opponent_ranges + block;
    Vector *result = results + block;
    State *state = root_states + 28 * block;
    Vector *scratch = scratches + 10 * block;
    State *next = state->next_states[0];
    for (int c = 0; c < 52; c++) {
        long river = 1l << c;
        if (river & state->cards) continue;
        Vector *new_range = scratch;
        copy(opponent_range, new_range);
        remove_collisions(new_range, river | state->cards);
        long set = (state->cards | river) ^ evaluator->flop;
        int eval_index = get_index(set);
        short *eval = evaluator->eval + eval_index * (1326 + 256 * 2);
        short *coll_vec = evaluator->coll_vec + eval_index * 52 * 51;
        evaluate_river(new_range, next, scratch + 1, state->cards | river, evaluator->card_indexes, eval,
                       coll_vec, evaluator->abstractions + eval_index * 1326,
                       updating_player, calc_exploit);
        remove_collisions(scratch + 1, river | state->cards);
        add_assign(result, scratch + 1);
    }
    divide(result, 48.0f);
    //remove_collisions(result, state->cards);
}

#ifdef TEST
#include "builder.cu"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <thread>

int main() {
    Evaluator *device_evaluator;
    cudaMalloc(&device_evaluator, sizeof(Evaluator));
    Evaluator *evaluator = (Evaluator *) calloc(1, sizeof(Evaluator));
    int file_evaluator = open("evaluator_test", O_RDWR | O_CREAT, 0666);
    void *src = mmap(NULL, sizeof(Evaluator), PROT_READ | PROT_WRITE, MAP_SHARED, file_evaluator, 0);

    memcpy(evaluator, src, sizeof(Evaluator));
    munmap(src, sizeof(Evaluator));
    close(file_evaluator);

    cudaMemcpy(device_evaluator, evaluator, sizeof(Evaluator), cudaMemcpyHostToDevice);
    DataType *range = (float *) calloc(1326, sizeof(DataType));
    for (int i = 0; i < 1326; i++) {
        if (evaluator->card_order[i] & 7l) {
            range[i] = 0.0;
        } else {
            range[i] = 1.0;
        }
    }

    State** states = build_turn_cuda(7l, 1.0);

    DataType *result = (float *) calloc(1326, sizeof(DataType));

    double start = cpuSecond();

    int THREADS = 100;
    std::thread threads[THREADS];
    for(int i = 0; i < THREADS; i++) threads[i] = std::thread(evaluate_turn_cuda, range, states, device_evaluator, 0, true, result);
    for(int i = 0; i < THREADS; i++) threads[i].join();

    double elapsed = cpuSecond() - start;


    float sum = 0;
    for (int i = 0; i < 1326; i++) {
        sum += result[i];
    }
    printf("sum: %f elapsed: %f\n", sum, elapsed);
    free(range);
    free(result);
    for (int i = 0; i < 49; i++) {
        cudaFree(states[i]);
    }
    cudaFree(device_evaluator);
}
#endif