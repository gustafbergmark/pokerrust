#include <stdio.h>
#include <cuda_runtime.h>
#include "math.h"
#include "structs.h"
#include "evaluator.cuh"
#include <sys/time.h>

//#define TPB 128
//#define ITERS 11
//#define FRAGMENTS 1
#define TPB 32
#define ITERS 42
#define FRAGMENTS 4

__device__ void multiply(Vector *__restrict__ v1, Vector *__restrict__ v2, Vector *__restrict__ res) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int b = 0; b < ITERS; b++) {
        int index = i + TPB * b;
        if (index < 1326) {
            res->values[index] = v1->values[index] * v2->values[index];
        }
    }
}

__device__ void fma(Vector *__restrict__ v1, Vector *__restrict__ v2, Vector *__restrict__ res) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int b = 0; b < ITERS; b++) {
        int index = i + TPB * b;
        if (index < 1326) {
            res->values[index] += v1->values[index] * v2->values[index];
        }
    }
}

__device__ void add_assign(Vector *__restrict__ v1, Vector *__restrict__ v2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int b = 0; b < ITERS; b++) {
        int index = i + TPB * b;
        if (index < 1326) {
            v1->values[index] += v2->values[index];
        }
    }
}

__device__ void sub_assign(Vector *__restrict__ v1, Vector *__restrict__ v2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int b = 0; b < ITERS; b++) {
        int index = i + TPB * b;
        if (index < 1326) {
            v1->values[index] -= v2->values[index];
        }
    }
}

__device__ void copy(Vector *__restrict__ from, Vector *__restrict__ into) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int b = 0; b < ITERS; b++) {
        int index = i + TPB * b;
        if (index < 1326) {
            into->values[index] = from->values[index];
        }
    }
}


__device__ void zero(Vector *v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int b = 0; b < ITERS; b++) {
        int index = i + TPB * b;
        if (index < 1326) {
            v->values[index] = 0;
        }
    }
}

__device__ void p_sum(DataType *input, int i) {
    int offset = 1;
    for (int d = TPB/2; d > 0; d >>= 1) {
        __syncthreads();
        if (i < d) {
            int ai = offset * (2 * i + 1) - 1;
            int bi = offset * (2 * i + 2) - 1;
            input[bi] += input[ai];
        }
        offset *= 2;
    }
    if (i == 0) {
        input[TPB-1] = 0;
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

__device__ void cuda_prefix_sum(DataType *input, DataType *temp) {
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    temp[i] = 0;
    for (int b = 0; b < ITERS; b++) {
        int index = i * ITERS + b;
        if (index < 1326 && i < 127) {
            temp[i] += input[index];
        }
    }
    p_sum(temp, i);

    DataType prefix = temp[i];
    for (int b = 0; b < ITERS; b++) {
        int index = i * ITERS + b;
        if (index < 1326) {
            DataType t = input[index];
            input[index] = prefix;
            prefix += t;
        }
    }
    __syncthreads();
}

__device__ DataType reduce_sum(DataType *vector, DataType *temp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    temp[i] = 0;
    for (int b = 0; b < ITERS; b++) {
        int index = i + TPB * b;
        if (index < 1326) {
            temp[i] += vector[index];
        }
    }
    __syncthreads();
    for (int k = 64; k > 0; k /= 2) {
        if (i < k) {
            temp[i] += temp[i + k];
        }
        __syncthreads();
    }
    return temp[0];
}

__device__ void get_strategy(State *state, Vector *scratch, Vector *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    Vector *sum = scratch;
    zero(sum);
    for (int i = 0; i < state->transitions; i++) {
        add_assign(sum, state->card_strategies[i]);
    }
    for (int i = 0; i < state->transitions; i++) {
        for (int b = 0; b < ITERS; b++) {
            int index = tid + TPB * b;
            if (index < 1326) {
                if (sum->values[index] <= 1e-4) {
                    result[i].values[index] = 1.0 / ((DataType) state->transitions);
                } else {
                    result[i].values[index] = state->card_strategies[i]->values[index] / sum->values[index];
                }
            }
        }
    }
}

__device__ void update_strategy(State *__restrict__ state, Vector *__restrict__ update) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < state->transitions; i++) {
        add_assign(state->card_strategies[i], update + i);
        for (int b = 0; b < ITERS; b++) {
            int index = tid + TPB * b;
            if (index < 1326) {
                state->card_strategies[i]->values[index] = max(state->card_strategies[i]->values[index], 0.0);
            }
        }
    }
}

__device__ void
handle_collisions( long communal_cards, short *eval, short *coll_vec,
                  DataType *sorted_range, DataType *sorted_eval) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __syncthreads();
    // Handle collisions before prefix sum consumes sorted_range
    // First two warps handles forward direction
    if (i < 52) {
        int offset = i * 51;
        DataType sum = 0.0;
        DataType group_sum = 0.0;
        for (int c = 0; c < 51; c++) {
            int index = coll_vec[offset + c];
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
evaluate_showdown_kernel_inner(DataType *opponent_range, long communal_cards, short *eval,
                               short *coll_vec, DataType bet, DataType *result, DataType *sorted_range,
                               DataType *sorted_eval,
                               DataType *temp) {
    __syncthreads();
    // Setup
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Sort hands by eval
    for (int b = 0; b < ITERS; b++) {
        int index = i + TPB * b;
        if (index < 1326) {
            // reset values
            sorted_eval[index] = 0;
            result[index] = 0;
            sorted_range[index] = opponent_range[eval[index] & 2047];
        }
        if (index == 1326) {
            sorted_range[index] = 0;
        }
    }

    // Handle card collisions
    handle_collisions(communal_cards, eval, coll_vec, sorted_range, sorted_eval);

    // Calculate prefix sum
    cuda_prefix_sum(sorted_range, temp);
    if (i == 0) {
        sorted_range[1326] = sorted_range[1325] + opponent_range[eval[1325] & 2047];
    }
    __syncthreads();

    // Calculate showdown value of all hands
    int prev_group = eval[1326 + i];
    for (int b = 0; b < ITERS; b++) {
        int index = i * ITERS + b;
        if (index < 1326) {
            // Impossible hand since overlap with communal cards
            if (eval[index] & 2048) { prev_group = index; }
            DataType worse = sorted_range[prev_group];
            sorted_eval[index] += worse;
        }
    }

    int next_group = eval[1326 + 128 + i];
    for (int b = 10; b >= 0; b--) {
        int index = i * ITERS + b;
        if (index < 1326) {
            DataType better = sorted_range[1326] - sorted_range[next_group];
            sorted_eval[index] -= better;
            // Observe reverse order because of reverse iteration
            if (eval[index] & 2048) { next_group = index; }
        }
    }

    // Write result
    __syncthreads();
    for (int b = 0; b < ITERS; b++) {
        int index = i + TPB * b;
        if (index < 1326) {
            result[eval[index] & 2047] = sorted_eval[index] * bet;
        }
    }
    __syncthreads();
}

__global__ void
evaluate_showdown_kernel(DataType *opponent_range, long communal_cards, short *eval,
                         short *coll_vec, DataType bet, DataType *result, Evaluator *evaluator) {
    __shared__ DataType sorted_range[1327];
    __shared__ DataType sorted_eval[1326];
    __shared__ DataType temp[TPB];
    evaluate_showdown_kernel_inner(opponent_range, communal_cards, eval, coll_vec, bet, result,
                                   sorted_range, sorted_eval, temp);
}

__device__ void
evaluate_fold_kernel_inner(DataType *opponent_range, long communal_cards, short *card_indexes,
                           short updating_player, short folding_player, DataType bet, DataType *result,
                           DataType *local_range,
                           DataType *temp) {
    __syncthreads();
    // Setup
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int b = 0; b < ITERS; b++) {
        int index = i + TPB * b;
        if (index < 1326) {
            local_range[index] = opponent_range[index];
            result[index] = opponent_range[index];
        }
    }

    DataType total = reduce_sum(local_range, temp);

    __syncthreads();
    temp[i] = 0;
    DataType card_sum = 0.0;
    if (i < 52) {
        for (int c = 0; c < 26; c++) {
            short index = card_indexes[i * 51 + c];
            card_sum += local_range[index];
        }
        atomicAdd(&temp[i], card_sum);
    } else if ((i>=64) && (i < (64+52))){
        for (int c = 26; c < 51; c++) {
            short index = card_indexes[(i-64) * 51 + c];
            card_sum += local_range[index];
        }
        atomicAdd(&temp[i-64], card_sum);
    }
    __syncthreads();
    for (int b = 0; b < ITERS; b++) {
        int index = i + TPB * b;
        if (index < 1326) {
            long cards = from_index(index);
            int card1 = __ffsll(cards)-1;
            cards -= 1l<<card1;
            int card2 = __ffsll(cards)-1;
            local_range[index] -= temp[card1] + temp[card2];
        }
    }
    __syncthreads();
    if (updating_player == folding_player) {
        bet = -bet;
    }
    total *= bet;
    for (int b = 0; b < ITERS; b++) {
        int index = i + TPB * b;
        if (index < 1326) {
            result[index] = local_range[index] * bet + total;
        }
    }
}

__global__ void
evaluate_fold_kernel(DataType *opponent_range, long communal_cards, short *card_indexes,
                     short updating_player, short folding_player, DataType bet, DataType *result) {
    __shared__ DataType range_sum[1326];
    __shared__ DataType temp[TPB];
    evaluate_fold_kernel_inner(opponent_range, communal_cards, card_indexes, updating_player,
                               folding_player, bet, result, range_sum, temp);
}

__device__ void remove_collisions(Vector *vector, int card, Evaluator *evaluator) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int b = 0; b < ITERS; b++) {
        int index = tid + TPB * b;
        if (index < 1326) {
            if (from_index(index) & (1l << card)) vector->values[index] = 0.0f;
        }
    }
    __syncthreads();
}

__device__ void evaluate_post_turn_kernel_inner(Vector *opponent_range_root,
                                                State *root_state,
                                                Evaluator *evaluator,
                                                Player updating_player,
                                                Vector *scratch_root,
                                                DataType *temp) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    Context contexts[14];
    contexts[0] = {root_state, opponent_range_root, 0};
    int depth = 0;

    while (depth >= 0) {
        //if(tid==0) printf("depth %d\n", depth);
        Vector *scratch = scratch_root + depth * 10;
        Context *context = &contexts[depth];
        State *state = context->state;
        Vector *opponent_range = context->opponent_range;

        switch (state->terminal) {
            case Showdown : {
                long set = state->cards ^ evaluator->flop;
                int eval_index = get_index(set);
                short *eval = evaluator->eval + eval_index * (1326 + 128 * 2);
                short *coll_vec = evaluator->coll_vec + eval_index * 52 * 51;
                evaluate_showdown_kernel_inner(opponent_range->values, state->cards, eval,
                                               coll_vec, state->sbbet, (DataType *) scratch, (DataType*) (scratch+1), (DataType*) (scratch+2),
                                               temp);
                depth--;
            }
                break;
            case SBWins :
                evaluate_fold_kernel_inner(opponent_range->values, state->cards,
                                           evaluator->card_indexes,
                                           updating_player, 1, state->bbbet, (DataType *) scratch, (DataType*) (scratch+1), temp);
                depth--;
                break;
            case BBWins :
                evaluate_fold_kernel_inner(opponent_range->values, state->cards,
                                           evaluator->card_indexes,
                                           updating_player, 0, state->sbbet, (DataType *) scratch, (DataType*) (scratch+1), temp);
                depth--;
                break;
            case NonTerminal : {
                Vector *average_strategy = scratch;
                scratch += 1;
                Vector *action_probs = scratch;
                scratch += state->transitions; // + state->transitions
                Vector *results = scratch;
                scratch += state->transitions; // + state-> transitions
                if (context->transition == 0) {
                    zero(average_strategy);
                    get_strategy(state, scratch, action_probs);
                } else {
                    int i = context->transition - 1;
                    Vector *new_result = average_strategy + 10;
                    copy(new_result, results + i);
                    if (updating_player == state->next_to_act) {
                        fma(results + i, action_probs + i, average_strategy);
                    } else {
                        add_assign(average_strategy, results + i);
                    }
                }

                if (context->transition == context->state->transitions) {
                    if (state->next_to_act == updating_player) {
                        for (int i = 0; i < state->transitions; i++) {
                            Vector *util = results + i;
                            sub_assign(util, average_strategy);
                        }
                        update_strategy(state, results);
                    }
                    depth--;
                } else {
                    int i = context->transition;
                    State *next = context->state->next_states[i];
                    Vector *new_range;
                    if (state->next_to_act == updating_player) {
                        new_range = opponent_range;
                    } else {
                        new_range = scratch;
                        scratch += 1; // + 1
                        multiply(opponent_range, action_probs + i, new_range);
                    }
                    contexts[depth + 1] = {next, new_range, 0};
                    context->transition++;
                    depth++;
                }
                break;
            }
            case River:
                Vector *result = scratch;
                scratch += 1;
                if (context->transition == 0) {
                    zero(result);
                } else {
                    // offset to next depths result
                    int dealt_card = __ffsll(state->next_states[context->transition - 1]->cards ^ state->cards) - 1;
                    remove_collisions(result + 10, dealt_card, evaluator);
                    add_assign(result, result + 10);
                }
                if (context->transition == context->state->transitions) {
                    for (int b = 0; b < ITERS; b++) {
                        int index = tid + TPB * b;
                        if (index < 1326) {
                            if (from_index(index) & state->cards) {
                                result->values[index] = 0.0;
                            } else {
                                result->values[index] /= ((DataType) state->transitions);
                            }
                        }
                    }
                    depth--;
                } else {
                    int i = context->transition;
                    State *next = state->next_states[i];
                    int dealt_card = __ffsll(next->cards ^ state->cards) - 1;
                    //if(tid == 0) printf("dc %d old %lu new %lu\n", dealt_card, state->cards, next->cards);
                    Vector *new_range = scratch;
                    scratch += 1;
                    copy(opponent_range, new_range);
                    remove_collisions(new_range, dealt_card, evaluator);
                    contexts[depth + 1] = {next, new_range, 0};
                    context->transition += 1;
                    depth++;
                }
                break;
        }
    }
}

__global__ void evaluate_post_turn_kernel(Vector *opponent_range,
                                          State *state,
                                          Evaluator *evaluator,
                                          Player updating_player,
                                          Vector *scratch) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ DataType temp[128];
    evaluate_post_turn_kernel_inner(opponent_range, state, evaluator, updating_player, scratch,temp);
    // Remove utility of impossible hands
    for (int b = 0; b < ITERS; b++) {
        int index = tid + TPB * b;
        if (index < 1326) {
            if (from_index(index) & state->cards) {
                scratch->values[index] = 0.0;
            }
        }
    }
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
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

    evaluate_showdown_kernel<<<1, 128>>>(device_opponent_range, communal_cards, device_eval,
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

    evaluate_fold_kernel<<<1, 128>>>(device_opponent_range, communal_cards, device_card_indexes,
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
    Vector *device_opponent_range;
    cudaMalloc(&device_opponent_range, sizeof(Vector));
    cudaMemcpy(device_opponent_range, opponent_range, 1326 * sizeof(DataType), cudaMemcpyHostToDevice);


    Vector *device_scratch;
    // Max depth less than 14 i think, and max 8 vecs allocated per level
    int scratch_size = sizeof(Vector) * 10 * 14;
    cudaMalloc(&device_scratch, scratch_size);
    cudaMemset(device_scratch, 0, scratch_size);
    // Result will always be put in scratch[0..1326]
    double start = cpuSecond();
    evaluate_post_turn_kernel<<<1, TPB>>>(device_opponent_range, state, evaluator, updating_player == 0 ? Small : Big,
                                          device_scratch);

    cudaMemcpy(result, device_scratch, 1326 * sizeof(DataType), cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
    double elapsed = cpuSecond() - start;
    printf("Kernel time: %fs\n", elapsed);
    fflush(stdout);


    cudaFree(device_opponent_range);
    cudaFree(device_scratch);
}
}

//#include "builder.cu"
//#include <fcntl.h>
//#include <sys/mman.h>
//#include <unistd.h>
//
//int main() {
//    Evaluator *device_evaluator;
//    cudaMalloc(&device_evaluator, sizeof(Evaluator));
//    Evaluator *evaluator = (Evaluator *) calloc(1, sizeof(Evaluator));
//    int file_evaluator = open("evaluator_test", O_RDWR | O_CREAT, 0666);
//    void *src = mmap(NULL, sizeof(Evaluator), PROT_READ | PROT_WRITE, MAP_SHARED, file_evaluator, 0);
//
//    memcpy(evaluator, src, sizeof(Evaluator));
//    munmap(src, sizeof(Evaluator));
//    close(file_evaluator);
//
//    cudaMemcpy(device_evaluator, evaluator, sizeof(Evaluator), cudaMemcpyHostToDevice);
//    DataType *range = (float *) calloc(1326, sizeof(DataType));
//    for (int i = 0; i < 1326; i++) {
//        if (evaluator->card_order[i] & 15l) {
//            range[i] = 0.0;
//        } else {
//            range[i] = 1.0;
//        }
//    }
//    int ns = 1;
//    State *states[ns];
//    for (int i = 0; i < ns; i++) {
//        states[i] = build_post_turn_cuda(15l, 1.0);
//    }
//    DataType *result = (float *) calloc(1326, sizeof(DataType));
//
//
//    for (int i = 0; i < ns; i++) {
//        evaluate_post_turn_cuda(range, states[i], device_evaluator, 0, result);
//    }
//
//    float sum = 0;
//    for (int i = 0; i < 1326; i++) {
//        sum += result[i];
//    }
//    printf("sum: %f\n", sum);
//    free(range);
//    free(result);
//    for (int i = 0; i < ns; i++) {
//        cudaFree(states[i]);
//    }
//    cudaFree(device_evaluator);
//}