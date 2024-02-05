#include <stdio.h>
#include <cuda_runtime.h>
#include "math.h"
#include "structs.h"
#include "evaluator.cuh"
#include <sys/time.h>
#include <cmath>

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
    val = 1.0f/val;
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

__device__ void cuda_prefix_sum(DataType *input, DataType *temp) {
    __syncthreads();
    int i = threadIdx.x;
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
    int i = threadIdx.x;
    temp[i] = 0;
    for (int b = 0; b < ITERS; b++) {
        int index = i + TPB * b;
        if (index < 1326) {
            temp[i] += vector[index];
        }
    }
    __syncthreads();
    for (int k = 64; k > 0; k >>= 1) {
        if (i < k) {
            temp[i] += temp[i + k];
        }
        __syncthreads();
    }
    return temp[0];
}

__device__ void get_strategy(State *state, Vector *scratch, Vector *result) {
    int tid = threadIdx.x;
    Vector *sum = scratch;
    zero(sum);
    int transitions = state->transitions;
    for (int i = 0; i < transitions; i++) {
        add_assign(sum, state->card_strategies[i]);
    }
    for (int i = 0; i < transitions; i++) {
        for (int b = 0; b < ITERS; b++) {
            int index = tid + TPB * b;
            if (index < 1326) {
                if (sum->values[index] <= 1e-4f) {
                    result[i].values[index] = 1.0f / ((DataType) transitions);
                } else {
                    result[i].values[index] = state->card_strategies[i]->values[index] / sum->values[index];
                }
            }
        }
    }
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

__device__ void update_strategy(State *__restrict__ state, Vector *__restrict__ update) {
    int tid = threadIdx.x;
    for (int i = 0; i < state->transitions; i++) {
        add_assign(state->card_strategies[i], update + i);
        for (int b = 0; b < ITERS; b++) {
            int index = tid + TPB * b;
            if (index < 1326) {
                state->card_strategies[i]->values[index] = max(state->card_strategies[i]->values[index], 0.0f);
            }
        }
    }
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
                  DataType *sorted_range, DataType *sorted_eval) {
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
            atomicAdd(&sorted_eval[index & 2047], -sum);
            group_sum += sorted_range[index & 2047];
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
            atomicAdd(&sorted_eval[index & 2047], sum);
            group_sum += sorted_range[index & 2047];

            // 2048 bit set => new group
            if (index & 2048) {
                sum += group_sum;
                group_sum = 0.0f;
            }
        }
    }
    __syncthreads();
}

__device__ void
evaluate_showdown(DataType *opponent_range, short *eval,
                  short *coll_vec, DataType bet, Vector *scratch,
                  DataType *temp) {
    __syncthreads();
    DataType *result = (DataType *) scratch;
    DataType *sorted_range = (DataType *) (scratch + 1);
    DataType *sorted_eval = (DataType *) (scratch + 2);

    // Setup
    int i = threadIdx.x;
    // Sort hands by eval
    for (int b = 0; b < ITERS; b++) {
        int index = i + TPB * b;
        if (index < 1326) {
            // reset values
            sorted_eval[index] = 0.0f;
            result[index] = 0.0f;
            sorted_range[index] = opponent_range[eval[index] & 2047];
        }
        if (index == 1326) {
            sorted_range[index] = 0.0f;
        }
    }

    // Handle card collisions
    handle_collisions(coll_vec, sorted_range, sorted_eval);

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


__device__ void
evaluate_fold(Vector *opponent_range, short *card_indexes, DataType bet, Vector *result,
              DataType *temp) {
    __syncthreads();
    // Setup
    int i = threadIdx.x;
    copy(opponent_range, result);


    DataType total = reduce_sum(opponent_range->values, temp);

    __syncthreads();
    temp[i] = 0;
    DataType card_sum = 0.0f;
    if (i < 52) {
        for (int c = 0; c < 26; c++) {
            short index = card_indexes[i * 51 + c];
            card_sum += opponent_range->values[index];
        }
        atomicAdd(&temp[i], card_sum);
    } else if ((i >= 64) && (i < (64 + 52))) {
        for (int c = 26; c < 51; c++) {
            short index = card_indexes[(i - 64) * 51 + c];
            card_sum += opponent_range->values[index];
        }
        atomicAdd(&temp[i - 64], card_sum);
    }
    __syncthreads();
    for (int b = 0; b < ITERS; b++) {
        int index = i + TPB * b;
        if (index < 1326) {
            long cards = from_index(index);
            int card1 = __ffsll(cards) - 1;
            cards -= 1l << card1;
            int card2 = __ffsll(cards) - 1;
            result->values[index] -= temp[card1] + temp[card2];
        }
    }
    __syncthreads();
    for (int b = 0; b < ITERS; b++) {
        int index = i + TPB * b;
        if (index < 1326) {
            result->values[index] += total;
            result->values[index] *= bet;
        }
    }
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

__device__ void handle_node(Vector *scratch, Context *contexts, short updating_player, bool calc_exploit, int *depth,
                            long communal_cards, short *abstractions) {
    int tid = threadIdx.x;
    Context *context = &contexts[*depth];
    State *state = context->state;
    int transitions = state->transitions;
    Vector *opponent_range = context->opponent_range;
    Vector *average_strategy = scratch;
    scratch += 1;
    Vector *action_probs = scratch;
    scratch += transitions; // + transitions
    Vector *results = scratch;
    scratch += transitions; // + state-> transitions
    if (context->transition == 0) {
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
        if (__popcll(communal_cards) < 5) {
            get_strategy(state, scratch, action_probs);
        } else {
            get_strategy_abstract(state, scratch, action_probs, abstractions);
        }
    } else {
        int i = context->transition - 1;
        Vector *new_result = average_strategy + 10;
        copy(new_result, results + i);
        if (updating_player == state->next_to_act) {
            if (!calc_exploit) {
                fma(results + i, action_probs + i, average_strategy);
            } else {
                for (int b = 0; b < ITERS; b++) {
                    int index = tid + TPB * b;
                    if (index < 1326) {
                        average_strategy->values[index] = max(average_strategy->values[index],
                                                              (results + i)->values[index]);
                    }
                }
            }
        } else {
            add_assign(average_strategy, results + i);
        }
    }

    if (context->transition == transitions) {
        if ((state->next_to_act == updating_player) && !calc_exploit) {
            for (int i = 0; i < transitions; i++) {
                Vector *util = results + i;
                sub_assign(util, average_strategy);
            }
            if (__popcll(communal_cards) < 5) {
                update_strategy(state, results);
            } else {
                update_strategy_abstract(state, results, abstractions);
            }
        }
        (*depth)--;
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
        contexts[*depth + 1] = {next, new_range, 0};
        context->transition++;
        (*depth)++;
    }
}

__device__ void evaluate_river(Vector *opponent_range_root,
                               State *root_state,
                               Evaluator *evaluator,
                               long communal_cards,
                               short *card_indexes,
                               short *eval,
                               short *coll_vec,
                               Player updating_player,
                               bool calc_exploit,
                               Vector *scratch_root,
                               DataType *temp,
                               short *abstractions) {
    int tid = threadIdx.x;
    __shared__ short local_abstractions[1326];
    for (int b = 0; b < ITERS; b++) {
        int index = tid + 128 * b;
        if (index < 1326) {
            local_abstractions[index] = abstractions[index];
        }
    }
    __syncthreads();
    Context contexts[7];
    contexts[0] = {root_state, opponent_range_root, 0};
    int depth = 0;
    while (depth >= 0) {
        Vector *scratch = scratch_root + depth * 10;
        Context *context = &contexts[depth];
        State *state = context->state;
        Vector *opponent_range = context->opponent_range;

        switch (state->terminal) {
            case Showdown :
                evaluate_showdown(opponent_range->values, eval,
                                  coll_vec, state->sbbet, scratch,
                                  temp);
                depth--;
                break;
            case SBWins :
                evaluate_fold(opponent_range,
                              card_indexes, updating_player == 1 ? -state->bbbet : state->bbbet, scratch,
                              temp);
                depth--;
                break;
            case BBWins :
                evaluate_fold(opponent_range,
                              card_indexes,
                              updating_player == 0 ? -state->sbbet : state->sbbet, scratch, temp);
                depth--;
                break;
            case NonTerminal : {
                handle_node(scratch, contexts, updating_player, calc_exploit, &depth, communal_cards,
                            local_abstractions);
                break;
            }
        }
    }
}

__device__ void evaluate_turn(Vector *opponent_range_root,
                              State *root_state,
                              Evaluator *evaluator,
                              Player updating_player,
                              bool calc_exploit,
                              Vector *scratch_root) {
    __shared__ DataType temp[128];
    // Remove possibility of impossible hands
    remove_collisions(opponent_range_root, root_state->cards);
    Context contexts[7];
    contexts[0] = {root_state, opponent_range_root, 0};
    int depth = 0;

    while (depth >= 0) {
        Vector *scratch = scratch_root + depth * 10;
        Context *context = &contexts[depth];
        State *state = context->state;
        Vector *opponent_range = context->opponent_range;

        switch (state->terminal) {
            case SBWins :
                evaluate_fold(opponent_range,
                              evaluator->card_indexes, updating_player == 1 ? -state->bbbet : state->bbbet, scratch,
                              temp);
                depth--;
                break;
            case BBWins :
                evaluate_fold(opponent_range,
                              evaluator->card_indexes,
                              updating_player == 0 ? -state->sbbet : state->sbbet, scratch, temp);
                depth--;
                break;
            case NonTerminal : {
                handle_node(scratch, contexts, updating_player, calc_exploit, &depth, state->cards, nullptr);
                break;
            }
            case River:
                Vector *result = scratch;
                scratch += 1;
                zero(result);
                State *next = state->next_states[0];
                for (int c = 0; c < 52; c++) {
                    long river = 1l << c;
                    if (river & state->cards) continue;
                    Vector *new_range = scratch;
                    copy(opponent_range, new_range);
                    remove_collisions(new_range, river);
                    long set = (state->cards | river) ^ evaluator->flop;
                    int eval_index = get_index(set);
                    short *eval = evaluator->eval + eval_index * (1326 + 128 * 2);
                    short *coll_vec = evaluator->coll_vec + eval_index * 52 * 51;
                    evaluate_river(new_range, next, evaluator, state->cards | river, evaluator->card_indexes, eval,
                                   coll_vec,
                                   updating_player,
                                   calc_exploit, scratch_root + (depth + 1) * 10, temp,
                                   evaluator->abstractions + eval_index * 1326);
                    remove_collisions(result + 10, river);
                    add_assign(result, result + 10);
                }
                divide(result, 48.0f);
                depth--;
                break;
        }
    }
    // Remove utility of impossible hands
    remove_collisions(scratch_root, root_state->cards);
}

__global__ void evaluate_all(Vector *root_scratch, Vector *opponent_ranges, State *root_states, Evaluator *evaluator,
                             Player updating_player, bool calc_exploit) {
    int block = blockIdx.x;
    Vector *scratch = root_scratch + 10 * 14 * block;
    Vector *opponent_range = opponent_ranges + block / 49;
    State *states = root_states + 270 * block;
    evaluate_turn(opponent_range, states, evaluator, updating_player, calc_exploit, scratch);
}

__global__ void aggregate(Vector *root_scratch, Vector *root_result) {
    int i = threadIdx.x;
    int block = blockIdx.x;
    Vector *scratch = root_scratch + block * 10 * 14 * 49;
    Vector *result = root_result + block;
    for (int i = 1; i < 49; i++) {
        add_assign(result, scratch + 10 * 14 * i);
    }
    for (int b = 0; b < ITERS; b++) {
        int index = i + TPB * b;
        if (index < 1326) {
            result->values[index] /= 49.0f;
        }
    }
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

extern "C" {

void evaluate_cuda(Builder *builder,
                   Evaluator *evaluator,
                   short updating_player,
                   bool calc_exploit) {
    cudaError_t err;
    Vector *device_scratch;
    size_t scratch_size = sizeof(Vector) * 10 * 14 * 49 * 63;
    cudaMalloc(&device_scratch, scratch_size);
    cudaMemset(device_scratch, 0, scratch_size);
    cudaMemcpy(builder->opponent_ranges, builder->communication, 63 * sizeof(Vector), cudaMemcpyHostToDevice);
    cudaMemset(builder->results, 0, 63 * sizeof(Vector));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Setup error: %s\n", cudaGetErrorString(err));
        fflush(stdout);
    }
    evaluate_all<<<49 * 63, 128>>>(device_scratch, builder->opponent_ranges, builder->states, evaluator,
                                   updating_player == 0 ? Small : Big, calc_exploit);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Main execution error: %s\n", cudaGetErrorString(err));
        fflush(stdout);
    }
    aggregate<<<63, 128>>>(device_scratch, builder->results);
    cudaMemcpy(builder->communication, builder->results, 63 * sizeof(Vector), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Aggregation error: %s\n", cudaGetErrorString(err));
        fflush(stdout);
    }
    cudaFree(device_scratch);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Execution error: %s\n", cudaGetErrorString(err));
        fflush(stdout);
    }
}
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