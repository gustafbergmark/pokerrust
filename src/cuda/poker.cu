#include <stdio.h>
#include <cuda_runtime.h>

__device__ void p_sum(float *input, int i) {
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
            float t = input[ai];
            input[ai] = input[bi];
            input[bi] += t;
        }
    }
    __syncthreads();
}

__global__ void gpu_prefix_sum(float *input) {
    __shared__ float temp[128];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    temp[i] = 0;
    for (int b = 0; b < 11; b++) {
        if (i * 11 + b < 1326 && i < 127) {
            temp[i] += input[i * 11 + b];
        }
    }

    __syncthreads();
    p_sum(temp, i);

    float prefix = temp[i];
    for (int b = 0; b < 11; b++) {
        if (i * 11 + b < 1326) {
            float temp = input[i * 11 + b];
            input[i * 11 + b] = prefix;
            prefix += temp;
        }
    }
}

__device__ void cuda_prefix_sum(float *input) {
    __shared__ float temp[128];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    temp[i] = 0;
    for (int b = 0; b < 11; b++) {
        if (i * 11 + b < 1326 && i < 127) {
            temp[i] += input[i * 11 + b];
        }
    }
    __syncthreads();
    p_sum(temp, i);

    float prefix = temp[i];
    for (int b = 0; b < 11; b++) {
        if (i * 11 + b < 1326) {
            float t = input[i * 11 + b];
            input[i * 11 + b] = prefix;
            prefix += t;
        }
    }
}

__device__ void
handle_collisions(int i, long communal_cards, long *card_order, short *eval, short *coll_vec,
                  float *sorted_range, float *sorted_eval) {
    // Handle collisions before prefix sum consumes sorted_range
    // First two warps handles forward direction
    if (i < 52) {
        int offset = i * 51;
        float sum = 0.0f;
        float group_sum = 0.0f;
        for (int c = 0; c < 51; c++) {
            int index = coll_vec[offset + c];
            // Skip impossible hands, unnecessary here, but consistent
            if ((communal_cards & card_order[eval[index & 2047] & 2047]) > 0) continue;
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
        float sum = 0.0f;
        float group_sum = 0.0f;
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
                group_sum = 0.0f;
            }
        }
    }
    __syncthreads();
}

__global__ void
evaluate_showdown_kernel(float *opponent_range, long communal_cards, long *card_order, short *eval,
                         short *coll_vec, float bet, float *result) {
    // Setup
    __shared__ float sorted_range[1327];
    __shared__ float sorted_eval[1326];
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Sort hands by eval
    for (int b = 0; b < 11; b++) {
        int index = i * 11 + b;
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
    __syncthreads();

    // Handle card collisions
    handle_collisions(i, communal_cards, card_order, eval, coll_vec, sorted_range, sorted_eval);

    __syncthreads();

    // Calculate prefix sum
    cuda_prefix_sum(sorted_range);
    __syncthreads();
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
            float worse = sorted_range[prev_group];
            sorted_eval[index] += worse;
        }
    }

    int next_group = eval[1326 + 128 + i];
    for (int b = 10; b >= 0; b--) {
        int index = i * 11 + b;
        if (index < 1326) {
            // Impossible hand since overlap with communal cards
            if ((communal_cards & card_order[eval[index] & 2047]) > 0) continue;
            float better = sorted_range[1326] - sorted_range[next_group];
            sorted_eval[index] -= better;
            // Observe reverse order because of reverse iteration
            if (eval[index] & 2048) { next_group = index; }
        }
    }

    // Write result
    __syncthreads();
    for (int b = 0; b < 11; b++) {
        int index = i * 11 + b;
        if (index < 1326) {
            result[eval[index] & 2047] = sorted_eval[index] * bet;
        }
    }
    __syncthreads();
}


extern "C" {
void prefix_sum_cuda(float *v) {
    size_t input_size = 1326 * sizeof(float);
    float *deviceInput;
    cudaMalloc(&deviceInput, input_size);
    cudaMemcpy(deviceInput, v, input_size, cudaMemcpyHostToDevice);
    gpu_prefix_sum<<<1, 128>>>(deviceInput);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
    cudaMemcpy(v, deviceInput, input_size, cudaMemcpyDeviceToHost);
    cudaFree(deviceInput);
}

void evaluate_showdown_cuda(float *opponent_range, long communal_cards, long *card_order, short *eval,
                            short *coll_vec, float bet, float *result) {
    float *device_opponent_range;
    cudaMalloc(&device_opponent_range, 1326 * sizeof(float));
    cudaMemcpy(device_opponent_range, opponent_range, 1326 * sizeof(float), cudaMemcpyHostToDevice);

    float *device_result;
    cudaMalloc(&device_result, 1326 * sizeof(float));

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
                                         device_coll_vec, bet, device_result);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));    cudaDeviceSynchronize();
    cudaMemcpy(result, device_result, 1326 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_opponent_range);
    cudaFree(device_result);
    cudaFree(device_card_order);
    cudaFree(device_eval);
    cudaFree(device_coll_vec);
}
}


//int main(int argc, char **argv) {
//    size_t input_size = 1326*sizeof(float);
//    float* input = (float*) malloc(input_size);
//    for(int i = 0; i < 1326; i++) {
//        input[i] = 1.0;
//    }
//    prefix_sum_cuda(input);
//    for(int i = 0; i < 1326; i++) {
//        printf("%d %f\n", i, input[i]);
//    }
//}