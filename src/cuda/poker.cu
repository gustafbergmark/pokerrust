#include <stdio.h>
#include <cuda_runtime.h>

__device__ void p_sum(float* input, int i) {
    int offset = 1;
    for(int d = 16; d > 0; d>>=1) {
        __syncthreads();
        if(i<d) {
            int ai = offset*(2*i+1)-1;
            int bi = offset*(2*i+2)-1;
            input[bi] += input[ai];
        }
        offset *= 2;
    }
    if(i==0) {
        input[31] = 0;
    }
    for(int d = 1; d < 32; d*=2) {
        offset >>= 1;
        __syncthreads();
        if(i<d) {
            int ai = offset*(2*i+1)-1;
            int bi = offset*(2*i+2)-1;
            float t = input[ai];
            input[ai] = input[bi];
            input[bi] += t;
        }
    }
    __syncthreads();
}

__global__ void gpu_prefix_sum(float* input) {
    __shared__ float temp[32];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for(int b = 0; b < 42; b++) {
        if(i*42+b < 1326 && i < 31) {
            temp[i] += input[i*42 + b];
        }
    }

    __syncthreads();
    p_sum(temp, i);

    float prefix = temp[i];
    for(int b = 0; b < 42; b++) {
        if(i*42+b < 1326) {
            float temp = input[i*42+b];
            input[i*42+b] = prefix;
            prefix += temp;
        }
    }
}

__device__ void cuda_prefix_sum(float* input) {
    __shared__ float temp[32];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    temp[i] = 0;
    for(int b = 0; b < 42; b++) {
        if(i*42+b < 1326 && i < 31) {
            temp[i] += input[i*42 + b];
        }
    }
    __syncthreads();
    p_sum(temp, i);

    float prefix = temp[i];
    for(int b = 0; b < 42; b++) {
        if(i*42+b < 1326) {
            float t = input[i*42+b];
            input[i*42+b] = prefix;
            prefix += t;
        }
    }
}

// returns the largest value smaller than or equal to val in list
__device__ short lower_bound(short val, short* list, int groups_size) {
    int u = groups_size;
    int l = 0;
    while (true) {
        if (u == l) {
            return list[l];
        } else if (l+1 == u) {
            if (list[u] <= val){
                return list[u];
            } else {
                return list[l];
            }
        }
        int m = (l+u) / 2;
        if (val > list[m]) {
            l = m;
        } else {
            u = m;
        }
    }
}

// returns the smallest value larger than val in list
__device__ short upper_bound(short val, short* list, int groups_size) {
    int u = groups_size;
    int l = 0;
    while (true) {
        if (u == l) {
            return list[l];
        } else if (l+1 == u) {
            if (list[l] > val){
                return list[l];
            } else {
                return list[u];
            }
        }
        int m = (l+u) / 2;
        if (val < list[m]) {
            u = m;
        } else {
            l = m;
        }
    }
}

__global__ void evaluate_showdown_kernel(float* opponent_range, long communal_cards, long* card_order, short* eval, short* groups, int groups_size, float* result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float sorted_range[1327];
    __shared__ float sorted_eval[1326];
    for(int b = 0; b < 42; b++) {
        int index = i*42+b;
        if(index < 1326) {
            // reset value?
            sorted_range[index] = 0;
            // Impossible hand since overlap with communal cards
            if ((communal_cards & card_order[eval[index]]) > 0) continue;
            sorted_range[index] = opponent_range[eval[index]];

        }
    }
    __syncthreads();

    cuda_prefix_sum(sorted_range);
    if (i == 0) {
        sorted_range[1326] = sorted_range[1325] + opponent_range[eval[1325]];
    }
    __syncthreads();

    for(int b = 0; b < 42; b++) {
        int index = i*42+b;
        if(index < 1326) {
            // reset value?
            sorted_eval[index] = 0;
            // Impossible hand since overlap with communal cards
            if ((communal_cards & card_order[eval[index]]) > 0) continue;
            int prev_group = lower_bound(index, groups, groups_size);
            float worse = sorted_range[prev_group];
            int next_group = upper_bound(index, groups, groups_size);
            float better = sorted_range[1326] - sorted_range[next_group];
            sorted_eval[index] = worse-better;
        }
    }
    __syncthreads();
    for(int b = 0; b < 42; b++) {
        int index = i*42+b;
        if(index < 1326) {
            result[eval[index]] = sorted_eval[index];
        }
    }
}


extern "C" {
    void prefix_sum_cuda(float *v) {
        size_t input_size = 1326*sizeof(float);
        float* deviceInput;
        cudaMalloc(&deviceInput, input_size);
        cudaMemcpy(deviceInput, v, input_size, cudaMemcpyHostToDevice);
        gpu_prefix_sum<<<1,32>>>(deviceInput);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString(err));
        cudaDeviceSynchronize();
        cudaMemcpy(v, deviceInput, input_size, cudaMemcpyDeviceToHost);
        cudaFree(deviceInput);
    }

    void evaluate_showdown_cuda(float* opponent_range, long communal_cards, long* card_order, short* eval, short* groups, int groups_size, float* result) {
        float* device_opponent_range;
        cudaMalloc(&device_opponent_range, 1326*sizeof(float));
        cudaMemcpy(device_opponent_range, opponent_range, 1326*sizeof(float), cudaMemcpyHostToDevice);

        float* device_result;
        cudaMalloc(&device_result, 1326*sizeof(float));

        long* device_card_order;
        cudaMalloc(&device_card_order, 1326*sizeof(long));
        cudaMemcpy(device_card_order, card_order, 1326*sizeof(long), cudaMemcpyHostToDevice);

        short* device_eval;
        cudaMalloc(&device_eval, 1326*sizeof(short));
        cudaMemcpy(device_eval, eval, 1326*sizeof(short), cudaMemcpyHostToDevice);

        short* device_groups;
        cudaMalloc(&device_groups, groups_size*sizeof(short));
        cudaMemcpy(device_groups, groups, groups_size*sizeof(short), cudaMemcpyHostToDevice);

        evaluate_showdown_kernel<<<1,32>>>(device_opponent_range, communal_cards, device_card_order, device_eval, device_groups, groups_size, device_result);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString(err));
        cudaMemcpy(result, device_result, 1326*sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(device_opponent_range);
        cudaFree(device_result);
        cudaFree(device_card_order);
        cudaFree(device_groups);
        cudaFree(device_eval);
    }
}


/*int main(int argc, char **argv) {
    size_t input_size = 1326*sizeof(float);
    float* input = (float*) malloc(input_size);
    for(int i = 0; i < 1326; i++) {
        input[i] = 1.0;
    }
    prefix_sum(input);
}*/