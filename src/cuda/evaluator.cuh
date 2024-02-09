//
// Created by gustaf on 2024-01-22.
//

#ifndef POKERRUST_EVALUATOR_CUH
#define POKERRUST_EVALUATOR_CUH

struct Evaluator {
    long flop;
    long card_order[1326];
    short card_indexes[52*51];
    short eval[(1326 + 256*2) * 1326];
    short coll_vec[(52*51) * 1326];
    short abstractions[1326*1326];
};

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

#endif //POKERRUST_EVALUATOR_CUH
