//
// Created by gustaf on 2024-01-22.
//

#ifndef POKERRUST_EVALUATOR_CUH
#define POKERRUST_EVALUATOR_CUH

struct Evaluator {
    long flop;
    long card_order[1326];
    short card_indexes[52*51];
    short eval[(1326 + 128*2) * 1326];
    short coll_vec[(52*51) * 1326];
    short abstractions[1326*1326];
};

__device__ int get_index(long set);

__device__ long from_index(int index);

#endif //POKERRUST_EVALUATOR_CUH
