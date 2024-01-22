//
// Created by gustaf on 2024-01-22.
//

#ifndef POKERRUST_EVALUATOR_CUH
#define POKERRUST_EVALUATOR_CUH

struct Evaluator {
    long card_order[1326];
    short card_indexes[52*51];
    short eval[(1326 + 128*2) * 1326];
    short coll_vec[(52*51) * 1326];
};

#endif //POKERRUST_EVALUATOR_CUH
