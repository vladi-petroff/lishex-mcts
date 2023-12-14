#ifndef SGD_H_
#define SGD_H_

#include <string>
#include <vector>

#include "types.h"

typedef struct {
    std::string fen;
    double result;
    double error;
} datapoint_t;

// TODO: Multiple threads can handle separate batches
typedef struct {
    std::vector<double> errors;
    std::vector<datapoint_t> datapoints;
} batch_t;


typedef struct {
    std::string name;
    int *value;
} param_t;

void tune();

double winning_prob(int score);
int centipawn_from_prob(double p);

#endif // SGD_H_
