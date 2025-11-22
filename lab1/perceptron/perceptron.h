
#ifndef PERCEPTRON_H
#define PERCEPTRON_H

typedef struct
{
    double x1_;
    double x2_;
    int expected_;
} Data;

int activation(double x1, double x2, double w1, double w2, double w3);
void initialize_data(Data *data);
void perceptron_train(Data *data);

#endif
