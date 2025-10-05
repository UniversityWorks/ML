#include <stdio.h>
#include <stdlib.h>
#define   DATA_SIZE  9

typedef struct 
{
  double x_;
  double y_;

} Data;

double sum_x(Data* data);
double sum_y(Data* data);
double mul_sum_xy(Data* data);
double power_sum_x(Data* data);

double k(Data* dt);
double b(Data* dt);
void initialize_data(Data* dt);

void print_data(Data* dt, double k, double b);
