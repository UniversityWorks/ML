#include <stdio.h>
#include "perceptron.h"

int main(int argc, char **argv)
{
  Node     data[DATA_SIZE];
  double   weight[WEIGHT_SIZE];

  initialize_data(data,weight);

  int step=0, func_res;
  while(step < DATA_SIZE)
  {

    print_data(&data[step], weight, step);
    func_res = activation_func(data, weight);
    printf("\nActivation function result: %d\n", func_res);
    if(func_res == data[step].expected_value_)
    {
      printf("\nstep %d. Done!\ndata:\n", step+1);
      step++; 
    }    
    else 
    {
        printf("\nstep %d. Unsatisfactory result. fixing our weight:\n", step+1);
        weight_updating(&data[step], weight, func_res);   
    }  
    
    
  } 

    return 0;
}

int activation_func(Node* node, double weight[])
{

  double res = node->x_ * weight[0] + node->y_ * weight[1] + BIAS * weight[2];
  printf("\nActivation function result:%lf.\n",res);
  return ((res >= 0) ? 1 : -1);
}

void weight_updating(Node* node, double *weight, int func_res)
{ 
  weight[0] = weight[0] + LEARNING_RATE * (node->expected_value_ - func_res)*node->x_; 
  weight[1] = weight[1] + LEARNING_RATE * (node->expected_value_ - func_res)*node->y_; 
  weight[2] = weight[2] + LEARNING_RATE * (node->expected_value_ - func_res)*BIAS; 
  
}

void initialize_data(Node* data, double* weight)
{
  int step;
  // INITIALIZING NODE
  printf("INITIALIZING DATA FROM USER..\n");
  for(step = 0; step < DATA_SIZE; step++)
  {
    printf("%d. Enter data (x1, x2, expected_value): ", step + 1);
    scanf("%lf %lf %d", &data[step].x_, &data[step].y_, &data[step].expected_value_);
  }
  // INITIALIZING WEIGHT
    printf("Enter weight(w1,w2,w3): ");
    scanf("%lf %lf %lf", &weight[0], &weight[1], &weight[2]);
}

void print_data(Node* data, double* weight, int step)
{
  printf("\npoint %d:\nx1 = %lf\nx2 = %lf\nexpected_value = %d\n", step+1, data->x_, data->y_, data->expected_value_);

  printf("\nCurrent weight: {%lf, %lf, %lf}", weight[0], weight[1], weight[2]);
}
