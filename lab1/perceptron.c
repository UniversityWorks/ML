/*
 *                                          __                        
______   ___________   ____  ____ _______/  |________  ____   ____  
\____ \_/ __ \_  __ \_/ ___\/ __ \\____ \   __\_  __ \/  _ \ /    \ 
|  |_> >  ___/|  | \/\  \__\  ___/|  |_> >  |  |  | \(  <_> )   |  \
|   __/ \___  >__|    \___  >___  >   __/|__|  |__|   \____/|___|  /
|__|        \/            \/    \/|__|                           \/ */




#include <stdio.h>

#define     DATA_SIZE        10
#define     WEIGHT_SIZE      3
#define     LEARNING_RATE    0.15
#define     BIAS             1

typedef struct
{
  double x_;
  double y_;
  short expected_value_;
} Node;


int activation_func(Node* node, double weight[]);

void weight_updating(Node* data, double* weight, int func_res);

void initialize_data(Node* data, double* weight);
 
int main(int argc, char **argv)
{
    Node     data[DATA_SIZE];
    double   weight[WEIGHT_SIZE];
    
    initialize_data(data,weight);
    
    

    return 0;
}





int activation_func(Node* node, double weight[])
{

  double res = node->x_ * weight[0] + node->y_ * weight[1] + BIAS * weight[2];
  return ((res >= 0) ? 1 : 0);
}

void weight_updating(Node* node, double *weight, int func_res)
{ 
  weight[0] = weight[0] + LEARNING_RATE * (node->expected_value_ - func_res)*node->x_; 
  weight[1] = weight[1] + LEARNING_RATE * (node->expected_value_ - func_res)*node->y_; 
  weight[2] = weight[2] + LEARNING_RATE * (node->expected_value_ - func_res)*BIAS; 
  
}

void initialize_data(Node* data, double* weight)
{
  int i;
  // INITIALIZING NODE
  printf("INITIALIZING DATA FROM USER..\n");
  for(i = 0; i < DATA_SIZE; i++)
  {
    printf("%d. Enter data (x1, x2, expected_value): ", i + 1);
    scanf("%lf %lf %d", &data[i].x_, &data[i].y_, &data[i].expected_value_);
  }
  // INITIALIZING WEIGHT
    printf("Enter weight(w1, w2,w3): ");
    scanf("%lf %lf %lf", &weight[0], &weight[1], &weight[2]);
}
