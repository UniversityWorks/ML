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

void print_data(Node* data, double* weight, int step);

