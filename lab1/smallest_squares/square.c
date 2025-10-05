#include "square.h"

 

int main(int argc, char **argv)
{
  Data data[DATA_SIZE];
  initialize_data(data);
  double k_ = k(data);
  double b_ = b(data);
  
  print_data(data, k_, b_);


  return 0;
}


void print_data(Data* dt, double k, double b)
{
  int step;
  double res;
  printf("\nCALCULATING RESULTS:\n");
  for(step = 0; step < DATA_SIZE; step++)
  {
    res = k * dt[step].x_ + b;
    printf("\npoint #%d: {x;y} = {%lf,%lf}:\n", step + 1, dt[step].x_, dt[step].y_);
    printf("function calculation result for point #%d: %lf\n",step+1, res);
    printf("difference between y: %lf\n", res - dt[step].y_);
  }
}


double k(Data* dt)
{
  return (DATA_SIZE * mul_sum_xy(dt) - sum_x(dt) * sum_y(dt)) / (DATA_SIZE * power_sum_x(dt) - sum_x(dt) * sum_x(dt));
}
double b(Data *dt)
{
  return(sum_y(dt) - k(dt) * sum_x(dt)) / DATA_SIZE;
}

double sum_x(Data* data)
{
  int i;
  double sum=0;
  for(i = 0; i < DATA_SIZE; i++) sum += data[i].x_;
  return sum;
}

double sum_y(Data* data)
{
  int i;
  double sum=0;
  for(i = 0; i < DATA_SIZE; i++) sum += data[i].y_;
  return sum;
}

double mul_sum_xy(Data* data)
{
  int i;
  double sum=0;
  for(i = 0; i < DATA_SIZE; i++) sum += data[i].y_ * data[i].x_;
  return sum;
}


double power_sum_x(Data* data)
{
  int i;
  double sum=0;
  for(i = 0; i < DATA_SIZE; i++) sum += data[i].x_ * data[i].x_;
  return sum;


}

void initialize_data(Data* data)
{
  int step;
  // INITIALIZING DATA
  printf("INITIALIZING DATA FROM USER..\n");
  for(step = 0; step < DATA_SIZE; step++)
  {
    printf("%d. Enter data (x, y): ", step + 1);
    scanf("%lf %lf", &data[step].x_, &data[step].y_);
  }
 }

