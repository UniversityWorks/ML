
#include <stdio.h>
#include <stdlib.h>
#include "perceptron.h"

#define DATA_SIZE 10
#define LEARNING_RATE 0.15
#define MAX_EPOCHS 100

int activation(double x1, double x2, double w1, double w2, double w3)
{
    double y = x1 * w1 + x2 * w2 + w3;
    return (y >= 0) ? 1 : -1;
}

void initialize_data(Data *data)
{
    printf("INITIALIZING DATA FROM USER..\n");
    for (int i = 0; i < DATA_SIZE; i++)
    {
        printf("%d. Enter data (x1, x2, expected_value): ", i + 1);
        scanf("%lf %lf %d", &data[i].x1_, &data[i].x2_, &data[i].expected_);
    }
}

void perceptron_train(Data *data)
{
    double w1, w2, w3;
    printf("Enter weight(w1,w2,w3): ");
    scanf("%lf %lf %lf", &w1, &w2, &w3);

    int epoch = 0;
    int all_correct;

    do
    {
        all_correct = 1;
        printf("\n--- EPOCH %d ---\n", epoch + 1);

        for (int i = 0; i < DATA_SIZE; i++)
        {
            int result = activation(data[i].x1_, data[i].x2_, w1, w2, w3);

            if (result != data[i].expected_)
            {
                // update weights
                w1 += LEARNING_RATE * data[i].expected_ * data[i].x1_;
                w2 += LEARNING_RATE * data[i].expected_ * data[i].x2_;
                w3 += LEARNING_RATE * data[i].expected_;

                all_correct = 0; // still need training
                printf("Adjusted weights -> w1=%.3f w2=%.3f w3=%.3f\n", w1, w2, w3);
            }
        }

        epoch++;

    } while (!all_correct && epoch < MAX_EPOCHS);

    printf("\nTraining complete!\n");
    printf("Final weights: w1=%.3f, w2=%.3f, w3=%.3f\n", w1, w2, w3);
}

int main(void)
{
    Data data[DATA_SIZE];
    initialize_data(data);
    perceptron_train(data);
    return 0;
}
