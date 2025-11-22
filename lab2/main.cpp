/**
 * @file main.cpp
 *
 * @mainpage Disease Classification using Softmax Neural Network
 *
 * Single-layer neural network with Softmax activation for classifying
 * medical conditions based on 20 clinical features.
 *
 * **Architecture:**
 * - Input: 20 neurons (medical features)
 * - Output: 4 neurons with Softmax (disease classes)
 * - Loss: Cross-Entropy
 * - Optimization: Gradient Descent
 *
 * **Classes:**
 * - 0: Healthy
 * - 1: Diabetes
 * - 2: Cardiovascular disease
 * - 3: Liver problems
 */

#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
