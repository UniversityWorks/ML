/**
 * @file neuralnetwork.h
 * Neural network with Softmax activation for disease classification
 */

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <cmath>
#include <random>

/**
 * @class NeuralNetwork
 * Single-layer neural network with Softmax output for multi-class classification
 */
class NeuralNetwork {
private:
    std::vector<std::vector<double>> W; ///< Матриця ваг розміру 20×4
    std::vector<double> z; ///< Вектор зважених сум (net inputs), розмір 4
    std::vector<double> y; ///< Вектор виходів після Softmax, розмір 4
    double learningRate; ///< Швидкість навчання (learning rate)

    int inputSize;  ///< Кількість вхідних нейронів (20)
    int outputSize; ///< Кількість вихідних нейронів (4)

    // Normalization parameters
    std::vector<double> featureMean; ///< Середні значення ознак для нормалізації
    std::vector<double> featureStd; ///< Стандартні відхилення ознак для нормалізації
    bool isNormalized; ///< Прапорець чи дані нормалізовані

    /**
     * @brief Нормалізує вхідні дані
     * @param x Вектор вхідних ознак
     * @return Нормалізований вектор (mean=0, std=1)
     */
    std::vector<double> normalizeInput(const std::vector<double>& x);

public:
    /** Constructor - initializes network with random weights */
    NeuralNetwork();

    /**
     * Forward pass - computes softmax outputs
     * @param x Input features (20)
     * @return Output probabilities (4)
     */
    std::vector<double> forward(const std::vector<double>& x);

    /**
     * Backward pass - updates weights using gradient descent
     * @param x Input features
     * @param target Target values (one-hot)
     */
    void backward(const std::vector<double>& x, const std::vector<double>& target);

    /**
     * Computes cross-entropy loss
     * @param target Target values (one-hot)
     * @return Loss value
     */
    double computeLoss(const std::vector<double>& target);

    /**
     * Trains network for one epoch on batch of data
     * @param X Input samples
     * @param targets Target values
     */
    void trainEpoch(const std::vector<std::vector<double>>& X,
                    const std::vector<std::vector<double>>& targets);

    /**
     * Computes normalization parameters from training data
     * @param X Training samples
     */
    void computeNormalization(const std::vector<std::vector<double>>& X);

    /** Returns current weights */
    std::vector<std::vector<double>> getWeights() const { return W; }

    /** Returns last computed outputs */
    std::vector<double> getOutputs() const { return y; }

    /** Returns last computed net inputs */
    std::vector<double> getNetInputs() const { return z; }

    /** Returns predicted class index */
    int getPredictedClass() const;
};

#endif // NEURALNETWORK_H
