/**
 * @file neuralnetwork.cpp
 * Implementation of neural network class
 */

#include "neuralnetwork.h"
#include <cmath>
#include <algorithm>
#include <limits>

NeuralNetwork::NeuralNetwork() {
    inputSize = 20;
    outputSize = 4;
    learningRate = 0.5; // Increased for normalized data

    isNormalized = false;
    featureMean.resize(inputSize, 0.0);
    featureStd.resize(inputSize, 1.0);

    // Initialize weights with small random values (Xavier initialization scaled down)
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    double scale = 0.01; // Small initialization for stability
    std::uniform_real_distribution<> dis(-scale, scale);

    W.resize(inputSize);
    for (int i = 0; i < inputSize; i++) {
        W[i].resize(outputSize);
        for (int j = 0; j < outputSize; j++) {
            W[i][j] = dis(gen);
        }
    }

    z.resize(outputSize);
    y.resize(outputSize);
}

std::vector<double> NeuralNetwork::normalizeInput(const std::vector<double>& x) {
    if (!isNormalized) return x;

    std::vector<double> normalized(inputSize);
    for (int i = 0; i < inputSize; i++) {
        if (featureStd[i] > 1e-10) {
            normalized[i] = (x[i] - featureMean[i]) / featureStd[i];
        } else {
            normalized[i] = 0.0;
        }
    }
    return normalized;
}

void NeuralNetwork::computeNormalization(const std::vector<std::vector<double>>& X) {
    int n = X.size();
    if (n == 0) return;

    // Compute mean
    for (int i = 0; i < inputSize; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += X[j][i];
        }
        featureMean[i] = sum / n;
    }

    // Compute std
    for (int i = 0; i < inputSize; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            double diff = X[j][i] - featureMean[i];
            sum += diff * diff;
        }
        featureStd[i] = std::sqrt(sum / n);
        if (featureStd[i] < 1e-10) {
            featureStd[i] = 1.0; // Avoid division by zero
        }
    }

    isNormalized = true;
}

std::vector<double> NeuralNetwork::forward(const std::vector<double>& x) {
    // Normalize input
    std::vector<double> x_norm = normalizeInput(x);

    // Compute z = W^T * x
    for (int i = 0; i < outputSize; i++) {
        z[i] = 0.0;
        for (int k = 0; k < inputSize; k++) {
            z[i] += W[k][i] * x_norm[k];
        }
    }

    // Compute softmax with numerical stability: y_i = e^(z_i - max(z)) / sum(e^(z_j - max(z)))
    // Find max z for numerical stability
    double max_z = z[0];
    for (int i = 1; i < outputSize; i++) {
        if (z[i] > max_z) {
            max_z = z[i];
        }
    }

    // Compute exp(z - max_z)
    double sum_exp = 0.0;
    std::vector<double> exp_z(outputSize);

    for (int i = 0; i < outputSize; i++) {
        exp_z[i] = std::exp(z[i] - max_z);
        sum_exp += exp_z[i];
    }

    // Normalize
    for (int i = 0; i < outputSize; i++) {
        y[i] = exp_z[i] / sum_exp;

        // Safety check for NaN
        if (std::isnan(y[i]) || std::isinf(y[i])) {
            y[i] = 1.0 / outputSize; // Fallback to uniform distribution
        }
    }

    return y;
}

void NeuralNetwork::backward(const std::vector<double>& x, const std::vector<double>& target) {
    // Normalize input
    std::vector<double> x_norm = normalizeInput(x);

    // Compute gradient and update weights
    // ΔW_ki = c * (d_i - y_i) * x_k

    for (int k = 0; k < inputSize; k++) {
        for (int i = 0; i < outputSize; i++) {
            double gradient = (target[i] - y[i]) * x_norm[k];
            W[k][i] += learningRate * gradient;
        }
    }
}

double NeuralNetwork::computeLoss(const std::vector<double>& target) {
    // Cross-entropy loss: L = -sum(d_j * ln(y_j))
    double loss = 0.0;

    for (int j = 0; j < outputSize; j++) {
        if (target[j] > 0) {
            loss -= target[j] * std::log(y[j] + 1e-10); // Add small epsilon to avoid log(0)
        }
    }

    return loss;
}

void NeuralNetwork::trainEpoch(const std::vector<std::vector<double>>& X,
                               const std::vector<std::vector<double>>& targets) {
    for (size_t i = 0; i < X.size(); i++) {
        forward(X[i]);
        backward(X[i], targets[i]);
    }
}

int NeuralNetwork::getPredictedClass() const {
    int maxIdx = 0;
    double maxVal = y[0];

    for (int i = 1; i < outputSize; i++) {
        if (y[i] > maxVal) {
            maxVal = y[i];
            maxIdx = i;
        }
    }

    return maxIdx;
}
