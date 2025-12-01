#include "Node.h"
#include <random>
#include <ctime>

Node::Node() : output(0.0), delta(0.0) {}

Node::Node(int numInputs) : output(0.0), delta(0.0) {
    initializeWeights(numInputs);
}

double Node::sigmoid(double x) {
    // f(net) = 1 / (1 + e^(-λ*net))

    return 1.0 / (1.0 + std::exp(-x));
}

double Node::sigmoidDerivative(double x) {
    // f'(net) = f(net) * (1 - f(net))
    double sig = sigmoid(x);
    return sig * (1.0 - sig);
}

double Node::calculateOutput(const std::vector<double>& inputs) {
    if (inputs.size() != weights.size()) {
        return 0.0;
    }

    double net = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        net += inputs[i] * weights[i];
    }

    output = sigmoid(net);
    return output;
}

void Node::setWeight(int index, double value) {
    if (index >= 0 && index < static_cast<int>(weights.size())) {
        weights[index] = value;
    }
}

double Node::getWeight(int index) const {
    if (index >= 0 && index < static_cast<int>(weights.size())) {
        return weights[index];
    }
    return 0.0;
}

void Node::initializeWeights(int numInputs) {
    weights.resize(numInputs);

    static std::mt19937 generator(static_cast<unsigned>(std::time(nullptr)));
    std::uniform_real_distribution<double> distribution(-0.5, 0.5);

    for (int i = 0; i < numInputs; ++i) {
        weights[i] = distribution(generator);
    }
}
