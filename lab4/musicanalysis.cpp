#include "musicanalysis.h"
#include <cmath>
#include <algorithm>

MusicAnalysis::MusicAnalysis(int numInputs, int numHidden1, int numHidden2, int numOutputs, double learningRate)
    : numInputs(numInputs), numHidden1(numHidden1), numHidden2(numHidden2),
    numOutputs(numOutputs), learningRate(learningRate) {

    hiddenLayer1 = Layer(numHidden1, numInputs);
    hiddenLayer2 = Layer(numHidden2, numHidden1);
    outputLayer = Layer(numOutputs, numHidden2);
}

std::vector<double> MusicAnalysis::feedForward(const std::vector<double>& inputs) {

    auto hidden1Outputs = hiddenLayer1.calculateOutputs(inputs);
    auto hidden2Outputs = hiddenLayer2.calculateOutputs(hidden1Outputs);
    auto finalOutputs = outputLayer.calculateOutputs(hidden2Outputs);

    return finalOutputs;
}

void MusicAnalysis::backpropagate(const std::vector<double>& targetOutputs) {

    // Δw_ij = c(d_j - O_j) * f'(net_j) * x_i
    for (int j = 0; j < numOutputs; ++j) {
        Node& outputNode = outputLayer.getNode(j);
        double output = outputNode.getOutput();
        double error = targetOutputs[j] - output;

        // δ = error * f'(net) = error * O_j * (1 - O_j)
        double delta = error * output * (1.0 - output);
        outputNode.setDelta(delta);
    }


    // Δw_ki = c * λ^2 * O_i * (1 - O_i) * x_k * Σ((d_j - O_j) * O_j * (1 - O_j) * w_ij)
    for (int i = 0; i < numHidden2; ++i) {
        Node& hiddenNode = hiddenLayer2.getNode(i);
        double output = hiddenNode.getOutput();

        double errorSum = 0.0;
        for (int j = 0; j < numOutputs; ++j) {
            const Node& outputNode = outputLayer.getNode(j);
            errorSum += outputNode.getDelta() * outputNode.getWeight(i);
        }

        double delta = output * (1.0 - output) * errorSum;
        hiddenNode.setDelta(delta);
    }


    for (int i = 0; i < numHidden1; ++i) {
        Node& hiddenNode = hiddenLayer1.getNode(i);
        double output = hiddenNode.getOutput();

        double errorSum = 0.0;
        for (int j = 0; j < numHidden2; ++j) {
            const Node& hidden2Node = hiddenLayer2.getNode(j);
            errorSum += hidden2Node.getDelta() * hidden2Node.getWeight(i);
        }

        double delta = output * (1.0 - output) * errorSum;
        hiddenNode.setDelta(delta);
    }
}

void MusicAnalysis::updateWeights(const std::vector<double>& inputs) {
    auto hidden1Outputs = hiddenLayer1.getOutputs();
    auto hidden2Outputs = hiddenLayer2.getOutputs();

    for (int j = 0; j < numOutputs; ++j) {
        Node& outputNode = outputLayer.getNode(j);
        for (int i = 0; i < numHidden2; ++i) {
            double weightUpdate = learningRate * outputNode.getDelta() * hidden2Outputs[i];
            outputNode.setWeight(i, outputNode.getWeight(i) + weightUpdate);
        }
    }


    for (int j = 0; j < numHidden2; ++j) {
        Node& hiddenNode = hiddenLayer2.getNode(j);
        for (int i = 0; i < numHidden1; ++i) {
            double weightUpdate = learningRate * hiddenNode.getDelta() * hidden1Outputs[i];
            hiddenNode.setWeight(i, hiddenNode.getWeight(i) + weightUpdate);
        }
    }


    for (int j = 0; j < numHidden1; ++j) {
        Node& hiddenNode = hiddenLayer1.getNode(j);
        for (int i = 0; i < numInputs; ++i) {
            double weightUpdate = learningRate * hiddenNode.getDelta() * inputs[i];
            hiddenNode.setWeight(i, hiddenNode.getWeight(i) + weightUpdate);
        }
    }
}

double MusicAnalysis::train(const std::vector<double>& inputs, const std::vector<double>& targetOutputs) {

    auto outputs = feedForward(inputs);

    double totalError = 0.0;
    for (size_t i = 0; i < outputs.size(); ++i) {
        double error = targetOutputs[i] - outputs[i];
        totalError += error * error;
    }
    totalError *= 0.5; // MSE


    backpropagate(targetOutputs);

    updateWeights(inputs);

    return totalError;
}

double MusicAnalysis::trainEpoch(const std::vector<std::vector<double>>& inputs,
                                 const std::vector<std::vector<double>>& targetOutputs) {
    double totalError = 0.0;

    for (size_t i = 0; i < inputs.size(); ++i) {
        totalError += train(inputs[i], targetOutputs[i]);
    }

    return totalError / inputs.size();
}

int MusicAnalysis::classify(const std::vector<double>& inputs) {
    auto outputs = feedForward(inputs);


    int maxIndex = 0;
    double maxValue = outputs[0];

    for (size_t i = 1; i < outputs.size(); ++i) {
        if (outputs[i] > maxValue) {
            maxValue = outputs[i];
            maxIndex = i;
        }
    }

    return maxIndex;
}

double MusicAnalysis::getConfidence(const std::vector<double>& outputs, int classIndex) const {
    if (classIndex < 0 || classIndex >= static_cast<int>(outputs.size())) {
        return 0.0;
    }


    return outputs[classIndex] * 100.0;
}
