#ifndef MUSICANALYSIS_H
#define MUSICANALYSIS_H

#include "layer.h"
#include <vector>

class MusicAnalysis {
private:
    Layer inputLayer;
    Layer hiddenLayer1;
    Layer hiddenLayer2;
    Layer outputLayer;

    double learningRate;
    int numInputs;
    int numHidden1;
    int numHidden2;
    int numOutputs;


    void backpropagate(const std::vector<double>& targetOutputs);


    void updateWeights(const std::vector<double>& inputs);

public:
    MusicAnalysis(int numInputs, int numHidden1, int numHidden2, int numOutputs, double learningRate = 0.1);


    std::vector<double> feedForward(const std::vector<double>& inputs);


    double train(const std::vector<double>& inputs, const std::vector<double>& targetOutputs);


    double trainEpoch(const std::vector<std::vector<double>>& inputs,
                      const std::vector<std::vector<double>>& targetOutputs);


    int classify(const std::vector<double>& inputs);


    double getConfidence(const std::vector<double>& outputs, int classIndex) const;


    double getLearningRate() const { return learningRate; }
    void setLearningRate(double rate) { learningRate = rate; }
};

#endif // MUSICANALYSIS_H
