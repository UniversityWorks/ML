#ifndef NODE_H
#define NODE_H

#include <vector>

class Node {
private:
    double output;
    double delta;
    std::vector<double> weights;

public:
    Node();
    Node(int numInputs);


    static double sigmoid(double x);


    static double sigmoidDerivative(double x);


    double calculateOutput(const std::vector<double>& inputs);


    double getOutput() const { return output; }
    void setOutput(double value) { output = value; }

    double getDelta() const { return delta; }
    void setDelta(double value) { delta = value; }

    std::vector<double>& getWeights() { return weights; }
    const std::vector<double>& getWeights() const { return weights; }

    void setWeight(int index, double value);
    double getWeight(int index) const;


    void initializeWeights(int numInputs);
};

#endif // NODE_H
