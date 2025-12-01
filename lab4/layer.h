#ifndef LAYER_H
#define LAYER_H

#include "Node.h"
#include <vector>

class Layer {
private:
    std::vector<Node> nodes;
    int numNodes;

public:
    Layer();
    Layer(int numNodes, int numInputsPerNode);


    std::vector<double> calculateOutputs(const std::vector<double>& inputs);


    std::vector<double> getOutputs() const;


    int getNumNodes() const { return numNodes; }
    Node& getNode(int index) { return nodes[index]; }
    const Node& getNode(int index) const { return nodes[index]; }

    std::vector<Node>& getNodes() { return nodes; }
    const std::vector<Node>& getNodes() const { return nodes; }
};

#endif // LAYER_H
