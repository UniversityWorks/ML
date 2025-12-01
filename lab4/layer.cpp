#include "layer.h"

Layer::Layer() : numNodes(0) {}

Layer::Layer(int numNodes, int numInputsPerNode) : numNodes(numNodes) {
    nodes.reserve(numNodes);
    for (int i = 0; i < numNodes; ++i) {
        nodes.emplace_back(numInputsPerNode);
    }
}

std::vector<double> Layer::calculateOutputs(const std::vector<double>& inputs) {
    std::vector<double> outputs;
    outputs.reserve(numNodes);

    for (int i = 0; i < numNodes; ++i) {
        outputs.push_back(nodes[i].calculateOutput(inputs));
    }

    return outputs;
}

std::vector<double> Layer::getOutputs() const {
    std::vector<double> outputs;
    outputs.reserve(numNodes);

    for (const auto& node : nodes) {
        outputs.push_back(node.getOutput());
    }

    return outputs;
}
