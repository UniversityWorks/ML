#include "minmaxnormalizer.h"
#include <algorithm>
#include <limits>
MinMaxNormalizer::MinMaxNormalizer() : isFitted(false) {}

void MinMaxNormalizer::fit(const std::vector<std::vector<double>>& data) {
    if (data.empty() || data[0].empty()) {
        return;
    }

    size_t numFeatures = data[0].size();
    minValues.resize(numFeatures, std::numeric_limits<double>::max());
    maxValues.resize(numFeatures, std::numeric_limits<double>::lowest());

    // Знаходимо мінімум і максимум для кожної ознаки
    for (const auto& sample : data) {
        for (size_t i = 0; i < numFeatures && i < sample.size(); ++i) {
            minValues[i] = std::min(minValues[i], sample[i]);
            maxValues[i] = std::max(maxValues[i], sample[i]);
        }
    }

    isFitted = true;
}

std::vector<double> MinMaxNormalizer::transform(const std::vector<double>& data) const {
    if (!isFitted || data.size() != minValues.size()) {
        return data;
    }

    std::vector<double> normalized;
    normalized.reserve(data.size());

    for (size_t i = 0; i < data.size(); ++i) {
        double range = maxValues[i] - minValues[i];
        if (range > 0.0001) { // Уникаємо ділення на нуль
            normalized.push_back((data[i] - minValues[i]) / range);
        } else {
            normalized.push_back(0.0);
        }
    }

    return normalized;
}

std::vector<double> MinMaxNormalizer::inverseTransform(const std::vector<double>& normalizedData) const {
    if (!isFitted || normalizedData.size() != minValues.size()) {
        return normalizedData;
    }

    std::vector<double> original;
    original.reserve(normalizedData.size());

    for (size_t i = 0; i < normalizedData.size(); ++i) {
        double range = maxValues[i] - minValues[i];
        original.push_back(normalizedData[i] * range + minValues[i]);
    }

    return original;
}
