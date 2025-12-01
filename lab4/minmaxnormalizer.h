#ifndef MINMAXNORMALIZER_H
#define MINMAXNORMALIZER_H

#include <vector>


class MinMaxNormalizer {
private:
    std::vector<double> minValues;
    std::vector<double> maxValues;
    bool isFitted;

public:
    MinMaxNormalizer();

    // Навчання нормалізатора на даних
    void fit(const std::vector<std::vector<double>>& data);

    // Нормалізація вектора: x' = (x - x_min) / (x_max - x_min)
    std::vector<double> transform(const std::vector<double>& data) const;

    // Денормалізація
    std::vector<double> inverseTransform(const std::vector<double>& normalizedData) const;

    // Getters
    bool getIsFitted() const { return isFitted; }
    const std::vector<double>& getMinValues() const { return minValues; }
    const std::vector<double>& getMaxValues() const { return maxValues; }
};

#endif // MINMAXNORMALIZER_H
