#ifndef MUSICCLASSIFIER_H
#define MUSICCLASSIFIER_H

#include "musicanalysis.h"
#include "minmaxnormalizer.h"
#include <QString>
#include <vector>
#include <map>

struct MusicFeatures {
    double tempo;
    double rhythm;
    double energy;
    double loudness;
};

enum MusicGenre {
    ROCK = 0,
    JAZZ = 1,
    ELECTRONIC = 2,
    BLUES = 3,
    UNKNOWN = 4
};

class MusicClassifier {
private:
    MusicAnalysis* network;
    MinMaxNormalizer normalizer;
    bool isTrained;

    std::map<int, QString> genreNames;


    void generateTrainingData(std::vector<std::vector<double>>& inputs,
                              std::vector<std::vector<double>>& outputs);


    std::vector<double> featuresToVector(const MusicFeatures& features) const;


    std::vector<double> genreToOneHot(MusicGenre genre) const;

public:
    MusicClassifier();
    ~MusicClassifier();


    void trainNetwork(int epochs = 1000);


    MusicGenre classifyMusic(const MusicFeatures& features, double& confidence);


    QString getGenreName(MusicGenre genre) const;


    bool getIsTrained() const { return isTrained; }


    MusicFeatures analyzeAudioFile(const QString& filePath);
};

#endif // MUSICCLASSIFIER_H
