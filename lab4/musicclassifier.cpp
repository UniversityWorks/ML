#include "musicclassifier.h"
#include <random>
#include <ctime>
#include <QFileInfo>

MusicClassifier::MusicClassifier() : isTrained(false) {

    network = new MusicAnalysis(4, 10, 10, 4, 0.1);


    genreNames[ROCK] = "Рок";
    genreNames[JAZZ] = "Джаз";
    genreNames[ELECTRONIC] = "Електронна";
    genreNames[BLUES] = "Блюз";
    genreNames[UNKNOWN] = "Невідомо";
}

MusicClassifier::~MusicClassifier() {
    delete network;
}

std::vector<double> MusicClassifier::featuresToVector(const MusicFeatures& features) const {
    return {features.tempo, features.rhythm, features.energy, features.loudness};
}

std::vector<double> MusicClassifier::genreToOneHot(MusicGenre genre) const {
    std::vector<double> oneHot(4, 0.0);
    if (genre >= 0 && genre < 4) {
        oneHot[genre] = 1.0;
    }
    return oneHot;
}

void MusicClassifier::generateTrainingData(std::vector<std::vector<double>>& inputs,
                                           std::vector<std::vector<double>>& outputs) {
    std::mt19937 generator(static_cast<unsigned>(std::time(nullptr)));


    std::uniform_real_distribution<double> rockTempo(120.0, 180.0);
    std::uniform_real_distribution<double> rockRhythm(70.0, 100.0);
    std::uniform_real_distribution<double> rockEnergy(70.0, 100.0);
    std::uniform_real_distribution<double> rockLoudness(70.0, 100.0);


    std::uniform_real_distribution<double> jazzTempo(80.0, 140.0);
    std::uniform_real_distribution<double> jazzRhythm(40.0, 70.0);
    std::uniform_real_distribution<double> jazzEnergy(40.0, 70.0);
    std::uniform_real_distribution<double> jazzLoudness(40.0, 70.0);


    std::uniform_real_distribution<double> electronicTempo(130.0, 180.0);
    std::uniform_real_distribution<double> electronicRhythm(80.0, 100.0);
    std::uniform_real_distribution<double> electronicEnergy(80.0, 100.0);
    std::uniform_real_distribution<double> electronicLoudness(60.0, 90.0);


    std::uniform_real_distribution<double> bluesTempo(60.0, 100.0);
    std::uniform_real_distribution<double> bluesRhythm(30.0, 60.0);
    std::uniform_real_distribution<double> bluesEnergy(30.0, 60.0);
    std::uniform_real_distribution<double> bluesLoudness(40.0, 70.0);

    int samplesPerGenre = 50;


    for (int i = 0; i < samplesPerGenre; ++i) {

        inputs.push_back({rockTempo(generator), rockRhythm(generator),
                          rockEnergy(generator), rockLoudness(generator)});
        outputs.push_back(genreToOneHot(ROCK));


        inputs.push_back({jazzTempo(generator), jazzRhythm(generator),
                          jazzEnergy(generator), jazzLoudness(generator)});
        outputs.push_back(genreToOneHot(JAZZ));


        inputs.push_back({electronicTempo(generator), electronicRhythm(generator),
                          electronicEnergy(generator), electronicLoudness(generator)});
        outputs.push_back(genreToOneHot(ELECTRONIC));


        inputs.push_back({bluesTempo(generator), bluesRhythm(generator),
                          bluesEnergy(generator), bluesLoudness(generator)});
        outputs.push_back(genreToOneHot(BLUES));
    }
}

void MusicClassifier::trainNetwork(int epochs) {
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> outputs;


    generateTrainingData(inputs, outputs);


    normalizer.fit(inputs);

    for (auto& input : inputs) {
        input = normalizer.transform(input);
    }


    for (int epoch = 0; epoch < epochs; ++epoch) {
        double error = network->trainEpoch(inputs, outputs);

    }

    isTrained = true;
}

MusicGenre MusicClassifier::classifyMusic(const MusicFeatures& features, double& confidence) {
    if (!isTrained) {
        confidence = 0.0;
        return UNKNOWN;
    }


    std::vector<double> input = featuresToVector(features);

       input = normalizer.transform(input);

       auto outputs = network->feedForward(input);
    int classIndex = network->classify(input);


    confidence = network->getConfidence(outputs, classIndex);


    if (confidence < 50.0) {
        return UNKNOWN;
    }

    return static_cast<MusicGenre>(classIndex);
}

QString MusicClassifier::getGenreName(MusicGenre genre) const {
    auto it = genreNames.find(genre);
    if (it != genreNames.end()) {
        return it->second;
    }
    return "Невідомо";
}

MusicFeatures MusicClassifier::analyzeAudioFile(const QString& filePath) {

    QFileInfo fileInfo(filePath);
    MusicFeatures features;


    std::mt19937 generator(static_cast<unsigned>(std::time(nullptr)));
    std::uniform_real_distribution<double> tempoDist(60.0, 180.0);
    std::uniform_real_distribution<double> otherDist(0.0, 100.0);

    features.tempo = tempoDist(generator);
    features.rhythm = otherDist(generator);
    features.energy = otherDist(generator);
    features.loudness = otherDist(generator);

    return features;
}
