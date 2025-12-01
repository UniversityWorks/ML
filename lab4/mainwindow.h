#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QProgressBar>
#include "musicclassifier.h"

class MainWindow : public QMainWindow {
    Q_OBJECT

private:
    MusicClassifier* classifier;


    QLineEdit* tempoInput;
    QLineEdit* rhythmInput;
    QLineEdit* energyInput;
    QLineEdit* loudnessInput;


    QPushButton* classifyButton;
    QPushButton* trainButton;
    QPushButton* loadAudioButton;


    QLabel* resultLabel;
    QLabel* confidenceLabel;
    QLabel* statusLabel;


    QProgressBar* progressBar;

    void setupUI();
    void connectSignals();

private slots:
    void onClassifyClicked();
    void onTrainClicked();
    void onLoadAudioClicked();

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
};

#endif // MAINWINDOW_H
