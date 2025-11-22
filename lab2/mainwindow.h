/**
 * @file mainwindow.h
 * Main window for disease classification application
 */

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QTableWidget>
#include <QComboBox>
#include <QFileDialog>
#include "neuralnetwork.h"

/** Patient medical data structure */
struct PatientData {
    std::vector<double> features; ///< 20 medical features
    int diagnosis; ///< Diagnosis: 0-Healthy, 1-Diabetes, 2-Cardiovascular, 3-Liver
    QString name; ///< Patient name
};

/**
 * @class MainWindow
 * GUI application for medical diagnosis classification
 */
class MainWindow : public QMainWindow {
    Q_OBJECT

private:
    NeuralNetwork* network; ///< Neural network instance
    std::vector<PatientData> patients; ///< Patient data
    int currentPatientIndex; ///< Current selected patient

    QTableWidget* dataTable; ///< Medical features table
    QComboBox* patientSelector; ///< Patient selector dropdown

    QLabel* outputY0; ///< Probability: Healthy
    QLabel* outputY1; ///< Probability: Diabetes
    QLabel* outputY2; ///< Probability: Cardiovascular
    QLabel* outputY3; ///< Probability: Liver
    QLabel* predictedClass; ///< Predicted class label

    QPushButton* btnClassify; ///< Classify button
    QPushButton* btnTrain; ///< Train button
    QPushButton* btnLoadCSV; ///< Load CSV button

    QTextEdit* trainingLog; ///< Training log text area
    int trainingIteration; ///< Training epoch counter

    QStringList featureNames; ///< Medical feature names
    QStringList classNames; ///< Disease class names

    /** Sets up user interface */
    void setupUI();

    /** Initializes hardcoded patient data */
    void initializeData();

    /** Updates output probability labels */
    void updateOutputs();

    /** Updates data table display */
    void updateDataTable();

    /**
     * Loads patient data into table
     * @param index Patient index
     */
    void loadPatientToTable(int index);

private slots:
    /** Classifies current patient */
    void onClassify();

    /** Trains network for one epoch */
    void onTrain();

    /** Loads patient data from CSV file */
    void onLoadCSV();

    /**
     * Handles patient selection change
     * @param index New patient index
     */
    void onPatientSelected(int index);

public:
    /** Constructor */
    MainWindow(QWidget *parent = nullptr);

    /** Destructor */
    ~MainWindow();
};

#endif // MAINWINDOW_H
