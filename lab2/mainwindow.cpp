/**
 * @file mainwindow.cpp
 * Implementation of main window
 */

#include "mainwindow.h"
#include <QMessageBox>
#include <QHeaderView>
#include <QFile>
#include <QTextStream>
#include <QSplitter>
#include <QScrollArea>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    network = new NeuralNetwork();

    // Initialize feature names (20 medical indicators)
    featureNames = {
        "Вік (років)", "Стать (0-ж, 1-ч)", "Систолічний тиск", "Діастолічний тиск",
        "Пульс (уд/хв)", "Глюкоза (ммоль/л)", "Гемоглобін (г/л)", "Холестерин (ммоль/л)",
        "ЛПНЩ (ммоль/л)", "ЛПВЩ (ммоль/л)", "Тригліцериди (ммоль/л)", "ІМТ",
        "Білірубін (мкмоль/л)", "АЛТ (Од/л)", "АСТ (Од/л)", "Креатинін (мкмоль/л)",
        "Сеча (кмоль/л)", "Лейкоцити (10⁹/л)", "Еритроцити (10¹²/л)", "ШОЕ (мм/год)"
    };

    // Class names
    classNames = {"Здоровий", "Діабет", "Серцево-судинні", "Печінкові проблеми"};

    currentPatientIndex = 0;
    trainingIteration = 0;

    initializeData();
    setupUI();

    setWindowTitle("Класифікація захворювань - Softmax нейронна мережа");
    resize(1200, 800);
}

MainWindow::~MainWindow() {
    delete network;
}

void MainWindow::initializeData() {

    patients.push_back({
        {35, 1, 120, 80, 72, 5.0, 145, 4.5, 2.5, 1.5, 1.2, 23.5, 15, 25, 28, 85, 5.5, 6.5, 4.8, 8},
        0, "Пацієнт 1 (Здоровий)"
    });


    patients.push_back({
        {28, 0, 115, 75, 68, 4.8, 135, 4.2, 2.3, 1.6, 1.0, 21.0, 12, 22, 25, 80, 5.2, 6.0, 4.5, 7},
        0, "Пацієнт 2 (Здоровий)"
    });


    patients.push_back({
        {52, 1, 135, 88, 78, 12.5, 140, 5.8, 3.5, 1.2, 2.8, 28.5, 18, 35, 40, 95, 8.5, 7.2, 4.6, 12},
        1, "Пацієнт 3 (Діабет)"
    });

    patients.push_back({
        {48, 0, 140, 90, 82, 13.2, 138, 6.0, 3.8, 1.1, 3.0, 30.0, 20, 38, 42, 98, 9.0, 7.5, 4.7, 14},
        1, "Пацієнт 4 (Діабет)"
    });


    patients.push_back({
        {65, 1, 165, 105, 95, 6.5, 130, 7.2, 4.5, 0.9, 2.5, 27.0, 16, 30, 35, 92, 6.0, 6.8, 4.2, 18},
        2, "Пацієнт 5 (Серцево-судинні)"
    });


    patients.push_back({
        {58, 0, 170, 110, 98, 6.8, 128, 7.5, 4.8, 0.8, 2.7, 29.0, 17, 32, 38, 94, 6.2, 7.0, 4.3, 20},
        2, "Пацієнт 6 (Серцево-судинні)"
    });

    patients.push_back({
        {45, 1, 125, 82, 75, 5.5, 125, 5.0, 2.8, 1.3, 1.8, 25.0, 45, 85, 95, 88, 5.8, 6.5, 4.0, 22},
        3, "Пацієнт 7 (Печінкові)"
    });


    patients.push_back({
        {50, 0, 128, 85, 77, 5.8, 122, 5.2, 2.9, 1.2, 1.9, 26.5, 48, 90, 100, 90, 6.0, 6.7, 4.1, 25},
        3, "Пацієнт 8 (Печінкові)"
    });
}

void MainWindow::setupUI() {
    QWidget* centralWidget = new QWidget(this);
    QHBoxLayout* mainLayout = new QHBoxLayout(centralWidget);


    QVBoxLayout* leftPanel = new QVBoxLayout();

    QGroupBox* dataGroup = new QGroupBox("Медичні дані пацієнта");
    QVBoxLayout* dataLayout = new QVBoxLayout();


    QHBoxLayout* selectorLayout = new QHBoxLayout();
    selectorLayout->addWidget(new QLabel("Вибрати пацієнта:"));
    patientSelector = new QComboBox();
    for (const auto& patient : patients) {
        patientSelector->addItem(patient.name);
    }
    connect(patientSelector, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onPatientSelected);
    selectorLayout->addWidget(patientSelector);
    dataLayout->addLayout(selectorLayout);


    dataTable = new QTableWidget(20, 2);
    dataTable->setHorizontalHeaderLabels({"Показник", "Значення"});
    dataTable->horizontalHeader()->setStretchLastSection(true);
    dataTable->setColumnWidth(0, 250);
    dataTable->setEditTriggers(QAbstractItemView::DoubleClicked);
    dataLayout->addWidget(dataTable);


    btnLoadCSV = new QPushButton("Завантажити CSV");
    btnLoadCSV->setStyleSheet("QPushButton { background-color: #FF9800; color: white; padding: 8px; }");
    connect(btnLoadCSV, &QPushButton::clicked, this, &MainWindow::onLoadCSV);
    dataLayout->addWidget(btnLoadCSV);

    dataGroup->setLayout(dataLayout);
    leftPanel->addWidget(dataGroup);

    QVBoxLayout* rightPanel = new QVBoxLayout();


    QGroupBox* outputGroup = new QGroupBox("Результат класифікації (Softmax Output)");
    QVBoxLayout* outputLayout = new QVBoxLayout();

    QHBoxLayout* y0Layout = new QHBoxLayout();
    y0Layout->addWidget(new QLabel("P(Здоровий):"));
    outputY0 = new QLabel("0.000");
    outputY0->setStyleSheet("QLabel { font-weight: bold; color: green; font-size: 14px; }");
    y0Layout->addWidget(outputY0);
    outputLayout->addLayout(y0Layout);

    QHBoxLayout* y1Layout = new QHBoxLayout();
    y1Layout->addWidget(new QLabel("P(Діабет):"));
    outputY1 = new QLabel("0.000");
    outputY1->setStyleSheet("QLabel { font-weight: bold; color: orange; font-size: 14px; }");
    y1Layout->addWidget(outputY1);
    outputLayout->addLayout(y1Layout);

    QHBoxLayout* y2Layout = new QHBoxLayout();
    y2Layout->addWidget(new QLabel("P(Серцево-судинні):"));
    outputY2 = new QLabel("0.000");
    outputY2->setStyleSheet("QLabel { font-weight: bold; color: red; font-size: 14px; }");
    y2Layout->addWidget(outputY2);
    outputLayout->addLayout(y2Layout);

    QHBoxLayout* y3Layout = new QHBoxLayout();
    y3Layout->addWidget(new QLabel("P(Печінкові):"));
    outputY3 = new QLabel("0.000");
    outputY3->setStyleSheet("QLabel { font-weight: bold; color: purple; font-size: 14px; }");
    y3Layout->addWidget(outputY3);
    outputLayout->addLayout(y3Layout);

    outputLayout->addSpacing(10);
    QHBoxLayout* predLayout = new QHBoxLayout();
    predLayout->addWidget(new QLabel("Прогноз:"));
    predictedClass = new QLabel("---");
    predictedClass->setStyleSheet("QLabel { font-weight: bold; color: blue; font-size: 16px; }");
    predLayout->addWidget(predictedClass);
    outputLayout->addLayout(predLayout);

    outputGroup->setLayout(outputLayout);
    rightPanel->addWidget(outputGroup);


    QHBoxLayout* buttonLayout = new QHBoxLayout();

    btnClassify = new QPushButton("Класифікувати");
    btnClassify->setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 12px; font-size: 14px; }");
    connect(btnClassify, &QPushButton::clicked, this, &MainWindow::onClassify);
    buttonLayout->addWidget(btnClassify);

    btnTrain = new QPushButton("Навчити (1 епоха)");
    btnTrain->setStyleSheet("QPushButton { background-color: #2196F3; color: white; padding: 12px; font-size: 14px; }");
    connect(btnTrain, &QPushButton::clicked, this, &MainWindow::onTrain);
    buttonLayout->addWidget(btnTrain);

    rightPanel->addLayout(buttonLayout);


    QGroupBox* logGroup = new QGroupBox("Журнал навчання");
    QVBoxLayout* logLayout = new QVBoxLayout();

    trainingLog = new QTextEdit();
    trainingLog->setReadOnly(true);
    trainingLog->append("Нейронна мережа ініціалізована:");
    trainingLog->append("- Вхідний шар: 20 нейронів (медичні показники)");
    trainingLog->append("- Вихідний шар: 4 нейрони (Softmax)");
    trainingLog->append("- Класи: 0-Здоровий, 1-Діабет, 2-Серцево-судинні, 3-Печінкові");
    trainingLog->append("- Швидкість навчання: c = 0.1");
    trainingLog->append("- Функція втрат: Cross-Entropy");
    trainingLog->append("- Навчальних зразків: " + QString::number(patients.size()));
    trainingLog->append("========================================\n");

    logLayout->addWidget(trainingLog);
    logGroup->setLayout(logLayout);
    rightPanel->addWidget(logGroup);


    mainLayout->addLayout(leftPanel, 1);
    mainLayout->addLayout(rightPanel, 1);

    setCentralWidget(centralWidget);


    loadPatientToTable(0);
}

void MainWindow::loadPatientToTable(int index) {
    if (index < 0 || index >= patients.size()) return;

    const auto& patient = patients[index];

    for (int i = 0; i < 20; i++) {
        QTableWidgetItem* nameItem = new QTableWidgetItem(featureNames[i]);
        nameItem->setFlags(nameItem->flags() & ~Qt::ItemIsEditable);
        dataTable->setItem(i, 0, nameItem);

        QTableWidgetItem* valueItem = new QTableWidgetItem(QString::number(patient.features[i], 'f', 1));
        dataTable->setItem(i, 1, valueItem);
    }

    currentPatientIndex = index;
}

void MainWindow::onPatientSelected(int index) {
    loadPatientToTable(index);
}

void MainWindow::onClassify() {

    std::vector<double> features(20);
    for (int i = 0; i < 20; i++) {
        bool ok;
        double value = dataTable->item(i, 1)->text().toDouble(&ok);
        if (!ok) {
            QMessageBox::warning(this, "Помилка", "Некоректне значення в рядку " + QString::number(i + 1));
            return;
        }
        features[i] = value;
    }


    network->forward(features);
    updateOutputs();

    trainingLog->append("--- Класифікація ---");
    trainingLog->append("Пацієнт: " + patientSelector->currentText());

    std::vector<double> y = network->getOutputs();
    trainingLog->append(QString("Ймовірності: [%1, %2, %3, %4]")
                            .arg(y[0], 6, 'f', 3).arg(y[1], 6, 'f', 3).arg(y[2], 6, 'f', 3).arg(y[3], 6, 'f', 3));

    int predicted = network->getPredictedClass();
    trainingLog->append("Прогноз: " + classNames[predicted] + "\n");
}

void MainWindow::onTrain() {

    std::vector<std::vector<double>> X;
    std::vector<std::vector<double>> targets;

    for (const auto& patient : patients) {
        X.push_back(patient.features);


        std::vector<double> target(4, 0.0);
        target[patient.diagnosis] = 1.0;
        targets.push_back(target);
    }


    if (trainingIteration == 0) {
        network->computeNormalization(X);
        trainingLog->append("Дані нормалізовані (mean=0, std=1) для кращого навчання\n");
    }


    double totalLossBefore = 0.0;
    for (size_t i = 0; i < X.size(); i++) {
        network->forward(X[i]);
        totalLossBefore += network->computeLoss(targets[i]);
    }
    double avgLossBefore = totalLossBefore / X.size();

    trainingLog->append(QString("========== Епоха %1 ==========").arg(trainingIteration + 1));
    trainingLog->append(QString("Функція втрат до навчання: L = %1").arg(avgLossBefore, 0, 'f', 4));


    network->trainEpoch(X, targets);


    double totalLossAfter = 0.0;
    int correctPredictions = 0;

    for (size_t i = 0; i < X.size(); i++) {
        network->forward(X[i]);
        totalLossAfter += network->computeLoss(targets[i]);

        if (network->getPredictedClass() == patients[i].diagnosis) {
            correctPredictions++;
        }
    }
    double avgLossAfter = totalLossAfter / X.size();
    double accuracy = (double)correctPredictions / X.size() * 100.0;

    trainingLog->append(QString("Функція втрат після навчання: L = %1").arg(avgLossAfter, 0, 'f', 4));
    trainingLog->append(QString("Точність: %1% (%2/%3)")
                            .arg(accuracy, 0, 'f', 1).arg(correctPredictions).arg(X.size()));
    trainingLog->append("========================================\n");

    trainingIteration++;


    onClassify();
}

void MainWindow::onLoadCSV() {
    QString fileName = QFileDialog::getOpenFileName(this, "Завантажити CSV", "", "CSV Files (*.csv)");

    if (fileName.isEmpty()) return;

    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox::critical(this, "Помилка", "Не вдалося відкрити файл!");
        return;
    }

    QTextStream in(&file);
    patients.clear();
    patientSelector->clear();


    if (!in.atEnd()) {
        in.readLine();
    }

    int patientNum = 1;
    while (!in.atEnd()) {
        QString line = in.readLine();
        QStringList parts = line.split(',');

        if (parts.size() < 21) continue; // 20 features + 1 diagnosis

        PatientData patient;
        patient.features.resize(20);

        for (int i = 0; i < 20; i++) {
            patient.features[i] = parts[i].toDouble();
        }

        patient.diagnosis = parts[20].toInt();
        patient.name = "Пацієнт " + QString::number(patientNum++) + " (" + classNames[patient.diagnosis] + ")";

        patients.push_back(patient);
        patientSelector->addItem(patient.name);
    }

    file.close();

    if (!patients.empty()) {
        loadPatientToTable(0);
        trainingLog->append(QString("Завантажено %1 пацієнтів з файлу %2\n")
                                .arg(patients.size()).arg(fileName));
    } else {
        QMessageBox::warning(this, "Помилка", "Файл порожній або має некоректний формат!");
    }
}

void MainWindow::updateOutputs() {
    std::vector<double> y = network->getOutputs();


    outputY0->setText(QString::number(y[0], 'f', 3));
    outputY1->setText(QString::number(y[1], 'f', 3));
    outputY2->setText(QString::number(y[2], 'f', 3));
    outputY3->setText(QString::number(y[3], 'f', 3));

    int predicted = network->getPredictedClass();
    predictedClass->setText(classNames[predicted]);


    QString styles[4] = {
        "QLabel { font-weight: bold; color: green; font-size: 14px; }",
        "QLabel { font-weight: bold; color: orange; font-size: 14px; }",
        "QLabel { font-weight: bold; color: red; font-size: 14px; }",
        "QLabel { font-weight: bold; color: purple; font-size: 14px; }"
    };

    QLabel* outputs[4] = {outputY0, outputY1, outputY2, outputY3};
    for (int i = 0; i < 4; i++) {
        if (i == predicted) {
            outputs[i]->setStyleSheet(styles[i] + " background-color: #FFEB3B;");
        } else {
            outputs[i]->setStyleSheet(styles[i]);
        }
    }
}
