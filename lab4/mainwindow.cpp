#include "mainwindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QThread>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    classifier = new MusicClassifier();
    setupUI();
    connectSignals();

    setWindowTitle("Класифікатор музичних жанрів");
    resize(600, 500);
}

MainWindow::~MainWindow() {
    delete classifier;
}

void MainWindow::setupUI() {
    QWidget* centralWidget = new QWidget(this);
    QVBoxLayout* mainLayout = new QVBoxLayout(centralWidget);


    QLabel* titleLabel = new QLabel("Класифікація музичних жанрів", this);
    QFont titleFont = titleLabel->font();
    titleFont.setPointSize(16);
    titleFont.setBold(true);
    titleLabel->setFont(titleFont);
    titleLabel->setAlignment(Qt::AlignCenter);
    mainLayout->addWidget(titleLabel);

    mainLayout->addSpacing(20);


    QGroupBox* inputGroup = new QGroupBox("Параметри музики", this);
    QVBoxLayout* inputLayout = new QVBoxLayout(inputGroup);


    QHBoxLayout* tempoLayout = new QHBoxLayout();
    QLabel* tempoLabel = new QLabel("Темп (BPM, 60-180):", this);
    tempoLabel->setMinimumWidth(150);
    tempoInput = new QLineEdit(this);
    tempoInput->setPlaceholderText("Наприклад: 120");
    tempoLayout->addWidget(tempoLabel);
    tempoLayout->addWidget(tempoInput);
    inputLayout->addLayout(tempoLayout);


    QHBoxLayout* rhythmLayout = new QHBoxLayout();
    QLabel* rhythmLabel = new QLabel("Ритм (0-100):", this);
    rhythmLabel->setMinimumWidth(150);
    rhythmInput = new QLineEdit(this);
    rhythmInput->setPlaceholderText("Наприклад: 75");
    rhythmLayout->addWidget(rhythmLabel);
    rhythmLayout->addWidget(rhythmInput);
    inputLayout->addLayout(rhythmLayout);


    QHBoxLayout* energyLayout = new QHBoxLayout();
    QLabel* energyLabel = new QLabel("Енергія (0-100):", this);
    energyLabel->setMinimumWidth(150);
    energyInput = new QLineEdit(this);
    energyInput->setPlaceholderText("Наприклад: 80");
    energyLayout->addWidget(energyLabel);
    energyLayout->addWidget(energyInput);
    inputLayout->addLayout(energyLayout);


    QHBoxLayout* loudnessLayout = new QHBoxLayout();
    QLabel* loudnessLabel = new QLabel("Гучність (0-100):", this);
    loudnessLabel->setMinimumWidth(150);
    loudnessInput = new QLineEdit(this);
    loudnessInput->setPlaceholderText("Наприклад: 70");
    loudnessLayout->addWidget(loudnessLabel);
    loudnessLayout->addWidget(loudnessInput);
    inputLayout->addLayout(loudnessLayout);

    mainLayout->addWidget(inputGroup);

    mainLayout->addSpacing(10);


    loadAudioButton = new QPushButton("Завантажити аудіофайл", this);
    mainLayout->addWidget(loadAudioButton);

    mainLayout->addSpacing(10);


    QHBoxLayout* buttonLayout = new QHBoxLayout();
    trainButton = new QPushButton("Навчити мережу", this);
    classifyButton = new QPushButton("Класифікувати", this);
    classifyButton->setEnabled(false);

    buttonLayout->addWidget(trainButton);
    buttonLayout->addWidget(classifyButton);
    mainLayout->addLayout(buttonLayout);

    mainLayout->addSpacing(10);


    progressBar = new QProgressBar(this);
    progressBar->setVisible(false);
    mainLayout->addWidget(progressBar);


    QGroupBox* resultGroup = new QGroupBox("Результат класифікації", this);
    QVBoxLayout* resultLayout = new QVBoxLayout(resultGroup);

    resultLabel = new QLabel("Жанр: -", this);
    QFont resultFont = resultLabel->font();
    resultFont.setPointSize(12);
    resultFont.setBold(true);
    resultLabel->setFont(resultFont);
    resultLayout->addWidget(resultLabel);

    confidenceLabel = new QLabel("Впевненість: -", this);
    resultLayout->addWidget(confidenceLabel);

    mainLayout->addWidget(resultGroup);

    mainLayout->addSpacing(10);


    statusLabel = new QLabel("Статус: Мережа не навчена", this);
    statusLabel->setStyleSheet("QLabel { color: red; }");
    mainLayout->addWidget(statusLabel);

    mainLayout->addStretch();

    setCentralWidget(centralWidget);
}

void MainWindow::connectSignals() {
    connect(trainButton, &QPushButton::clicked, this, &MainWindow::onTrainClicked);
    connect(classifyButton, &QPushButton::clicked, this, &MainWindow::onClassifyClicked);
    connect(loadAudioButton, &QPushButton::clicked, this, &MainWindow::onLoadAudioClicked);
}

void MainWindow::onTrainClicked() {
    trainButton->setEnabled(false);
    classifyButton->setEnabled(false);
    progressBar->setVisible(true);
    progressBar->setRange(0, 0);

    statusLabel->setText("Статус: Навчання мережі...");
    statusLabel->setStyleSheet("QLabel { color: orange; }");


    classifier->trainNetwork(1000);

    progressBar->setVisible(false);
    trainButton->setEnabled(true);
    classifyButton->setEnabled(true);

    statusLabel->setText("Статус: Мережа навчена та готова до роботи");
    statusLabel->setStyleSheet("QLabel { color: green; }");

    QMessageBox::information(this, "Навчання завершено",
                             "Нейронна мережа успішно навчена!\nТепер можна класифікувати музику.");
}

void MainWindow::onClassifyClicked() {
    if (!classifier->getIsTrained()) {
        QMessageBox::warning(this, "Помилка", "Спочатку навчіть мережу!");
        return;
    }


    bool ok;
    double tempo = tempoInput->text().toDouble(&ok);
    if (!ok || tempo < 60 || tempo > 180) {
        QMessageBox::warning(this, "Помилка", "Введіть коректний темп (60-180 BPM)");
        return;
    }

    double rhythm = rhythmInput->text().toDouble(&ok);
    if (!ok || rhythm < 0 || rhythm > 100) {
        QMessageBox::warning(this, "Помилка", "Введіть коректний ритм (0-100)");
        return;
    }

    double energy = energyInput->text().toDouble(&ok);
    if (!ok || energy < 0 || energy > 100) {
        QMessageBox::warning(this, "Помилка", "Введіть коректну енергію (0-100)");
        return;
    }

    double loudness = loudnessInput->text().toDouble(&ok);
    if (!ok || loudness < 0 || loudness > 100) {
        QMessageBox::warning(this, "Помилка", "Введіть коректну гучність (0-100)");
        return;
    }


    MusicFeatures features;
    features.tempo = tempo;
    features.rhythm = rhythm;
    features.energy = energy;
    features.loudness = loudness;


    double confidence = 0.0;
    MusicGenre genre = classifier->classifyMusic(features, confidence);


    QString genreName = classifier->getGenreName(genre);
    resultLabel->setText("Жанр: " + genreName);
    confidenceLabel->setText(QString("Впевненість: %1%").arg(confidence, 0, 'f', 1));

    if (genre == UNKNOWN) {
        resultLabel->setStyleSheet("QLabel { color: orange; }");
    } else {
        resultLabel->setStyleSheet("QLabel { color: green; }");
    }
}

void MainWindow::onLoadAudioClicked() {
    QString fileName = QFileDialog::getOpenFileName(this,
                                                    "Виберіть аудіофайл",
                                                    "",
                                                    "Audio Files (*.mp3 *.wav *.flac *.ogg)");

    if (fileName.isEmpty()) {
        return;
    }


    MusicFeatures features = classifier->analyzeAudioFile(fileName);


    tempoInput->setText(QString::number(features.tempo, 'f', 1));
    rhythmInput->setText(QString::number(features.rhythm, 'f', 1));
    energyInput->setText(QString::number(features.energy, 'f', 1));
    loudnessInput->setText(QString::number(features.loudness, 'f', 1));

    QMessageBox::information(this, "Аналіз завершено",
                             "Аудіофайл проаналізовано!\nПараметри заповнено автоматично.\n\n"
                             "Примітка: це спрощений аналіз для демонстрації.");
}
