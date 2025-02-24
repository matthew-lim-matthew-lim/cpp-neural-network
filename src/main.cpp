#include "NeuralNetwork/NeuralNetwork.h"
#include "Utility/NeuralNetworkUtility.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <iostream>
#include <random>

int main() {
  std::string filename = "../data/train.csv";
  Eigen::MatrixXd data = NeuralNetworkUtility::loadCSV(filename);
  int m = data.rows();
  int n = data.cols();

  // Shuffle rows (samples)
  std::vector<int> indices(m);
  for (int i = 0; i < m; ++i) {
    indices[i] = i;
  }
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(indices.begin(), indices.end(), g);

  Eigen::MatrixXd shuffledData(m, n);
  for (int i = 0; i < m; ++i) {
    shuffledData.row(i) = data.row(indices[i]);
  }

  // Split data into development (first 1000 samples) and training sets
  int devSize = 1000;

  // For the dev set: take the first 1000 rows and then transpose.
  // After transpose data_dev has shape (n, devSize)
  Eigen::MatrixXd data_dev = shuffledData.topRows(devSize).transpose();
  // Y_dev is the first row (labels)
  Eigen::MatrixXd Y_dev = data_dev.row(0);
  // X_dev is the remaining rows (features)
  Eigen::MatrixXd X_dev = data_dev.bottomRows(data_dev.rows() - 1);
  // Normalize X_dev by dividing by 255.
  X_dev /= 255.0;

  // For the training set: take the rest of the rows and then transpose.
  // After transpose data_train has shape (n, m - devSize)
  Eigen::MatrixXd data_train = shuffledData.bottomRows(m - devSize).transpose();
  Eigen::MatrixXd Y_train = data_train.row(0);
  Eigen::MatrixXd X_train = data_train.bottomRows(data_train.rows() - 1);
  X_train /= 255.0;

  int m_train = X_train.cols();

  std::cout << "X_dev shape: " << X_dev.rows() << " x " << X_dev.cols()
            << std::endl;
  std::cout << "Y_dev shape: " << Y_dev.rows() << " x " << Y_dev.cols()
            << std::endl;
  std::cout << "X_train shape: " << X_train.rows() << " x " << X_train.cols()
            << std::endl;
  std::cout << "Y_train shape: " << Y_train.rows() << " x " << Y_train.cols()
            << std::endl;
  std::cout << "m_train: " << m_train << std::endl;

  NeuralNetwork network;

  network.gradient_descent(X_train, Y_train, 0.10, 2000);

  return 0;
}