#include "NeuralNetworkUtility.h"

// Function to load CSV data into an Eigen::MatrixXd
Eigen::MatrixXd NeuralNetworkUtility::loadCSV(const std::string &path) {
  std::vector<std::vector<double>> data;
  std::ifstream in(path);
  std::string line;

  // Skip header row
  if (std::getline(in, line)) {
  }

  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::string cell;
    std::vector<double> row;
    while (std::getline(ss, cell, ',')) {
      row.push_back(std::stod(cell));
    }
    data.push_back(row);
  }
  if (data.empty())
    return Eigen::MatrixXd();

  int rows = data.size();
  int cols = data[0].size();
  Eigen::MatrixXd mat(rows, cols);
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      mat(i, j) = data[i][j];
  return mat;
}