#ifndef NEURAL_NETWORK_UTILITY_HPP
#define NEURAL_NETWORK_UTILITY_HPP

#include <eigen3/Eigen/Dense>
#include <fstream>
#include <sstream>

class NeuralNetworkUtility {
public:
  // Function to load CSV data into an Eigen::MatrixXd
  static Eigen::MatrixXd loadCSV(const std::string &path);
};

#endif // NEURAL_NETWORK_UTILITY_HPP