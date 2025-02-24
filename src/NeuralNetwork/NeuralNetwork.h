#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <eigen3/Eigen/Dense>

class NeuralNetwork {
public:
  Eigen::MatrixXd W1, b1, W2, b2;

  NeuralNetwork() {
    W1 = Eigen::MatrixXd::Random(20, 784) * 0.5;
    b1 = Eigen::MatrixXd::Random(20, 1) * 0.5;
    W2 = Eigen::MatrixXd::Random(10, 20) * 0.5;
    b2 = Eigen::MatrixXd::Random(10, 1) * 0.5;
  }

  void update(Eigen::MatrixXd &W1, Eigen::MatrixXd &b1, Eigen::MatrixXd &W2,
              Eigen::MatrixXd &b2, Eigen::MatrixXd &dW1, Eigen::MatrixXd &db1,
              Eigen::MatrixXd &dW2, Eigen::MatrixXd &db2, double alpha);

  std::vector<Eigen::MatrixXd> gradient_descent(Eigen::MatrixXd &X,
                                                Eigen::MatrixXd &Y,
                                                double alpha, int iterations);
};

#endif // NEURAL_NETWORK_HPP