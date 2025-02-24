#include "NeuralNetwork.h"
#include <iostream>

// Math ////////////////////////////////////////////////////////

// Operates on each element. If less than 0, makes the element 0.
// Introduces non-linearity into the network.
Eigen::MatrixXd ReLU(const Eigen::MatrixXd &z) {
  // Use vectorised version for performance. Returns z element-wise max(0, z).
  return z.array().max(0.0).matrix();
}

// The derivative is simply 1 if the element is bigger than 0, and 0 otherwise.
Eigen::MatrixXd ReLuDeriv(const Eigen::MatrixXd &z) {
  // Returns elementwise the value of (z > 0) - either 1 or 0.
  return (z.array() > 0.0).cast<double>().matrix();
}

// For each column, caclulate e^{z_i} / (sum e^{z_i}).
// Produces outputs that added together, sum to 1.
Eigen::MatrixXd softMax(Eigen::MatrixXd &z) {
  std::vector<double> colSums(z.cols(), 0.0);
  for (int i = 0; i < z.rows(); i++) {
    for (int j = 0; j < z.cols(); j++) {
      colSums[j] += std::exp(z(i, j));
    }
  }

  Eigen::MatrixXd softMaxMatrix(z.rows(), z.cols());
  for (int i = 0; i < z.rows(); i++) {
    for (int j = 0; j < z.cols(); j++) {
      softMaxMatrix(i, j) = std::exp(z(i, j)) / colSums[j];
    }
  }

  return softMaxMatrix;
}

// Forward Propagation /////////////////////////////////////

// Calculate raw node activations
Eigen::MatrixXd forwardPropZ1(Eigen::MatrixXd &W1, Eigen::MatrixXd &b1,
                              Eigen::MatrixXd &X) {
  return W1 * X + b1.replicate(1, X.cols());
}

// Apply ReLU to node activations
Eigen::MatrixXd forwardPropA1(Eigen::MatrixXd &Z1) { return ReLU(Z1); }

// Calculate edge weights
Eigen::MatrixXd forwardPropZ2(Eigen::MatrixXd &W2, Eigen::MatrixXd &A1,
                              Eigen::MatrixXd &b2) {
  return W2 * A1 + b2.replicate(1, A1.cols());
}

// Calculate output layer node activations
Eigen::MatrixXd forwardPropA2(Eigen::MatrixXd &Z2) { return softMax(Z2); }

// Backward Propagation /////////////////////////////////////

// One hot encode to convert our array of results into a matrix
// where each row (testcase) has a column 1-hot-encoded.
Eigen::MatrixXd oneHotEncode(Eigen::MatrixXd &Y) {
  double maxVal = 0;
  for (int i = 0; i < Y.rows(); i++) {
    for (int j = 0; j < Y.cols(); j++) {
      maxVal = std::max(maxVal, Y(i, j));
    }
  }

  Eigen::MatrixXd oneHotY(Y.cols(), static_cast<int>(maxVal) + 1);
  oneHotY.setZero();
  for (int i = 0; i < Y.cols(); i++) {
    oneHotY(i, static_cast<int>(Y(0, i))) = 1.0;
  }

  // '.eval()' forces transpose to complete before assigning back (avoid
  // aliasing problem)
  oneHotY = oneHotY.transpose().eval();

  return oneHotY;
}

std::vector<Eigen::MatrixXd>
backwardProp(Eigen::MatrixXd &Z1, Eigen::MatrixXd &A1, Eigen::MatrixXd &A2,
             Eigen::MatrixXd &W2, Eigen::MatrixXd &X, Eigen::MatrixXd &Y) {
  Eigen::MatrixXd oneHotY = oneHotEncode(Y);
  Eigen::MatrixXd dZ2 = A2 - oneHotY;
  // Both X and Y have 'm' columns
  int m = Y.cols();
  Eigen::MatrixXd dW2 = 1.0 / m * (dZ2 * A1.transpose());
  // Sum for each node accross 'm' testcases.
  Eigen::MatrixXd db2 = dZ2.rowwise().sum() / m;
  // Use the Hadamard Product 'cwiseProduct()'
  Eigen::MatrixXd dZ1 = (W2.transpose() * dZ2).cwiseProduct(ReLuDeriv(Z1));
  Eigen::MatrixXd dW1 = 1.0 / m * dZ1 * X.transpose();
  Eigen::MatrixXd db1 = dZ1.rowwise().sum() / m;

  return {dW1, db1, dW2, db2};
}

void NeuralNetwork::update(Eigen::MatrixXd &W1, Eigen::MatrixXd &b1,
                           Eigen::MatrixXd &W2, Eigen::MatrixXd &b2,
                           Eigen::MatrixXd &dW1, Eigen::MatrixXd &db1,
                           Eigen::MatrixXd &dW2, Eigen::MatrixXd &db2,
                           double alpha) {
  W1 = W1 - alpha * dW1;
  b1 = b1 - alpha * db1;
  W2 = W2 - alpha * dW2;
  b2 = b2 - alpha * db2;
}

Eigen::MatrixXd getPredictions(Eigen::MatrixXd &A2) {
  Eigen::MatrixXd predictions(1, A2.cols());
  for (int j = 0; j < A2.cols(); ++j) {
    int maxIndex;
    // For each column, find the index of the maximum element
    A2.col(j).maxCoeff(&maxIndex);
    predictions(0, j) = maxIndex;
  }
  return predictions;
}

double getAccuracy(const Eigen::MatrixXd &predictions,
                   const Eigen::MatrixXd &Y) {
  double correctCount = 0;
  for (int i = 0; i < predictions.rows(); ++i) {
    for (int j = 0; j < predictions.cols(); ++j) {
      if (predictions(i, j) == Y(i, j)) {
        correctCount++;
      }
    }
  }
  double totalElements = predictions.size();
  return correctCount / totalElements;
}

std::vector<Eigen::MatrixXd> NeuralNetwork::gradient_descent(Eigen::MatrixXd &X,
                                                             Eigen::MatrixXd &Y,
                                                             double alpha,
                                                             int iterations) {
  for (int i = 0; i < iterations; ++i) {
    Eigen::MatrixXd Z1 = forwardPropZ1(W1, b1, X);
    Eigen::MatrixXd A1 = forwardPropA1(Z1);
    Eigen::MatrixXd Z2 = forwardPropZ2(W2, A1, b2);
    Eigen::MatrixXd A2 = forwardPropA2(Z2);

    std::vector<Eigen::MatrixXd> bwdProp = backwardProp(Z1, A1, A2, W2, X, Y);
    Eigen::MatrixXd dW1 = bwdProp[0];
    Eigen::MatrixXd db1 = bwdProp[1];
    Eigen::MatrixXd dW2 = bwdProp[2];
    Eigen::MatrixXd db2 = bwdProp[3];

    update(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha);
    if (i % 10 == 0) {
      Eigen::MatrixXd predictions = getPredictions(A2);
      std::cout << "Iteration: " << i << "\n"
                << "Accuracy: " << getAccuracy(predictions, Y) << "\n";
    }
  }
  return {W1, b1, W2, b2};
}