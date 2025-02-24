#include <eigen3/Eigen/Dense> // Main header for Eigen
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

// Initialisation //////////////////////////////////////////////
Eigen::MatrixXd initW1() {
  Eigen::MatrixXd W1 = Eigen::MatrixXd::Random(15, 784);
  W1 *= 0.5;
  return W1;
}

Eigen::MatrixXd initb1() {
  Eigen::MatrixXd b1 = Eigen::MatrixXd::Random(15, 1);
  b1 *= 0.5;
  return b1;
}

Eigen::MatrixXd initW2() {
  Eigen::MatrixXd W2 = Eigen::MatrixXd::Random(10, 15);
  W2 *= 0.5;
  return W2;
}

Eigen::MatrixXd initb2() {
  Eigen::MatrixXd b2 = Eigen::MatrixXd::Random(10, 1);
  b2 *= 0.5;
  return b2;
}

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

void update(Eigen::MatrixXd &W1, Eigen::MatrixXd &b1, Eigen::MatrixXd &W2,
            Eigen::MatrixXd &b2, Eigen::MatrixXd &dW1, Eigen::MatrixXd &db1,
            Eigen::MatrixXd &dW2, Eigen::MatrixXd &db2, double alpha) {
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

std::vector<Eigen::MatrixXd> gradient_descent(Eigen::MatrixXd &X,
                                              Eigen::MatrixXd &Y, double alpha,
                                              int iterations) {
  Eigen::MatrixXd W1 = initW1();
  Eigen::MatrixXd b1 = initb1();
  Eigen::MatrixXd W2 = initW2();
  Eigen::MatrixXd b2 = initb2();
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

// Function to load CSV data into an Eigen::MatrixXd
Eigen::MatrixXd loadCSV(const std::string &path) {
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

int main() {
  std::string filename = "../data/train.csv";
  Eigen::MatrixXd data = loadCSV(filename);
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

  gradient_descent(X_train, Y_train, 0.10, 2000);

  return 0;
}