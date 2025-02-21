#include <eigen3/Eigen/Dense> // Main header for Eigen
#include <iostream>

int main() {
  Eigen::Matrix2d A;
  Eigen::Vector2d b, x;

  A << 3, 2, 1, 4;
  b << 5, 6;

  x = A.colPivHouseholderQr().solve(b); // Solve Ax = b

  std::cout << "Solution x:\n" << x << std::endl;
  return 0;
}