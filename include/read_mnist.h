#ifndef READ_MNIST_H
#define READ_MNIST_H

#include <vector>
#include <Eigen/Dense>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdint>

// Function declarations
uint32_t read_uint32(std::ifstream &file);
Eigen::MatrixXd read_mnist_images(const std::string &filepath);
std::vector<int> read_mnist_labels(const std::string &filepath);

#endif // READ_MNIST_H