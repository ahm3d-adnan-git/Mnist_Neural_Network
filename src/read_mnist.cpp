#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
#include <Eigen/Dense>
#include "../include/read_mnist.h"

using Eigen::MatrixXd;
using std::vector;

// Function to read a 32-bit unsigned integer from the file in big-endian format
uint32_t read_uint32(std::ifstream &file) {
    uint32_t value = 0;
    file.read(reinterpret_cast<char *>(&value), 4);
    return __builtin_bswap32(value); // Convert big-endian to little-endian
}

// Function to read MNIST images
MatrixXd read_mnist_images(const std::string &filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    // Read header
    uint32_t magic_number = read_uint32(file);
    uint32_t num_images = read_uint32(file);
    uint32_t rows = read_uint32(file);
    uint32_t cols = read_uint32(file);

    if (magic_number != 2051) { // Check magic number for images
        throw std::runtime_error("Invalid magic number in file: " + filepath);
    }

    // Read image data
    MatrixXd images(num_images, rows * cols);
    for (uint32_t i = 0; i < num_images; ++i) {
        for (uint32_t j = 0; j < rows * cols; ++j) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char *>(&pixel), 1);
            images(i, j) = static_cast<double>(pixel) / 255.0; // Normalize to [0, 1]
        }
    }

    file.close();
    return images;
}

// Function to read MNIST labels
vector<int> read_mnist_labels(const std::string &filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    // Read header
    uint32_t magic_number = read_uint32(file);
    uint32_t num_labels = read_uint32(file);

    if (magic_number != 2049) { // Check magic number for labels
        throw std::runtime_error("Invalid magic number in file: " + filepath);
    }

    // Read label data
    vector<int> labels(num_labels);
    for (uint32_t i = 0; i < num_labels; ++i) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char *>(&label), 1);
        labels[i] = static_cast<int>(label);
    }

    file.close();
    return labels;
}