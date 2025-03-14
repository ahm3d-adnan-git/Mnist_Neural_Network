#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <arpa/inet.h> // For ntohl (converting big-endian to little-endian)


// Function to read a 32-bit unsigned integer from a file (big-endian)
uint32_t read_uint32(std::ifstream& file) {
    uint32_t value;
    file.read(reinterpret_cast<char*>(&value), sizeof(value));
    return ntohl(value); // Convert from big-endian to little-endian
}

// Function to load a specific MNIST image as a 2D tensor
Eigen::MatrixXd load_image_at_index(const std::string& file_path, int index, int& num_rows, int& num_cols) {

    std::ifstream file(file_path , std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        exit(1);
    }

    // Read the header
    uint32_t magic_number = read_uint32(file); // Magic number (2051)
    uint32_t num_images = read_uint32(file);   // Number of images
    num_rows = read_uint32(file);              // Number of rows (28)
    num_cols = read_uint32(file);              // Number of columns (28)

    // Check if the index is valid
    if (index < 0 || index >= static_cast<int>(num_images)) {
        std::cerr << "Invalid image index: " << index << ". Must be between 0 and " << num_images - 1 << std::endl;
        exit(1);
    }

    // Skip to the desired image
    size_t image_size = num_rows * num_cols; // Size of one image
    file.seekg(16 + index * image_size, std::ios::beg);

    // Read the image data
    Eigen::MatrixXd image(num_rows, num_cols);
    for (int r = 0; r < num_rows; ++r) {
        for (int c = 0; c < num_cols; ++c) {
            unsigned char pixel;
            file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            image(r, c) = static_cast<double>(pixel) / 255.0; // Normalize pixel values to [0, 1]
        }
    }

    return image;
}

// Function to write the image tensor to a file
void write_image_tensor(const std::string& output_path, const Eigen::MatrixXd& image_tensor) {
    std::ofstream file(output_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << output_path << std::endl;
        exit(1);
    }

    file << 2 << std::endl;
    file << image_tensor.rows() << std::endl << image_tensor.cols() << std::endl;

    // Write the tensor values row by row
    for (int r = 0; r < image_tensor.rows(); ++r) {
        for (int c = 0; c < image_tensor.cols(); ++c) {
            file << image_tensor(r, c);
            if (c < image_tensor.cols() - 1) file << std::endl;
        }
        file << std::endl;
    }
}

int main(int argc, char* argv[]) {

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <image_dataset_input> <image_tensor_output> <image_index>" << std::endl;
        return 1;
    }

    // Parse command-line arguments
    std::string image_dataset_input = argv[1];
    std::string image_tensor_output = argv[2];
    int image_index = std::stoi(argv[3]);

    // Load the MNIST image
    int num_rows, num_cols;
    Eigen::MatrixXd image_tensor = load_image_at_index(image_dataset_input, image_index, num_rows, num_cols);

    // Write the image tensor to the output file
    write_image_tensor(image_tensor_output, image_tensor);

    std::cout << "Successfully wrote image tensor to " << image_tensor_output << std::endl;
    return 0;
}