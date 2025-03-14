#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <arpa/inet.h> // For ntohl

using namespace std;
using namespace Eigen;

// Function to read a 32-bit unsigned integer from a file (big-endian)
uint32_t read_uint32(ifstream& file) {
    uint32_t value;
    file.read(reinterpret_cast<char*>(&value), sizeof(value));
    return ntohl(value); // Convert from big-endian to little-endian
}

// Function to load a label at a specific index and one-hot encode it
VectorXd load_label_at_index(const string& file_path, int index, int& num_labels) {
    ifstream file(file_path, ios::binary);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << file_path << endl;
        exit(1);
    }

    // Read the header
    uint32_t magic_number = read_uint32(file); // Magic number (2049)
    num_labels = read_uint32(file);            // Number of labels

    // Check if the index is valid
    if (index < 0 || index >= static_cast<int>(num_labels)) {
        cerr << "Invalid label index: " << index << ". Must be between 0 and " << num_labels - 1 << endl;
        exit(1);
    }

    // Skip to the desired label
    file.seekg(8 + index, ios::beg);

    // Read the label
    unsigned char label;
    file.read(reinterpret_cast<char*>(&label), sizeof(label));

    // One-hot encode the label
    VectorXd one_hot_label = VectorXd::Zero(10); // Initialize with zeros
    one_hot_label(static_cast<int>(label)) = 1.0; // Set the index corresponding to the label to 1.0

    return one_hot_label;
}

// Function to write the label tensor to a file
void write_label_tensor(const string& output_path, const VectorXd& label_tensor) {
    ofstream file(output_path);
    if (!file.is_open()) {
        cerr << "Failed to open file for writing: " << output_path << endl;
        exit(1);
    }

    // Write the tensor type, shape, and values
    file << 1 << endl;
    file << label_tensor.size() << endl;
    for (int i = 0; i < label_tensor.size(); ++i) {
        file << label_tensor(i) << endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <label_dataset_input> <label_tensor_output> <label_index>" << endl;
        return 1;
    }

    // Parse command-line arguments
    string label_dataset_input = argv[1];
    string label_tensor_output = argv[2];
    int label_index = stoi(argv[3]);

    // Load the label
    int num_labels;
    VectorXd label_tensor = load_label_at_index(label_dataset_input, label_index, num_labels);

    // Write the label tensor to the output file
    write_label_tensor(label_tensor_output, label_tensor);

    cout << "Successfully wrote label tensor to " << label_tensor_output << endl;
    return 0;
}