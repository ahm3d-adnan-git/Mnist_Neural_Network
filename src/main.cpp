#include <iostream>
#include <map>
#include "neural_network.cpp"
#include "../include/read_mnist.h"
#include <Eigen/Core>
#include <thread>

using namespace std;

// Function to parse the input.config file
map<string, string> parse_config(const string& file_path);

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <input_config>" << endl;
        return 1;
    }

    // Parse the configuration file
    string config_path = argv[1];
    map<string, string> config = parse_config(config_path);

    // Check for missing keys
    vector<string> required_keys = {
        "rel_path_train_images",
        "rel_path_train_labels",
        "rel_path_test_images",
        "rel_path_test_labels",
        "rel_path_log_file",
        "num_epochs",
        "batch_size",
        "hidden_size",
        "learning_rate"
    };

    for (const string& key : required_keys) {
        if (config.find(key) == config.end()) {
            cerr << "Error: Missing key in config file: " << key << endl;
            return 1;
        }
    }

   // Enable Eigen multi-threading
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 1;  // Ensure at least one thread is used
    Eigen::setNbThreads(num_threads);
    cout << "Using " << num_threads << " threads for Eigen operations." << endl;


    // Extract settings from the config
    string train_images_path = config["rel_path_train_images"];
    string train_labels_path = config["rel_path_train_labels"];
    string test_images_path = config["rel_path_test_images"];
    string test_labels_path = config["rel_path_test_labels"];
    string log_file_path = config["rel_path_log_file"];
    int num_epochs = stoi(config["num_epochs"]);
    int batch_size = stoi(config["batch_size"]);
    int hidden_size = stoi(config["hidden_size"]);
    double learning_rate = stod(config["learning_rate"]);

    try {
        // Load the training dataset
        cout << "Loading training data..." << endl;
        Eigen::MatrixXd train_images = read_mnist_images(train_images_path);
        vector<int> train_labels = read_mnist_labels(train_labels_path);

        // Load the testing dataset
        cout << "Loading testing data..." << endl;
        Eigen::MatrixXd test_images = read_mnist_images(test_images_path);
        vector<int> test_labels = read_mnist_labels(test_labels_path);

        // Initialize the neural network
        cout << "Initializing neural network..." << endl;
        NeuralNetwork nn(train_images.cols(), hidden_size, 10, learning_rate);

        // Train the neural network
        cout << "Training the network..." << endl;
        nn.train(train_images, train_labels, num_epochs, batch_size);

        // Test the neural network
        cout << "Testing the network..." << endl;
        nn.test(test_images, test_labels, log_file_path, batch_size);

        cout << "Predictions logged to " << log_file_path << endl;
    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}


// Implementation of parse_config
map<string, string> parse_config(const string& file_path) {
    map<string, string> config;
    ifstream file(file_path);

    if (!file.is_open()) {
        cerr << "Error: Failed to open configuration file: " << file_path << endl;
        exit(1);
    }

    string line;
    while (getline(file, line)) {
        // Remove comments
        line = line.substr(0, line.find('#'));
        line.erase(0, line.find_first_not_of(" \t")); // Trim leading whitespace
        line.erase(line.find_last_not_of(" \t") + 1); // Trim trailing whitespace
        if (line.empty()) continue;

        // Parse key-value pair
        size_t delimiter_pos = line.find('=');
        if (delimiter_pos == string::npos) {
            cerr << "Error: Invalid configuration line: " << line << endl;
            exit(1);
        }

        string key = line.substr(0, delimiter_pos);
        string value = line.substr(delimiter_pos + 1);

        // Trim whitespace around key and value
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        // Remove surrounding quotes from value, if present
        if (!value.empty() && value.front() == '"' && value.back() == '"') {
            value = value.substr(1, value.size() - 2);
        }

        config[key] = value;
    }

    file.close();
    return config;
}