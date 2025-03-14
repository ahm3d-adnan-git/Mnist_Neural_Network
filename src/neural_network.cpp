#define EIGEN_DONT_PARALLELIZE
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include <omp.h>

using namespace std;
using namespace Eigen;

class NeuralNetwork {
private:
    MatrixXd W1, W2; // Weights
    VectorXd b1, b2; // Biases
    double learning_rate;
    int input_size, hidden_size, output_size;

    // Activation functions
    VectorXd relu(const VectorXd& x) {
        return x.unaryExpr([](double val) { return max(0.0, val); });
    }

    VectorXd relu_derivative(const VectorXd& x) {
        return x.unaryExpr([](double val) { return val > 0 ? 1.0 : 0.0; });
    }

    VectorXd softmax(const VectorXd& x) {
        VectorXd exp_vals = x.array().exp();
        return exp_vals / exp_vals.sum();
    }

    double cross_entropy_loss(const VectorXd& predictions, int label) {
        return -log(predictions(label) + 1e-9); // Avoid log(0)
    }

    void initialize_weights(MatrixXd& matrix, int rows, int cols) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(-sqrt(6.0 / (rows + cols)), sqrt(6.0 / (rows + cols)));
        matrix.resize(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                matrix(i, j) = dist(gen);
            }
        }
    }

    
    // Small random initialization for biases
    void initialize_biases(VectorXd& vector, int size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(-0.01, 0.01);
        vector.resize(size);
        for (int i = 0; i < size; ++i) {
            vector(i) = dist(gen);
        }
    }


public:
    NeuralNetwork(int input, int hidden, int output, double lr)
        : input_size(input), hidden_size(hidden), output_size(output), learning_rate(lr) {
        initialize_weights(W1, hidden_size, input_size);
        initialize_weights(W2, output_size, hidden_size);
        initialize_biases(b1, hidden_size);
        initialize_biases(b2, output_size);
    }

    VectorXd forward(const VectorXd& input, VectorXd& hidden_output) {
        // Input to hidden layer
        VectorXd z1 = W1 * input + b1;
        hidden_output = relu(z1);

        // Hidden to output layer
        VectorXd z2 = W2 * hidden_output + b2;
        VectorXd output = softmax(z2);

        return output; // Return predictions
    }

    void backward(const VectorXd& input, const VectorXd& hidden_output, const VectorXd& predictions, int label) {
        // Gradient for output layer
        VectorXd y_true = VectorXd::Zero(output_size);
        y_true(label) = 1.0; // One-hot encoding for the true label

        VectorXd dz2 = predictions - y_true;  // dL/dz2
        MatrixXd dW2 = dz2 * hidden_output.transpose(); // dL/dW2
        VectorXd db2 = dz2;                             // dL/db2

        // Gradient for hidden layer
        VectorXd dz1 = (W2.transpose() * dz2).cwiseProduct(relu_derivative(hidden_output)); // dL/dz1
        MatrixXd dW1 = dz1 * input.transpose(); // dL/dW1
        VectorXd db1 = dz1;                     // dL/db1

        #pragma omp critical
        {
            // Update weights and biases
            W2 -= learning_rate * dW2;
            b2 -= learning_rate * db2;
            W1 -= learning_rate * dW1;
            b1 -= learning_rate * db1;
        }
    }

    void train(const MatrixXd& X, const vector<int>& Y, int epochs, int batch_size) {
        assert(X.rows() == Y.size());
        int num_samples = X.rows();

        for (int epoch = 0; epoch < epochs; ++epoch) {
            double epoch_loss = 0.0;

            #pragma omp parallel for reduction(+:epoch_loss)
            for (int i = 0; i < num_samples; i += batch_size) {
                int batch_end = min(i + batch_size, num_samples);
                double local_loss = 0.0;

                for (int j = i; j < batch_end; ++j) {
                    VectorXd input = X.row(j).transpose();
                    int label = Y[j];

                    // Forward pass
                    VectorXd hidden_output;
                    VectorXd predictions = forward(input, hidden_output);

                    // Compute loss
                    double loss = cross_entropy_loss(predictions, label);
                    local_loss += loss;

                    // Backward pass
                    backward(input, hidden_output, predictions, label);
                }

                #pragma omp atomic
                epoch_loss += local_loss;
            }

            // Print loss for the epoch
            cout << "Epoch " << epoch + 1 << "/" << epochs << ", Loss: " << epoch_loss / num_samples << endl;
        }
    }

    vector<int> test(const MatrixXd& X, const vector<int>& Y, const string& log_file, int batch_size) {
        assert(X.rows() == Y.size());
        int num_samples = X.rows();
        vector<int> predictions;
        ofstream log(log_file);

        if (!log.is_open()) {
            cerr << "Failed to open log file: " << log_file << endl;
            exit(1);
        }

        int num_batches = (num_samples + batch_size - 1) / batch_size; // Calculate the number of batches

        for (int batch = 0; batch < num_batches; ++batch) {
            log << "Current batch: " << batch << endl;

            int start_index = batch * batch_size;
            int end_index = min(start_index + batch_size, num_samples);

            #pragma omp parallel for
            for (int i = start_index; i < end_index; ++i) {
                VectorXd input = X.row(i).transpose();

                // Forward pass
                VectorXd hidden_output;
                VectorXd output = forward(input, hidden_output);

                // Predicted class
                int predicted_label = static_cast<int>(distance(output.data(), max_element(output.data(), output.data() + output.size())));

                #pragma omp critical
                {
                    predictions.push_back(predicted_label);
                    log << " - image " << i << ": Prediction=" << predicted_label << ". Label=" << Y[i] << endl;
                }
            }
        }

        log.close();
        return predictions;
    }
};