# Ensure the input configuration file is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_config>"
    exit 1
fi

INPUT_CONFIG=$1
EXECUTABLE=main_program

EIGEN_PATH="./include/eigen-3.4.0"

# Step 1: Compile the program
echo "Compiling the program..."
g++ -O3 -march=native -mtune=native  -fopenmp -DEIGEN_USE_THREADS -o $EXECUTABLE src/main.cpp src/neural_network.cpp src/read_mnist.cpp -I $EIGEN_PATH -std=c++17

# Check if the compilation was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed. Please check for errors."
    exit 1
fi

echo "Compilation successful."

# Step 2: Run the program
echo "Running the program with configuration file: $INPUT_CONFIG"
./$EXECUTABLE "$INPUT_CONFIG"

# Check if the program ran successfully
if [ $? -ne 0 ]; then
    echo "Program execution failed. Please check for errors."
    exit 1
fi

echo "Program executed successfully."
