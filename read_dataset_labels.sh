# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <label_dataset_input> <label_tensor_output> <label_index>"
    exit 1
fi

# Assign arguments to variables
label_dataset_input=$1
label_tensor_output=$2
label_index=$3

# Set the Eigen include path (modify it as per your Eigen installation)
EIGEN_PATH="./include/eigen-3.4.0"

# Compile the C++ program
g++ -std=c++11 -I $EIGEN_PATH src/read_dataset_labels.cpp -o read_dataset_labels
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# Run the compiled program
./read_dataset_labels "$label_dataset_input" "$label_tensor_output" "$label_index"
if [ $? -ne 0 ]; then
    echo "Execution failed!"
    exit 1
fi

echo "MNIST image processing completed successfully. Output written to $label_tensor_output."