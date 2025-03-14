# Check if the correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <image_dataset_input> <image_tensor_output> <image_index>"
    exit 1
fi

# Assign input arguments to variables
image_dataset_input=$1
image_tensor_output=$2
image_index=$3

# Set the Eigen include path (modify it as per your Eigen installation)
EIGEN_PATH="./include/eigen-3.4.0"

# Compile the C++ program
g++ -std=c++11 -I $EIGEN_PATH src/read_dataset_images.cpp -o read_dataset_images
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# Execute the program with the provided arguments
./read_dataset_images "$image_dataset_input" "$image_tensor_output" "$image_index"
if [ $? -ne 0 ]; then
    echo "Execution failed!"
    exit 1
fi

echo "MNIST image processing completed successfully. Output written to $image_tensor_output."