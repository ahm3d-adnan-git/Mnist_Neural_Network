#!/bin/bash
echo "Building the project now..."

# Clean the build directory if it exists
if [ -d "build" ]; then
    echo "Cleaning build directory"
    rm -rf build
fi

# Create a build directory
mkdir -p build

# Navigate to the build directory
cd build

# Run CMake to configure the project
cmake ..

# Build the project
cmake --build .

# Check if the build was successful
if [ $? -eq 0 ]; 
then
    echo "Build successful!"
else
    echo "Build failed!"
    exit 1
fi