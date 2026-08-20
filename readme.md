# LonexGrad

<p align="center">
  <strong>A modern C++20 tensor and automatic differentiation library for deep learning.</strong>
</p>

<p align="center">
  PyTorch-inspired API · Automatic Differentiation · CPU Backend · CUDA Acceleration
</p>

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-20-blue">
  <img src="https://img.shields.io/badge/CMake-3.25%2B-blue">
  <img src="https://img.shields.io/badge/CUDA-12%2B-green">
</p>

---

## Overview

**LonexGrad** is a C++20 deep learning library built around a tensor-based programming model with automatic differentiation.

The API is inspired by frameworks such as PyTorch, while the implementation stays focused on modern C++ and a modular backend architecture.

LonexGrad provides:

- Multidimensional tensors
- Automatic differentiation
- Tensor views and strided layouts
- Neural network modules
- Loss functions
- Optimizers
- Dataset and dataloader utilities
- CPU execution
- Optional CUDA acceleration
- CMake-based integration
- C++20 support

The goal is to provide the core building blocks required to construct, train, and experiment with neural networks directly in C++.

---

## Features

### Tensor Engine

- Multidimensional tensor representation
- Arbitrary tensor shapes
- Strided storage
- Tensor indexing
- Tensor slicing
- Tensor views
- Reshaping
- Broadcasting
- Type conversion
- Device transfers
- Random tensor generation
- Reduction operations
- Element-wise operations
- Matrix multiplication

### Automatic Differentiation

LonexGrad provides an automatic differentiation engine that builds a computational graph during forward execution and calculates gradients during backpropagation.

```cpp
Tensor x = uniform_real(0, 1, {10});
Tensor y = x * x;
Tensor loss = y.sum();
loss.backward();
```

This allows gradients to be propagated through supported tensor operations without manually deriving and implementing each gradient.

### Neural Networks

LonexGrad includes neural network building blocks for constructing models.

Supported components include:

- Linear layers
- Convolution layers
- RNN
- LSTM
- GRU
- Embedding
- Dropout
- Pooling
- Batch Normalization
- Layer Normalization
- Instance Normalization
- Activation functions
- Loss functions

### Optimizers

Available optimization algorithms include:

- SGD
- Adam
- AdamW
- Adagrad
- RMSprop

### Device Support

LonexGrad provides a device abstraction allowing tensors and neural networks to execute on supported backends.

```cpp
#ifdef LONEXGRAD_CUDA
constexpr Device device = kCUDA;
#else
constexpr Device device = kCPU;
#endif
```

The same high-level tensor and neural-network API can be used while changing the execution device.

---

## Backend Architecture

The tensor implementation is separated from backend-specific execution.

```
                    LonexGrad
                        │
                 Tensor Operations
                        │
              ┌─────────┴─────────┐
              │                   │
             CPU                CUDA
              │                   │
          CPU Kernels        CUDA Kernels
```

This architecture allows backend implementations to remain independent from the high-level tensor API.

---

## Why LonexGrad?

LonexGrad is designed around a few core ideas:

**Modern C++**
The library is implemented using modern C++20 features and follows a strongly typed C++ API.

**PyTorch-Inspired API**
The tensor and neural-network interfaces are designed to feel familiar to developers who have worked with PyTorch.

**Explicit Tensor Shapes**
LonexGrad does not automatically broadcast most operations. Instead, tensors can be explicitly reshaped or broadcast when required:

```cpp
Tensor x = ...;
Tensor y = x.reshape(...);
```

This makes tensor shape transformations explicit and helps avoid unexpected broadcasting behavior.

**Strided Tensor Storage**
Tensor storage uses strides, allowing supported view operations to reference existing storage instead of unnecessarily copying tensor data.

---

## Quick Start

### Requirements

**CPU**
- C++20 compatible compiler
- CMake 3.25+
- Standard C++ development environment

**CUDA** (for acceleration)
- CUDA Toolkit 12+
- Compatible NVIDIA GPU
- C++20 compatible compiler
- CMake 3.25+

### Installation

Clone the repository:

```bash
git clone https://github.com/Doves788/LonexGrad.git
cd LonexGrad
```

### Building

**Basic CPU Build**

```bash
cmake -S . -B build
cmake --build build
```

**Release Build**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

**CUDA Build**

```bash
cmake -S . -B build -DLONEXGRAD_BUILD_CUDA=ON
cmake --build build
```

CUDA support is optional. The CPU backend can be built without CUDA.

**Static and Shared Libraries**

Static library:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Shared library:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
cmake --build build
```

---

## Examples

### Basic Tensor Example

```cpp
#include <lonexgrad/lonexgrad.h>

using namespace lonexgrad;

int main() {
    Tensor x = uniform_real(0, 1, {4, 4});
    Tensor y = x * x;
    Tensor loss = y.sum();

    loss.backward();

    return 0;
}
```

### Neural Network Example

A simple neural network can be created using the `nn` module system.

```cpp
#include <lonexgrad/lonexgrad.h>

using namespace lonexgrad;

#ifdef LONEXGRAD_CUDA
constexpr Device device = kCUDA;
#else
constexpr Device device = kCPU;
#endif

class Net : public nn::Module {
public:
    Net() : linear1(4, 32), linear2(32, 10) {
        register_module(linear1);
        register_module(linear2);
    }

    [[nodiscard]] auto name() const -> std::string override {
        return "Net";
    }

    [[nodiscard]] auto forward(Tensor input) -> Tensor {
        Tensor result = relu(linear1.forward(input));
        return linear2.forward(result);
    }

private:
    nn::Linear linear1;
    nn::Linear linear2;
};

int main() {
    Tensor inputs = uniform_real(0, 1, {100, 4});
    Tensor targets = uniform_real(0, 1, {100, 10});

    Net net;
    net.to(device);

    optim::Adam optimizer(net.parameters_for_optimizer(), 3e-4);

    net.train();
    optimizer.zero_grad();

    Tensor output = net.forward(inputs);
    Tensor loss = nn::mse_loss(output, targets);

    loss.backward();
    optimizer.step();

    return 0;
}
```

### Training Loop

LonexGrad provides the components required to construct a standard training pipeline.

```cpp
Net net;
net.to(device);

optim::Adam optimizer(net.parameters_for_optimizer(), 3e-4);

net.train();

for (auto [input, target] : train_loader) {
    input = input.to(device);
    target = target.to(device);

    optimizer.zero_grad();

    Tensor output = net.forward(input);
    Tensor loss = nn::mse_loss(output, target);

    loss.backward();
    optimizer.step();
}
```

For evaluation, gradients can be disabled using `NoGradGuard`:

```cpp
{
    const autograd::NoGradGuard guard;

    net.eval();

    Tensor output = net.forward(input);
    Tensor loss = nn::mse_loss(output, target);
}
```

---

## Tensor Operations

LonexGrad supports a variety of tensor operations.

**Element-wise Operations**

```cpp
Tensor c = a + b;
Tensor d = a - b;
Tensor e = a * b;
Tensor f = a / b;
```

**Matrix Multiplication**

```cpp
Tensor result = matmul(a, b);
```

**Reshaping**

```cpp
Tensor reshaped = tensor.reshape(...);
```

**Device Transfer**

```cpp
Tensor gpu_tensor = tensor.to(kCUDA);
```

**Reduction**

```cpp
Tensor result = tensor.sum();
```

The exact supported operations are documented in the `docs/` directory.

---

## Automatic Differentiation

Automatic differentiation is integrated directly into tensor operations.

```cpp
Tensor x = uniform_real(0, 1, {10});
Tensor y = x * x;
Tensor loss = y.sum();

loss.backward();
```

Conceptually, LonexGrad constructs a computational graph similar to:

```
        x
       / \
      x   x
       \ /
        *
        │
       sum
        │
      loss
        │
    backward()
```

The backward pass propagates gradients through the supported operations in the graph.

---

## Datasets and DataLoaders

LonexGrad provides dataset and dataloader utilities for training neural networks.

```cpp
data::TensorDataset dataset(inputs, targets);

auto [train_data, validate_data, test_data] =
    data::random_split(std::move(dataset), seed, 70, 20, 10);

auto train_loader =
    data::DataLoader(train_data, batch_size, true, seed);
```

This allows datasets to be split into training, validation, and testing subsets and iterated in batches.

---

## CMake Integration

LonexGrad can be used as a dependency in another CMake project.

```cmake
cmake_minimum_required(VERSION 3.25)

project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(lonexgrad CONFIG REQUIRED)

add_executable(main main.cpp)

target_link_libraries(main PRIVATE lonexgrad::lonexgrad)
```

### FetchContent

LonexGrad can also be included directly through CMake `FetchContent`.

```cmake
include(FetchContent)

FetchContent_Declare(
    lonexgrad
    GIT_REPOSITORY https://github.com/Doves788/LonexGrad.git
    GIT_TAG main
)

FetchContent_MakeAvailable(lonexgrad)

add_executable(main main.cpp)

target_link_libraries(main PRIVATE lonexgrad::lonexgrad)
```

---

## Project Structure

```
LonexGrad/
│
├── include/
│   └── lonexgrad/
│       ├── autograd.h
│       ├── concepts.h
│       ├── device.h
│       ├── exception.h
│       ├── grad_mode.h
│       ├── index.h
│       ├── random.h
│       ├── scalar.h
│       ├── shape.h
│       ├── tensor.h
│       │
│       ├── data/
│       ├── nn/
│       └── optim/
│
├── lonexgrad/
│   ├── autograd/
│   ├── data/
│   ├── nn/
│   ├── optim/
│   └── tensor/
│       └── backend/
│           ├── common/
│           ├── cpu/
│           └── cuda/
│
├── examples/
├── tests/
├── docs/
│
├── CMakeLists.txt
├── CMakePresets.json
└── LICENSE
```

---

## Architecture

LonexGrad is organized into several major components.

**Tensor** — Tensor storage, shapes, strides, indexing, views, tensor operations, device management.

**Autograd** — Computational graph construction, gradient tracking, backpropagation, gradient propagation, gradient mode management.

**Neural Network** — Modules, layers, activations, normalization, recurrent networks, convolution, pooling, loss functions.

**Optimizers** — Responsible for updating trainable parameters using calculated gradients.

**Data** — Datasets, tensor datasets, dataloaders, dataset splitting, data handling utilities.

**Backends** — Contain the low-level kernels used to execute tensor operations. Currently supported backends include CPU and CUDA.

---

## Testing

LonexGrad uses CTest for its test suite.

```bash
cmake -S . -B build -DLONEXGRAD_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

---

## Examples Directory

Examples are located in `examples/`. They demonstrate different parts of the LonexGrad API, including tensor operations, automatic differentiation, neural networks, and training.

## Documentation

Additional documentation is available in `docs/`, covering:

- Tensor creation
- Tensor operations
- Neural networks
- Datasets and dataloaders
- Random number generation
- Custom autograd functions
- Backend development

---

## Design Principles

1. **Modern C++** — Use C++20 language features and strong compile-time abstractions.
2. **Explicit Tensor Operations** — Tensor transformations should be explicit and predictable.
3. **Backend Separation** — High-level tensor functionality should remain independent of low-level execution backends.
4. **Reusable Components** — Neural networks, optimizers, datasets, and tensor operations should be composable.
5. **PyTorch-Inspired Usability** — Developers familiar with PyTorch should find the API concepts familiar while working entirely within C++.

---

## Current Scope

LonexGrad currently focuses on:

- Tensor computation
- Automatic differentiation
- Neural network construction
- Model training
- CPU execution
- CUDA execution
- C++ integration

The library is structured so additional tensor operations and execution backends can be added over time.

---
