cmake_minimum_required(VERSION 4.3)
project(neural_network)

set(CMAKE_CXX_STANDARD 20)

add_executable(neural_network main.cpp
        neuron/neuron.cpp
        neuron/neuron.h
        neural_network.cpp
        neural_network.h
        training/train.cpp
        training/train.h)
