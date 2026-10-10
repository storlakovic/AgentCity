cmake_minimum_required(VERSION 4.3)
project(agent_brain)

set(CMAKE_CXX_STANDARD 20)

add_executable(agent_brain main.cpp
        layer/layer.cpp
        layer/layer.hpp
        brain/neural_network.cpp
        brain/neural_network.hpp
        training/train.cpp
        training/train.hpp
        environment/action.cpp
        environment/action.hpp
        environment/state.cpp
        environment/state.hpp
        environment/environment.cpp
        environment/environment.hpp
        environment/step_result.cpp
        environment/step_result.hpp
        training/experience.cpp
        training/experience.hpp
)
