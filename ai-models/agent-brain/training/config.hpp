#pragma once

namespace train::config {
    inline constexpr int epochs = 10000000;
    inline constexpr double learning_rate = 0.01;
    inline constexpr double gamma = 0.95;
    inline constexpr int max_steps = 100;
    inline constexpr int log_interval = 100;
    inline constexpr int evaluation_interval = 1000;
    inline constexpr int evaluation_episodes = 100;
    inline constexpr char checkpoint_path[] = "best_model.json";
}
