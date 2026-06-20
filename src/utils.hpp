#pragma once

#include <chrono>
#include <cstddef>
#include <iostream>
#include <string>

class Timer {
public:
    Timer() : start_(std::chrono::steady_clock::now()) {}

    double seconds() const {
        const auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - start_).count();
    }

private:
    std::chrono::steady_clock::time_point start_;
};

inline std::string join_path(const std::string& dir, const std::string& file) {
    if (dir.empty() || dir.back() == '/' || dir.back() == '\\') {
        return dir + file;
    }
    return dir + "/" + file;
}

inline size_t effective_limit(size_t requested, size_t available) {
    if (requested == 0 || requested > available) {
        return available;
    }
    return requested;
}

inline void print_progress(size_t current, size_t total, double loss_sum, double seconds) {
    const double avg_loss = current > 0 ? loss_sum / static_cast<double>(current) : 0.0;
    std::cout << "  sample " << current << "/" << total
              << "  avg_loss=" << avg_loss
              << "  time=" << seconds << "s\n";
}
