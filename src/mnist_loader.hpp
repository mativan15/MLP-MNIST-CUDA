#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

constexpr float MNIST_MEAN = 0.1307f;
constexpr float MNIST_STD = 0.3081f;

struct MNISTDataset {
    std::vector<float> images;
    std::vector<uint8_t> labels;
    uint32_t rows = 0;
    uint32_t cols = 0;

    size_t size() const;
    size_t image_size() const;
    const float* image_ptr(size_t index) const;
};

MNISTDataset load_mnist_dataset(const std::string& image_path,
                                const std::string& label_path);
