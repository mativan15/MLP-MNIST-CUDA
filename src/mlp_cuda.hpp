#pragma once

#include "mnist_loader.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct EvaluationResult {
    size_t correct = 0;
    size_t total = 0;
    float accuracy = 0.0f;
};

class MLPCuda {
public:
    explicit MLPCuda(int device_id = 0, uint32_t seed = 42);
    ~MLPCuda();

    MLPCuda(const MLPCuda&) = delete;
    MLPCuda& operator=(const MLPCuda&) = delete;

    float train_epoch(const MNISTDataset& dataset,
                      float learning_rate,
                      float dropout_rate,
                      bool augment_training,
                      size_t sample_limit,
                      size_t progress_interval);

    EvaluationResult evaluate(const MNISTDataset& dataset,
                              size_t sample_limit);

    void save_model(const std::string& path) const;
    void load_model(const std::string& path);

    int predict(const float* image);
    std::vector<float> predict_logits(const float* image);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
