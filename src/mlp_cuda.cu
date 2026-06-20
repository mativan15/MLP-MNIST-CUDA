#include "mlp_cuda.hpp"

#include "utils.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

constexpr std::array<uint32_t, 6> kLayerSizes = {784, 512, 256, 128, 64, 10};
constexpr size_t kLayerCount = kLayerSizes.size() - 1;

// Kept equal to the Metal project so the same .bin weights can be loaded.
constexpr char kModelMagic[8] = {'M', 'L', 'P', 'M', 'T', 'L', '1', '\0'};
constexpr uint32_t kModelVersion = 2;

constexpr uint32_t kImageWidth = 28;
constexpr uint32_t kImageHeight = 28;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kAugmentDegrees = 10.0f;
constexpr float kAugmentTranslate = 0.10f;
constexpr float kAugmentMinScale = 0.85f;
constexpr float kAugmentMaxScale = 1.15f;
constexpr float kAugmentShearDegrees = 5.0f;
constexpr float kNormalizedBlack = (0.0f - MNIST_MEAN) / MNIST_STD;
constexpr int kThreadsPerBlock = 256;

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

int blocks_for(size_t count) {
    return static_cast<int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
}

struct DeviceBuffer {
    float* ptr = nullptr;
    size_t count = 0;

    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t value_count) {
        allocate(value_count);
    }

    ~DeviceBuffer() {
        if (ptr) {
            cudaFree(ptr);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr(other.ptr),
          count(other.count) {
        other.ptr = nullptr;
        other.count = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr) {
                cudaFree(ptr);
            }
            ptr = other.ptr;
            count = other.count;
            other.ptr = nullptr;
            other.count = 0;
        }
        return *this;
    }

    void allocate(size_t value_count) {
        if (value_count == 0) {
            return;
        }
        count = value_count;
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(float)), "cudaMalloc");
    }

    void zero() {
        cuda_check(cudaMemset(ptr, 0, count * sizeof(float)), "cudaMemset");
    }
};

void copy_to_buffer(const DeviceBuffer& buffer, const float* source, size_t count) {
    if (count > buffer.count) {
        throw std::runtime_error("Host-to-device copy exceeds CUDA buffer size");
    }
    cuda_check(cudaMemcpy(buffer.ptr,
                          source,
                          count * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy host to device");
}

std::vector<float> copy_from_buffer(const DeviceBuffer& buffer, size_t count) {
    if (count > buffer.count) {
        throw std::runtime_error("Device-to-host copy exceeds CUDA buffer size");
    }
    std::vector<float> values(count);
    cuda_check(cudaMemcpy(values.data(),
                          buffer.ptr,
                          count * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy device to host");
    return values;
}

float sample_bilinear(const float* image, float x, float y, float fill_value) {
    if (x < 0.0f || y < 0.0f ||
        x > static_cast<float>(kImageWidth - 1) ||
        y > static_cast<float>(kImageHeight - 1)) {
        return fill_value;
    }

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min<int>(x0 + 1, kImageWidth - 1);
    const int y1 = std::min<int>(y0 + 1, kImageHeight - 1);
    const float dx = x - static_cast<float>(x0);
    const float dy = y - static_cast<float>(y0);

    const float v00 = image[y0 * kImageWidth + x0];
    const float v10 = image[y0 * kImageWidth + x1];
    const float v01 = image[y1 * kImageWidth + x0];
    const float v11 = image[y1 * kImageWidth + x1];

    const float top = v00 * (1.0f - dx) + v10 * dx;
    const float bottom = v01 * (1.0f - dx) + v11 * dx;
    return top * (1.0f - dy) + bottom * dy;
}

template <typename T>
void write_binary(std::ofstream& file, const T& value) {
    file.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!file) {
        throw std::runtime_error("Could not write model file");
    }
}

template <typename T>
void read_binary(std::ifstream& file, T& value) {
    file.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!file) {
        throw std::runtime_error("Unexpected end of model file");
    }
}

__global__ void dense_forward_kernel(const float* weights,
                                     const float* input,
                                     const float* bias,
                                     float* output,
                                     uint32_t input_size,
                                     uint32_t output_size) {
    const uint32_t gid = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid >= output_size) {
        return;
    }

    float sum = bias[gid];
    const uint32_t row_start = gid * input_size;
    for (uint32_t i = 0; i < input_size; ++i) {
        sum += weights[row_start + i] * input[i];
    }
    output[gid] = sum;
}

__global__ void relu_kernel(float* values, uint32_t size) {
    const uint32_t gid = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid < size) {
        values[gid] = fmaxf(values[gid], 0.0f);
    }
}

__global__ void dropout_kernel(float* values, const float* mask, uint32_t size) {
    const uint32_t gid = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid < size) {
        values[gid] *= mask[gid];
    }
}

__global__ void softmax_cross_entropy_kernel(const float* logits,
                                             float* probabilities,
                                             float* loss,
                                             uint32_t label,
                                             uint32_t output_size) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    float max_logit = logits[0];
    for (uint32_t i = 1; i < output_size; ++i) {
        max_logit = fmaxf(max_logit, logits[i]);
    }

    float sum_exp = 0.0f;
    for (uint32_t i = 0; i < output_size; ++i) {
        const float value = expf(logits[i] - max_logit);
        probabilities[i] = value;
        sum_exp += value;
    }

    const float inv_sum = 1.0f / sum_exp;
    for (uint32_t i = 0; i < output_size; ++i) {
        probabilities[i] *= inv_sum;
    }

    const float selected = fmaxf(probabilities[label], 1.0e-20f);
    loss[0] = -logf(selected);
}

__global__ void output_delta_kernel(const float* probabilities,
                                    float* delta,
                                    uint32_t label,
                                    uint32_t output_size) {
    const uint32_t gid = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid < output_size) {
        delta[gid] = probabilities[gid] - (gid == label ? 1.0f : 0.0f);
    }
}

__global__ void dense_input_delta_kernel(const float* weights,
                                         const float* delta_out,
                                         float* delta_input,
                                         uint32_t input_size,
                                         uint32_t output_size) {
    const uint32_t gid = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid >= input_size) {
        return;
    }

    float sum = 0.0f;
    for (uint32_t out = 0; out < output_size; ++out) {
        sum += weights[out * input_size + gid] * delta_out[out];
    }
    delta_input[gid] = sum;
}

__global__ void relu_backward_kernel(const float* delta_activation,
                                     const float* activation,
                                     float* delta_z,
                                     uint32_t size) {
    const uint32_t gid = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid < size) {
        delta_z[gid] = activation[gid] > 0.0f ? delta_activation[gid] : 0.0f;
    }
}

__global__ void relu_dropout_backward_kernel(const float* delta_activation,
                                             const float* activation,
                                             const float* mask,
                                             float* delta_z,
                                             uint32_t size) {
    const uint32_t gid = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid < size) {
        delta_z[gid] = activation[gid] > 0.0f ? delta_activation[gid] * mask[gid] : 0.0f;
    }
}

__global__ void dense_backward_kernel(const float* delta_out,
                                      const float* input_activation,
                                      float* grad_weights,
                                      float* grad_bias,
                                      uint32_t input_size,
                                      uint32_t output_size) {
    const uint32_t gid = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t total_weights = input_size * output_size;

    if (gid < total_weights) {
        const uint32_t out = gid / input_size;
        const uint32_t in = gid - out * input_size;
        grad_weights[gid] = delta_out[out] * input_activation[in];
    } else if (gid < total_weights + output_size) {
        const uint32_t out = gid - total_weights;
        grad_bias[out] = delta_out[out];
    }
}

__global__ void sgd_update_kernel(float* weights,
                                  float* bias,
                                  const float* grad_weights,
                                  const float* grad_bias,
                                  float learning_rate,
                                  uint32_t total_weights,
                                  uint32_t output_size) {
    const uint32_t gid = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid < total_weights) {
        weights[gid] -= learning_rate * grad_weights[gid];
    } else if (gid < total_weights + output_size) {
        const uint32_t out = gid - total_weights;
        bias[out] -= learning_rate * grad_bias[out];
    }
}

} // namespace

struct MLPCuda::Impl {
    struct LayerBuffers {
        uint32_t input_size = 0;
        uint32_t output_size = 0;
        DeviceBuffer weights;
        DeviceBuffer bias;
        DeviceBuffer grad_weights;
        DeviceBuffer grad_bias;

        LayerBuffers() = default;
        LayerBuffers(const LayerBuffers&) = delete;
        LayerBuffers& operator=(const LayerBuffers&) = delete;
        LayerBuffers(LayerBuffers&&) noexcept = default;
        LayerBuffers& operator=(LayerBuffers&&) noexcept = default;
    };

    std::vector<LayerBuffers> layers;
    std::vector<DeviceBuffer> activations;
    std::vector<DeviceBuffer> deltas;
    std::vector<DeviceBuffer> delta_inputs;
    std::vector<DeviceBuffer> dropout_masks;
    std::vector<std::vector<float>> host_dropout_masks;
    DeviceBuffer probabilities;
    DeviceBuffer loss;
    std::mt19937 dropout_rng;
    std::mt19937 order_rng;
    std::mt19937 augment_rng;

    Impl(int device_id, uint32_t seed)
        : dropout_rng(seed + 1),
          order_rng(seed + 2),
          augment_rng(seed + 3) {
        int device_count = 0;
        cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count == 0) {
            throw std::runtime_error("No CUDA device found");
        }
        if (device_id >= device_count) {
            throw std::runtime_error("CUDA device id is out of range");
        }
        cuda_check(cudaSetDevice(device_id), "cudaSetDevice");

        initialize_buffers(seed);
    }

    void initialize_buffers(uint32_t seed) {
        std::mt19937 rng(seed);

        layers.reserve(kLayerCount);
        for (size_t i = 0; i < kLayerCount; ++i) {
            LayerBuffers layer;
            layer.input_size = kLayerSizes[i];
            layer.output_size = kLayerSizes[i + 1];

            const size_t weight_count =
                static_cast<size_t>(layer.input_size) * layer.output_size;
            layer.weights = DeviceBuffer(weight_count);
            layer.bias = DeviceBuffer(layer.output_size);
            layer.grad_weights = DeviceBuffer(weight_count);
            layer.grad_bias = DeviceBuffer(layer.output_size);

            const float stddev = std::sqrt(2.0f / static_cast<float>(layer.input_size));
            std::normal_distribution<float> normal(0.0f, stddev);

            std::vector<float> host_weights(weight_count);
            for (float& weight : host_weights) {
                weight = normal(rng);
            }
            copy_to_buffer(layer.weights, host_weights.data(), host_weights.size());
            layer.bias.zero();
            layer.grad_weights.zero();
            layer.grad_bias.zero();

            layers.push_back(std::move(layer));
        }

        activations.reserve(kLayerSizes.size());
        for (uint32_t size : kLayerSizes) {
            activations.emplace_back(size);
        }

        dropout_masks.reserve(kLayerCount - 1);
        host_dropout_masks.reserve(kLayerCount - 1);
        for (size_t i = 1; i < kLayerCount; ++i) {
            dropout_masks.emplace_back(kLayerSizes[i]);
            host_dropout_masks.emplace_back(kLayerSizes[i], 1.0f);
        }

        deltas.reserve(kLayerCount);
        delta_inputs.reserve(kLayerCount);
        for (size_t i = 0; i < kLayerCount; ++i) {
            deltas.emplace_back(kLayerSizes[i + 1]);
            delta_inputs.emplace_back(kLayerSizes[i]);
        }

        probabilities = DeviceBuffer(kLayerSizes.back());
        loss = DeviceBuffer(1);
    }

    void launch_dense_forward(const LayerBuffers& layer,
                              const DeviceBuffer& input,
                              const DeviceBuffer& output) {
        dense_forward_kernel<<<blocks_for(layer.output_size), kThreadsPerBlock>>>(
            layer.weights.ptr,
            input.ptr,
            layer.bias.ptr,
            output.ptr,
            layer.input_size,
            layer.output_size);
    }

    void launch_relu(const DeviceBuffer& values, uint32_t size) {
        relu_kernel<<<blocks_for(size), kThreadsPerBlock>>>(values.ptr, size);
    }

    void launch_dropout(const DeviceBuffer& values,
                        const DeviceBuffer& mask,
                        uint32_t size) {
        dropout_kernel<<<blocks_for(size), kThreadsPerBlock>>>(values.ptr, mask.ptr, size);
    }

    void generate_dropout_masks(float dropout_rate) {
        const float keep_probability = 1.0f - dropout_rate;
        const float scale = 1.0f / keep_probability;
        std::bernoulli_distribution keep(keep_probability);

        for (size_t i = 0; i < host_dropout_masks.size(); ++i) {
            std::vector<float>& mask = host_dropout_masks[i];
            for (float& value : mask) {
                value = keep(dropout_rng) ? scale : 0.0f;
            }
            copy_to_buffer(dropout_masks[i], mask.data(), mask.size());
        }
    }

    void launch_forward(bool training, float dropout_rate) {
        for (size_t i = 0; i < kLayerCount; ++i) {
            launch_dense_forward(layers[i], activations[i], activations[i + 1]);
            if (i + 1 < kLayerCount) {
                launch_relu(activations[i + 1], layers[i].output_size);
                if (training && dropout_rate > 0.0f) {
                    launch_dropout(activations[i + 1],
                                   dropout_masks[i],
                                   layers[i].output_size);
                }
            }
        }
    }

    void launch_softmax_loss(uint32_t label) {
        softmax_cross_entropy_kernel<<<1, 1>>>(
            activations.back().ptr,
            probabilities.ptr,
            loss.ptr,
            label,
            kLayerSizes.back());
    }

    void launch_output_delta(uint32_t label) {
        output_delta_kernel<<<blocks_for(kLayerSizes.back()), kThreadsPerBlock>>>(
            probabilities.ptr,
            deltas.back().ptr,
            label,
            kLayerSizes.back());
    }

    void launch_dense_input_delta(const LayerBuffers& layer,
                                  const DeviceBuffer& delta_out,
                                  const DeviceBuffer& delta_input) {
        dense_input_delta_kernel<<<blocks_for(layer.input_size), kThreadsPerBlock>>>(
            layer.weights.ptr,
            delta_out.ptr,
            delta_input.ptr,
            layer.input_size,
            layer.output_size);
    }

    void launch_relu_backward(const DeviceBuffer& delta_activation,
                              const DeviceBuffer& activation,
                              const DeviceBuffer& delta_z,
                              uint32_t size) {
        relu_backward_kernel<<<blocks_for(size), kThreadsPerBlock>>>(
            delta_activation.ptr,
            activation.ptr,
            delta_z.ptr,
            size);
    }

    void launch_relu_dropout_backward(const DeviceBuffer& delta_activation,
                                      const DeviceBuffer& activation,
                                      const DeviceBuffer& mask,
                                      const DeviceBuffer& delta_z,
                                      uint32_t size) {
        relu_dropout_backward_kernel<<<blocks_for(size), kThreadsPerBlock>>>(
            delta_activation.ptr,
            activation.ptr,
            mask.ptr,
            delta_z.ptr,
            size);
    }

    void launch_dense_backward(LayerBuffers& layer,
                               const DeviceBuffer& delta_out,
                               const DeviceBuffer& input_activation) {
        const uint32_t total_weights = layer.input_size * layer.output_size;
        const uint32_t count = total_weights + layer.output_size;
        dense_backward_kernel<<<blocks_for(count), kThreadsPerBlock>>>(
            delta_out.ptr,
            input_activation.ptr,
            layer.grad_weights.ptr,
            layer.grad_bias.ptr,
            layer.input_size,
            layer.output_size);
    }

    void launch_sgd_update(LayerBuffers& layer, float learning_rate) {
        const uint32_t total_weights = layer.input_size * layer.output_size;
        const uint32_t count = total_weights + layer.output_size;
        sgd_update_kernel<<<blocks_for(count), kThreadsPerBlock>>>(
            layer.weights.ptr,
            layer.bias.ptr,
            layer.grad_weights.ptr,
            layer.grad_bias.ptr,
            learning_rate,
            total_weights,
            layer.output_size);
    }

    float train_sample(const float* image, uint8_t label, float learning_rate, float dropout_rate) {
        copy_to_buffer(activations.front(), image, kLayerSizes.front());
        if (dropout_rate > 0.0f) {
            generate_dropout_masks(dropout_rate);
        }

        launch_forward(true, dropout_rate);
        launch_softmax_loss(static_cast<uint32_t>(label));
        launch_output_delta(static_cast<uint32_t>(label));

        for (int layer_index = static_cast<int>(kLayerCount) - 1; layer_index >= 0; --layer_index) {
            LayerBuffers& layer = layers[static_cast<size_t>(layer_index)];

            if (layer_index > 0) {
                launch_dense_input_delta(layer,
                                         deltas[static_cast<size_t>(layer_index)],
                                         delta_inputs[static_cast<size_t>(layer_index)]);
            }

            launch_dense_backward(layer,
                                  deltas[static_cast<size_t>(layer_index)],
                                  activations[static_cast<size_t>(layer_index)]);
            launch_sgd_update(layer, learning_rate);

            if (layer_index > 0) {
                if (dropout_rate > 0.0f) {
                    launch_relu_dropout_backward(delta_inputs[static_cast<size_t>(layer_index)],
                                                 activations[static_cast<size_t>(layer_index)],
                                                 dropout_masks[static_cast<size_t>(layer_index - 1)],
                                                 deltas[static_cast<size_t>(layer_index - 1)],
                                                 layer.input_size);
                } else {
                    launch_relu_backward(delta_inputs[static_cast<size_t>(layer_index)],
                                         activations[static_cast<size_t>(layer_index)],
                                         deltas[static_cast<size_t>(layer_index - 1)],
                                         layer.input_size);
                }
            }
        }

        cuda_check(cudaGetLastError(), "CUDA kernel launch");

        float host_loss = 0.0f;
        cuda_check(cudaMemcpy(&host_loss,
                              loss.ptr,
                              sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy loss to host");
        return host_loss;
    }

    void augment_image(const float* source, float* target) {
        std::uniform_real_distribution<float> angle_dist(-kAugmentDegrees, kAugmentDegrees);
        std::uniform_real_distribution<float> translate_dist(-kAugmentTranslate, kAugmentTranslate);
        std::uniform_real_distribution<float> scale_dist(kAugmentMinScale, kAugmentMaxScale);
        std::uniform_real_distribution<float> shear_dist(-kAugmentShearDegrees, kAugmentShearDegrees);

        const float angle = angle_dist(augment_rng) * kPi / 180.0f;
        const float translate_x = translate_dist(augment_rng) * static_cast<float>(kImageWidth);
        const float translate_y = translate_dist(augment_rng) * static_cast<float>(kImageHeight);
        const float scale = scale_dist(augment_rng);
        const float shear = std::tan(shear_dist(augment_rng) * kPi / 180.0f);

        const float c = std::cos(angle);
        const float s = std::sin(angle);

        const float a00 = c * scale;
        const float a01 = (c * shear - s) * scale;
        const float a10 = s * scale;
        const float a11 = (s * shear + c) * scale;
        const float determinant = a00 * a11 - a01 * a10;

        const float inv00 = a11 / determinant;
        const float inv01 = -a01 / determinant;
        const float inv10 = -a10 / determinant;
        const float inv11 = a00 / determinant;

        const float center_x = (static_cast<float>(kImageWidth) - 1.0f) * 0.5f;
        const float center_y = (static_cast<float>(kImageHeight) - 1.0f) * 0.5f;

        for (uint32_t y = 0; y < kImageHeight; ++y) {
            for (uint32_t x = 0; x < kImageWidth; ++x) {
                const float out_x = static_cast<float>(x) - center_x - translate_x;
                const float out_y = static_cast<float>(y) - center_y - translate_y;

                const float in_x = inv00 * out_x + inv01 * out_y + center_x;
                const float in_y = inv10 * out_x + inv11 * out_y + center_y;
                target[y * kImageWidth + x] = sample_bilinear(source, in_x, in_y, kNormalizedBlack);
            }
        }
    }

    std::vector<float> forward_logits(const float* image) {
        copy_to_buffer(activations.front(), image, kLayerSizes.front());
        launch_forward(false, 0.0f);
        cuda_check(cudaGetLastError(), "CUDA forward kernel launch");
        return copy_from_buffer(activations.back(), kLayerSizes.back());
    }

    void save_model(const std::string& path) const {
        const std::filesystem::path output_path(path);
        if (output_path.has_parent_path()) {
            std::filesystem::create_directories(output_path.parent_path());
        }

        std::ofstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Could not open model file for writing: " + path);
        }

        file.write(kModelMagic, sizeof(kModelMagic));
        write_binary(file, kModelVersion);

        const uint32_t layer_count = static_cast<uint32_t>(kLayerCount);
        write_binary(file, layer_count);
        for (uint32_t size : kLayerSizes) {
            write_binary(file, size);
        }

        for (const LayerBuffers& layer : layers) {
            write_binary(file, layer.input_size);
            write_binary(file, layer.output_size);

            const size_t weight_count =
                static_cast<size_t>(layer.input_size) * layer.output_size;
            const std::vector<float> weights = copy_from_buffer(layer.weights, weight_count);
            const std::vector<float> bias = copy_from_buffer(layer.bias, layer.output_size);
            file.write(reinterpret_cast<const char*>(weights.data()),
                       static_cast<std::streamsize>(weights.size() * sizeof(float)));
            file.write(reinterpret_cast<const char*>(bias.data()),
                       static_cast<std::streamsize>(bias.size() * sizeof(float)));
            if (!file) {
                throw std::runtime_error("Could not write model parameters");
            }
        }
    }

    void load_model(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Could not open model file for reading: " + path);
        }

        char magic[sizeof(kModelMagic)] = {};
        file.read(magic, sizeof(magic));
        if (std::memcmp(magic, kModelMagic, sizeof(kModelMagic)) != 0) {
            throw std::runtime_error("Invalid model file magic: " + path);
        }

        uint32_t version = 0;
        read_binary(file, version);
        if (version != kModelVersion) {
            throw std::runtime_error("Unsupported model file version");
        }

        uint32_t layer_count = 0;
        read_binary(file, layer_count);
        if (layer_count != kLayerCount) {
            throw std::runtime_error("Model layer count does not match this program");
        }

        for (uint32_t expected_size : kLayerSizes) {
            uint32_t file_size = 0;
            read_binary(file, file_size);
            if (file_size != expected_size) {
                throw std::runtime_error("Model architecture does not match this program");
            }
        }

        for (LayerBuffers& layer : layers) {
            uint32_t input_size = 0;
            uint32_t output_size = 0;
            read_binary(file, input_size);
            read_binary(file, output_size);
            if (input_size != layer.input_size || output_size != layer.output_size) {
                throw std::runtime_error("Layer shape mismatch in model file");
            }

            const size_t weight_count =
                static_cast<size_t>(layer.input_size) * layer.output_size;
            std::vector<float> weights(weight_count);
            std::vector<float> bias(layer.output_size);
            file.read(reinterpret_cast<char*>(weights.data()),
                      static_cast<std::streamsize>(weights.size() * sizeof(float)));
            file.read(reinterpret_cast<char*>(bias.data()),
                      static_cast<std::streamsize>(bias.size() * sizeof(float)));
            if (!file) {
                throw std::runtime_error("Unexpected end while reading model parameters");
            }

            copy_to_buffer(layer.weights, weights.data(), weights.size());
            copy_to_buffer(layer.bias, bias.data(), bias.size());
            layer.grad_weights.zero();
            layer.grad_bias.zero();
        }
    }
};

MLPCuda::MLPCuda(int device_id, uint32_t seed)
    : impl_(std::make_unique<Impl>(device_id, seed)) {}

MLPCuda::~MLPCuda() = default;

float MLPCuda::train_epoch(const MNISTDataset& dataset,
                           float learning_rate,
                           float dropout_rate,
                           bool augment_training,
                           size_t sample_limit,
                           size_t progress_interval) {
    const size_t limit = effective_limit(sample_limit, dataset.size());
    std::vector<size_t> sample_order(limit);
    std::iota(sample_order.begin(), sample_order.end(), size_t{0});
    std::shuffle(sample_order.begin(), sample_order.end(), impl_->order_rng);

    double loss_sum = 0.0;
    Timer timer;
    std::vector<float> augmented_image(kLayerSizes.front());

    for (size_t i = 0; i < limit; ++i) {
        const size_t sample_index = sample_order[i];
        const float* image = dataset.image_ptr(sample_index);
        if (augment_training) {
            impl_->augment_image(image, augmented_image.data());
            image = augmented_image.data();
        }

        const float loss = impl_->train_sample(image,
                                               dataset.labels[sample_index],
                                               learning_rate,
                                               dropout_rate);
        loss_sum += static_cast<double>(loss);

        if (progress_interval > 0 && (i + 1) % progress_interval == 0) {
            print_progress(i + 1, limit, loss_sum, timer.seconds());
        }
    }

    return limit > 0 ? static_cast<float>(loss_sum / static_cast<double>(limit)) : 0.0f;
}

EvaluationResult MLPCuda::evaluate(const MNISTDataset& dataset, size_t sample_limit) {
    const size_t limit = effective_limit(sample_limit, dataset.size());
    EvaluationResult result;
    result.total = limit;

    for (size_t i = 0; i < limit; ++i) {
        const int predicted = predict(dataset.image_ptr(i));
        if (predicted == static_cast<int>(dataset.labels[i])) {
            ++result.correct;
        }
    }

    result.accuracy = limit > 0
        ? static_cast<float>(result.correct) / static_cast<float>(limit)
        : 0.0f;
    return result;
}

void MLPCuda::save_model(const std::string& path) const {
    impl_->save_model(path);
}

void MLPCuda::load_model(const std::string& path) {
    impl_->load_model(path);
}

int MLPCuda::predict(const float* image) {
    const std::vector<float> logits = predict_logits(image);
    return static_cast<int>(std::distance(logits.begin(),
        std::max_element(logits.begin(), logits.end())));
}

std::vector<float> MLPCuda::predict_logits(const float* image) {
    return impl_->forward_logits(image);
}
