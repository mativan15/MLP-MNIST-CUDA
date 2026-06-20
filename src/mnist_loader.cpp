#include "mnist_loader.hpp"

#include <fstream>
#include <stdexcept>

namespace {

uint32_t read_be_u32(std::ifstream& file) {
    unsigned char bytes[4] = {0, 0, 0, 0};
    file.read(reinterpret_cast<char*>(bytes), 4);
    if (!file) {
        throw std::runtime_error("Unexpected end of IDX file");
    }

    return (static_cast<uint32_t>(bytes[0]) << 24) |
           (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8) |
           static_cast<uint32_t>(bytes[3]);
}

void require_file_open(const std::ifstream& file, const std::string& path) {
    if (!file) {
        throw std::runtime_error("Could not open file: " + path);
    }
}

} // namespace

size_t MNISTDataset::size() const {
    return labels.size();
}

size_t MNISTDataset::image_size() const {
    return static_cast<size_t>(rows) * static_cast<size_t>(cols);
}

const float* MNISTDataset::image_ptr(size_t index) const {
    return images.data() + index * image_size();
}

MNISTDataset load_mnist_dataset(const std::string& image_path, const std::string& label_path) {
    std::ifstream image_file(image_path, std::ios::binary);
    std::ifstream label_file(label_path, std::ios::binary);
    require_file_open(image_file, image_path);
    require_file_open(label_file, label_path);

    const uint32_t image_magic = read_be_u32(image_file);
    const uint32_t image_count = read_be_u32(image_file);
    const uint32_t rows = read_be_u32(image_file);
    const uint32_t cols = read_be_u32(image_file);

    const uint32_t label_magic = read_be_u32(label_file);
    const uint32_t label_count = read_be_u32(label_file);

    if (image_magic != 2051) {
        throw std::runtime_error("Invalid image IDX magic in " + image_path);
    }
    if (label_magic != 2049) {
        throw std::runtime_error("Invalid label IDX magic in " + label_path);
    }
    if (image_count != label_count) {
        throw std::runtime_error("Image/label count mismatch");
    }
    if (rows != 28 || cols != 28) {
        throw std::runtime_error("Expected MNIST images with shape 28x28");
    }

    MNISTDataset dataset;
    dataset.rows = rows;
    dataset.cols = cols;
    dataset.labels.resize(label_count);
    dataset.images.resize(static_cast<size_t>(image_count) * rows * cols);

    std::vector<unsigned char> raw_images(dataset.images.size());
    image_file.read(reinterpret_cast<char*>(raw_images.data()),static_cast<std::streamsize>(raw_images.size()));
    label_file.read(reinterpret_cast<char*>(dataset.labels.data()),static_cast<std::streamsize>(dataset.labels.size()));

    if (!image_file || !label_file) {
        throw std::runtime_error("Unexpected end while reading MNIST data");
    }

    for (size_t i = 0; i < raw_images.size(); ++i) {
        const float pixel = static_cast<float>(raw_images[i]) / 255.0f;
        dataset.images[i] = (pixel - MNIST_MEAN) / MNIST_STD;
    }

    return dataset;
}
