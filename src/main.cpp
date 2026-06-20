#include "mlp_cuda.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Options {
    std::string data_dir = "data";
    std::string save_model_path = "models/mlp_mnist.bin";
    std::string load_model_path;
    std::string predict_vector_path;
    int epochs = 3;
    float learning_rate = 0.01f;
    float learning_rate_decay = 0.5f;
    int learning_rate_decay_every = 3;
    float dropout_rate = 0.0f;
    bool augment_training = false;
    size_t train_limit = 0;
    size_t test_limit = 0;
    size_t examples = 10;
    size_t progress_interval = 1000;
    uint32_t seed = 42;
    int device_id = 0;
};

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "Options:\n"
        << "  --data DIR          MNIST IDX directory (default: data)\n"
        << "  --device N          CUDA device id (default: 0)\n"
        << "  --epochs N          Training epochs (default: 3)\n"
        << "  --lr VALUE          Initial SGD learning rate (default: 0.01)\n"
        << "  --lr-decay VALUE    Multiply lr by VALUE every --lr-decay-every epochs (default: 0.5)\n"
        << "  --lr-decay-every N  Epoch interval for lr decay (default: 3)\n"
        << "  --dropout VALUE     Hidden-layer dropout during training, 0 disables it (default: 0)\n"
        << "  --augment           Apply MNIST-like random affine augmentation during training\n"
        << "  --save PATH         Save trained model (default: models/mlp_mnist.bin)\n"
        << "  --load PATH         Load model weights before training/evaluation/prediction\n"
        << "  --predict P         Predict one 784-float vector produced by scripts/preprocess_image.py\n"
        << "  --train-limit N     Train on first N samples, 0 means all (default: 0)\n"
        << "  --test-limit N      Evaluate on first N samples, 0 means all (default: 0)\n"
        << "  --examples N        Number of predictions to print (default: 10)\n"
        << "  --progress N        Print training progress every N samples (default: 1000)\n"
        << "  --seed N            Random seed for He init (default: 42)\n"
        << "  --help              Show this help\n";
}

std::string require_value(int& index, int argc, char** argv) {
    if (index + 1 >= argc) {
        throw std::runtime_error(std::string("Missing value for ") + argv[index]);
    }
    ++index;
    return argv[index];
}

Options parse_options(int argc, char** argv) {
    Options options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--data") {
            options.data_dir = require_value(i, argc, argv);
        } else if (arg == "--device") {
            options.device_id = std::stoi(require_value(i, argc, argv));
        } else if (arg == "--epochs") {
            options.epochs = std::stoi(require_value(i, argc, argv));
        } else if (arg == "--lr") {
            options.learning_rate = std::stof(require_value(i, argc, argv));
        } else if (arg == "--lr-decay") {
            options.learning_rate_decay = std::stof(require_value(i, argc, argv));
        } else if (arg == "--lr-decay-every") {
            options.learning_rate_decay_every = std::stoi(require_value(i, argc, argv));
        } else if (arg == "--dropout") {
            options.dropout_rate = std::stof(require_value(i, argc, argv));
        } else if (arg == "--augment") {
            options.augment_training = true;
        } else if (arg == "--save") {
            options.save_model_path = require_value(i, argc, argv);
        } else if (arg == "--load") {
            options.load_model_path = require_value(i, argc, argv);
        } else if (arg == "--predict") {
            options.predict_vector_path = require_value(i, argc, argv);
        } else if (arg == "--train-limit") {
            options.train_limit = static_cast<size_t>(std::stoull(require_value(i, argc, argv)));
        } else if (arg == "--test-limit") {
            options.test_limit = static_cast<size_t>(std::stoull(require_value(i, argc, argv)));
        } else if (arg == "--examples") {
            options.examples = static_cast<size_t>(std::stoull(require_value(i, argc, argv)));
        } else if (arg == "--progress") {
            options.progress_interval = static_cast<size_t>(std::stoull(require_value(i, argc, argv)));
        } else if (arg == "--seed") {
            options.seed = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }

    if (options.device_id < 0) {
        throw std::runtime_error("--device must be >= 0");
    }
    if (options.epochs < 0) {
        throw std::runtime_error("--epochs must be >= 0");
    }
    if (options.learning_rate <= 0.0f) {
        throw std::runtime_error("--lr must be > 0");
    }
    if (options.learning_rate_decay <= 0.0f || options.learning_rate_decay > 1.0f) {
        throw std::runtime_error("--lr-decay must be > 0 and <= 1");
    }
    if (options.learning_rate_decay_every < 1) {
        throw std::runtime_error("--lr-decay-every must be >= 1");
    }
    if (options.dropout_rate < 0.0f || options.dropout_rate >= 1.0f) {
        throw std::runtime_error("--dropout must be >= 0 and < 1");
    }
    if (!options.predict_vector_path.empty() && options.epochs == 0 &&
        options.load_model_path.empty()) {
        throw std::runtime_error("--predict with --epochs 0 requires --load");
    }

    return options;
}

float learning_rate_for_epoch(const Options& options, int epoch) {
    const int completed_decay_steps = (epoch - 1) / options.learning_rate_decay_every;
    return options.learning_rate *
           std::pow(options.learning_rate_decay, static_cast<float>(completed_decay_steps));
}

std::vector<float> load_image_vector(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Could not open image vector: " + path);
    }

    std::vector<float> image;
    image.reserve(784);

    float value = 0.0f;
    while (file >> value) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("Image vector contains a non-finite value");
        }
        image.push_back(value);
    }

    if (image.size() != 784) {
        throw std::runtime_error("Image vector must contain exactly 784 float values");
    }
    return image;
}

std::vector<float> softmax(const std::vector<float>& logits) {
    const float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probabilities(logits.size());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = std::exp(logits[i] - max_logit);
        sum += probabilities[i];
    }
    for (float& value : probabilities) {
        value /= sum;
    }
    return probabilities;
}

void print_prediction(MLPCuda& model, const std::string& vector_path) {
    const std::vector<float> image = load_image_vector(vector_path);
    const std::vector<float> logits = model.predict_logits(image.data());
    const std::vector<float> probabilities = softmax(logits);
    const int prediction = static_cast<int>(
        std::distance(probabilities.begin(),
                      std::max_element(probabilities.begin(), probabilities.end())));

    std::cout << "\nPREDICCION: " << prediction << "\n"
              << "  confidence=" << probabilities[static_cast<size_t>(prediction)] * 100.0f
              << "%\n";
}

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

        MLPCuda model(options.device_id, options.seed);

        if (!options.load_model_path.empty()) {
            model.load_model(options.load_model_path);
            std::cout << "Loaded model from " << options.load_model_path << "\n";
        }

        if (!options.predict_vector_path.empty() && options.epochs == 0) {
            print_prediction(model, options.predict_vector_path);
            return 0;
        }

        const std::string train_images = join_path(options.data_dir, "train-images-idx3-ubyte");
        const std::string train_labels = join_path(options.data_dir, "train-labels-idx1-ubyte");
        const std::string test_images = join_path(options.data_dir, "t10k-images-idx3-ubyte");
        const std::string test_labels = join_path(options.data_dir, "t10k-labels-idx1-ubyte");

        std::cout << "Loading MNIST from " << options.data_dir << "\n";
        MNISTDataset train = load_mnist_dataset(train_images, train_labels);
        MNISTDataset test = load_mnist_dataset(test_images, test_labels);

        std::cout << "Model: 784 -> 512 -> 256 -> 128 -> 64 -> 10\n";
        std::cout << "Dropout: " << options.dropout_rate
                  << (options.dropout_rate == 0.0f ? " (disabled)" : " (training only)")
                  << "\n";
        std::cout << "Learning rate: initial=" << options.learning_rate
                  << " decay=" << options.learning_rate_decay
                  << " every " << options.learning_rate_decay_every << " epochs\n";
        std::cout << "Augmentation: "
                  << (options.augment_training ? "enabled (training only)" : "disabled")
                  << "\n";
        std::cout << "Train samples: " << effective_limit(options.train_limit, train.size())
                  << "  Test samples: " << effective_limit(options.test_limit, test.size())
                  << "\n";

        for (int epoch = 1; epoch <= options.epochs; ++epoch) {
            const float epoch_learning_rate = learning_rate_for_epoch(options, epoch);
            std::cout << "\nEpoch " << epoch << "/" << options.epochs
                      << "  lr=" << epoch_learning_rate << "\n";
            Timer timer;
            const float avg_loss = model.train_epoch(train,
                                                     epoch_learning_rate,
                                                     options.dropout_rate,
                                                     options.augment_training,
                                                     options.train_limit,
                                                     options.progress_interval);
            std::cout << "Epoch " << epoch << " average loss: " << avg_loss
                      << "  time: " << timer.seconds() << "s\n";
        }

        if (options.epochs > 0 && !options.save_model_path.empty()) {
            model.save_model(options.save_model_path);
            std::cout << "\nSaved model to " << options.save_model_path << "\n";
        }

        std::cout << "\nEvaluating on test set...\n";
        const EvaluationResult result = model.evaluate(test, options.test_limit);
        std::cout << "Final accuracy: " << (result.accuracy * 100.0f)
                  << "% (" << result.correct << "/" << result.total << ")\n";

        const size_t example_count = effective_limit(options.examples, test.size());
        std::cout << "\nExample predictions:\n";
        for (size_t i = 0; i < example_count; ++i) {
            const int prediction = model.predict(test.image_ptr(i));
            std::cout << "  sample " << i
                      << "  label=" << static_cast<int>(test.labels[i])
                      << "  prediction=" << prediction << "\n";
        }

        if (!options.predict_vector_path.empty()) {
            std::cout << "\nCustom image prediction:\n";
            print_prediction(model, options.predict_vector_path);
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
