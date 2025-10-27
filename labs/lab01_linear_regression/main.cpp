#include <iostream>
#include <memory>
#include <filesystem>
#include <chrono>
#include <fmt/format.h>
#include <fmt/chrono.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <cxxopts.hpp>

#include "ml_labs/core/tensor.hpp"

// Forward declarations for our implementations
#include "src/data_loader.cpp"
#include "src/gradient_descent.cpp"
#include "src/visualization.cpp"

using namespace ml_labs::core;
using namespace ml_labs::lab01;

namespace fs = std::filesystem;

// Setup logging
void setup_logging(const std::string& log_level = "info") {
    // Create color multi-threaded logger
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::debug);
    console_sink->set_pattern("[%H:%M:%S.%e] [%^%l%$] [thread %t] %v");
    
    // Create file sink
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("lab01_linear_regression.log", true);
    file_sink->set_level(spdlog::level::trace);
    file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [thread %t] %v");
    
    // Create logger with multiple sinks
    std::vector<spdlog::sink_ptr> sinks {console_sink, file_sink};
    auto logger = std::make_shared<spdlog::logger>("lab01", sinks.begin(), sinks.end());
    
    // Set log level based on input
    if (log_level == "trace") {
        logger->set_level(spdlog::level::trace);
    } else if (log_level == "debug") {
        logger->set_level(spdlog::level::debug);
    } else if (log_level == "info") {
        logger->set_level(spdlog::level::info);
    } else if (log_level == "warn") {
        logger->set_level(spdlog::level::warn);
    } else if (log_level == "error") {
        logger->set_level(spdlog::level::err);
    } else {
        logger->set_level(spdlog::level::info);
    }
    
    // Register as default logger
    spdlog::set_default_logger(logger);
    spdlog::flush_every(std::chrono::seconds(3));
    
    spdlog::info("=================================================");
    spdlog::info("ML Labs - Lab 01: Linear Regression");
    spdlog::info("=================================================");
}

// Print dataset statistics
void print_dataset_stats(const Dataset& dataset, const std::string& name) {
    spdlog::info("");
    spdlog::info("{} Dataset Statistics:", name);
    spdlog::info("  Samples: {}", dataset.n_samples);
    spdlog::info("  Features: {}", dataset.n_features);
    
    if (!dataset.feature_names.empty()) {
        spdlog::info("  Feature names: {}", fmt::join(dataset.feature_names, ", "));
    }
    spdlog::info("  Target: {}", dataset.target_name);
    
    // Compute basic statistics
    auto& X = dataset.features;
    auto& y = dataset.targets;
    
    spdlog::info("  Features range: [{:.4f}, {:.4f}]", X.min(), X.max());
    spdlog::info("  Target range: [{:.4f}, {:.4f}]", y.min(), y.max());
    spdlog::info("  Target mean: {:.4f}", y.mean());
}

// Print model coefficients
void print_model_coefficients(const LinearRegression& model, const std::vector<std::string>& feature_names) {
    auto weights = model.get_weights();
    auto bias = model.get_bias();
    
    spdlog::info("");
    spdlog::info("Model Coefficients:");
    spdlog::info("  Intercept (bias): {:.6f}", bias[0]);
    
    if (!feature_names.empty()) {
        spdlog::info("  Weights:");
        for (size_t i = 0; i < weights.shape()[0]; ++i) {
            std::string name = (i < feature_names.size()) ? feature_names[i] : fmt::format("feature_{}", i);
            spdlog::info("    {}: {:.6f}", name, weights.at({i, 0}));
        }
    } else {
        spdlog::info("  Weights shape: [{}, {}]", weights.shape()[0], weights.shape()[1]);
    }
}

// Evaluate model performance
void evaluate_model(LinearRegression& model, const Dataset& dataset, const std::string& dataset_name) {
    auto predictions = model.predict(dataset.features);
    
    float mse_val = model.compute_loss(predictions, dataset.targets);
    float rmse_val = model.rmse(dataset.targets, predictions);
    float mae_val = model.mae(dataset.targets, predictions);
    float r2_val = model.r2_score(dataset.targets, predictions);
    
    spdlog::info("");
    spdlog::info("{} Set Performance:", dataset_name);
    spdlog::info("  MSE:  {:.6f}", mse_val);
    spdlog::info("  RMSE: {:.6f}", rmse_val);
    spdlog::info("  MAE:  {:.6f}", mae_val);
    spdlog::info("  R²:   {:.6f}", r2_val);
}

// Run gradient descent experiment
void run_gradient_descent_experiment(const Dataset& train_data, const Dataset& test_data) {
    spdlog::info("");
    spdlog::info("=== Gradient Descent Experiment ===");
    
    // Configure optimizer
    OptimizerConfig config;
    config.learning_rate = 0.01f;
    config.max_epochs = 1000;
    config.batch_size = 32;
    config.tolerance = 1e-6f;
    config.verbose = true;
    config.print_every = 100;
    config.l2_lambda = 0.001f;  // Small L2 regularization
    config.use_lr_decay = true;
    config.lr_decay_rate = 0.99f;
    config.lr_decay_steps = 100;
    config.grad_clip_value = 5.0f;
    config.early_stopping = true;
    config.patience = 50;
    config.min_delta = 1e-5f;
    
    // Create and train model
    LinearRegression model(config);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Train with validation
    auto history = model.fit(
        train_data.features, train_data.targets,
        &test_data.features, &test_data.targets
    );
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    spdlog::info("Training completed in {:.3f} seconds", duration.count() / 1000.0);
    
    // Print coefficients
    print_model_coefficients(model, train_data.feature_names);
    
    // Evaluate on both sets
    evaluate_model(model, train_data, "Training");
    evaluate_model(model, test_data, "Test");
    
    // Plot training history
    TrainingVisualizer visualizer;
    visualizer.plot_training_history(history, "gradient_descent_history.png");
    
    // Plot predictions vs actual
    auto train_predictions = model.predict(train_data.features);
    auto test_predictions = model.predict(test_data.features);
    
    visualizer.plot_predictions(
        train_data.targets, train_predictions,
        "gradient_descent_train_predictions.png", "Training Set"
    );
    
    visualizer.plot_predictions(
        test_data.targets, test_predictions,
        "gradient_descent_test_predictions.png", "Test Set"
    );
    
    // Plot residuals
    visualizer.plot_residuals(
        test_data.targets, test_predictions,
        "gradient_descent_residuals.png"
    );
}

// Compare different optimizers
void run_optimizer_comparison(const Dataset& train_data, const Dataset& test_data) {
    spdlog::info("");
    spdlog::info("=== Optimizer Comparison ===");
    
    struct OptimizerExperiment {
        std::string name;
        float learning_rate;
        size_t batch_size;
        float l2_lambda;
    };
    
    std::vector<OptimizerExperiment> experiments = {
        {"SGD", 0.01f, 32, 0.001f},
        {"SGD_high_lr", 0.1f, 32, 0.001f},
        {"SGD_full_batch", 0.01f, train_data.n_samples, 0.001f},
        {"SGD_mini_batch", 0.01f, 64, 0.001f},
        {"SGD_no_reg", 0.01f, 32, 0.0f},
        {"SGD_high_reg", 0.01f, 32, 0.01f},
    };
    
    std::vector<TrainingHistory> histories;
    std::vector<float> final_test_scores;
    
    for (const auto& exp : experiments) {
        spdlog::info("");
        spdlog::info("Running experiment: {}", exp.name);
        
        OptimizerConfig config;
        config.learning_rate = exp.learning_rate;
        config.max_epochs = 500;
        config.batch_size = exp.batch_size;
        config.l2_lambda = exp.l2_lambda;
        config.verbose = false;
        config.early_stopping = true;
        config.patience = 20;
        
        LinearRegression model(config);
        auto history = model.fit(
            train_data.features, train_data.targets,
            &test_data.features, &test_data.targets
        );
        
        histories.push_back(history);
        
        // Evaluate
        auto test_pred = model.predict(test_data.features);
        float r2 = model.r2_score(test_data.targets, test_pred);
        float rmse = model.rmse(test_data.targets, test_pred);
        
        final_test_scores.push_back(r2);
        
        spdlog::info("  Final test R²: {:.6f}, RMSE: {:.6f}", r2, rmse);
        spdlog::info("  Training time: {:.2f}s", history.training_time_seconds);
        spdlog::info("  Epochs trained: {}", history.epochs_trained);
    }
    
    // Find best model
    auto best_idx = std::distance(final_test_scores.begin(), 
                                 std::max_element(final_test_scores.begin(), final_test_scores.end()));
    spdlog::info("");
    spdlog::info("Best model: {} with R² = {:.6f}", 
                experiments[best_idx].name, final_test_scores[best_idx]);
    
    // Plot comparison
    TrainingVisualizer visualizer;
    visualizer.plot_optimizer_comparison(histories, experiments, "optimizer_comparison.png");
}

// Run learning rate analysis
void run_learning_rate_analysis(const Dataset& train_data, const Dataset& test_data) {
    spdlog::info("");
    spdlog::info("=== Learning Rate Analysis ===");
    
    std::vector<float> learning_rates = {0.0001f, 0.001f, 0.01f, 0.1f, 1.0f};
    std::vector<float> final_losses;
    std::vector<size_t> convergence_epochs;
    
    for (float lr : learning_rates) {
        spdlog::info("Testing learning rate: {}", lr);
        
        OptimizerConfig config;
        config.learning_rate = lr;
        config.max_epochs = 200;
        config.batch_size = 32;
        config.verbose = false;
        
        try {
            LinearRegression model(config);
            auto history = model.fit(train_data.features, train_data.targets);
            
            final_losses.push_back(history.train_losses.back());
            convergence_epochs.push_back(history.epochs_trained);
            
            spdlog::info("  Final loss: {:.6f}, Epochs: {}", 
                        history.train_losses.back(), history.epochs_trained);
        } catch (const std::exception& e) {
            spdlog::warn("  Failed with error: {}", e.what());
            final_losses.push_back(std::numeric_limits<float>::max());
            convergence_epochs.push_back(config.max_epochs);
        }
    }
    
    // Find optimal learning rate
    auto best_idx = std::distance(final_losses.begin(), 
                                 std::min_element(final_losses.begin(), final_losses.end()));
    if (best_idx >= 0 && best_idx < learning_rates.size()) {
        spdlog::info("");
        spdlog::info("Optimal learning rate: {} with final loss: {:.6f}", 
                    learning_rates[best_idx], final_losses[best_idx]);
    }
}

// Main function
int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        cxxopts::Options options("lab01_linear_regression", 
                                "ML Labs - Linear Regression with Gradient Descent");
        
        options.add_options()
            ("train", "Path to training data CSV", cxxopts::value<std::string>())
            ("test", "Path to test data CSV", cxxopts::value<std::string>())
            ("mode", "Execution mode: train, compare, analyze", 
             cxxopts::value<std::string>()->default_value("train"))
            ("epochs", "Maximum number of epochs", cxxopts::value<size_t>()->default_value("1000"))
            ("lr", "Learning rate", cxxopts::value<float>()->default_value("0.01"))
            ("batch-size", "Batch size", cxxopts::value<size_t>()->default_value("32"))
            ("l2", "L2 regularization strength", cxxopts::value<float>()->default_value("0.0"))
            ("log-level", "Logging level: trace, debug, info, warn, error", 
             cxxopts::value<std::string>()->default_value("info"))
            ("plot", "Generate plots", cxxopts::value<bool>()->default_value("true"))
            ("h,help", "Print usage");
        
        auto result = options.parse(argc, argv);
        
        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            return 0;
        }
        
        // Setup logging
        setup_logging(result["log-level"].as<std::string>());
        
        // Check for data files
        std::string train_path = "data/california_housing_train.csv";
        std::string test_path = "data/california_housing_test.csv";
        
        if (result.count("train")) {
            train_path = result["train"].as<std::string>();
        }
        if (result.count("test")) {
            test_path = result["test"].as<std::string>();
        }
        
        // Verify files exist
        if (!fs::exists(train_path)) {
            spdlog::error("Training file not found: {}", train_path);
            return 1;
        }
        if (!fs::exists(test_path)) {
            spdlog::error("Test file not found: {}", test_path);
            return 1;
        }
        
        spdlog::info("Loading data from:");
        spdlog::info("  Train: {}", train_path);
        spdlog::info("  Test:  {}", test_path);
        
        // Load data
        auto train_data = CaliforniaHousingLoader::load_train(train_path);
        auto test_data = CaliforniaHousingLoader::load_test(test_path);
        
        // Print statistics
        print_dataset_stats(train_data, "Training");
        print_dataset_stats(test_data, "Test");
        
        // Run experiments based on mode
        std::string mode = result["mode"].as<std::string>();
        
        if (mode == "train") {
            run_gradient_descent_experiment(train_data, test_data);
        } else if (mode == "compare") {
            run_optimizer_comparison(train_data, test_data);
        } else if (mode == "analyze") {
            run_learning_rate_analysis(train_data, test_data);
        } else if (mode == "all") {
            run_gradient_descent_experiment(train_data, test_data);
            run_optimizer_comparison(train_data, test_data);
            run_learning_rate_analysis(train_data, test_data);
        } else {
            spdlog::error("Unknown mode: {}", mode);
            return 1;
        }
        
        spdlog::info("");
        spdlog::info("=================================================");
        spdlog::info("Lab 01 completed successfully!");
        spdlog::info("=================================================");
        
        return 0;
        
    } catch (const std::exception& e) {
        spdlog::error("Fatal error: {}", e.what());
        return 1;
    }
}
