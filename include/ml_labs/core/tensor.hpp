#pragma once

#include <Eigen/Dense>
#include <concepts>
#include <expected>
#include <format>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <ranges>
#include <algorithm>
#include <numeric>

namespace ml_labs::core {

// Forward declarations
class Tensor;
class TensorView;

// Error types for tensor operations
enum class TensorError {
    InvalidShape,
    ShapeMismatch,
    InvalidIndex,
    InvalidOperation,
    AllocationFailure,
    BroadcastError
};

// Error message helper
inline std::string tensor_error_message(TensorError error) {
    switch (error) {
        case TensorError::InvalidShape: return "Invalid tensor shape";
        case TensorError::ShapeMismatch: return "Tensor shapes do not match";
        case TensorError::InvalidIndex: return "Index out of bounds";
        case TensorError::InvalidOperation: return "Invalid operation for tensor";
        case TensorError::AllocationFailure: return "Failed to allocate memory";
        case TensorError::BroadcastError: return "Cannot broadcast tensors";
    }
    return "Unknown error";
}

// Result type for tensor operations
template<typename T>
using TensorResult = std::expected<T, TensorError>;

// Concept for numeric types suitable for tensors
template<typename T>
concept TensorNumeric = std::floating_point<T> || std::integral<T>;

// Shape type
using Shape = std::vector<size_t>;
using Strides = std::vector<size_t>;

// Compute strides from shape (row-major order)
inline Strides compute_strides(const Shape& shape) {
    if (shape.empty()) return {};
    
    Strides strides(shape.size());
    strides.back() = 1;
    
    for (auto i = static_cast<std::ptrdiff_t>(shape.size() - 2); i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    
    return strides;
}

// Compute total number of elements
inline size_t compute_size(const Shape& shape) {
    return std::reduce(shape.begin(), shape.end(), size_t(1), std::multiplies<>());
}

// Main Tensor class - owning container
template<TensorNumeric T = float>
class TensorImpl {
public:
    using value_type = T;
    using eigen_matrix_t = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using eigen_vector_t = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using eigen_map_t = Eigen::Map<eigen_matrix_t>;
    using const_eigen_map_t = Eigen::Map<const eigen_matrix_t>;

private:
    std::unique_ptr<T[]> data_;  // Owning pointer to data
    Shape shape_;                 // Tensor dimensions
    Strides strides_;            // Strides for indexing
    size_t size_;                // Total number of elements
    bool requires_grad_;         // For autodiff

public:
    // Constructors
    TensorImpl() : size_(0), requires_grad_(false) {}
    
    explicit TensorImpl(const Shape& shape, bool requires_grad = false) 
        : shape_(shape)
        , strides_(compute_strides(shape))
        , size_(compute_size(shape))
        , requires_grad_(requires_grad) {
        if (size_ > 0) {
            data_ = std::make_unique<T[]>(size_);
            std::fill_n(data_.get(), size_, T{0});
        }
    }
    
    // Construct from initializer list (2D)
    TensorImpl(std::initializer_list<std::initializer_list<T>> init) 
        : requires_grad_(false) {
        size_t rows = init.size();
        size_t cols = rows > 0 ? init.begin()->size() : 0;
        
        shape_ = {rows, cols};
        strides_ = compute_strides(shape_);
        size_ = rows * cols;
        
        if (size_ > 0) {
            data_ = std::make_unique<T[]>(size_);
            size_t idx = 0;
            for (const auto& row : init) {
                for (const auto& val : row) {
                    data_[idx++] = val;
                }
            }
        }
    }
    
    // Construct from data pointer (non-owning view)
    TensorImpl(T* data, const Shape& shape, bool copy = true) 
        : shape_(shape)
        , strides_(compute_strides(shape))
        , size_(compute_size(shape))
        , requires_grad_(false) {
        if (size_ > 0) {
            if (copy) {
                data_ = std::make_unique<T[]>(size_);
                std::copy_n(data, size_, data_.get());
            } else {
                // For views, we'd need a different design
                // This constructor always copies for now
                data_ = std::make_unique<T[]>(size_);
                std::copy_n(data, size_, data_.get());
            }
        }
    }
    
    // Copy constructor
    TensorImpl(const TensorImpl& other) 
        : shape_(other.shape_)
        , strides_(other.strides_)
        , size_(other.size_)
        , requires_grad_(other.requires_grad_) {
        if (size_ > 0 && other.data_) {
            data_ = std::make_unique<T[]>(size_);
            std::copy_n(other.data_.get(), size_, data_.get());
        }
    }
    
    // Move constructor
    TensorImpl(TensorImpl&&) noexcept = default;
    
    // Assignment operators
    TensorImpl& operator=(const TensorImpl& other) {
        if (this != &other) {
            shape_ = other.shape_;
            strides_ = other.strides_;
            size_ = other.size_;
            requires_grad_ = other.requires_grad_;
            
            if (size_ > 0 && other.data_) {
                data_ = std::make_unique<T[]>(size_);
                std::copy_n(other.data_.get(), size_, data_.get());
            } else {
                data_.reset();
            }
        }
        return *this;
    }
    
    TensorImpl& operator=(TensorImpl&&) noexcept = default;
    
    // Factory methods
    static TensorImpl zeros(const Shape& shape) {
        return TensorImpl(shape);
    }
    
    static TensorImpl ones(const Shape& shape) {
        TensorImpl tensor(shape);
        if (tensor.size_ > 0) {
            std::fill_n(tensor.data_.get(), tensor.size_, T{1});
        }
        return tensor;
    }
    
    static TensorImpl random(const Shape& shape, T min_val = 0, T max_val = 1) {
        TensorImpl tensor(shape);
        if (tensor.size_ > 0) {
            std::random_device rd;
            std::mt19937 gen(rd());
            
            if constexpr (std::is_floating_point_v<T>) {
                std::uniform_real_distribution<T> dis(min_val, max_val);
                std::generate_n(tensor.data_.get(), tensor.size_, 
                               [&gen, &dis]() { return dis(gen); });
            } else {
                std::uniform_int_distribution<T> dis(min_val, max_val);
                std::generate_n(tensor.data_.get(), tensor.size_, 
                               [&gen, &dis]() { return dis(gen); });
            }
        }
        return tensor;
    }
    
    static TensorImpl randn(const Shape& shape, T mean = 0, T stddev = 1) {
        static_assert(std::is_floating_point_v<T>, "randn requires floating point type");
        
        TensorImpl tensor(shape);
        if (tensor.size_ > 0) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<T> dis(mean, stddev);
            std::generate_n(tensor.data_.get(), tensor.size_, 
                           [&gen, &dis]() { return dis(gen); });
        }
        return tensor;
    }
    
    static TensorImpl eye(size_t n) {
        TensorImpl tensor({n, n});
        for (size_t i = 0; i < n; ++i) {
            tensor.at({i, i}) = T{1};
        }
        return tensor;
    }
    
    // Accessors
    T* data() noexcept { return data_.get(); }
    const T* data() const noexcept { return data_.get(); }
    
    const Shape& shape() const noexcept { return shape_; }
    const Strides& strides() const noexcept { return strides_; }
    size_t size() const noexcept { return size_; }
    size_t ndim() const noexcept { return shape_.size(); }
    bool requires_grad() const noexcept { return requires_grad_; }
    void set_requires_grad(bool value) noexcept { requires_grad_ = value; }
    
    // Element access
    T& operator[](size_t idx) {
        if (idx >= size_) {
            throw std::out_of_range(std::format("Index {} out of range for size {}", idx, size_));
        }
        return data_[idx];
    }
    
    const T& operator[](size_t idx) const {
        if (idx >= size_) {
            throw std::out_of_range(std::format("Index {} out of range for size {}", idx, size_));
        }
        return data_[idx];
    }
    
    // Multi-dimensional indexing
    T& at(const std::vector<size_t>& indices) {
        size_t flat_idx = compute_flat_index(indices);
        return data_[flat_idx];
    }
    
    const T& at(const std::vector<size_t>& indices) const {
        size_t flat_idx = compute_flat_index(indices);
        return data_[flat_idx];
    }
    
    // Reshape (returns new tensor with shared data in real implementation)
    TensorResult<TensorImpl> reshape(const Shape& new_shape) const {
        size_t new_size = compute_size(new_shape);
        if (new_size != size_) {
            return std::unexpected(TensorError::InvalidShape);
        }
        
        TensorImpl result(new_shape);
        std::copy_n(data_.get(), size_, result.data_.get());
        result.requires_grad_ = requires_grad_;
        return result;
    }
    
    // View as Eigen matrix (2D tensors only)
    eigen_map_t as_eigen_matrix() {
        if (shape_.size() != 2) {
            throw std::runtime_error("Tensor must be 2D for matrix view");
        }
        return eigen_map_t(data_.get(), shape_[0], shape_[1]);
    }
    
    const_eigen_map_t as_eigen_matrix() const {
        if (shape_.size() != 2) {
            throw std::runtime_error("Tensor must be 2D for matrix view");
        }
        return const_eigen_map_t(data_.get(), shape_[0], shape_[1]);
    }
    
    // View as Eigen vector (1D tensors only)
    Eigen::Map<eigen_vector_t> as_eigen_vector() {
        if (shape_.size() != 1) {
            throw std::runtime_error("Tensor must be 1D for vector view");
        }
        return Eigen::Map<eigen_vector_t>(data_.get(), size_);
    }
    
    Eigen::Map<const eigen_vector_t> as_eigen_vector() const {
        if (shape_.size() != 1) {
            throw std::runtime_error("Tensor must be 1D for vector view");
        }
        return Eigen::Map<const eigen_vector_t>(data_.get(), size_);
    }
    
    // Arithmetic operations
    TensorImpl operator+(const TensorImpl& other) const {
        if (shape_ != other.shape_) {
            throw std::runtime_error("Shape mismatch for addition");
        }
        
        TensorImpl result(shape_);
        std::transform(data_.get(), data_.get() + size_, 
                      other.data_.get(), result.data_.get(),
                      std::plus<T>());
        return result;
    }
    
    TensorImpl operator-(const TensorImpl& other) const {
        if (shape_ != other.shape_) {
            throw std::runtime_error("Shape mismatch for subtraction");
        }
        
        TensorImpl result(shape_);
        std::transform(data_.get(), data_.get() + size_, 
                      other.data_.get(), result.data_.get(),
                      std::minus<T>());
        return result;
    }
    
    TensorImpl operator*(T scalar) const {
        TensorImpl result(shape_);
        std::transform(data_.get(), data_.get() + size_, 
                      result.data_.get(),
                      [scalar](T x) { return x * scalar; });
        return result;
    }
    
    TensorImpl operator/(T scalar) const {
        if (scalar == T{0}) {
            throw std::runtime_error("Division by zero");
        }
        
        TensorImpl result(shape_);
        std::transform(data_.get(), data_.get() + size_, 
                      result.data_.get(),
                      [scalar](T x) { return x / scalar; });
        return result;
    }
    
    // In-place operations
    TensorImpl& operator+=(const TensorImpl& other) {
        if (shape_ != other.shape_) {
            throw std::runtime_error("Shape mismatch for addition");
        }
        
        std::transform(data_.get(), data_.get() + size_, 
                      other.data_.get(), data_.get(),
                      std::plus<T>());
        return *this;
    }
    
    TensorImpl& operator-=(const TensorImpl& other) {
        if (shape_ != other.shape_) {
            throw std::runtime_error("Shape mismatch for subtraction");
        }
        
        std::transform(data_.get(), data_.get() + size_, 
                      other.data_.get(), data_.get(),
                      std::minus<T>());
        return *this;
    }
    
    TensorImpl& operator*=(T scalar) {
        std::transform(data_.get(), data_.get() + size_, data_.get(),
                      [scalar](T x) { return x * scalar; });
        return *this;
    }
    
    TensorImpl& operator/=(T scalar) {
        if (scalar == T{0}) {
            throw std::runtime_error("Division by zero");
        }
        
        std::transform(data_.get(), data_.get() + size_, data_.get(),
                      [scalar](T x) { return x / scalar; });
        return *this;
    }
    
    // Matrix multiplication (2D tensors only)
    TensorImpl matmul(const TensorImpl& other) const {
        if (shape_.size() != 2 || other.shape_.size() != 2) {
            throw std::runtime_error("matmul requires 2D tensors");
        }
        
        if (shape_[1] != other.shape_[0]) {
            throw std::runtime_error(std::format("Shape mismatch for matmul: ({}, {}) @ ({}, {})",
                                                shape_[0], shape_[1], 
                                                other.shape_[0], other.shape_[1]));
        }
        
        TensorImpl result({shape_[0], other.shape_[1]});
        
        auto a = as_eigen_matrix();
        auto b = other.as_eigen_matrix();
        auto c = result.as_eigen_matrix();
        
        c = a * b;  // Eigen handles the optimized matrix multiplication
        
        return result;
    }
    
    // Transpose (2D tensors only)
    TensorImpl transpose() const {
        if (shape_.size() != 2) {
            throw std::runtime_error("transpose requires 2D tensor");
        }
        
        TensorImpl result({shape_[1], shape_[0]});
        auto src = as_eigen_matrix();
        auto dst = result.as_eigen_matrix();
        dst = src.transpose();
        
        return result;
    }
    
    // Reduction operations
    T sum() const {
        return std::reduce(data_.get(), data_.get() + size_, T{0});
    }
    
    T mean() const {
        if (size_ == 0) return T{0};
        return sum() / static_cast<T>(size_);
    }
    
    T min() const {
        if (size_ == 0) {
            throw std::runtime_error("Cannot compute min of empty tensor");
        }
        return *std::min_element(data_.get(), data_.get() + size_);
    }
    
    T max() const {
        if (size_ == 0) {
            throw std::runtime_error("Cannot compute max of empty tensor");
        }
        return *std::max_element(data_.get(), data_.get() + size_);
    }
    
    // Apply function element-wise
    template<typename Func>
    TensorImpl apply(Func&& func) const {
        TensorImpl result(shape_);
        std::transform(data_.get(), data_.get() + size_, 
                      result.data_.get(), std::forward<Func>(func));
        return result;
    }
    
    // Fill with value
    void fill(T value) {
        std::fill_n(data_.get(), size_, value);
    }
    
    // Clone
    TensorImpl clone() const {
        return *this;  // Copy constructor handles the cloning
    }
    
    // String representation
    std::string to_string() const {
        std::string result = "Tensor(shape=[";
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (i > 0) result += ", ";
            result += std::to_string(shape_[i]);
        }
        result += "], dtype=" + std::string(typeid(T).name());
        if (requires_grad_) {
            result += ", requires_grad=true";
        }
        result += ")";
        return result;
    }

private:
    size_t compute_flat_index(const std::vector<size_t>& indices) const {
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument("Number of indices does not match tensor dimensions");
        }
        
        size_t flat_idx = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range(std::format("Index {} out of range for dimension {} with size {}", 
                                                   indices[i], i, shape_[i]));
            }
            flat_idx += indices[i] * strides_[i];
        }
        return flat_idx;
    }
};

// Type aliases
using Tensor = TensorImpl<float>;
using DoubleTensor = TensorImpl<double>;
using IntTensor = TensorImpl<int32_t>;
using LongTensor = TensorImpl<int64_t>;

// Non-member operators
template<TensorNumeric T>
TensorImpl<T> operator*(T scalar, const TensorImpl<T>& tensor) {
    return tensor * scalar;
}

// Print operator
template<TensorNumeric T>
std::ostream& operator<<(std::ostream& os, const TensorImpl<T>& tensor) {
    os << tensor.to_string();
    return os;
}

} // namespace ml_labs::core
