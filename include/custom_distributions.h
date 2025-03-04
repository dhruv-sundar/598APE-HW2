#ifndef CUSTOM_DISTRIBUTIONS_H
#define CUSTOM_DISTRIBUTIONS_H

#include <cstdint>
#include <limits>
#include <stdexcept>

// Custom uniform integer distribution using rejection sampling to avoid bias.
// Template parameter IntType is the integer type (e.g. int).
template <typename IntType = int> class uniform_int_distribution_custom {
public:
    using result_type = IntType;

    uniform_int_distribution_custom() : a_(0), b_(std::numeric_limits<IntType>::max()) {}

    // Constructor takes inclusive lower and upper bounds.
    uniform_int_distribution_custom(result_type a, result_type b) : a_(a), b_(b) {
        if (b < a) {
            throw std::invalid_argument("Range error: b must be >= a");
        }
    }

    // When called with a random engine, returns a uniformly distributed integer.
    // The engine is expected to provide a uint32_t via operator().
    template <typename Engine> result_type operator()(Engine& eng) {
        // Compute the range size.
        uint32_t range = static_cast<uint32_t>(b_ - a_ + 1);
        // Calculate the largest multiple of 'range' that fits in a uint32_t.
        uint32_t limit =
            std::numeric_limits<uint32_t>::max() - (std::numeric_limits<uint32_t>::max() % range);
        uint32_t x;
        // Rejection sampling to avoid bias.
        do {
            x = eng();
        } while (x >= limit);
        return a_ + static_cast<result_type>(x % range);
    }

private:
    result_type a_;
    result_type b_;
};

// Custom uniform real distribution mapping the raw uint32_t to a double in [0,
// 1), then scaling to the desired range [a, b).
template <typename RealType = double> class uniform_real_distribution_custom {
public:
    using result_type = RealType;

    // Default constructor: range [0, 1)
    uniform_real_distribution_custom() : a_(0), b_(1) {}

    // Constructor takes lower bound 'a' and upper bound 'b'; b must be > a.
    uniform_real_distribution_custom(result_type a, result_type b) : a_(a), b_(b) {
        if (b <= a) {
            throw std::invalid_argument("Range error: b must be > a");
        }
    }

    // When called with a random engine, returns a uniformly distributed real
    // number. The engine is expected to provide a uint32_t via operator().
    template <typename Engine> result_type operator()(Engine& eng) {
        // Obtain a raw uint32_t from the engine.
        uint32_t x = eng();
        // Map it to a value in [0, 1) using a standard conversion.
        double u = x * (1.0 / (static_cast<double>(std::numeric_limits<uint32_t>::max()) + 1.0));
        return a_ + static_cast<result_type>((b_ - a_) * u);
    }

private:
    result_type a_;
    result_type b_;
};

// Custom Bernoulli distribution that returns a boolean value.
// It uses a probability p in [0,1] to decide the outcome.
template <typename RealType = double> class bernoulli_distribution_custom {
public:
    using result_type = bool;

    // Default constructor: probability 0.5.
    bernoulli_distribution_custom() : p_(0.5) {}

    // Constructor takes a probability p; defaults to 0.5.
    explicit bernoulli_distribution_custom(RealType p = RealType(0.5)) : p_(p) {
        if (p < 0 || p > 1) {
            throw std::invalid_argument("Probability must be in [0,1].");
        }
    }

    // When called with a random engine, returns true with probability p.
    // The engine is expected to provide a uint32_t via operator().
    template <typename Engine> result_type operator()(Engine& eng) {
        uint32_t x = eng();
        double   u = x * (1.0 / (static_cast<double>(std::numeric_limits<uint32_t>::max()) + 1.0));
        return u < p_;
    }

    // Accessor for probability.
    RealType p() const { return p_; }

private:
    RealType p_;
};

#endif // CUSTOM_DISTRIBUTIONS_H
