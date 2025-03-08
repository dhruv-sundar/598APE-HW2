#pragma once

#include "src/constants.h"
#include "src/reg_stack.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>


// using FnPtr = float (*)(float, float);
using FnPtr = void(*)(genetic::stack<float, genetic::MAX_STACK_SIZE>&);

#define bin_func(F) \
    [](genetic::stack<float, genetic::MAX_STACK_SIZE>& eval_stack) { \
        float a = eval_stack.pop(); \
        float b = eval_stack.pop(); \
        float c = F; \
        eval_stack.push(c); \
    }

#define un_func(F) \
    [](genetic::stack<float, genetic::MAX_STACK_SIZE>& eval_stack) { \
        float a = eval_stack.pop(); \
        float c = F; \
        eval_stack.push(c); \
    }

/*
need to account for the (underflow?) checking that exists currently in the
switch statement but this is faster as of now.
*/

static constexpr float EPSILON = 0.001f;

static const FnPtr function_table[] = {
    // Skip variable (0) and constant (1)
    [](auto){}, [](auto){},
    // Binary functions
    bin_func(a + b),  // add
    bin_func(std::atan2(a, b)),                     // atan2
    bin_func(std::abs(b) < EPSILON ? 1.0f : a / b), // div
    bin_func(std::fdim(a, b)),                      // fdim
    bin_func(std::max(a, b)),                       // max
    bin_func(std::min(a, b)),                       // min
    bin_func(a * b),                                // mul
    bin_func(std::pow(a, b)),                       // pow
    bin_func(a - b),                                // sub
    // Unary functions
    un_func(std::abs(a)),                           // abs
    un_func(std::acos(a)),                          // acos
    un_func(std::acosh(a)),                         // acosh
    un_func(std::asin(a)),                          // asin
    un_func(std::asinh(a)),                         // asinh
    un_func(std::atan(a)),                          // atan
    un_func(std::atanh(a)),                         // atanh
    un_func(std::cbrt(a)),                          // cbrt
    un_func(std::cos(a)),                           // cos
    un_func(std::cosh(a)),                          // cosh
    un_func(a * a * a),                             // cube
    un_func(std::exp(a)),                           // exp
    un_func(std::abs(a) < EPSILON ? 0.f : 1.f / a), // inv
    un_func(std::abs(a) < EPSILON ? 0.f : std::log(std::abs(a))),  // log
    un_func(-a),                            // neg
    un_func(1.0f / std::cbrt(a)),           // rcbrt
    un_func(1.0f / std::sqrt(std::abs(a))), // rsqrt
    un_func(std::sin(a)),                   // sin
    un_func(std::sinh(a)),                  // sinh
    un_func(a * a),                         // sq
    un_func(std::sqrt(std::abs(a))),        // sqrt
    un_func(std::tan(a)),                   // tan
    un_func(std::tanh(a))                   // tanh
};