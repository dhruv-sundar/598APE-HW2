#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>

using FnPtr = float (*)(float, float);

/*
need to account for the (underflow?) checking that exists currently in the
switch statement but this is faster as of now.
*/

static constexpr float EPSILON = 0.001f;

static const FnPtr function_table[] = {
    // Skip variable (0) and constant (1)
    nullptr, nullptr,
    // Binary functions
    [](float a, float b) { return a + b; },                                // add
    [](float a, float b) { return std::atan2(a, b); },                     // atan2
    [](float a, float b) { return std::abs(b) < EPSILON ? 1.0f : a / b; }, // div
    [](float a, float b) { return std::fdim(a, b); },                      // fdim
    [](float a, float b) { return std::max(a, b); },                       // max
    [](float a, float b) { return std::min(a, b); },                       // min
    [](float a, float b) { return a * b; },                                // mul
    [](float a, float b) { return std::pow(a, b); },                       // pow
    [](float a, float b) { return a - b; },                                // sub
    // Unary functions
    [](float a, float) { return std::abs(a); },                           // abs
    [](float a, float) { return std::acos(a); },                          // acos
    [](float a, float) { return std::acosh(a); },                         // acosh
    [](float a, float) { return std::asin(a); },                          // asin
    [](float a, float) { return std::asinh(a); },                         // asinh
    [](float a, float) { return std::atan(a); },                          // atan
    [](float a, float) { return std::atanh(a); },                         // atanh
    [](float a, float) { return std::cbrt(a); },                          // cbrt
    [](float a, float) { return std::cos(a); },                           // cos
    [](float a, float) { return std::cosh(a); },                          // cosh
    [](float a, float) { return a * a * a; },                             // cube
    [](float a, float) { return std::exp(a); },                           // exp
    [](float a, float) { return std::abs(a) < EPSILON ? 0.f : 1.f / a; }, // inv
    [](float a, float) {
        float abs_a = std::abs(a);
        return abs_a < EPSILON ? 0.f : std::log(abs_a);
    },                                                            // log
    [](float a, float) { return -a; },                            // neg
    [](float a, float) { return 1.0f / std::cbrt(a); },           // rcbrt
    [](float a, float) { return 1.0f / std::sqrt(std::abs(a)); }, // rsqrt
    [](float a, float) { return std::sin(a); },                   // sin
    [](float a, float) { return std::sinh(a); },                  // sinh
    [](float a, float) { return a * a; },                         // sq
    [](float a, float) { return std::sqrt(std::abs(a)); },        // sqrt
    [](float a, float) { return std::tan(a); },                   // tan
    [](float a, float) { return std::tanh(a); },                  // tanh
};