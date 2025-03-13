
#include "node_detail.h"
#include <memory>
#include <stdexcept>

#define THROW(fmt, ...)                                                                            \
    do {                                                                                           \
        int size1 = std::snprintf(nullptr, 0, "exception occurred! file=%s line=%d: ", __FILE__,   \
                                  __LINE__);                                                       \
        int size2 = std::snprintf(nullptr, 0, fmt, ##__VA_ARGS__);                                 \
        if (size1 < 0 || size2 < 0)                                                                \
            throw std::runtime_error("Error in snprintf, cannot handle exception.");               \
        auto size = size1 + size2 + 1; /* +1 for final '\0' */                                     \
        auto buf  = std::make_unique<char[]>(size_t(size));                                        \
        std::snprintf(buf.get(), size1 + 1 /* +1 for '\0' */,                                      \
                      "exception occurred! file=%s line=%d: ", __FILE__, __LINE__);                \
        std::snprintf(buf.get() + size1, size2 + 1 /* +1 for '\0' */, fmt, ##__VA_ARGS__);         \
        std::string msg(buf.get(), buf.get() + size - 1); /* -1 to remove final '\0' */            \
        throw std::runtime_error(msg);                                                             \
    } while (0)

/** macro to check for a conditional and assert on failure */
#define ASSERT(check, fmt, ...)                                                                    \
    do {                                                                                           \
        if (!(check))                                                                              \
            THROW(fmt, ##__VA_ARGS__);                                                             \
    } while (0)

namespace genetic {

    const int node::kInvalidFeatureId = -1;

    node::node() {
    }

    node::node(node::type ft) {
        flags.type_ = static_cast<uint8_t>(ft);
        ASSERT(is_nonterminal(), "node: ctor with `type` argument expects functions type only!");
        u.fid              = kInvalidFeatureId;
        flags.arity_       = detail::arity(ft);
    }

    node::node(int fid) {
        flags.type_ = static_cast<uint8_t>(node::type::variable);
        u.fid       = fid;
        flags.arity_ = detail::arity(static_cast<node::type>(flags.type_));
    }

    node::node(float val) {
        flags.type_ = static_cast<uint8_t>(node::type::constant);
        u.val       = val;
        flags.arity_ = detail::arity(static_cast<node::type>(flags.type_));
    }

    node::node(const node& src) {
        flags.type_ = src.flags.type_;
        u           = src.u;
        flags.arity_ = detail::arity(static_cast<node::type>(flags.type_));
    }

    node& node::operator=(const node& src) {
        flags.type_ = src.flags.type_;
        u           = src.u;
        flags.arity_ = detail::arity(static_cast<node::type>(flags.type_));
        return *this;
    }

    bool node::is_terminal() const {
        return detail::is_terminal(static_cast<node::type>(flags.type_));
    }

    bool node::is_nonterminal() const {
        return detail::is_nonterminal(static_cast<node::type>(flags.type_));
    }

    int node::arity() const {
        return detail::arity(static_cast<node::type>(flags.type_));
    }

#define CASE(str, val)                                                                             \
    if (#val == str)                                                                               \
    return node::type::val

    node::type node::from_str(const std::string& ntype) {
        CASE(ntype, variable);
        CASE(ntype, constant);
        // note: keep the case statements in alphabetical order under each category of
        // operators.
        // binary operators
        CASE(ntype, add);
        CASE(ntype, atan2);
        CASE(ntype, div);
        CASE(ntype, fdim);
        CASE(ntype, max);
        CASE(ntype, min);
        CASE(ntype, mul);
        CASE(ntype, pow);
        CASE(ntype, sub);
        // unary operators
        CASE(ntype, abs);
        CASE(ntype, acos);
        CASE(ntype, asin);
        CASE(ntype, atan);
        CASE(ntype, acosh);
        CASE(ntype, asinh);
        CASE(ntype, atanh);
        CASE(ntype, cbrt);
        CASE(ntype, cos);
        CASE(ntype, cosh);
        CASE(ntype, cube);
        CASE(ntype, exp);
        CASE(ntype, inv);
        CASE(ntype, log);
        CASE(ntype, neg);
        CASE(ntype, rcbrt);
        CASE(ntype, rsqrt);
        CASE(ntype, sq);
        CASE(ntype, sqrt);
        CASE(ntype, sin);
        CASE(ntype, sinh);
        CASE(ntype, tan);
        CASE(ntype, tanh);
        ASSERT(false, "node::from_str: Bad type passed '%s'!", ntype.c_str());
    }
#undef CASE

} // namespace genetic
