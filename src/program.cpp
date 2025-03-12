#include "constants.h"
#include "custom_distributions.h"
#include "node_detail.h"
#include "philox_engine.h"
#include "reg_stack.h"
#include <algorithm>
#include <cstdint>
#include <fitness.h>
#include <node.h>
#include <numeric>
#include <program.h>
#include <random>
#include <stack>
#include <cassert>

namespace genetic {

    /**
     * Execution kernel for a single program. We assume that the input data
     * is stored in column major format.
     */
    template <int MaxSize = MAX_STACK_SIZE>
    void execute_kernel(const program_t d_progs, const float* data, float* y_pred,
                        const uint64_t n_rows, const uint64_t n_progs) {
        stack<float, MaxSize> eval_stack;
#pragma omp parallel for schedule(dynamic) private(eval_stack)
        for (uint64_t pid = 0; pid < n_progs; ++pid) {
            const program&        curr_p = d_progs[pid]; // Current program
            // for (int i = 0; i < curr_p.len; ++i) {
            //     assert(curr_p.nodes[i].t >= node::type::variable && curr_p.nodes[i].t <= node::type::functions_end);
            // }
            for (uint64_t row_id = 0; row_id < n_rows; ++row_id) {
                eval_stack.clear();

                // float res   = 0.0f;
                // float in[2] = {0.0f, 0.0f};

                for (int i = 0; i < curr_p.len; ++i) {
                    const node& curr_node = curr_p.nodes[i];
                    // if (curr_node.flags.is_terminal_ == false) {
                    //     int ar = curr_node.flags.arity_;
                    //     in[0]  = eval_stack.pop(); // Min arity of function is 1
                    //     if (ar > 1)
                    //         in[1] = eval_stack.pop();
                    // }
                    detail::evaluate_node_lookup(curr_node, data, n_rows, row_id, eval_stack);
                }

                // Outputs stored in col-major format
                y_pred[pid * n_rows + row_id] = eval_stack.pop();
            }
        }
    }

    program::program()
        : nodes(nullptr), len(0), depth(0), raw_fitness_(0.0f), metric(metric_t::mse),
          mut_type(mutation_t::none) {
    }

    program::~program() {
        // delete[] nodes;
        // delete[] nodes;
    }

    program::program(const program& src)
        : len(src.len), depth(src.depth), raw_fitness_(src.raw_fitness_), metric(src.metric),
          mut_type(src.mut_type) {
        nodes = std::make_unique<node[]>(len);

        std::copy(src.nodes.get(), src.nodes.get() + src.len, nodes.get());
    }

    program& program::operator=(const program& src) {
        len          = src.len;
        depth        = src.depth;
        raw_fitness_ = src.raw_fitness_;
        metric       = src.metric;
        mut_type     = src.mut_type;

        // Copy nodes
        // delete[] nodes;
        // delete[] nodes;
        nodes = std::make_unique<node[]>(len);
        std::copy(src.nodes.get(), src.nodes.get() + src.len, nodes.get());

        return *this;
    }

    void compute_metric(int n_rows, int n_progs, const float* y, const float* y_pred,
                        const float* w, float* score, const param& params) {
        // Call appropriate metric function based on metric defined in params
        if (params.metric == metric_t::pearson) {
            weightedPearson(n_rows, n_progs, y, y_pred, w, score);
        } else if (params.metric == metric_t::spearman) {
            weightedSpearman(n_rows, n_progs, y, y_pred, w, score);
        } else if (params.metric == metric_t::mae) {
            meanAbsoluteError(n_rows, n_progs, y, y_pred, w, score);
        } else if (params.metric == metric_t::mse) {
            meanSquareError(n_rows, n_progs, y, y_pred, w, score);
        } else if (params.metric == metric_t::rmse) {
            rootMeanSquareError(n_rows, n_progs, y, y_pred, w, score);
        } else if (params.metric == metric_t::logloss) {
            logLoss(n_rows, n_progs, y, y_pred, w, score);
        } else {
            // This should not be reachable
        }
    }

    void execute(const program_t& d_progs, const int n_rows, const int n_progs, const float* data,
                 float* y_pred) {
        execute_kernel(d_progs, data, y_pred, static_cast<uint64_t>(n_rows),
                       static_cast<uint64_t>(n_progs));
    }

    void find_fitness(program_t d_prog, float* score, const param& params, const int n_rows,
                      const float* data, const float* y, const float* sample_weights) {
        // Compute predicted values
        std::vector<float> y_pred(n_rows);
        execute(d_prog, n_rows, 1, data, y_pred.data());

        // Compute error
        compute_metric(n_rows, 1, y, y_pred.data(), sample_weights, score, params);
    }

    void find_batched_fitness(int n_progs, program_t d_progs, float* score, const param& params,
                              const int n_rows, const float* data, const float* y,
                              const float* sample_weights) {
        std::vector<float> y_pred((uint64_t)n_rows * (uint64_t)n_progs);
        execute(d_progs, n_rows, n_progs, data, y_pred.data());

        // Compute error
        compute_metric(n_rows, n_progs, y, y_pred.data(), sample_weights, score, params);
    }

    void set_fitness(program& h_prog, const param& params, const int n_rows, const float* data,
                     const float* y, const float* sample_weights) {
        std::vector<float> score(1);

        find_fitness(&h_prog, score.data(), params, n_rows, data, y, sample_weights);

        // Update host and device score for program
        h_prog.raw_fitness_ = score[0];
    }

    void set_batched_fitness(int n_progs, std::vector<program>& h_progs, const param& params,
                             const int n_rows, const float* data, const float* y,
                             const float* sample_weights) {
        std::vector<float> score(n_progs);

        find_batched_fitness(n_progs, h_progs.data(), score.data(), params, n_rows, data, y,
                             sample_weights);

        // Update scores on host and device
        // TODO: Find a way to reduce the number of implicit memory transfers
        for (auto i = 0; i < n_progs; ++i) {
            h_progs[i].raw_fitness_ = score[i];
        }
    }

    float get_fitness(const program& prog, const param& params) {
        int   crit    = params.criterion();
        float penalty = params.parsimony_coefficient * prog.len * (2 * crit - 1);
        return (prog.raw_fitness_ - penalty);
    }

    /**
     * @brief Get a random subtree of the current program nodes (on CPU)
     *
     * @param pnodes  AST represented as a list of nodes
     * @param len     The total number of nodes in the AST
     * @param rng     Random number generator for subtree selection
     * @return A tuple [first,last) which contains the required subtree
     */
    std::pair<int, int> get_subtree(node* pnodes, int nodes_arr_len, int len, PhiloxEngine& rng) {
        
        assert(pnodes[0].is_terminal());
        
        // Calculate actual end idx from len
        int start_idx = nodes_arr_len - 1;
        int end_idx = nodes_arr_len - 1 - len;
        int offset = end_idx + 1;

        // Specify RNG
        uniform_real_distribution_custom<float> dist_uniform(0.0f, 1.0f);
        float                                   bound = dist_uniform(rng);

        // Specify subtree start probs acc to Koza's selection approach
        std::vector<float> node_probs(len, 0.1);
        float              sum = 0.1 * len;

        for (int i = start_idx; i > end_idx; --i) {
            if (pnodes[i].is_nonterminal()) {
                node_probs[i - offset] = 0.9;
                sum += 0.8;
            }
        }

        // Normalize vector
        for (int i = 0; i < len; ++i) {
            node_probs[i] /= sum;
        }

        // Compute cumulative sum
        std::partial_sum(node_probs.rbegin(), node_probs.rend(), node_probs.rbegin());

        auto it = std::lower_bound(node_probs.rbegin(), node_probs.rend(), bound);
        int start = node_probs.rend() - it - 1 + offset;
        int end   = start;

        // Iterate until all function arguments are satisfied in current subtree
        int num_args = 1;
        while (num_args > start - end) {
            if (pnodes[end].is_nonterminal())
                num_args += pnodes[end].arity();
            --end;
        }

        return std::make_pair(start, end);
    }

    int get_depth(const program& p_out) {
        int             depth = 0;
        std::stack<int> arity_stack;
        for (auto i = p_out.len - 1; i >= 0; --i) {
            node curr(p_out.nodes[i]);

            // Update depth
            int sz = arity_stack.size();
            depth  = std::max(depth, sz);

            // Update stack
            if (curr.is_nonterminal()) {
                arity_stack.push(curr.arity());
            } else {
                // Only triggered for a depth 0 node
                if (arity_stack.empty())
                    break;

                int e = arity_stack.top();
                arity_stack.pop();
                arity_stack.push(e - 1);

                while (arity_stack.top() == 0) {
                    arity_stack.pop();
                    if (arity_stack.empty())
                        break;

                    e = arity_stack.top();
                    arity_stack.pop();
                    arity_stack.push(e - 1);
                }
            }
        }

        return depth;
    }

    void build_program(program& p_out, const param& params, PhiloxEngine& rng) {
        // Define data structures needed for tree
        std::stack<int>   arity_stack;
        std::vector<node> nodelist;
        nodelist.reserve(1 << (MAX_STACK_SIZE));

        // Specify Distributions with parameters
        uniform_int_distribution_custom<int>    dist_function(0, params.function_set.size() - 1);
        uniform_int_distribution_custom<int>    dist_initDepth(params.init_depth[0],
                                                               params.init_depth[1]);
        uniform_int_distribution_custom<int>    dist_terminalChoice(0, params.num_features);
        uniform_real_distribution_custom<float> dist_constVal(params.const_range[0],
                                                              params.const_range[1]);
        bernoulli_distribution_custom           dist_nodeChoice(params.terminalRatio);
        bernoulli_distribution_custom           dist_coinToss(0.5);

        // Initialize nodes
        int        max_depth = dist_initDepth(rng);
        node::type func      = params.function_set[dist_function(rng)];
        node       curr_node(func);
        nodelist.push_back(curr_node);
        arity_stack.push(curr_node.arity());

        init_method_t method = params.init_method;
        if (method == init_method_t::half_and_half) {
            // Choose either grow or full for this tree
            bool choice = dist_coinToss(rng);
            method      = choice ? init_method_t::grow : init_method_t::full;
        }

        // Fill tree
        while (!arity_stack.empty()) {
            int depth        = arity_stack.size();
            p_out.depth      = std::max(depth, p_out.depth);
            bool node_choice = dist_nodeChoice(rng);

            if ((node_choice == false || method == init_method_t::full) && depth < max_depth) {
                // Add a function to node list
                curr_node = node(params.function_set[dist_function(rng)]);
                nodelist.push_back(curr_node);
                arity_stack.push(curr_node.arity());
            } else {
                // Add terminal
                int terminal_choice = dist_terminalChoice(rng);
                if (terminal_choice == params.num_features) {
                    // Add constant
                    float val = dist_constVal(rng);
                    curr_node = node(val);
                } else {
                    // Add variable
                    int fid   = terminal_choice;
                    curr_node = node(fid);
                }

                // Modify nodelist
                nodelist.push_back(curr_node);

                // Modify stack
                int e = arity_stack.top();
                arity_stack.pop();
                arity_stack.push(e - 1);
                while (arity_stack.top() == 0) {
                    arity_stack.pop();
                    if (arity_stack.empty()) {
                        break;
                    }

                    e = arity_stack.top();
                    arity_stack.pop();
                    arity_stack.push(e - 1);
                }
            }
        }

        // Set new program parameters - need to do a copy as
        // nodelist will be deleted using RAII semantics
        p_out.nodes = std::make_unique<node[]>(nodelist.size());
        std::copy(nodelist.rbegin(), nodelist.rend(), p_out.nodes.get());

        p_out.len          = nodelist.size();
        p_out.metric       = params.metric;
        p_out.raw_fitness_ = 0.0f;
    }

    void point_mutation(const program& prog, program& p_out, const param& params,
                        PhiloxEngine& rng) {
        // deep-copy program
        p_out = prog;

        // Specify RNGs
        uniform_real_distribution_custom<float> dist_uniform(0.0f, 1.0f);
        uniform_int_distribution_custom<int>    dist_terminalChoice(0, params.num_features);
        uniform_real_distribution_custom<float> dist_constantVal(params.const_range[0],
                                                                 params.const_range[1]);

        // Fill with uniform numbers
        std::vector<float> node_probs(p_out.len);
        std::generate(node_probs.begin(), node_probs.end(),
                      [&dist_uniform, &rng] { return dist_uniform(rng); });

        // Mutate nodes
        int len = p_out.len;
        for (int i = 0; i < len; ++i) {
            node curr(prog.nodes[i]);

            if (node_probs[i] < params.p_point_replace) {
                if (curr.is_terminal()) {
                    int choice = dist_terminalChoice(rng);

                    if (choice == params.num_features) {
                        // Add a randomly generated constant
                        curr = node(dist_constantVal(rng));
                    } else {
                        // Add a variable with fid=choice
                        curr = node(choice);
                    }
                } else if (curr.is_nonterminal()) {
                    // Replace current function with another function of the same arity
                    int ar = curr.arity();
                    // CUML_LOG_DEBUG("Arity is %d, curr function is
                    // %d",ar,static_cast<std::underlying_type<node::type>::type>(curr.t));
                    std::vector<node::type>              fset = params.arity_set[ar];
                    uniform_int_distribution_custom<int> dist_fset(0, fset.size() - 1);
                    int                                  choice = dist_fset(rng);
                    curr                                        = node(fset[choice]);
                }

                // Update p_out with updated value
                p_out.nodes[i] = curr;
            }
        }
    }

    void crossover(const program& prog, const program& donor, program& p_out,
                   [[maybe_unused]] const param& params, PhiloxEngine& rng) {
        // Get a random subtree of prog to replace
        std::pair<int, int> prog_slice = get_subtree(prog.nodes.get(), prog.len, prog.len, rng);
        int                 prog_start = prog_slice.first;
        int                 prog_end   = prog_slice.second;

        assert(prog.depth < MAX_STACK_SIZE);

        // Set metric of output program
        p_out.metric = prog.metric;

        // MAX_STACK_SIZE can only handle tree of depth MAX_STACK_SIZE -
        // max(func_arity=2) + 1 Thus we continuously hoist the donor subtree. Actual
        // indices in donor
        int donor_start  = donor.len - 1;
        int donor_end    = -1;
        int output_depth = 0;
        do {
            // Get donor subtree
            std::pair<int, int> donor_slice =
                get_subtree(donor.nodes.get(), donor_start + 1, donor_start - donor_end, rng);

            // Get indices w.r.t current subspace [donor_start,donor_end)
            int donor_substart = donor_slice.first;
            int donor_subend   = donor_slice.second;

            // Update relative indices to global indices
            // int offset = donor.len - donor_start - 1;
            // donor_substart -= offset;
            // donor_subend -= offset;

            // Update to new subspace
            donor_start = donor_substart;
            donor_end   = donor_subend;

            // Evolve on current subspace
            p_out.len = (prog_end + 1) + (donor_start - donor_end) + (prog.len - prog_start - 1);
            // delete[] p_out.nodes;
            p_out.nodes = std::make_unique<node[]>(p_out.len);

            // Copy slices using std::copy
            // std::copy(prog.nodes.get(), prog.nodes.get() + prog_start, p_out.nodes.get());
            // std::copy(donor.nodes.get() + donor_start, donor.nodes.get() + donor_end,
            //           p_out.nodes.get() + prog_start);
            // std::copy(prog.nodes.get() + prog_end, prog.nodes.get() + prog.len,
            //           p_out.nodes.get() + (prog_start) + (donor_end - donor_start));
            
            std::copy(prog.nodes.get(), prog.nodes.get() + prog_end + 1, p_out.nodes.get());
            std::copy(donor.nodes.get() + donor_end + 1, donor.nodes.get() + donor_start + 1, p_out.nodes.get() + prog_end + 1);
            std::copy(prog.nodes.get() + prog_start + 1, prog.nodes.get() + prog.len, p_out.nodes.get() + prog_end + 1 + (donor_start - donor_end));

            output_depth = get_depth(p_out);
        } while (output_depth >= MAX_STACK_SIZE);

        // Set the depth of the final program
        p_out.depth = output_depth;
    }

    void subtree_mutation(const program& prog, program& p_out, const param& params,
                          PhiloxEngine& rng) {
        // Generate a random program and perform crossover
        program new_program;
        build_program(new_program, params, rng);
        crossover(prog, new_program, p_out, params, rng);
    }

    void hoist_mutation(const program& prog, program& p_out, [[maybe_unused]] const param& params,
                        PhiloxEngine& rng) {
        // Replace program subtree with a random sub-subtree

        std::pair<int, int> prog_slice = get_subtree(prog.nodes.get(), prog.len, prog.len, rng);
        int                 prog_start = prog_slice.first;
        int                 prog_end   = prog_slice.second;

        std::pair<int, int> sub_slice =
            get_subtree(prog.nodes.get(), prog_start + 1, prog_start - prog_end, rng);
        int sub_start = sub_slice.first;
        int sub_end   = sub_slice.second;

        // Update subtree indices to global indices
        // sub_start += prog_start;
        // sub_end += prog_start;

        p_out.len    = (prog_end + 1) + (sub_start - sub_end) + (prog.len - prog_start - 1);
        p_out.nodes  = std::make_unique<node[]>(p_out.len);
        p_out.metric = prog.metric;


        // Copy node slices using std::copy
        // std::copy(prog.nodes.get(), prog.nodes.get() + prog_start, p_out.nodes.get());
        // std::copy(prog.nodes.get() + sub_start, prog.nodes.get() + sub_end,
        //           p_out.nodes.get() + prog_start);
        // std::copy(prog.nodes.get() + prog_end, prog.nodes.get() + prog.len,
        //           p_out.nodes.get() + (prog_start) + (sub_end - sub_start));

        std::copy(prog.nodes.get(), prog.nodes.get() + prog_end + 1, p_out.nodes.get());
        std::copy(prog.nodes.get() + sub_end + 1, prog.nodes.get() + sub_start + 1, p_out.nodes.get() + prog_end + 1);
        std::copy(prog.nodes.get() + prog_start + 1, prog.nodes.get() + prog.len, p_out.nodes.get() + prog_end + 1 + (sub_start - sub_end));

        // Update depth
        p_out.depth = get_depth(p_out);
        assert(p_out.depth < MAX_STACK_SIZE);
    }

} // namespace genetic
