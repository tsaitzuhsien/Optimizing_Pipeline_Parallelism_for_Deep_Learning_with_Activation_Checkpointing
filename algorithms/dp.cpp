#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace std;

#ifndef DP_CORE_ONLY
namespace py = pybind11;
#endif

namespace {

constexpr float NEG = -1.0F;
constexpr float INF = numeric_limits<float>::infinity();
constexpr int64_t PARAMETER_COPIES = 4;  // weight, grad, Adam m, Adam v

int64_t gcd(int64_t a, int64_t b) {
    while (b != 0) {
        const int64_t r = a % b;
        a = b;
        b = r;
    }
    return a;
}

int64_t gcd_vector(const vector<int64_t>& d) {
    int64_t result = 0;

    for (size_t i = 1; i + 1 < d.size(); i++) {
        result = gcd(result, d[i]);
    }

    return result > 0 ? result : 1;
}

void validate_inputs(
    const vector<float>& f,
    const vector<float>& b,
    const vector<int64_t>& param,
    const vector<int64_t>& d,
    const vector<int>& U,
    int64_t mem,
    int p,
    int micro_batches) {

    if (d.size() < 2 || f.size() != d.size() || b.size() != d.size() || param.size() != d.size()) {
        throw invalid_argument("f, b, param, and d must have the same length and include index 0");
    }

    const int n = static_cast<int>(d.size()) - 1;
    if (p <= 0 || p > n) {
        throw invalid_argument("p must be in [1, number_of_layers]");
    }
    if (micro_batches <= 0 || mem <= 0) {
        throw invalid_argument("micro_batches and mem must be positive");
    }

    for (size_t i = 0; i < d.size(); ++i) {
        if (!isfinite(f[i]) || f[i] < 0 || !isfinite(b[i]) || b[i] < 0) {
            throw invalid_argument("execution times must be finite and non-negative");
        }
        if (param[i] < 0 || d[i] < 0) {
            throw invalid_argument("memory sizes must be non-negative");
        }
    }

    int previous = -1;
    for (const int pivot : U) {
        if (pivot < 0 || pivot > n || pivot <= previous) {
            throw invalid_argument("U must be strictly increasing and lie in [0, n]");
        }
        previous = pivot;
    }
}

pair<vector<int>, vector<int>> dynamic_programming(
    const vector<float>& f,
    const vector<float>& b,
    const vector<int64_t>& param,
    const vector<int64_t>& d,
    const vector<int>& U,
    int64_t mem_bytes,
    int p,
    int micro_batches) {

    validate_inputs(f, b, param, d, U, mem_bytes, p, micro_batches);

    const int layer_n = static_cast<int>(d.size()) - 1;

    vector<int64_t> psum(layer_n + 1, 0);
    vector<float> fsum(layer_n + 1, 0.0F), bsum(layer_n + 1, 0.0F);
    for (int i = 1; i <= layer_n; ++i) {
        psum[i] = psum[i - 1] + param[i];
        fsum[i] = fsum[i - 1] + f[i];
        bsum[i] = bsum[i - 1] + b[i];
    }

    const int64_t atom = gcd_vector(d);
    const int64_t mem = mem_bytes / atom;
    vector<int64_t> rounded_d(d.size());
    for (size_t i = 0; i < rounded_d.size(); ++i) {
        rounded_d[i] = d[i] / atom + ((d[i] % atom != 0)? 1:0);
    }

    const size_t memory_states = static_cast<size_t> (mem + 1);

    vector<bool> endpoints(layer_n + 1, false);
    vector<bool> splits(layer_n + 1, false);
    vector<int> u(layer_n + 1, layer_n);
    vector<int64_t> residual_edge_size(layer_n + 1, 0);

    if (U.empty()) {
        fill(splits.begin(), splits.end(), true);
    } else {
        int pos = U.size() - 1;
        int next_endpoint = layer_n;

        for (int i = layer_n; i >= 0; i--) {
            u[i] = next_endpoint;
            if (pos >= 0 && i == U[pos]) {
                endpoints[i] = true;
                next_endpoint = U[pos];
                pos--;
            }
            if (endpoints[i] || i < U.front() || i > U.back()) {
                splits[i] = true;
            }
        }

        int64_t previous_source_size = 0;
        for (int i = 0; i <= U.back(); ++i) {
            residual_edge_size[i] = previous_source_size;
            if (endpoints[i]) {
                previous_source_size = rounded_d[i];
            }
        }
    }

    vector<vector<int64_t>> segment(layer_n + 1, vector<int64_t>(layer_n + 1, 0));
    for (int i = 0; i < layer_n; ++i) {
        int64_t sum = 0;
        int64_t max_grad = rounded_d[i];
        int64_t max_sum = 0;

        for (int j = i + 1; j <= layer_n; j++) {
            max_sum = max(max_sum, sum + residual_edge_size[j]);
            segment[i][j] = max_sum + max_grad;

            max_grad = max(max_grad, rounded_d[j]);
            sum += rounded_d[j];
        }
    }

    vector<vector<float>> T(layer_n + 1, vector<float>(memory_states, NEG));
    vector<vector<float>> T_star(layer_n + 1, vector<float>(memory_states, NEG));
    /* Trace the next checkpoint for each layer */
    vector<vector<int>> next(layer_n + 1, vector<int>(memory_states, -1));
    vector<vector<int>> next_star(layer_n + 1, vector<int>(memory_states, -1));
    /* Trace that whether the update of T uses T or T_star*/
    vector<vector<char>> uses_star(layer_n + 1, vector<char>(memory_states, false));
    vector<vector<char>> star_uses_star(layer_n + 1, vector<char>(memory_states, false));

    vector<vector<vector<float>>> cost(p, vector<vector<float>>(layer_n + 1, vector<float>(layer_n + 1, INF)));

    vector<vector<vector<vector<int>>>> trace(p, vector<vector<vector<int>>>(layer_n + 1, vector<vector<int>>(layer_n + 1)));

    for (int stage = 0; stage < p; stage++) {
        const int l = min(micro_batches, p - stage);

        /* j is the possible last layer of a stage*/
        for (int j = 1; j <= layer_n; j++) {
            if (!splits[j]) {
                continue;
            }

            fill(T[j].begin(), T[j].end(), NEG);
            fill(T_star[j].begin(), T_star[j].end(), NEG);

            if (mem >= l * rounded_d[j]) {
                for (int64_t m = l * rounded_d[j]; m <= mem; m++) {
                    T[j][m] = f[j];
                    T_star[j][m] = f[j];
                }
            }
            /* Each iteration computes the checkpoint computation time from layer i to layer j. */
            for (int i = j - 1; i >= 0; i--) {
                fill(T[i].begin(), T[i].end(), NEG);
                fill(T_star[i].begin(), T_star[i].end(), NEG);
                fill(next[i].begin(), next[i].end(), -1);
                fill(next_star[i].begin(), next_star[i].end(), -1);
                fill(uses_star[i].begin(), uses_star[i].end(), false);
                fill(star_uses_star[i].begin(), star_uses_star[i].end(), false);

                const int64_t checkpoint_memory = static_cast<int64_t>(l) * rounded_d[i];

                for (int64_t m = checkpoint_memory; m <= mem; m++) {
                    const int64_t r = m - checkpoint_memory;
                    const int next_endpoint = min(u[i], j);

                    // Compute T^*(i,m).
                    for (int k = i + 1; k < next_endpoint; ++k) {
                        if (T_star[k][r] != NEG && segment[i][k] <= r) {
                            const float candidate = f[i] + T_star[k][r];

                            if (candidate >= T_star[i][m]) {
                                T_star[i][m] = candidate;
                                next_star[i][m] = k;
                                star_uses_star[i][m] = true;
                            }
                        }
                    }
                    
                    // Compute T^*(i,m) when the next checkpoint after i is next_endpoint.
                    if (T[next_endpoint][r] != NEG && segment[i][next_endpoint] <= r) {
                        const float candidate = f[i] + T[next_endpoint][r];
                        if (candidate >= T_star[i][m]) {
                            T_star[i][m] = candidate;
                            next_star[i][m] = next_endpoint;
                            star_uses_star[i][m] = false;
                        }
                    }

                    // Compute T(i,m).
                    for (int k = i + 1; k <= j; ++k) {
                        const float previous_checkpoint_computation_time = (k <= u[i] || endpoints[k])? T[k][r] : T_star[k][r];

                        if (previous_checkpoint_computation_time == NEG || segment[i][k] > r) {
                            continue;
                        }

                        const float candidate = f[i] + previous_checkpoint_computation_time;

                        if (candidate >= T[i][m]) {
                            T[i][m] = candidate;
                            next[i][m] = k;
                            uses_star[i][m] = !(k <= u[i] || endpoints[k]);
                        }
                    }
                }
            }

            /* Construct the optimal checkpointing strategy for each stage.*/
            for (int stage_start = 1; stage_start <= j; stage_start++) {
                if (!splits[stage_start - 1]) {
                    continue;
                }

                int64_t available = mem_bytes - (psum[j] - psum[stage_start - 1]) * PARAMETER_COPIES;

                const int64_t input_size = rounded_d[stage_start - 1] * atom;
                const int64_t output_size = rounded_d[j] * atom;

                available -= input_size * ((stage == 0)? (micro_batches - l): (2 * micro_batches - l));
                available -= output_size * ((stage == p - 1) ? 3:(2 * micro_batches - l));

                if (available < 0) {
                    continue;
                }

                const int64_t m = available / atom;

                if (m > mem || T[stage_start - 1][m] == NEG) {
                    continue;
                }

                cost[stage][stage_start][j] = (bsum[j] - bsum[stage_start - 1]) /* backward time */
                                                + 2 * (fsum[j] - fsum[stage_start - 1]) /* Two times of forward time */
                                                - (T[stage_start - 1][m] - f[stage_start - 1]); /* Recomputation time*/

                bool in_star = false;
                int k = stage_start - 1;
                int64_t current_m = m;

                vector<int>& selected = trace[stage][stage_start][j];
                selected.clear();

                while (k != j) {
                    const int64_t r = current_m - l * rounded_d[k];
                    if (r < 0) {
                        throw runtime_error("negative memory during checkpoint backtracking");
                    }

                    const int old_k = k;
                    bool next_is_star = false;

                    if (in_star) {
                        k = next_star[old_k][current_m];
                        next_is_star = star_uses_star[old_k][current_m];
                    } else {
                        k = next[old_k][current_m];
                        next_is_star = uses_star[old_k][current_m];
                    }

                    if (k <= old_k || k > j) {
                        throw runtime_error("invalid checkpoint trace");
                    }

                    selected.push_back(k);
                    current_m = r;
                    in_star = next_is_star;
                }
            }
        }
    }

    vector<vector<float>> P(p, vector<float>(layer_n + 1, INF));

    vector<vector<int>> previous(p, vector<int>(layer_n + 1, -1));

    for (int end = 1; end <= layer_n; ++end) {
        if (splits[end]) {
            P[0][end] = cost[0][1][end];
        }
    }

    for (int stage = 1; stage < p; ++stage) {
        for (int end = stage + 1; end <= layer_n; ++end) {
            if (!splits[end]) {
                continue;
            }

            for (int start = stage + 1; start <= end; ++start) {

                const int split = start - 1;

                if (!splits[split] ||
                    !isfinite(P[stage - 1][split]) ||
                    !isfinite(cost[stage][start][end])) {
                    continue;
                }

                const float candidate = max(P[stage - 1][split], cost[stage][start][end]);

                if (candidate < P[stage][end]) {
                    P[stage][end] = candidate;
                    previous[stage][end] = split;
                }
            }
        }
    }
    
    if (!isfinite(P[p - 1][layer_n])) {
        ostringstream message;

        message << "No feasible " << p << "-stage partition exists under the memory budget of " << mem_bytes << " bytes";

        throw runtime_error(message.str());
    }

    vector<int> split_idx(p - 1, -1);

    int end = layer_n;
    for (int stage = p - 1; stage > 0; --stage) {
        const int split = previous[stage][end];

        if (split < stage || split >= end) {
            throw runtime_error("invalid pipeline trace");
        }

        split_idx[stage - 1] = split;
        end = split;
    }

    vector<int> checkpoint_idx{0};

    int start = 1;
    for (int stage = 0; stage < p; ++stage) {
        const int stage_end = stage == p - 1 ? layer_n : split_idx[stage];

        const vector<int>& selected = trace[stage][start][stage_end];

        if (selected.empty() || selected.back() != stage_end) {
            throw runtime_error("missing checkpoint trace for selected partition");
        }

        checkpoint_idx.insert(checkpoint_idx.end(), selected.begin(), selected.end());

        start = stage_end + 1;
    }

    for (size_t i = 1; i < checkpoint_idx.size(); ++i) {

        if (checkpoint_idx[i] <= checkpoint_idx[i - 1]) {

            throw runtime_error("checkpoint indices are not strictly increasing");
        }
    }

    return {split_idx, checkpoint_idx};
}

}  // namespace

#ifndef DP_CORE_ONLY
PYBIND11_MODULE(dp, m) {
    m.def(
        "dynamic_programming",
        &dynamic_programming,
        py::arg("f"),
        py::arg("b"),
        py::arg("param"),
        py::arg("d"),
        py::arg("U"),
        py::arg("mem"),
        py::arg("p"),
        py::arg("micro_batches"),
        "Optimal pipeline partitioning and checkpoint selection");
    m.def(
        "dynamic_programming",
        &dynamic_programming,
        py::arg("f"),
        py::arg("b"),
        py::arg("param"),
        py::arg("d"),
        py::arg("U"),
        py::arg("mem"),
        py::arg("p"),
        py::arg("micro_batches"),
        "Optimal pipeline partitioning and checkpoint selection");
}
#endif
