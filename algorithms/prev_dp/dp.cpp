#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <limits>

using namespace std;
namespace py = pybind11;

long gcd(long a, long b) {
    if (b == 0) return a;
    else return gcd(b, a % b);
}

long gcd_d(vector<long> &d) {
    int n = d.size() - 1;
    long res = d[1];
    // forget about d_0 and d_n since they are must-saved.
    for (int i = 1; i < n; i++)
        res = gcd(res, d[i]);
    return res;
}

pair<vector<int>, vector<int>> dynamic_programming(
    vector<float> &f, // forward times
    vector<float> &b, // backward times
    vector<long> &param, // memory size of layer parameter
    vector<long> &d, // memory size of layer output
    vector<int> &U, // pivots in fast-forward serial graph
    long mem, // device memory in bytes
    int p, // pipeline depth
    int micro_batches // number of micro_batches
) {
    int n = d.size() - 1; // number of layers, minus 1 because d contains d_0
    // stage optimal checkpointed cost and correspoding checkpoints (input is not included)
    vector<vector<vector<float>>> cost(p, vector<vector<float>>(n + 1, vector<float>(n + 1, 0.0)));
    vector<vector<vector<vector<int>>>> trace(p, vector<vector<vector<int>>>(n + 1, vector<vector<int>>(n + 1, vector<int>())));

    // prefix sums for f, b, param
    vector<long> psum(n + 1, 0);
    vector<float> fsum(n + 1, 0), bsum(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        psum[i] = psum[i - 1] + param[i];
        fsum[i] = fsum[i - 1] + f[i];
        bsum[i] = bsum[i - 1] + b[i];
    }

    long M = mem; // M is original divice memory.
    // Speed up for psuedo-polynomial time dp.
    long atom = gcd_d(d);
    mem /= atom;
    for (auto &d_i: d)
        d_i = (d_i + atom - 1) / atom; // ceiling for the last output

    // dp table for checkpoint selection, under score stands for tilde in paper's formulation
    vector<vector<float>> T(n + 1, vector<float>(mem + 1, -1.0)), _T(n + 1, vector<float>(mem + 1, -1.0));
    vector<vector<int>> next(n + 1, vector<int>(mem + 1, -1)), _next(n + 1, vector<int>(mem + 1, -1));
    vector<vector<bool>> tilde(n + 1, vector<bool>(mem + 1)), _tilde(n + 1, vector<bool>(mem + 1));
    vector<vector<long>> C(n + 1, vector<long>(mem + 1, 0)), _C(n + 1, vector<long>(mem + 1, 0));

    vector<bool> pivots(n + 1, false); // layer that contain fast-forward edges
    vector<bool> splits(n + 1, false); // valid split layers stage can endded with
    vector<int> u(n + 1); // next pivot greater than i as the paper defines.
    for (int i = n, j = U.size() - 1, u_i = -1; i >= 0; i--) {
        u[i] = u_i;
        if (j >= 0 && i == U[j]) {
            u_i = U[j--];
            pivots[i] = true;
        }
        if (pivots[i] || i > U.back() || i < U.front())
            splits[i] = true;
    }

    // segment[i][j] is s(i, j) defined in the paper.6
    vector<vector<long>> segment(n + 1, vector<long>(n + 1, 0));
    for (int i = 0; i < n; i++) {
        long sum = 0, max_grad = d[i];
        for (int j = i + 1; j <= n; j++) {
            segment[i][j] = sum + max_grad;
            max_grad = max(max_grad, d[j]);
            sum += d[j];
        }
    }

    for (int s = 0; s < p; s++) {
        for (int j = 1; j <= n; j++) {
            if (!splits[j]) continue;
            // Dynamic programming for optimal checkpoint selection.
            int r_s = p - s;
            for (int i = j - 1; i >= 0; i--) {
                for (int m = 0; m <= mem; m++) {
                    T[i][m] = _T[i][m] = -1.0;
                    if (!splits[i]) { // d_i is between pivots.
                        // Compututation of _T.
                        for (int k = i + 1; k < u[i]; k++) {
                            if (r_s * d[k] <= m
                                && _T[k][m - r_s * d[k]] != -1.0
                                && segment[i][k] + d[k] + (r_s - 1) * _C[k][m - r_s * d[k]] <= m
                                && f[k] + _T[k][m - r_s * d[k]] >= _T[i][m]) {
                                _T[i][m] = f[k] + _T[k][m - r_s * d[k]];
                                _C[i][m] = d[i] + _C[k][m - r_s * d[k]];
                                _next[i][m] = k;
                                _tilde[i][m] = true;
                            }
                        }
                        for (int k = u[i] + 1; k < j; k++) {
                            if (r_s * d[k] <= m
                                && T[k][m - r_s * d[k]] != -1.0
                                && segment[i][k] + d[k] + (r_s - 1) * C[k][m - r_s * d[k]] <= m
                                && f[k] + T[k][m - r_s * d[k]] >= _T[i][m]) {
                                _T[i][m] = f[k] + T[k][m - r_s * d[k]];
                                _C[i][m] = d[i] + C[k][m - r_s * d[k]];
                                _next[i][m] = k;
                                _tilde[i][m] = false;
                            }
                        }
                        if (segment[i][j] <= m && f[j] >= _T[i][m]) {
                            _T[i][m] = f[j];
                            _C[i][m] = d[i];
                            _next[i][m] = j;
                            _tilde[i][m] = false;
                        }
                    }
                    // Computation of T.
                    if (!pivots[i]) {
                        if (u[i] == -1 || u[i] >= j) {
                            for (int k = i + 1; k < j; k++) {
                                if (r_s * d[k] <= m
                                    && T[k][m - r_s * d[k]] != -1.0
                                    && segment[i][k] + d[k] + (r_s - 1) * C[k][m - r_s * d[k]] <= m
                                    && f[k] + T[k][m - r_s * d[k]] >= T[i][m]) {
                                    T[i][m] = f[k] + T[k][m - r_s * d[k]];
                                    C[i][m] = d[i] + C[k][m - r_s * d[k]];
                                    next[i][m] = k;
                                    tilde[i][m] = false;
                                }
                            }
                            if (segment[i][j] <= m && f[j] >= T[i][m]) {
                                T[i][m] = f[j];
                                C[i][m] = d[i];
                                next[i][m] = j;
                                tilde[i][m] = false;
                            }
                        } else {
                            for (int k = i + 1; k <= u[i]; k++) {
                                if (r_s * d[k] <= m
                                    && T[k][m - r_s * d[k]] != -1.0
                                    && segment[i][k] + d[k] + (r_s - 1) * C[k][m - r_s * d[k]] <= m
                                    && f[k] + T[k][m - r_s * d[k]] >= T[i][m]) {
                                    T[i][m] = f[k] + T[k][m - r_s * d[k]];
                                    C[i][m] = d[i] + C[k][m - r_s * d[k]];
                                    next[i][m] = k;
                                    tilde[i][m] = false;
                                }
                            }
                        }
                    } else {
                        if (u[i] != -1) {
                            for (int k = i + 1; k < u[i]; k++) {
                                if (r_s * d[k] <= m
                                    && _T[k][m - r_s * d[k]] != -1.0
                                    && segment[i][k] + d[k] + (r_s - 1) * _C[k][m - r_s * d[k]] <= m
                                    && f[k] + _T[k][m - r_s * d[k]] >= T[i][m]) {
                                    T[i][m] = f[k] + _T[k][m - r_s * d[k]];
                                    C[i][m] = d[i] + _C[k][m - r_s * d[k]];
                                    next[i][m] = k;
                                    tilde[i][m] = true;
                                }
                            }
                        }

                        for (int k = i + 1; k < j; k++) {
                            if (r_s * d[k] <= m
                                && T[k][m - r_s * d[k]] != -1.0
                                && segment[i][k] + d[k] + (r_s - 1) * C[k][m - r_s * d[k]] <= m
                                && f[k] + T[k][m - r_s * d[k]] >= T[i][m]) {
                                T[i][m] = f[k] + T[k][m - r_s * d[k]];
                                C[i][m] = d[i] + C[k][m - r_s * d[k]];
                                next[i][m] = k;
                                tilde[i][m] = false;
                            }
                        }
                        if (segment[i][j] <= m && f[j] >= T[i][m]) {
                            T[i][m] = f[j];
                            C[i][m] = d[i];
                            next[i][m] = j;
                            tilde[i][m] = false;
                        }
                    }
                }
            }

            // Computation of the stage backward execution time.
            for (int i = 1; i <= j; i++) {
                if (!splits[i - 1]) continue;
                long m = M - (psum[j] - psum[i - 1]) * 4; // excludes weights, gradients and optimizer states.
                if (s == 0)
                    m -= d[i - 1] * atom * micro_batches; // excludes input data.
                else
                    m -= d[i - 1] * atom * micro_batches * 2; // excludes act_recv communication buffers, and cached grad_ins.
                if (s == p - 1)
                    m -= d[j] * atom * 4; // excludes output and loss function overhead.
                else
                    m -= d[j] * atom * micro_batches * 2; // excludes grad_recv communication buffers and cached outputs.
                m /= atom;
                if (m <= 0 || T[i - 1][m] == -1.0) {
                    cost[s][i][j] = numeric_limits<float>::max();
                } else {
                    cost[s][i][j] = (bsum[j] - bsum[i - 1]) + 2 * (fsum[j] - fsum[i - 1]) - T[i - 1][m];
                    // Backtracking selected checkpoints.
                    bool t = false; // current table is tilde or not
                    for (int k = i - 1; k != j;) {
                        if (t) {
                            t = _tilde[k][m];
                            k = _next[k][m];
                        } else {
                            t = tilde[k][m];
                            k = next[k][m];
                        }
                        m -= r_s * d[k];
                        trace[s][i][j].push_back(k);
                    }
                }
            }
        }
    }

    vector<vector<float>> P(p, vector<float>(n + 1, numeric_limits<float>::max()));
    vector<vector<int>> prev(p, vector<int>(n + 1, -1));

    for (int j = 1; j <= n; j++)
        if (splits[j]) P[0][j] = cost[0][1][j];

    for (int s = 1; s < p; s++) {
        for (int j = s + 1; j <= n; j++) {
            if (!splits[j]) continue;
            for (int i = s + 1; i <= j; i++) {
                if(!splits[i - 1]) continue;
                float curr = max(P[s - 1][i - 1], cost[s][i][j]);
                if (curr < P[s][j]) {
                    P[s][j] = curr;
                    prev[s][j] = i - 1;
                }
            }
        }
    }

    vector<int> split_idx(3);
    for (int s = p - 1, j = n; s > 0; s--) {
        split_idx[s - 1] = prev[s][j];
        j = prev[s][j];
    }

    vector<int> checkpoint_idx = {0};
    for (int s = 0, i = 1; s < p; s++) {
        int j = (s == p - 1)? n: split_idx[s];
        checkpoint_idx.insert(checkpoint_idx.end(), trace[s][i][j].begin(), trace[s][i][j].end());
        i = j + 1;
    }

    return {split_idx, checkpoint_idx};
}


PYBIND11_MODULE(dp, m) {
    m.def("dynamic_programming", &dynamic_programming, "Dynamic programing algorithm");
}

