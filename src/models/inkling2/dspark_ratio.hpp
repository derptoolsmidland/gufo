// Lab diagnostic, not offered upstream: speculative acceptance by probability
// for sampled requests (Leviathan et al. 2022, Chen et al. 2023).
//
// A drafted token x was drawn from the drafter's distribution q. The target's
// distribution at that position is p. Keep x with probability
// min(1, p(x) / q(x)); otherwise draw from the residual max(0, p - q),
// normalised. The token that comes out has distribution p exactly, provided
// the decision to check a draft at all did not depend on which token was
// drawn. The chance of keeping the draft is sum_i min(p_i, q_i), the overlap
// of the two distributions, which no other rule can beat.
//
// Host only, standard library only, so the rule can be tested without a GPU
// (tests/spec/ratio_test.cpp).
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace gufo::models::inkling2::ratio {

/// Weights of the drafter's proposal distribution over the head rows, as the
/// device computes them (hip/dspark.hip, cs_weight with no min-p floor):
/// w[k] = exp((softcap(logit_k) - max) / t), here in F64. Returns their sum.
inline double HeadWeights(std::span<const float> logits, float cap, float t,
                          std::vector<double>* w) {
  w->resize(logits.size());
  float m = -std::numeric_limits<float>::infinity();
  for (std::size_t k = 0; k < logits.size(); ++k) {
    const float c = cap > 0.f ? cap * std::tanh(logits[k] / cap) : logits[k];
    (*w)[k] = static_cast<double>(c);
    m = c > m ? c : m;
  }
  double total = 0;
  for (std::size_t k = 0; k < logits.size(); ++k) {
    (*w)[k] = std::exp(((*w)[k] - static_cast<double>(m)) /
                       static_cast<double>(t));
    total += (*w)[k];
  }
  return total;
}

/// The head row the device's walk picks for the uniform u (CoupledSample):
/// the first row whose running sum passes u * total.
inline std::size_t HeadPick(std::span<const double> w, double total, double u) {
  const double target = u * total;
  double run = 0;
  for (std::size_t k = 0; k < w.size(); ++k) {
    run += w[k];
    if (target < run)
      return k;
  }
  return w.empty() ? 0 : w.size() - 1;
}

/// Whether a draft with drafter probability qx and target probability px is
/// kept, for the uniform u.
inline bool Accept(double px, double qx, double u) {
  return u * qx < px;
}

struct Residual {
  std::int32_t token = -1;
  double mass = 0;  // sum of max(0, p - q) = 1 - overlap
};

/// The residual draw for the uniform u, walking the vocabulary in token
/// order. pw / ptotal: the target's weights over the vocabulary; qw / qtotal:
/// the drafter's over the head rows; ids: the token id of each head row,
/// ascending. A token outside the head has q = 0.
inline Residual SampleResidual(std::span<const double> pw, double ptotal,
                               std::span<const double> qw, double qtotal,
                               std::span<const int> ids, double u) {
  const auto walk = [&](auto&& visit) {
    std::size_t k = 0;
    for (std::size_t i = 0; i < pw.size(); ++i) {
      double q = 0;
      if (k < ids.size() && static_cast<std::size_t>(ids[k]) == i) {
        q = qw[k] / qtotal;
        ++k;
      }
      const double r = pw[i] / ptotal - q;
      if (r > 0 && !visit(i, r))
        return;
    }
  };
  Residual out;
  walk([&](std::size_t, double r) {
    out.mass += r;
    return true;
  });
  if (!(out.mass > 0)) {
    // p <= q everywhere (rounding only): fall back to a draw from p
    double run = 0;
    const double target = u * ptotal;
    for (std::size_t i = 0; i < pw.size(); ++i) {
      run += pw[i];
      out.token = static_cast<std::int32_t>(i);
      if (target < run)
        break;
    }
    return out;
  }
  const double target = u * out.mass;
  double run = 0;
  walk([&](std::size_t i, double r) {
    run += r;
    out.token = static_cast<std::int32_t>(i);
    return !(target < run);
  });
  return out;
}

}  // namespace gufo::models::inkling2::ratio
