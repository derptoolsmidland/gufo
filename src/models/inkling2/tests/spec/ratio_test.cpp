// Lab diagnostic (host only): the probability rule of dspark_ratio.hpp gives
// tokens with the target's distribution.
//
// Each case fixes a target distribution p over a small vocabulary and a
// drafter distribution q over a subset of it (the "proposal head"), draws a
// proposal from q the way the device does (HeadPick over the weights, with a
// 24-bit uniform and the proposal's probability rounded to float), applies
// Accept and SampleResidual, and compares the histogram of what comes out
// with p by a chi-square test. The acceptance rate must equal the overlap
// sum(min(p, q)). Two wrong rules are run as controls and must fail the same
// test: redrawing from p after a rejection, and deciding whether to use the
// proposal from the drawn token's own probability.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "src/models/inkling2/dspark_ratio.hpp"

namespace {
using namespace gufo::models::inkling2;

struct Case {
  const char* name;
  int vocab, head;       // head rows: every token id with id % 3 != 2, capped
  float temperature;
  double target_scale;   // spread of the target's logits
  double draft_noise;    // how far the drafter's logits are from the target's
};

struct Rng {
  std::mt19937_64 g;
  double U() { return static_cast<double>(g() >> 11U) * 0x1.0p-53; }
  float U24() { return static_cast<float>(g() >> 40U) * 0x1.0p-24F; }
  double N() {
    std::normal_distribution<double> d;
    return d(g);
  }
};

// Upper critical value of chi-square with df degrees of freedom at a tail of
// about 1e-4 (Wilson-Hilferty).
double Critical(double df) {
  const double z = 3.72, a = 2.0 / (9.0 * df);
  const double c = 1.0 - a + z * std::sqrt(a);
  return df * c * c * c;
}

double ChiSquare(const std::vector<std::uint64_t>& count,
                 const std::vector<double>& p, std::uint64_t n, double* df) {
  double chi = 0, rest_e = 0, rest_o = 0;
  int bins = 0;
  for (std::size_t i = 0; i < p.size(); ++i) {
    const double e = p[i] * static_cast<double>(n);
    if (e < 20) {  // pool the rare tokens
      rest_e += e;
      rest_o += static_cast<double>(count[i]);
      continue;
    }
    const double d = static_cast<double>(count[i]) - e;
    chi += d * d / e;
    ++bins;
  }
  if (rest_e > 0) {
    const double d = rest_o - rest_e;
    chi += d * d / rest_e;
    ++bins;
  }
  *df = bins - 1;
  return chi;
}

enum class Rule { kCorrect, kRedrawFromTarget, kUseByDrawnProbability };

bool Run(const Case& c, Rule rule, std::uint64_t trials, bool expect_pass) {
  Rng rng{std::mt19937_64(0x1234 + static_cast<std::uint64_t>(c.vocab) * 7 +
                          static_cast<std::uint64_t>(rule))};
  const std::size_t V = static_cast<std::size_t>(c.vocab);
  std::vector<float> target_logits(V);
  for (auto& l : target_logits)
    l = static_cast<float>(c.target_scale * rng.N());
  // the target's weights as the sampler takes them
  std::vector<double> pw(V);
  double pmax = -1e300, ptotal = 0;
  for (std::size_t i = 0; i < V; ++i)
    pmax = std::max(pmax, static_cast<double>(target_logits[i]));
  for (std::size_t i = 0; i < V; ++i) {
    pw[i] = std::exp((static_cast<double>(target_logits[i]) - pmax) /
                     c.temperature);
    ptotal += pw[i];
  }
  std::vector<double> p(V);
  for (std::size_t i = 0; i < V; ++i)
    p[i] = pw[i] / ptotal;
  // the head and the drafter's logits over it
  std::vector<int> ids;
  for (int i = 0; i < c.vocab && static_cast<int>(ids.size()) < c.head; ++i)
    if (i % 3 != 2)
      ids.push_back(i);
  std::vector<float> head_logits(ids.size());
  for (std::size_t k = 0; k < ids.size(); ++k)
    head_logits[k] = target_logits[static_cast<std::size_t>(ids[k])] +
                     static_cast<float>(c.draft_noise * rng.N());
  std::vector<double> qw;
  const double qtotal = ratio::HeadWeights(head_logits, 30.f, c.temperature, &qw);
  double overlap = 0;
  {
    std::size_t k = 0;
    for (std::size_t i = 0; i < V; ++i) {
      double q = 0;
      if (k < ids.size() && static_cast<std::size_t>(ids[k]) == i)
        q = qw[k++] / qtotal;
      overlap += std::min(p[i], q);
    }
  }
  double qmax = 0;
  for (const double w : qw)
    qmax = std::max(qmax, w / qtotal);
  std::vector<std::uint64_t> count(V);
  std::uint64_t kept = 0, drafted = 0;
  for (std::uint64_t t = 0; t < trials; ++t) {
    const std::size_t k = ratio::HeadPick(qw, qtotal, rng.U24());
    const std::size_t x = static_cast<std::size_t>(ids[k]);
    const double qx = static_cast<double>(static_cast<float>(qw[k] / qtotal));
    std::int32_t token = -1;
    if (rule == Rule::kUseByDrawnProbability && qx < 0.5 * qmax) {
      // wrong: the proposal is dropped because of the token that was drawn
      token = ratio::SampleResidual(pw, ptotal, {}, 1.0, {}, rng.U()).token;
    } else {
      ++drafted;
      if (ratio::Accept(p[x], qx, rng.U())) {
        token = static_cast<std::int32_t>(x);
        ++kept;
      } else if (rule == Rule::kRedrawFromTarget) {
        // wrong: a plain draw instead of the residual
        token = ratio::SampleResidual(pw, ptotal, {}, 1.0, {}, rng.U()).token;
      } else {
        token =
            ratio::SampleResidual(pw, ptotal, qw, qtotal, ids, rng.U()).token;
      }
    }
    ++count[static_cast<std::size_t>(token)];
  }
  double df = 0;
  const double chi = ChiSquare(count, p, trials, &df);
  const double crit = Critical(df);
  const double rate = static_cast<double>(kept) / static_cast<double>(drafted);
  const double sd = std::sqrt(overlap * (1 - overlap) /
                              static_cast<double>(drafted));
  bool ok = (chi < crit) == expect_pass;
  if (rule == Rule::kCorrect && std::abs(rate - overlap) > 4.5 * sd + 1e-6)
    ok = false;
  const char* label = rule == Rule::kCorrect             ? "probability rule"
                      : rule == Rule::kRedrawFromTarget  ? "control: plain redraw"
                                                         : "control: use by drawn q";
  std::printf("%-22s %-24s chi2 %9.1f (limit %6.1f, df %3.0f)  kept %.4f  "
              "overlap %.4f  %s\n",
              c.name, label, chi, crit, df, rate, overlap,
              ok ? (expect_pass ? "ok" : "ok (fails, as it must)") : "FAILED");
  return ok;
}
}  // namespace

int main() {
  const Case cases[] = {
      {"close, T=1", 60, 40, 1.0f, 1.5, 0.5},
      {"close, T=0.7", 60, 40, 0.7f, 1.5, 0.5},
      {"far apart, T=1", 60, 40, 1.0f, 1.5, 2.0},
      {"peaked, T=1", 60, 40, 1.0f, 4.0, 1.0},
      {"small head, T=1", 90, 20, 1.0f, 1.0, 0.7},
      {"flat, T=1.3", 200, 130, 1.3f, 0.8, 0.8},
  };
  constexpr std::uint64_t kTrials = 4'000'000;
  bool ok = true;
  for (const Case& c : cases) {
    ok &= Run(c, Rule::kCorrect, kTrials, true);
    ok &= Run(c, Rule::kRedrawFromTarget, kTrials, false);
    ok &= Run(c, Rule::kUseByDrawnProbability, kTrials, false);
  }
  std::printf(ok ? "ALL PASS\n" : "FAILED\n");
  return ok ? 0 : 1;
}
