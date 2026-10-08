/**
 * @file main.cpp
 * @brief Phase-0 / Phase-1 CLI driver.
 *
 *   --model ising  : 2D Ising β-sweep (Onsager transition near β = 0.4407)
 *   --model phi4   : 2D φ⁴ m²-sweep at fixed λ (Z₂ broken phase below m_c²)
 *
 * Both sweeps dump CSV to stdout; eyeball checks for the corresponding
 * symmetry-breaking transition are the same as the test-suite checks but
 * over a wider parameter range.
 */

#include "fields/link_field.hpp"
#include "fields/site_field.hpp"
#include "fields/spinor_field.hpp"
#include "lattice/lattice.hpp"
#include "math/dirac_2d.hpp"
#include "models/ising.hpp"
#include "models/phi4.hpp"
#include "models/u1.hpp"
#include "math/su2.hpp"
#include "models/su2.hpp"
#include "monte_carlo/hmc.hpp"
#include "monte_carlo/metropolis.hpp"
#include "monte_carlo/schwinger_hmc.hpp"
#include "monte_carlo/su2_heatbath.hpp"
#include "monte_carlo/tuning.hpp"
#include "monte_carlo/u1_heatbath.hpp"
#include "observables/condensate.hpp"
#include "observables/observables.hpp"
#include "rng/rng.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace
{

enum class Model { Ising, Phi4, U1, SU2, SchwingerQuenched, SchwingerHMC };

struct Cli
{
    Model         model          = Model::Ising;
    int           dim            = 2;        // only honored by --model u1 for now
    int           L              = 32;
    std::uint64_t seed           = 1234567ull;
    int           therm          = 500;
    int           measure_sweeps = 2000;
    int           sample_every   = 10;

    // Ising β-sweep
    double beta_min       = 0.30;
    double beta_max       = 0.55;
    int    beta_steps     = 11;

    // φ⁴ m²-sweep
    double m_sq_min       = -1.5;
    double m_sq_max       = +1.0;
    int    m_sq_steps     = 11;
    double lambda         = 1.0;

    // U(1) β-sweep (overlaps beta_* fields above; same flags reused)

    // Continuous-DOF step size: -1 ⇒ autotune to 50% acceptance.
    double step_size      = -1.0;

    // Quenched Schwinger fermion knobs
    double mass           = 0.5;
    int    n_sources      = 8;
    double cg_tol         = 1e-9;
    int    cg_max_iters   = 2000;

    // HMC knobs (used by schwinger-hmc)
    double hmc_dt         = 0.04;
    int    hmc_n_steps    = 20;

    // Use U(1) heat-bath sweep instead of Metropolis (--model u1 / schwinger-quenched)
    bool   use_heatbath   = false;
};

void printUsage()
{
    std::printf(
        "usage: LatticeQFT [--model ising|phi4|u1|su2|schwinger-quenched|schwinger-hmc]\n"
        "                  [--L N] [--seed S] [--therm N] [--measure-sweeps N] [--sample-every K]\n"
        "  ising-specific:\n"
        "    [--beta-min B] [--beta-max B] [--beta-steps K]\n"
        "  phi4-specific:\n"
        "    [--m-sq-min M] [--m-sq-max M] [--m-sq-steps K]\n"
        "    [--lambda L] [--step-size S]   (step-size -1 ⇒ autotune to 50%% acc)\n"
        "  u1-specific:\n"
        "    [--beta-min B] [--beta-max B] [--beta-steps K] [--step-size S]\n"
        "    [--dim D]   (Dim ∈ {2, 3, 4}; default 2)\n"
        "    [--use-heatbath]\n"
        "  su2-specific:\n"
        "    [--beta-min B] [--beta-max B] [--beta-steps K] [--step-size S]\n"
        "    [--dim D]   (Dim ∈ {2, 3, 4}; default 3)\n"
        "    [--use-heatbath]   (Kennedy-Pendleton)\n"
        "  schwinger-quenched / schwinger-hmc (2D only):\n"
        "    [--beta-min B] [--beta-max B] [--beta-steps K] [--step-size S]\n"
        "    [--mass M] [--n-sources K] [--cg-tol T]\n"
        "    schwinger-hmc adds: [--hmc-dt T] [--hmc-n-steps N]\n"
        "\n"
        "CSV is emitted to stdout; per-sweep telemetry to stderr.\n"
    );
}

bool parseInt(const char* s, int& out)
{ try { out = std::stoi(s); return true; } catch (...) { return false; } }
bool parseU64(const char* s, std::uint64_t& out)
{ try { out = std::stoull(s); return true; } catch (...) { return false; } }
bool parseDouble(const char* s, double& out)
{ try { out = std::stod(s); return true; } catch (...) { return false; } }

bool parseCli(int argc, char** argv, Cli& cli)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto need  = [&](double& out)        { return i + 1 < argc && parseDouble(argv[++i], out); };
        auto needI = [&](int& out)           { return i + 1 < argc && parseInt   (argv[++i], out); };
        auto needU = [&](std::uint64_t& out) { return i + 1 < argc && parseU64   (argv[++i], out); };

        if      (a == "--help" || a == "-h") { printUsage(); std::exit(0); }
        else if (a == "--model")
        {
            if (i + 1 >= argc) return false;
            std::string v = argv[++i];
            if      (v == "ising")              cli.model = Model::Ising;
            else if (v == "phi4")               cli.model = Model::Phi4;
            else if (v == "u1")                 cli.model = Model::U1;
            else if (v == "su2")                { cli.model = Model::SU2; if (cli.dim == 2) cli.dim = 3; }
            else if (v == "schwinger-quenched") cli.model = Model::SchwingerQuenched;
            else if (v == "schwinger-hmc")      cli.model = Model::SchwingerHMC;
            else { std::fprintf(stderr, "unknown model: %s\n", v.c_str()); return false; }
        }
        else if (a == "--L")              { if (!needI(cli.L))               return false; }
        else if (a == "--dim")            { if (!needI(cli.dim))             return false; }
        else if (a == "--seed")           { if (!needU(cli.seed))            return false; }
        else if (a == "--therm")          { if (!needI(cli.therm))           return false; }
        else if (a == "--measure-sweeps") { if (!needI(cli.measure_sweeps))  return false; }
        else if (a == "--sample-every")   { if (!needI(cli.sample_every))    return false; }
        else if (a == "--beta-min")       { if (!need (cli.beta_min))        return false; }
        else if (a == "--beta-max")       { if (!need (cli.beta_max))        return false; }
        else if (a == "--beta-steps")     { if (!needI(cli.beta_steps))      return false; }
        else if (a == "--m-sq-min")       { if (!need (cli.m_sq_min))        return false; }
        else if (a == "--m-sq-max")       { if (!need (cli.m_sq_max))        return false; }
        else if (a == "--m-sq-steps")     { if (!needI(cli.m_sq_steps))      return false; }
        else if (a == "--lambda")         { if (!need (cli.lambda))          return false; }
        else if (a == "--step-size")      { if (!need (cli.step_size))       return false; }
        else if (a == "--mass")           { if (!need (cli.mass))            return false; }
        else if (a == "--n-sources")      { if (!needI(cli.n_sources))       return false; }
        else if (a == "--cg-tol")         { if (!need (cli.cg_tol))          return false; }
        else if (a == "--hmc-dt")         { if (!need (cli.hmc_dt))          return false; }
        else if (a == "--hmc-n-steps")    { if (!needI(cli.hmc_n_steps))     return false; }
        else if (a == "--use-heatbath")   { cli.use_heatbath = true; }
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return false; }
    }
    return true;
}

// -----------------------------------------------------------------------------
// Ising β-sweep
// -----------------------------------------------------------------------------

int runIsing(const Cli& cli)
{
    using namespace lqft;
    constexpr int Dim = 2;
    Lattice<Dim> lattice = Lattice<Dim>::cube(cli.L);
    SiteField<std::int8_t, Dim> field(lattice);
    Rng rng(cli.seed);
    ising::IsingModel<Dim>::hot(field, rng);

    std::fprintf(stderr,
        "[ising] 2D Ising  L=%d  V=%d  seed=%llu  therm=%d  measure=%d\n",
        cli.L, lattice.volume(), (unsigned long long)cli.seed,
        cli.therm, cli.measure_sweeps);
    std::fprintf(stderr,
        "[ising] Onsager β_c (2D) = %.10f\n", ising::kOnsagerBetaC2D);

    std::printf("beta,abs_m,abs_m_err,energy_density,acceptance\n");

    for (int k = 0; k < cli.beta_steps; ++k)
    {
        const double beta = cli.beta_min
            + (cli.beta_max - cli.beta_min) * static_cast<double>(k)
            / static_cast<double>(std::max(1, cli.beta_steps - 1));

        ising::IsingModel<Dim> model(lattice, beta);
        mc::MetropolisSweep<ising::IsingModel<Dim>> sweeper(model);
        sweeper.sweepN(field, rng, cli.therm);
        sweeper.resetCounters();

        obs::Mean abs_m, e_density;
        for (int s = 0; s < cli.measure_sweeps; ++s)
        {
            sweeper.sweep(field, rng);
            if ((s % cli.sample_every) == 0)
            {
                abs_m.add(obs::abs_magnetization(field));
                e_density.add(-obs::nn_pair_density(lattice, field));
            }
        }

        std::printf("%.5f,%.6f,%.6f,%.6f,%.6f\n",
            beta,
            abs_m.mean(), abs_m.naiveStdErr(),
            e_density.mean(),
            sweeper.cumulativeAcceptance());
        std::fflush(stdout);
    }
    return 0;
}

// -----------------------------------------------------------------------------
// φ⁴ m²-sweep at fixed λ
// -----------------------------------------------------------------------------

int runPhi4(const Cli& cli)
{
    using namespace lqft;
    constexpr int Dim = 2;
    Lattice<Dim> lattice = Lattice<Dim>::cube(cli.L);
    SiteField<double, Dim> field(lattice);
    Rng rng(cli.seed);
    phi4::Phi4Model<Dim>::hot(field, rng, 1.0);

    std::fprintf(stderr,
        "[phi4] 2D φ⁴  L=%d  V=%d  seed=%llu  λ=%.4f  step_size=%.4f%s\n",
        cli.L, lattice.volume(), (unsigned long long)cli.seed,
        cli.lambda, cli.step_size,
        cli.step_size < 0.0 ? " (autotune)" : "");

    std::printf("m_sq,abs_phi,abs_phi_err,phi2,phi2_err,action_density,step_size,acceptance\n");

    for (int k = 0; k < cli.m_sq_steps; ++k)
    {
        const double m_sq = cli.m_sq_min
            + (cli.m_sq_max - cli.m_sq_min) * static_cast<double>(k)
            / static_cast<double>(std::max(1, cli.m_sq_steps - 1));

        const double init_step = (cli.step_size > 0.0) ? cli.step_size : 1.0;
        phi4::Phi4Model<Dim> model(lattice, m_sq, cli.lambda, init_step);

        if (cli.step_size < 0.0)
        {
            // Autotune step size to ~50 % acceptance during thermalization.
            mc::autoTuneStepSize(model, field, rng,
                                 /*batch_sweeps*/100,
                                 /*max_iters*/20,
                                 /*target*/0.5,
                                 /*tol*/0.05);
        }

        mc::MetropolisSweep<phi4::Phi4Model<Dim>> sweeper(model);
        sweeper.sweepN(field, rng, cli.therm);
        sweeper.resetCounters();

        obs::Mean abs_phi, phi2, action_density;
        for (int s = 0; s < cli.measure_sweeps; ++s)
        {
            sweeper.sweep(field, rng);
            if ((s % cli.sample_every) == 0)
            {
                abs_phi.add(obs::abs_mean_field(field));
                phi2  .add(obs::mean_field_squared(field));
                action_density.add(model.totalAction(field)
                                  / static_cast<double>(lattice.volume()));
            }
        }

        std::printf("%.5f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            m_sq,
            abs_phi.mean(), abs_phi.naiveStdErr(),
            phi2   .mean(), phi2   .naiveStdErr(),
            action_density.mean(),
            model.stepSize(),
            sweeper.cumulativeAcceptance());
        std::fflush(stdout);
    }
    return 0;
}

// -----------------------------------------------------------------------------
// U(1) β-sweep — compact-U(1) Wilson plaquette gauge theory.
// Templated on Dim so the same body services 2D / 3D / 4D from one switch.
// In Dim = 2 the Bessel-ratio closed form is emitted alongside the MC value;
// in higher Dim there is no exact analytic, so those columns are omitted.
// -----------------------------------------------------------------------------

template<int Dim>
int runU1Dim(const Cli& cli)
{
    using namespace lqft;
    Lattice<Dim> lattice = Lattice<Dim>::cube(cli.L);
    LinkField<double, Dim> field(lattice);
    Rng rng(cli.seed);
    u1::U1Model<Dim>::hot(field, rng);

    std::fprintf(stderr,
        "[u1] %dD compact U(1)  L=%d  V=%d  links/site=%d  plaq-planes/site=%d\n",
        Dim, cli.L, lattice.volume(), Dim, Dim * (Dim - 1) / 2);
    std::fprintf(stderr,
        "[u1] seed=%llu  therm=%d  measure=%d\n",
        (unsigned long long)cli.seed, cli.therm, cli.measure_sweeps);

    if constexpr (Dim == 2)
        std::printf("beta,plaq,plaq_err,plaq_analytic,W22,W22_err,W22_analytic,step_size,acceptance\n");
    else
        std::printf("beta,plaq,plaq_err,W22,W22_err,step_size,acceptance\n");

    for (int k = 0; k < cli.beta_steps; ++k)
    {
        const double beta = cli.beta_min
            + (cli.beta_max - cli.beta_min) * static_cast<double>(k)
            / static_cast<double>(std::max(1, cli.beta_steps - 1));

        const double init_step = (cli.step_size > 0.0) ? cli.step_size : 1.0;
        u1::U1Model<Dim> model(lattice, beta, init_step);

        // ---- Thermalization ----
        if (cli.use_heatbath)
        {
            u1::heatBathSweepN(model, field, rng, cli.therm);
        }
        else
        {
            if (cli.step_size < 0.0)
                mc::autoTuneStepSize(model, field, rng,
                                     /*batch*/100, /*max_iters*/20, /*target*/0.5, /*tol*/0.05);
            mc::MetropolisSweep<u1::U1Model<Dim>> sweeper(model);
            sweeper.sweepN(field, rng, cli.therm);
        }

        // ---- Measurement (one sweeper persists across measurement loop for accept stats) ----
        obs::Mean plaq, w22;
        double cumulative_acc = 1.0;
        if (cli.use_heatbath)
        {
            for (int s = 0; s < cli.measure_sweeps; ++s)
            {
                u1::heatBathSweep(model, field, rng);
                if ((s % cli.sample_every) == 0)
                {
                    plaq.add(u1::averagePlaquette(lattice, field));
                    w22 .add(u1::averageWilsonLoop(lattice, field, 0, 1, 2, 2));
                }
            }
        }
        else
        {
            mc::MetropolisSweep<u1::U1Model<Dim>> sweeper(model);
            for (int s = 0; s < cli.measure_sweeps; ++s)
            {
                sweeper.sweep(field, rng);
                if ((s % cli.sample_every) == 0)
                {
                    plaq.add(u1::averagePlaquette(lattice, field));
                    w22 .add(u1::averageWilsonLoop(lattice, field, 0, 1, 2, 2));
                }
            }
            cumulative_acc = sweeper.cumulativeAcceptance();
        }

        if constexpr (Dim == 2)
        {
            const double r         = u1::besselRatio(beta);
            const double w22_exact = r * r * r * r;
            std::printf("%.5f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                beta,
                plaq.mean(), plaq.naiveStdErr(), r,
                w22 .mean(), w22 .naiveStdErr(), w22_exact,
                model.stepSize(), cumulative_acc);
        }
        else
        {
            std::printf("%.5f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                beta,
                plaq.mean(), plaq.naiveStdErr(),
                w22 .mean(), w22 .naiveStdErr(),
                model.stepSize(), cumulative_acc);
        }
        std::fflush(stdout);
    }
    return 0;
}

int runU1(const Cli& cli)
{
    switch (cli.dim)
    {
        case 2: return runU1Dim<2>(cli);
        case 3: return runU1Dim<3>(cli);
        case 4: return runU1Dim<4>(cli);
        default:
            std::fprintf(stderr, "U(1): unsupported --dim %d (allowed: 2, 3, 4)\n", cli.dim);
            return 1;
    }
}

// -----------------------------------------------------------------------------
// SU(2) pure-gauge β-sweep.
// At each β: thermalize, measure ⟨½ tr U_□⟩ and Wilson loops W(1,1), W(2,1),
// W(1,2), W(2,2), then derive the Creutz ratio χ(2,2) — the lattice
// estimator of the string tension when (R, T) → ∞.
// Templated on Dim so 2D / 3D / 4D run from the same code path.
// -----------------------------------------------------------------------------

template<int Dim>
int runSU2Dim(const Cli& cli)
{
    using namespace lqft;
    Lattice<Dim> lattice = Lattice<Dim>::cube(cli.L);
    LinkField<su2::Element, Dim> field(lattice);
    Rng rng(cli.seed);
    su2_model::SU2Model<Dim>::hot(field, rng);

    std::fprintf(stderr,
        "[su2] %dD SU(2) pure gauge  L=%d  V=%d  links/site=%d  plaq-planes/site=%d\n",
        Dim, cli.L, lattice.volume(), Dim, Dim * (Dim - 1) / 2);
    std::fprintf(stderr,
        "[su2] seed=%llu  therm=%d  measure=%d  algo=%s\n",
        (unsigned long long)cli.seed, cli.therm, cli.measure_sweeps,
        cli.use_heatbath ? "heat-bath (Kennedy-Pendleton)" : "Metropolis");

    std::printf("beta,plaq,plaq_err,W21,W21_err,W12,W12_err,W22,W22_err,creutz22,acceptance\n");

    for (int k = 0; k < cli.beta_steps; ++k)
    {
        const double beta = cli.beta_min
            + (cli.beta_max - cli.beta_min) * static_cast<double>(k)
            / static_cast<double>(std::max(1, cli.beta_steps - 1));

        const double init_step = (cli.step_size > 0.0) ? cli.step_size : 0.4;
        su2_model::SU2Model<Dim> model(lattice, beta, init_step);

        // ---- Thermalize ----
        double cumulative_acc = 1.0;
        if (cli.use_heatbath)
        {
            su2_model::heatBathSweepN(model, field, rng, cli.therm);
        }
        else
        {
            mc::MetropolisSweep<su2_model::SU2Model<Dim>> sweeper(model);
            sweeper.sweepN(field, rng, cli.therm);
        }

        // ---- Measure ----
        obs::Mean plaq, W21, W12, W22;
        if (cli.use_heatbath)
        {
            for (int s = 0; s < cli.measure_sweeps; ++s)
            {
                su2_model::heatBathSweep(model, field, rng);
                if ((s % cli.sample_every) == 0)
                {
                    plaq.add(su2_model::averagePlaquette(lattice, field));
                    W21 .add(su2_model::averageWilsonLoop(lattice, field, 0, 1, 2, 1));
                    W12 .add(su2_model::averageWilsonLoop(lattice, field, 0, 1, 1, 2));
                    W22 .add(su2_model::averageWilsonLoop(lattice, field, 0, 1, 2, 2));
                }
            }
        }
        else
        {
            mc::MetropolisSweep<su2_model::SU2Model<Dim>> sweeper(model);
            for (int s = 0; s < cli.measure_sweeps; ++s)
            {
                sweeper.sweep(field, rng);
                if ((s % cli.sample_every) == 0)
                {
                    plaq.add(su2_model::averagePlaquette(lattice, field));
                    W21 .add(su2_model::averageWilsonLoop(lattice, field, 0, 1, 2, 1));
                    W12 .add(su2_model::averageWilsonLoop(lattice, field, 0, 1, 1, 2));
                    W22 .add(su2_model::averageWilsonLoop(lattice, field, 0, 1, 2, 2));
                }
            }
            cumulative_acc = sweeper.cumulativeAcceptance();
        }

        const double chi22 = su2_model::creutzRatio(
            W22.mean(), plaq.mean(), W12.mean(), W21.mean());

        std::printf("%.5f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f\n",
            beta,
            plaq.mean(), plaq.naiveStdErr(),
            W21 .mean(), W21 .naiveStdErr(),
            W12 .mean(), W12 .naiveStdErr(),
            W22 .mean(), W22 .naiveStdErr(),
            chi22, cumulative_acc);
        std::fflush(stdout);
    }
    return 0;
}

int runSU2(const Cli& cli)
{
    switch (cli.dim)
    {
        case 2: return runSU2Dim<2>(cli);
        case 3: return runSU2Dim<3>(cli);
        case 4: return runSU2Dim<4>(cli);
        default:
            std::fprintf(stderr, "SU(2): unsupported --dim %d (allowed: 2, 3, 4)\n", cli.dim);
            return 1;
    }
}

// -----------------------------------------------------------------------------
// Quenched Schwinger β-sweep at fixed fermion mass.
// Reuses the U(1) Metropolis sweeper; at each measurement step also computes
// the stochastic chiral condensate via N_src Z₂ noise sources + CG.
// -----------------------------------------------------------------------------

int runSchwingerQuenched(const Cli& cli)
{
    using namespace lqft;
    constexpr int Dim = 2;
    Lattice<Dim> lattice = Lattice<Dim>::cube(cli.L);
    LinkField<double, Dim> field(lattice);
    Rng rng(cli.seed);
    u1::U1Model<Dim>::hot(field, rng);

    std::fprintf(stderr,
        "[schwinger] 2D quenched Schwinger  L=%d  V=%d  mass=%.4f\n",
        cli.L, lattice.volume(), cli.mass);
    std::fprintf(stderr,
        "[schwinger] seed=%llu  therm=%d  measure=%d  n_sources=%d  cg_tol=%.1e\n",
        (unsigned long long)cli.seed, cli.therm, cli.measure_sweeps,
        cli.n_sources, cli.cg_tol);

    std::printf("beta,plaq,plaq_err,condensate,condensate_se,avg_cg_iters,gauge_acceptance\n");

    dirac::WilsonDirac2D D(lattice, cli.mass);

    for (int k = 0; k < cli.beta_steps; ++k)
    {
        const double beta = cli.beta_min
            + (cli.beta_max - cli.beta_min) * static_cast<double>(k)
            / static_cast<double>(std::max(1, cli.beta_steps - 1));

        const double init_step = (cli.step_size > 0.0) ? cli.step_size : 1.0;
        u1::U1Model<Dim> model(lattice, beta, init_step);

        if (cli.step_size < 0.0)
            mc::autoTuneStepSize(model, field, rng,
                                 /*batch*/100, /*max*/20, /*target*/0.5, /*tol*/0.05);

        mc::MetropolisSweep<u1::U1Model<Dim>> sweeper(model);
        sweeper.sweepN(field, rng, cli.therm);
        sweeper.resetCounters();

        obs::Mean plaq, condensate;
        long  iters_total = 0;
        int   meas_count  = 0;

        for (int s = 0; s < cli.measure_sweeps; ++s)
        {
            sweeper.sweep(field, rng);
            if ((s % cli.sample_every) == 0)
            {
                plaq.add(u1::averagePlaquette(lattice, field));
                const auto r = obs::stochasticCondensate(
                    D, field, lattice, rng,
                    cli.n_sources, cli.cg_tol, cli.cg_max_iters);
                condensate.add(r.mean);
                iters_total += r.avg_cg_iters;
                ++meas_count;
            }
        }

        const int avg_iters = meas_count > 0
            ? static_cast<int>(iters_total / meas_count) : 0;
        std::printf("%.5f,%.6f,%.6f,%.6f,%.6f,%d,%.4f\n",
            beta,
            plaq.mean(), plaq.naiveStdErr(),
            condensate.mean(), condensate.naiveStdErr(),
            avg_iters,
            sweeper.cumulativeAcceptance());
        std::fflush(stdout);
    }
    return 0;
}

// -----------------------------------------------------------------------------
// Dynamical Schwinger model via pseudofermion HMC.
// β-sweep at fixed mass, measuring plaquette + condensate per trajectory.
// -----------------------------------------------------------------------------

int runSchwingerHMC(const Cli& cli)
{
    using namespace lqft;
    constexpr int Dim = 2;
    Lattice<Dim> lattice = Lattice<Dim>::cube(cli.L);
    LinkField<double, Dim> U(lattice);
    Rng rng(cli.seed);
    u1::U1Model<Dim>::hot(U, rng);

    std::fprintf(stderr,
        "[schwinger-hmc] 2D dynamical Schwinger HMC  L=%d  V=%d  mass=%.4f\n"
        "[schwinger-hmc] dt=%.4f  n_steps=%d  trajectories=%d  measure-every=%d\n",
        cli.L, lattice.volume(), cli.mass,
        cli.hmc_dt, cli.hmc_n_steps, cli.measure_sweeps, cli.sample_every);

    dirac::WilsonDirac2D D(lattice, cli.mass);

    std::printf("beta,plaq,plaq_err,condensate,condensate_se,acceptance,avg_dH\n");

    for (int k = 0; k < cli.beta_steps; ++k)
    {
        const double beta = cli.beta_min
            + (cli.beta_max - cli.beta_min) * static_cast<double>(k)
            / static_cast<double>(std::max(1, cli.beta_steps - 1));

        u1::U1Model<Dim> gauge(lattice, beta);
        schwinger::SchwingerProvider provider(
            gauge, D, lattice, cli.cg_tol, cli.cg_max_iters);
        hmc::GaugeHMC<schwinger::SchwingerProvider> H(
            provider, cli.hmc_dt, cli.hmc_n_steps);

        // Thermalize.
        for (int t = 0; t < cli.therm; ++t)
        {
            provider.refreshPseudofermion(U, rng);
            H.trajectory(U, rng);
        }
        H.resetCounters();

        obs::Mean plaq, condensate, dH;
        for (int t = 0; t < cli.measure_sweeps; ++t)
        {
            provider.refreshPseudofermion(U, rng);
            const auto r = H.trajectory(U, rng);
            dH.add(r.dH);
            if ((t % cli.sample_every) == 0)
            {
                plaq.add(u1::averagePlaquette(lattice, U));
                const auto c = obs::stochasticCondensate(
                    D, U, lattice, rng, cli.n_sources, cli.cg_tol, cli.cg_max_iters);
                condensate.add(c.mean);
            }
        }

        std::printf("%.5f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f\n",
            beta,
            plaq.mean(), plaq.naiveStdErr(),
            condensate.mean(), condensate.naiveStdErr(),
            H.cumulativeAcceptance(),
            dH.mean());
        std::fflush(stdout);
    }
    return 0;
}

} // anonymous namespace

int main(int argc, char** argv)
{
    Cli cli;
    if (!parseCli(argc, argv, cli)) { printUsage(); return 1; }

    switch (cli.model)
    {
        case Model::Ising:             return runIsing(cli);
        case Model::Phi4:              return runPhi4(cli);
        case Model::U1:                return runU1(cli);
        case Model::SU2:               return runSU2(cli);
        case Model::SchwingerQuenched: return runSchwingerQuenched(cli);
        case Model::SchwingerHMC:      return runSchwingerHMC(cli);
    }
    return 0;
}
