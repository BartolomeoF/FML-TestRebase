#ifndef GRAVITYMODEL_HiCOLA_HEADER
#define GRAVITYMODEL_HiCOLA_HEADER

#include "GravityModel.h"
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/NBody/NBody.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>
#include <FML/FileUtils/FileUtils.h>

/// HiCOLA coupling+screening factor model
template <int NDIM>
class GravityModelHiCOLA final : public GravityModel<NDIM> {
  public:
    template <int N>
    using FFTWGrid = FML::GRID::FFTWGrid<N>;
    using ParameterMap = FML::UTILS::ParameterMap;
    using Spline = FML::INTERPOLATION::SPLINE::Spline;
    using DVector = FML::INTERPOLATION::SPLINE::DVector;

    GravityModelHiCOLA() : GravityModel<NDIM>("HiCOLA") {}
    GravityModelHiCOLA(std::shared_ptr<Cosmology> cosmo) : GravityModel<NDIM>(cosmo, "HiCOLA") {}

    //========================================================================
    // Print some info
    //========================================================================
    void info() const override {
        GravityModel<NDIM>::info();
        if (FML::ThisTask == 0) {
                std::cout << "# HiCOLA_preforce_filename : " << HiCOLA_preforce_filename << "\n";
                std::cout << "# Screening method : " << use_screening_method << "\n";
            if (use_screening_method) {
                std::cout << "# Enforce correct linear evolution : " << screening_enforce_largescale_linear << "\n";
                std::cout << "# Scale for which we enforce this  : " << screening_linear_scale_hmpc << " h/Mpc\n";
            }
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }

    //========================================================================
    // Internal methods to get coupling and chi/delta from splines/files?
    //========================================================================

    void read_and_spline_chi_over_delta_chi() {
      DVector HiCOLA_a_arr;
      DVector chi_over_delta_arr;
      DVector coupling_arr;

      // Fileformat: [a, chi/delta, coupling]
      const int ncols = 3;
      const int col_a = 0;
      const int col_chi = 1;
      const int col_coupl = 2;
      std::vector<int> cols_to_keep{col_a, col_chi, col_coupl};
      const int nheaderlines = 0;

      auto HiCOLAdata = FML::FILEUTILS::read_regular_ascii(HiCOLA_preforce_filename, ncols, cols_to_keep, nheaderlines);

      HiCOLA_a_arr.resize(HiCOLAdata.size());
      chi_over_delta_arr.resize(HiCOLAdata.size());
      coupling_arr.resize(HiCOLAdata.size());

      for (size_t i = 0; i < HiCOLA_a_arr.size(); i++) {
          HiCOLA_a_arr[i] = HiCOLAdata[i][col_a];
          chi_over_delta_arr[i] = HiCOLAdata[i][col_chi];
          coupling_arr[i] = HiCOLAdata[i][col_coupl];
          //std::cout << "# HiCOLA input: " << i << " " << HiCOLA_a_arr[i] << " " << chi_over_delta_arr[i] << " " << coupling_arr[i] << "\n";
      }

      chi_over_delta_spline.create(HiCOLA_a_arr, chi_over_delta_arr, "chi/delta(a)");
      coupling_spline.create(HiCOLA_a_arr, coupling_arr, "coupling(a)");

    }

    double get_chi_over_delta(double a) const {
        return chi_over_delta_spline(a);
    }

    double get_coupling(double a) const {
        return coupling_spline(a);
    }

    //========================================================================
    // Effective newtonian constant
    //========================================================================
    double GeffOverG(double a, [[maybe_unused]] double koverH0 = 0.0) const override {
        return 1.0 + get_coupling(a);
    }

    // For the LPT equations there is a modified >= 2LPT factor //BILL: for HiCOLA we don't want to modify the 2LPT for now (i.e. just use the GR 2LPT equation for D_2 but with modified D_1)
    //========================================================================
    //========================================================================
    //double source_factor_2LPT(double a, [[maybe_unused]] double koverH0 = 0.0) const override {
    //    double betadgp = get_beta_dgp(a);
    //    // The base-class contains how to deal with massive neutrinos so we multiply this in
    //    // to avoid having to also implement it here
    //    return GravityModel<NDIM>::source_factor_2LPT(a, koverH0) *
    //           (1.0 - (2.0 * rcH0_DGP * rcH0_DGP * this->cosmo->get_OmegaM()) /
    //                      (9.0 * a * a * a * (betadgp * betadgp * betadgp) * (1 + 1.0 / (3.0 * betadgp))));
    //}

    //========================================================================
    // Compute the force DPhi from the density field delta in fourier space
    // We compute this from D^2 Phi_GR = norm_poisson_equation * delta (GR)
    // and D^2 phi_DGP = C delta where Phi = Phi_GR + phi_DGP
    // With screening we compute C as a function of delta
    //========================================================================
    void compute_force(double a,
                       double H0Box,
                       FFTWGrid<NDIM> & density_fourier,
                       std::string density_assignment_method_used,
                       std::array<FFTWGrid<NDIM>, NDIM> & force_real) const override {

        // Compute fifth-force
        const double norm_poisson_equation = 1.5 * this->cosmo->get_OmegaM() * a;
        auto coupling = [&]([[maybe_unused]] double kBox) { return GeffOverG(a, kBox / H0Box) - 1.0; };
        FFTWGrid<NDIM> density_fifth_force;

       if (use_screening_method) {

            // Approximate screening method
            const double OmegaM = this->cosmo->get_OmegaM();
            auto screening_function_HiCOLA = [=](double density_contrast) {
                double fac = get_chi_over_delta(a) * pow(a, 3.) * (density_contrast); //BILL: need to re-write the equation and read the spline/file for chi_over_delta
                return fac < 1e-5 ? 1.0 : 2.0 * (std::sqrt(1.0 + fac) - 1) / fac; //BILL: need to understand what this line does
            };
            std::cout << "At a= " << a << " chi/delta=" << get_chi_over_delta(a) << " coupling=" << coupling(a) << "\n";
            FML::NBODY::compute_delta_fifth_force_density_screening(density_fourier,
                                                                    density_fifth_force,
                                                                    coupling,
                                                                    screening_function_HiCOLA,
                                                                    smoothing_scale_over_boxsize,
                                                                    smoothing_filter);

            // Ensure that the large scales are behaving correctly
            // We set delta_fifth_force => A * (1-f) + B * f
            // i.e. we use the linear prediction on large sales and the
            // screened prediction on small scales
            if (screening_enforce_largescale_linear and screening_linear_scale_hmpc > 0.0) {

                if (FML::ThisTask == 0) {
                    std::cout << "Combining screened solution with linear solution\n";
                    std::cout << "We use linear solution for k < " << screening_linear_scale_hmpc << " h/Mpc\n";
                }

                // Make a spline of the low-pass filter we use
                const int npts = 1000;
                const double kBoxmin = M_PI;
                const double kBoxmax = M_PI * std::sqrt(NDIM) * density_fourier.get_nmesh();
                const double kcutBox = screening_linear_scale_hmpc / this->H0_hmpc * H0Box;
                std::vector<double> kBox(npts);
                std::vector<double> f(npts);
                for (int i = 0; i < npts; i++) {
                    kBox[i] = kBoxmin + (kBoxmax - kBoxmin) * i / double(npts - 1);
                    f[i] = std::exp(-0.5 * kBox[i] * kBox[i] / (kcutBox * kcutBox));
                }
                Spline f_spline(kBox, f, "Low-pass filter");

                // Combine the screened and the linear solution together
                auto Local_nx = density_fourier.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    [[maybe_unused]] std::array<double, NDIM> kvec;
                    double kmag;
                    for (auto && fourier_index : density_fourier.get_fourier_range(islice, islice + 1)) {
                        density_fourier.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag);
                        auto delta = density_fourier.get_fourier_from_index(fourier_index);
                        auto delta_fifth_force = density_fifth_force.get_fourier_from_index(fourier_index);
                        auto delta_fifth_force_linear = delta * FML::GRID::FloatType(GeffOverG(a, kmag / H0Box) - 1.0);
                        FML::GRID::FloatType filter = f_spline(kmag);
                        auto value = delta_fifth_force_linear * filter + delta_fifth_force * FML::GRID::FloatType(1.0 - filter);
                        density_fifth_force.set_fourier_from_index(fourier_index, value);
                    }
                }
            }

        } else {

            // No screening - linear equation
            FML::NBODY::compute_delta_fifth_force<NDIM>(density_fourier, density_fifth_force, coupling);
        }

        // Add on density corresponding to the gravitational force to get total force
        auto Local_nx = density_fourier.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (int islice = 0; islice < Local_nx; islice++) {
            for (auto && fourier_index : density_fourier.get_fourier_range(islice, islice + 1)) {
                auto delta = density_fourier.get_fourier_from_index(fourier_index);
                auto delta_fifth_force = density_fifth_force.get_fourier_from_index(fourier_index);
                density_fifth_force.set_fourier_from_index(fourier_index, delta + delta_fifth_force);
            }
        }

        // Compute total force
        FML::NBODY::compute_force_from_density_fourier<NDIM>(
            density_fifth_force, force_real, density_assignment_method_used, norm_poisson_equation);
    }

    //========================================================================
    // Read the parameters we need
    //========================================================================
    void read_parameters(ParameterMap & param) override {
        GravityModel<NDIM>::read_parameters(param);
        HiCOLA_preforce_filename = param.get<std::string>("HiCOLA_preforce_filename");
        use_screening_method = param.get<bool>("gravity_model_screening");
        if (use_screening_method) {
            smoothing_scale_over_boxsize = param.get<double>("gravity_model_HiCOLA_smoothing_scale_over_boxsize"); // BILL: need to understand whether we want to do any smoothing for HiCOLA
            smoothing_filter = param.get<std::string>("gravity_model_HiCOLA_smoothing_filter"); // BILL: need to understand whether we want to do any smoothing for HiCOLA
            screening_enforce_largescale_linear = param.get<bool>("gravity_model_screening_enforce_largescale_linear");
            screening_linear_scale_hmpc = param.get<double>("gravity_model_screening_linear_scale_hmpc");
        }
        this->scaledependent_growth = this->cosmo->get_OmegaMNu() > 0.0;
    }

  protected:
    //double rcH0_DGP; //BILL: may need to define coupling/screening factor variables here?

    std::string HiCOLA_preforce_filename; // The filename
    // For screening method
    bool use_screening_method{true};
    std::string smoothing_filter{"tophat"};
    double smoothing_scale_over_boxsize{0.0};
    bool screening_enforce_largescale_linear{false};
    double screening_linear_scale_hmpc{0.0};

    // For solving the exact equation //BILL: don't need for HiCOLA?
    bool solve_exact_equation{false};
    int multigrid_nsweeps_first_step{10};
    int multigrid_nsweeps{10};
    double multigrid_solver_residual_convergence{1e-7};

    Spline chi_over_delta_spline;//{"chi/delta(a)"};
    Spline coupling_spline;//{"coupling(a)"};
};

#endif
