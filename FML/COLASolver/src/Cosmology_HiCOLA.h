#ifndef COSMOLOGY_HiCOLA_HEADER
#define COSMOLOGY_HiCOLA_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>
#include <FML/FileUtils/FileUtils.h>

#include "Cosmology.h"

class CosmologyHiCOLA final : public Cosmology {
  public:
    CosmologyHiCOLA() { name = "HiCOLA"; }

    //========================================================================
    // Read the parameters we need
    //========================================================================
    void read_parameters(ParameterMap & param) override {
        Cosmology::read_parameters(param);
        HiCOLA_expansion_filename = param.get<std::string>("HiCOLA_expansion_filename");
    }

    //========================================================================
    // Initialize the cosmology
    //========================================================================
    void init() override { Cosmology::init(); }

    //========================================================================
    // Print some info
    //========================================================================
    void info() const override {
        Cosmology::info();
        if (FML::ThisTask == 0) {
            std::cout << "# HiCOLA_expansion_filename     : " << HiCOLA_expansion_filename << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }

    //==============================================================
    // Internal methods to get E(a) and dlogHdloga from file/splines
    //==============================================================

    void read_and_spline_expansion() {
      DVector HiCOLA_a_arr;
      DVector E_arr;
      DVector dlogHdloga_arr;

      // Fileformat: [a, chi/delta, coupling]
      const int ncols = 3;
      const int col_a = 0;
      const int col_E = 1;
      const int col_dlogHdloga = 2;
      std::vector<int> cols_to_keep{col_a, col_E, col_dlogHdloga};
      const int nheaderlines = 0;

      auto HiCOLAdata = FML::FILEUTILS::read_regular_ascii(HiCOLA_expansion_filename, ncols, cols_to_keep, nheaderlines);

      HiCOLA_a_arr.resize(HiCOLAdata.size());
      E_arr.resize(HiCOLAdata.size());
      dlogHdloga_arr.resize(HiCOLAdata.size());

      for (size_t i = 0; i < HiCOLA_a_arr.size(); i++) {
          HiCOLA_a_arr[i] = HiCOLAdata[i][col_a];
          E_arr[i] = HiCOLAdata[i][col_E];
          dlogHdloga_arr[i] = HiCOLAdata[i][col_dlogHdloga];
          //std::cout << "# HiCOLA input: " << i << " " << HiCOLA_a_arr[i] << " " << chi_over_delta_arr[i] << " " << coupling_arr[i] << "\n";
      }

      E_spline.create(HiCOLA_a_arr, E_arr, "E(a)");
      dlogHdloga_spline.create(HiCOLA_a_arr, dlogHdloga_arr, "dlogHdloga(a)");

    }

    double get_E(double a) const {
        return E_spline(a);
    }

    double get_dlogHdloga(double a) const {
        return dlogHdloga_spline(a);
    }

    //========================================================================
    // Hubble function -- now just read from splines
    //========================================================================
    double HoverH0_of_a(double a) const override {
        return get_E(a);
    }

    double dlogHdloga_of_a(double a) const override {
        return get_dlogHdloga(a);
    }

  protected:
    //========================================================================
    // Parameters specific to the HiCOLA model
    //========================================================================
    std::string HiCOLA_expansion_filename;
    Spline E_spline;
    Spline dlogHdloga_spline;
};
#endif
