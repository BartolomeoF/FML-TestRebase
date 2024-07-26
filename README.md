# Hi-COLA/Duncan

Please see 'README.md' in the Hi-COLA_public branch for general information on Hi-COLA.

This branch contains additional executable python files and modifications to the existing file: 'generate_simulation_input.py' within the Hi-COLA Frontend. 

Modifications to 'generate_simulation_input.py' allow solutions to cosmological background properties to be solved forwards in time starting from high redshift. Additional quantities including the effective equation of state and dark energy equation of state are computed as well as the three property functions used to describe the evolution history of the cosmological background. Simple parameterisations for the property functions are evaluated and compared to the full solutions. A stability check based on the conditions that Q_s > 0 and c_s^2 > 0 can be performed, computed using the property functions.

The first new executable file named 'model_checker.py' can sort models generated using random parameters into 'stable' or 'unstable' categories. The conditions that need to be met for solutions to be classified as stable include numerical continuity followed by: consistency with the Hubble parameter of LCDM within 20% and stability which is true if Q_s > 0 and c_s^2 > 0.

This is superseded by the second new executable file named 'model_sampler.py' which is able to explore the parameter space more efficiently using MCMC and can produce a higher proportion of stable models with smaller deviations from LCDM.

To run 'generate_simulation_input.py' the 'horndeski_parameters.ini' and 'numerical_parameters.ini' files are required and can be modified.
To run 'model_checker.py' the 'horndeski_generic.ini' and 'numerical_parameters.ini' files are required, and can be modified, as well as the number of models to be generated.
To run 'model_sampler.py' the 'horndeski_generic.ini' and 'numerical_parameters.ini' files are required as well as the number of walkers and iterations to be used.

The new functions added to modules called by 'generate_simulation_input.py' are as follows:
    In 'expression_builder.py':
        M_star_sqrd(),      alpha_M(),          alpha_B(),  alpha_K(),
        rhode() (reworked), Pde() (reworked),   Q_s(),      c_s_sq().
    In 'numerical_solver.py':
        make_timeout(),     comp_LCDM(),        run_solver_lite(),  comp_alphas(),
        comp_w_phi(),       comp_w_phi2(),      comp_w_eff(),       comp_stability(),
        alpha_X1(),         alpha_X2(),         alpha_X3(),         r_chi2(),
        parameterise1(),    parameterise2(),    parameterise3().

