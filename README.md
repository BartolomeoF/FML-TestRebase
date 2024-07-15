# Hi-COLA/Duncan

Please see 'README.md' in the Hi-COLA_public branch for general information on Hi-COLA.

This branch contains an additional executable python file and modifications to the existing file: 'generate_simulation_input.py' within the Hi-COLA Frontend. 

Modifications to 'generate_simulation_input.py' allow solutions to cosmological background properties to be solved forwards in time starting from high redshift. Additional quantities including the effective equation of state and dark energy equation of state are computed as well as the three property functions used to describe the evolution history of the cosmological background. Simple parameterisations for the property functions are evaluated and compared to the full solutions. A stability check based on the conditions that Q_s > 0 and c_s^2 > 0 can be performed, computed using the property functions.

The new executable file named 'model_checker.py' can sort models generated using random parameters into 'stable' or 'unstable' categories. The conditions that need to be met for solutions to be classified as stable include numerical continuity followed by: consistency with the Hubble parameter of LCDM within 20% and stability which is true if Q_s > 0 and c_s^2 > 0.

To run 'generate_simulation_input.py' the 'horndeski_parameters.ini' and 'numerical_parameters.ini' files are required and can be modified.
To run 'model_checker.py' the 'horndeski_generic.ini' and 'numerical_parameters.ini' files are required, and can be modified, as well as the number of models to be generated.