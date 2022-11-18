from configobj import ConfigObj
import sympy as sym
import numpy as np
import expression_builder as eb


def read_in_parameters(horndeski_path, numerical_path):
    read_out = {}
    horndeski_read = ConfigObj(horndeski_path)
    numerical_read = ConfigObj(numerical_path)
    
    h = numerical_read.as_float("h")
    
    Npoints = numerical_read.as_int('Npoints')
    max_redshift = numerical_read.as_float('max_redshift')
    
    suppression_flag = numerical_read.as_bool('suppression_flag')
    
    GR_flag = numerical_read.as_bool('GR_flag')
    
    threshold_value = numerical_read.as_float('threshold')
    
    simulation_parameters = [Npoints, max_redshift, suppression_flag, threshold_value, GR_flag]
    
    if numerical_read["Omega_m0"] == "None":
        Omega_b0h2 = numerical_read.as_float("Omega_b0h2")
        Omega_c0h2 = numerical_read.as_float("Omega_c0h2")
        
        
        Omega_m0 = Omega_b0h2/h/h + Omega_c0h2/h/h
    else:
        Omega_m0 = numerical_read.as_float("Omega_r0")
    
    if numerical_read["Omega_r0"] == "None":
        Omega_r0h2 = numerical_read.as_float("Omega_r0h2")
        
        Omega_r0 = Omega_r0h2/h/h
    else:
        Omega_r0 = numerical_read.as_float("Omega_r0")
    
    Omega_DE0 = 1. - Omega_m0 - Omega_r0
    
    f = numerical_read.as_float("f")
    Omega_l0 = (1. - f)*Omega_DE0
    
    Hubble0 = numerical_read.as_float("Hubble0")
    phiprime0 = numerical_read.as_float("phiprime0")
    
    if horndeski_read.as_bool("set_all_to_one") is True:
        M_pG4v, M_KG4v, M_G3sv, M_sG4v, M_G3G4v, M_Ksv, M_gpv = 1., 1., 1., 1., 1., 1., 1.
    else:
        mass_ratios = []
        for key in horndeski_read:
            if key[:2] == 'M_':
                print(key)
                mass_ratios.append(key)
        print(mass_ratios)
        [M_pG4v, M_KG4v, M_G3sv, M_sG4v, M_G3G4v, M_Ksv, M_gpv] = [horndeski_read[i] for i in mass_ratios]
    mass_ratio_list = [M_pG4v, M_KG4v, M_G3sv, M_sG4v, M_G3G4v, M_Ksv, M_gpv]
        
    parameters = []
    for key in horndeski_read:
        check = '_parameter'
        if key[-len(check):] == check:
            parameters.append(horndeski_read.as_float(key))
            
    symbol_list = []
    for key in horndeski_read:
        check = '_symbol'
        if key[-len(check):] == check:
            symbol_list.append(horndeski_read[key])
    symbol_list = [sym.sympify(j) for j in symbol_list]        
    
    K = sym.sympify(horndeski_read['K'])
    K = K.subs( [ ('M_pG4', M_pG4v), ('M_KG4',M_KG4v), ('M_G3s', M_G3sv), ('M_sG4',M_sG4v),('M_G3G4', M_G3G4v), ('M_Ks', M_Ksv)  ] )
    
    G3 = sym.sympify(horndeski_read['G3'])
    G3 = G3.subs( [ ('M_pG4', M_pG4v), ('M_KG4',M_KG4v), ('M_G3s', M_G3sv), ('M_sG4',M_sG4v), ('M_G3G4', M_G3G4v), ('M_Ks', M_Ksv)  ] )
    
    G4 = sym.sympify(horndeski_read['G4'])
    G4 = G4.subs( [ ('M_pG4', M_pG4v), ('M_KG4',M_KG4v), ('M_G3s', M_G3sv), ('M_sG4',M_sG4v), ('M_G3G4', M_G3G4v), ('M_Ks', M_Ksv)  ] )
    
    
    
    cosmological_parameters = [Omega_r0, Omega_m0, Omega_l0]
    initial_conditions = [Hubble0, phiprime0]
    
    closure_guess = horndeski_read.as_float('closure_guess_value')
    odeint_declaration = horndeski_read.as_bool('use_constraint_eq_on_odeint_parameters')
    if odeint_declaration:
        closure_decl_string = 'odeint_parameters'
        closure_decl_int = horndeski_read.as_int('which_odeint_par')
        initial_conditions[closure_decl_int] = closure_guess
    else:
        closure_decl_string = 'parameters'
        closure_decl_int = horndeski_read.as_int('which_Horndeski_par')
        parameters[closure_decl_int] = closure_guess
    closure_declaration = [closure_decl_string,closure_decl_int]
    
    model_name = horndeski_read['model_name']
    cosmo_name = numerical_read['cosmo_name']
    names = {'model_name':model_name, 'cosmo_name':cosmo_name}
    read_out.update(names)
    
    Horndeski_functions = { 'K':K, 'G3':G3, 'G4':G4}
    read_out.update(Horndeski_functions)
    
    lists = {'mass_ratio_list':mass_ratio_list, 'symbol_list':symbol_list, 'closure_declaration':closure_declaration}
    read_out.update(lists)
    
    simulation_dict = {'simulation_parameters':simulation_parameters, 'threshold_value':threshold_value,'GR_flag':GR_flag}
    read_out.update(simulation_dict)
    
    parameters_dict = {'cosmological_parameters':cosmological_parameters, 'Horndeski_parameters':parameters,'initial_conditions':initial_conditions}
    read_out.update(parameters_dict)
    
    return read_out
#####
# float_vars = ['a','b1','Om','Ol','Or']
# integer_vars = ['integer constant']

# params = params_read.copy()
# for key in float_vars:
#     params[key] = params_read.as_float(key)
# for key in integer_vars:
#     params[key] = params_read.as_int(key)