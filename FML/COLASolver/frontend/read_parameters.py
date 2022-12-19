from configobj import ConfigObj
import sympy as sym
import numpy as np
import expression_builder as eb
from support import generate_scan_array as gsa
from support import make_scan_array as msa

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

    f_phi = numerical_read.as_float("f_phi")
    Omega_l0 = (1. - f_phi)*Omega_DE0

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
    odeint_declaration = horndeski_read.as_bool('use_constraint_eq_on_odeint_variables')
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

def read_in_scan_parameters(horndeski_scanning_path, numerical_scanning_path):
    read_out = {}
    horndeski_read = ConfigObj(horndeski_scanning_path)
    numerical_read = ConfigObj(numerical_scanning_path)

    h = numerical_read.as_float("h")

    Npoints = numerical_read.as_int('Npoints')
    max_redshift = numerical_read.as_float('max_redshift')

    suppression_flag = numerical_read.as_bool('suppression_flag')

    GR_flag = numerical_read.as_bool('GR_flag')

    threshold_value = numerical_read.as_float('threshold')

    simulation_parameters = [Npoints, max_redshift, suppression_flag, threshold_value, GR_flag]

    Omega_m0_number = numerical_read.as_int("Omega_m0_number")
    Omega_r0_number = numerical_read.as_int("Omega_r0_number")

    if numerical_read["Omega_m0_min"] == "None":
        Omega_b0h2_min = numerical_read.as_float("Omega_b0h2_min")
        Omega_b0h2_max = numerical_read.as_float("Omega_b0h2_max")
        # Omega_b0h2_array = msa(Omega_b0h2_min, Omega_b0h2_max, Omega_m0_number)
        Omega_b0h2_array = msa(Omega_b0h2_min, Omega_b0h2_max, Omega_m0_number)


        Omega_c0h2_min = numerical_read.as_float("Omega_c0h2_min")
        Omega_c0h2_max = numerical_read.as_float("Omega_c0h2_max")
        # Omega_c0h2_array = msa(Omega_c0h2_min, Omega_c0h2_max, Omega_m0_number)
        Omega_c0h2_array = msa(Omega_c0h2_min, Omega_c0h2_max, Omega_m0_number)

        Omega_m0_array = Omega_b0h2_array/h/h + Omega_c0h2_array/h/h
    else:
        Omega_m0_min = numerical_read.as_float("Omega_m0_min")
        Omega_m0_max = numerical_read.as_float("Omega_m0_max")
        Omega_m0_array = msa(Omega_m0_min, Omega_m0_max, Omega_m0_number)

    if numerical_read["Omega_r0_min"] == "None":
        Omega_r0h2_min = numerical_read.as_float("Omega_r0h2_min")
        Omega_r0h2_max = numerical_read.as_float("Omega_r0h2_max")
        # Omega_r0h2_array = msa(Omega_r0h2_min, Omega_r0h2_max, Omega_m0_number)
        Omega_r0h2_array = msa(Omega_r0h2_min, Omega_r0h2_max, Omega_r0_number)

        Omega_r0_array = Omega_r0h2_array/h/h
    else:
        # Omega_r0_min = numerical_read.as_float("Omega_r0_min")
        # Omega_r0_max = numerical_read.as_float("Omega_r0_max")
        # Omega_r0_array = msa(Omega_r0_min, Omega_r0_max, Omega_r0_number)
        Omega_r0_array = gsa(numerical_read, "Omega_r0")

    Omega_DE0_array = 1. - Omega_m0_array - Omega_r0_array

    # f_phi_min = numerical_read.as_float("f_phi_min")
    # f_phi_max = numerical_read.as_float("f_phi_max")
    # f_phi_number = numerical_read.as_int("f_phi_number")
    # f_phi_array = msa(f_phi_min, f_phi_max, f_phi_number)
    f_phi_array = gsa(numerical_read,"f_phi")



    if numerical_read.as_bool("read_odeint_ICs_from_file") is True:
        path_to_odeint_ICs = numerical_read['path_to_odeint_ICs']
        odeint_ICs = dict( np.load(path_to_odeint_ICs) )
        Hubble0_array, phiprime0_array, f_phi_array = odeint_ICs.values()
    else:
        Hubble0_array = gsa(numerical_read,"Hubble0")
        phiprime0_array = gsa(numerical_read, "phiprime0")

    Omega_l0_array = (1. - f_phi_array)*Omega_DE0_array

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

    parameter_arrays = []
    if horndeski_read.as_bool('read_parameter_values_from_file') is True:
        path_to_parameter_values = horndeski_read['path_to_parameter_values']
        parameter_dict = dict( np.load(path_to_parameter_values) )
        for i in parameter_dict.values():
            parameter_arrays.append(i)
    else:
        for key in horndeski_read:
            check = '_parameter'
            if check in key:
                print(horndeski_read.as_list(key))
                [minv, maxv, numv] = [eval(i) for i in horndeski_read.as_list(key)]
                if minv == maxv:
                    numv = 1
                parameter_array = np.linspace(minv,maxv,numv,endpoint=True)
                parameter_arrays.append(parameter_array)

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



    cosmological_parameter_arrays = [Omega_r0_array, Omega_m0_array, Omega_l0_array]
    initial_condition_arrays = [Hubble0_array, phiprime0_array]

    closure_guess = horndeski_read.as_float('closure_guess_value')
    odeint_declaration = horndeski_read.as_bool('use_constraint_eq_on_odeint_variables')
    if odeint_declaration:
        closure_decl_string = 'odeint_parameters'
        closure_decl_int = horndeski_read.as_int('which_odeint_par')

        initial_condition_arrays[closure_decl_int] = [closure_guess]#*np.ones(size)
    else:
        closure_decl_string = 'parameters'
        closure_decl_int = horndeski_read.as_int('which_Horndeski_par')

        parameter_arrays[closure_decl_int] = [closure_guess]#*np.ones(size)
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

    parameters_dict = {'cosmological_parameter_arrays':cosmological_parameter_arrays, 'Horndeski_parameter_arrays':parameter_arrays,'initial_condition_arrays':initial_condition_arrays}
    read_out.update(parameters_dict)

    red_switch = numerical_read.as_bool("red_switch")
    blue_switch = numerical_read.as_bool("blue_switch")
    yellow_switch = numerical_read.as_bool("yellow_switch")

    Omega_m_crit = numerical_read.as_float("Omega_m_crit")

    early_DE_threshold = numerical_read.as_float("early_DE_threshold")
    tolerance = numerical_read.as_float("tolerance")
    scanning_dict = {'red_switch':red_switch, 'blue_switch':blue_switch, 'yellow_switch':yellow_switch, 'early_DE_threshold':early_DE_threshold, 'tolerance':tolerance, 'Omega_m_crit':Omega_m_crit}
    read_out.update(scanning_dict)

    return read_out

def read_in_scan_settings(path_to_settings):
    numerical_read = ConfigObj(path_to_settings)
    Npoints = numerical_read.as_int('Npoints')
    max_redshift = numerical_read.as_float('max_redshift')

    suppression_flag = numerical_read.as_bool('suppression_flag')

    GR_flag = numerical_read.as_bool('GR_flag')

    threshold_value = numerical_read.as_float('threshold')

    model_name = numerical_read['model_name']
    cosmo_name = numerical_read['cosmo_name']
    simulation_parameters = [Npoints, max_redshift, suppression_flag, threshold_value, GR_flag]

    red_switch = numerical_read.as_bool("red_switch")
    blue_switch = numerical_read.as_bool("blue_switch")
    yellow_switch = numerical_read.as_bool("yellow_switch")
    tolerance = numerical_read.as_float("tolerance")
    Omega_m_crit= numerical_read.as_float("Omega_m_crit")
    early_DE_threshold = numerical_read.as_float("early_DE_threshold")

    read_out_dict = {'simulation_parameters':simulation_parameters, 'model_name':model_name, 'cosmo_name':cosmo_name}

    scanning_dict = {'red_switch':red_switch, 'blue_switch':blue_switch, 'yellow_switch':yellow_switch, 'early_DE_threshold':early_DE_threshold, 'tolerance':tolerance, 'Omega_m_crit':Omega_m_crit}
    read_out_dict.update(scanning_dict)

    if numerical_read.as_bool("set_all_to_one") is True:
        M_pG4v, M_KG4v, M_G3sv, M_sG4v, M_G3G4v, M_Ksv, M_gpv = 1., 1., 1., 1., 1., 1., 1.
    else:
        mass_ratios = []
        for key in numerical_read:
            if key[:2] == 'M_':
                print(key)
                mass_ratios.append(key)
        print(mass_ratios)
        [M_pG4v, M_KG4v, M_G3sv, M_sG4v, M_G3G4v, M_Ksv, M_gpv] = [numerical_read[i] for i in mass_ratios]
    mass_ratio_list = [M_pG4v, M_KG4v, M_G3sv, M_sG4v, M_G3G4v, M_Ksv, M_gpv]

    symbol_list = []
    for key in numerical_read:
        check = '_symbol'
        if key[-len(check):] == check:
            symbol_list.append(numerical_read[key])
    symbol_list = [sym.sympify(j) for j in symbol_list]

    K = sym.sympify(numerical_read['K'])
    K = K.subs( [ ('M_pG4', M_pG4v), ('M_KG4',M_KG4v), ('M_G3s', M_G3sv), ('M_sG4',M_sG4v),('M_G3G4', M_G3G4v), ('M_Ks', M_Ksv)  ] )

    G3 = sym.sympify(numerical_read['G3'])
    G3 = G3.subs( [ ('M_pG4', M_pG4v), ('M_KG4',M_KG4v), ('M_G3s', M_G3sv), ('M_sG4',M_sG4v), ('M_G3G4', M_G3G4v), ('M_Ks', M_Ksv)  ] )

    G4 = sym.sympify(numerical_read['G4'])
    G4 = G4.subs( [ ('M_pG4', M_pG4v), ('M_KG4',M_KG4v), ('M_G3s', M_G3sv), ('M_sG4',M_sG4v), ('M_G3G4', M_G3G4v), ('M_Ks', M_Ksv)  ] )

    Horndeski_functions = { 'K':K, 'G3':G3, 'G4':G4}
    read_out_dict.update(Horndeski_functions)

    closure_guess = numerical_read.as_float('closure_guess_value')
    odeint_declaration = numerical_read.as_bool('use_constraint_eq_on_odeint_variables')
    if odeint_declaration:
        closure_decl_string = 'odeint_parameters'
        closure_decl_int = numerical_read.as_int('which_odeint_par')
    else:
        closure_decl_string = 'parameters'
        closure_decl_int = numerical_read.as_int('which_Horndeski_par')


    closure_declaration = [closure_decl_string,closure_decl_int]


    lists = {'mass_ratio_list':mass_ratio_list, 'symbol_list':symbol_list, 'closure_declaration':closure_declaration}
    read_out_dict.update(lists)

    return read_out_dict
#####
# float_vars = ['a','b1','Om','Ol','Or']
# integer_vars = ['integer constant']

# params = params_read.copy()
# for key in float_vars:
#     params[key] = params_read.as_float(key)
# for key in integer_vars:
#     params[key] = params_read.as_int(key)
