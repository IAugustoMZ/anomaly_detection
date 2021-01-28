"""
This file contains the functions used to simulate CSTR reactor operation
"""

# importing the necessary libraries

import numpy as np
import math as m
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------------------------------------------------------

# definition of constants and properties values
R = 8.314472                        # ideal gas constant (J/mol.K)
E = 15000                           # activation energy (J/mol)
A = 50                              # pre-exponential factor (m**6/h.mol**2)
V = 8.5                             # CSTR system volume (m³)
CpW = 75.2                          # water heat capacity (J/mol.K)
CpA = 1.2*CpW                       # solution A heat capacity (J/mol.K)
CpB = 1.95*CpW                      # solution B heat capacity (J/mol.K)
CpC = 1.07*CpW                      # outlet solution heat capacity (J/mol.K)
deltaHr = 60000                     # heat of reaction (J/mol)
eps = 1e-6                          # convergence tolerance
Cp_v = [CpA, CpW, CpB, CpC]         # list of heat capacities (J/mol)

# --------------------------------------------------------------------------------------------------------------

# input parameters (test)
Ca0 = 7                             # component A feed concentration (mol/m³)
Cb0 = 12.5                          # component B feed concentraion (mol/m³)
Ta = 30                             # solution A feed temperature (deg C)
Tb = 30                             # solution B feed temperature (deg C)
Tw = 10                             # water feed temperature (deg C)
v1 = 2.5                            # solution A flow rate (m³/h)
v2 = 7                              # water flow rate (m³/h)
v3 = 0.5                            # solution B flow rate (m³/h)
v_v = [v1, v2, v3]                  # list of inlet flow rates (m³/h)

# --------------------------------------------------------------------------------------------------------------

# calculated parameters
vt = v1 + v2 + v3                   # total outlet flow rate (m³/h)

# unit conversion
Ta = Ta + 273.15                    # conversion to K
Tb = Tb + 273.15                    # conversion to K
Tw = Tw + 273.15                    # conversion to K
T_v = [Ta, Tw, Tb]                  # list of inlet temperatures (K)

# --------------------------------------------------------------------------------------------------------------

# creation of response array with initial values
Ca = Ca0                                       # initial value for component A outlet concentration (mol/m³)
Cb = Cb0                                       # initial value for component B outlet concentration (mol/m³)
T = (v1*T_v[0]+v2*T_v[1]+v3*T_v[2])/vt         # initial value for outlet stream temperature (K)

resp = np.array([Ca, Cb, T])    # array of response variables

# --------------------------------------------------------------------------------------------------------------

# definitions of objective and auxiliary funcitons functions

def kinetic_rate(resp, act_ener = E, pre_exp = A, gas_const = R, volume = V):
    # this function calculates the rate of reaction of component A
    k1 = pre_exp*m.exp((-act_ener)/(gas_const*resp[2]))     # kinetic factor
    
    rate = k1*resp[0]*(resp[1]**2)*volume                   # rate of reaction (mol/h)
    
    return rate

rate = kinetic_rate(resp)
    
def F1(resp, vt = vt, v1 = v_v[0], Ca0 = Ca0):
    # this function represents the first objective function
    # which is the component A molar balance
    
    rate = kinetic_rate(resp)                   # kinetic rate of reaction of A
    
    return (vt*resp[0]) - (v1*Ca0) + rate       # molar balance of A

def F2(resp, vt = vt, v3 = v_v[2], Cb0 = Cb0):
    # this function represents the second objective function
    # which is the component B molar balance
    
    rate = kinetic_rate(resp)                       # kinetic rate of reaction of B
    
    return (vt*resp[1]) - (v3*Cb0) + (2*rate)       # molar balance of B

def F3(resp, v_v = v_v, Cp_v = Cp_v, T_v = T_v, deltaHr = deltaHr):
    # this function represents the third objective function
    # which is the energy balance
    
    rate = kinetic_rate(resp)
    
    vt = sum(v_v)
    
    en_out = vt*Cp_v[3]*resp[2]                     # energy output (J/h)
    en_A_in = v_v[0]*Cp_v[0]*T_v[0]                 # solution A energy input (J/h)
    en_W_in = v_v[1]*Cp_v[1]*T_v[1]                 # water energy input (J/h)
    en_B_in = v_v[2]*Cp_v[2]*T_v[2]                 # solution B energy input (J/h)
    en_reac = rate*deltaHr                          # reaction energy release (J/h)
    
    return (en_out)-(en_A_in + en_W_in + en_B_in + en_reac)

# F = np.array([F1(resp), F2(resp), F3(resp)])

# --------------------------------------------------------------------------------------------------------------

# definition of Jacobian function

def jacobian(eps, resp, Cp_v, v_v, T_v, Ca0, Cb0):
    # this function calculates the Jacobian matrix of the non-linear system
    
    epsA = [eps, 0, 0]                                                                                  # list for calculation of derivative in respect to Ca
    epsB = [0, eps, 0]                                                                                  # list for calculation of derivative in respect to Cb
    epsT = [0, 0, eps]                                                                                  # list for calculation of derivative in respect to T
    
    dF1dCa = (F1(resp+epsA, sum(v_v), v_v[0], Ca0)-F1(resp-epsA, sum(v_v), v_v[0], Ca0))/(2*eps)        # derivative of F1 in respect to Ca
    dF2dCa = (F2(resp+epsA, sum(v_v), v_v[2], Cb0)-F2(resp-epsA, sum(v_v), v_v[2], Cb0))/(2*eps)        # derivative of F2 in respect to Ca
    dF3dCa = (F3(resp+epsA, v_v, Cp_v, T_v)-F3(resp-epsA, v_v, Cp_v, T_v))/(2*eps)                      # derivative of F3 in respect to Ca
    
    dF1dCb = (F1(resp+epsB, sum(v_v), v_v[0], Ca0)-F1(resp-epsB, sum(v_v), v_v[0], Ca0))/(2*eps)        # derivative of F1 in respect to Cb
    dF2dCb = (F2(resp+epsB, sum(v_v), v_v[2], Cb0)-F2(resp-epsB, sum(v_v), v_v[2], Cb0))/(2*eps)        # derivative of F2 in respect to Cb
    dF3dCb = (F3(resp+epsB, v_v, Cp_v, T_v)-F3(resp-epsB, v_v, Cp_v, T_v))/(2*eps)                      # derivative of F3 in respect to Cb
    
    dF1dT = (F1(resp+epsT, sum(v_v), v_v[0], Ca0)-F1(resp-epsT, sum(v_v), v_v[0], Ca0))/(2*eps)         # derivative of F1 in respect to T
    dF2dT = (F2(resp+epsT, sum(v_v), v_v[2], Cb0)-F2(resp-epsT, sum(v_v), v_v[2], Cb0))/(2*eps)         # derivative of F2 in respect to T
    dF3dT = (F3(resp+epsT, v_v, Cp_v, T_v)-F3(resp-epsT, v_v, Cp_v, T_v))/(2*eps)                       # derivative of F3 in respect to T
    
    return np.array([[dF1dCa, dF1dCb, dF1dT], [dF2dCa, dF2dCb, dF2dT], [dF3dCa, dF3dCb, dF3dT]])

# J = jacobian(eps = eps)

# --------------------------------------------------------------------------------------------------------------

# implementation of extended Newton's method
    
def newton(resp, Cp_v, v_v, T_v, Ca0, Cb0):
    # this function implements the Newton's method for solving non linear algebraic system
    
    F = np.array([F1(resp, sum(v_v), v_v[0], Ca0), 
                  F2(resp, sum(v_v), v_v[2], Cb0), 
                  F3(resp, v_v, Cp_v, T_v)])                                                                 # first objective function
    J = jacobian(eps = eps, resp = resp, Cp_v = Cp_v, v_v = v_v, T_v = T_v, Ca0 = Ca0, Cb0 = Cb0)            # first Jacobian function
    
    S = 30                                                                                                   # stopping criteria
    i = 0                                                                                                    # iteration counter
    
    while (S >= eps):
        
        
        s = np.linalg.solve(J, -F)                                                                           # solve linear system Js = -F to obtain variable step vector
        resp = resp + s                                                                                      # update resp vector
        
        F = np.array([F1(resp, sum(v_v), v_v[0], Ca0), 
                  F2(resp, sum(v_v), v_v[2], Cb0), 
                  F3(resp, v_v, Cp_v, T_v)])                                                                 # new objective function
        J = jacobian(eps = eps, resp = resp, Cp_v = Cp_v, v_v = v_v, T_v = T_v, Ca0 = Ca0, Cb0 = Cb0)        # new Jacobian function
        
        S = sum(np.abs(F))                                                                                   # stopping criteria update
        i += 1                                                                                               # iteration counter increment
    
    return resp, S
        

# X = newton(resp)

# --------------------------------------------------------------------------------------------------------------

# dataset simulation

# definition of input variables
var = 0.05                                                      # input variables variance = 5 % of mean
n = 5000                                                           # simulation size
Ca0 = 7                                                         # component A feed concentration (mol/m³)
Cb0 = 12.5                                                      # component B feed concentraion (mol/m³)
Ta = 30                                                         # solution A feed temperature (deg C)
Tb = 30                                                         # solution B feed temperature (deg C)
Tw = 10                                                         # water feed temperature (deg C)
v1 = 2.5                                                        # solution A flow rate (m³/h)
v2 = 7                                                          # water flow rate (m³/h)
v3 = 0.5                                                        # solution B flow rate (m³/h)
Ca0 = Ca0 + np.random.normal(loc = 0, scale = var*Ca0, size=n)  # component A feed concentration (mol/m³)
Cb0 = Cb0 + np.random.normal(loc = 0, scale = var*Cb0, size=n)  # component B feed concentration (mol/m³)
Ta = Ta + np.random.normal(loc = 0, scale = var*Ta, size=n)     # solution A inlet temperature (K)
Tb = Tb + np.random.normal(loc = 0, scale = var*Tb, size=n)     # solution B inlet temperature (K)
Tw = Tw + np.random.normal(loc = 0, scale = var*Tw, size=n)     # water inlet temperature (K)
v1 = v1 + np.random.normal(loc = 0, scale = var*v1, size=n)     # solution A inlet flow rate (m³/h)
v2 = v2 + np.random.normal(loc = 0, scale = var*v2, size=n)     # water inlet flow rate (m³/h)
v3 = v3 + np.random.normal(loc = 0, scale = var*v3, size=n)    # solution B inlet flow rate (m³/h)
Ta = Ta +273.15
Tb = Tb +273.15
Tw = Tw +273.15

# test
# plt.plot(v1)
# plt.plot(v2)
# plt.plot(v3)
# plt.show()

# input vectors
v_v = [v1, v2, v3]
T_v = [Ta, Tw, Tb]

# initial values for output variables
Ca = Ca0                                                   # initial value for component A outlet concentration (mol/m³)
Cb = Cb0                                                   # initial value for component B outlet concentration (mol/m³)
T = (v_v[0]*T_v[0]+v_v[1]*T_v[1]+v_v[2]*T_v[2])/sum(v_v)   # initial value for outlet stream temperature (K)

resp = np.transpose(np.array([Ca, Cb, T]))                 # array of response variables

Ca_k = []                                                  # outlet concentration of component A (mol/m³)
Cb_k = []                                                  # outlet concentration of component B (mol/m³)
Cc_k = []                                                  # outlet concentration of component C (mol/m³)
Tout_k = []                                                # outlet temperature (K)
conv_k = []

for i in range(n):
    
    v1_i = v_v[0][i]
    v2_i = v_v[1][i]
    v3_i = v_v[2][i]
    
    v_k = [v1_i, v2_i, v3_i]                                                                  # inlet flow rates vector
    
    T1_i = T_v[0][i]
    T2_i = T_v[1][i]
    T3_i = T_v[2][i]
    
    T_k = [T1_i, T2_i, T3_i]                                                                  # inlet tempertures vector

    X, conv = newton(resp[i], Cp_v = Cp_v, v_v = v_k, T_v = T_k, Ca0 = Ca0[i], Cb0 = Cb0[i])  # solving non linear system
    # print(X)
    Ca_k.append(X[0])                                                                         # extraction of oulet concentration of A (mol/m³)
    Cb_k.append(X[1])                                                                         # extraction of outlet concentration of B (mol/m³)
    Tout_k.append(X[2]-273.15)                                                                       # extraction of outlet temperature (K)
    conv_k.append(conv)

    rate = kinetic_rate(X)                                                                    # rate of reaction of A (mol/h)

    Cc_k.append(2*rate/sum(v_k))                                                              # extraction of outlet concentration of C (mol/m³)


# test
plt.figure()
plt.plot(Ca_k)
plt.plot(Cb_k)
plt.plot(Cc_k)
plt.ylim(0,3)
text = ['Concentration of A', 'Concentration of B', 'Concentration of C']
plt.xlabel('# Run')
plt.ylabel('Concentration at Outlet (mol/m³)')
plt.legend(text)           
plt.show()

plt.figure()
plt.plot(Tout_k)
plt.xlabel('# Run')
plt.ylabel('Temperature at Outlet (deg C)')        
plt.show()     
   
plt.figure()
plt.plot(conv_k) 
plt.xlabel('# Run')
plt.ylabel('Newton Method Convergence')
plt.show()

# --------------------------------------------------------------------------------------------------------------

# csv file creation
data = pd.DataFrame({'solutionA_inletFlowRate': v1,
                     'water_inletFlowRate': v2,
                     'solutionB_inletFlowRate': v3,
                     'componentA_inletConcentration': Ca0,
                     'componentB_inletConcentration': Cb0,
                     'solutionA_inletTemperature': Ta-273.15,
                     'water_inletTemperature': Tw-273.15,
                     'solutionB_inletTemperature': Tb-273.15,
                     'componentA_outletConcentration': Ca_k,
                     'componentB_outletConcentration': Cb_k,
                     'componentC_outletConcentration': Cc_k,
                     'outletTemperature': Tout_k})

data.to_csv('process_data.csv', index = False)        