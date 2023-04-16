import numpy as np

alpha = np.array([4.04, 6.7, 4.39, 17.33, 0])*1e3 #s^(-1)
beta = np.array([0, 2.88, 6.39, 8.16, 1.84])*1e3 #s^(-1)
v_ref = np.array([0, 49.14, 42.08, 55.31, 26.55, 0])*1e-3 #V
conductance = 2.7 #pS
n_ch = 22


def rateForw(alpha, v_ref, v, i):
    if (i >= 0 and i < 4):
        return alpha[i]*np.exp(v/v_ref[i+1])
    return 0

def rateBackw(beta, v_ref, v, i):
    if (i > 0 and i < 5):
        return beta[i]*np.exp(-v/v_ref[i])
    return 0

def setTransition(mat, v):
    for idx, a_ij in np.ndenumerate(mat):
        i, j = idx
        if (i == j):
            mat[idx] = -rateForw(alpha, v_ref, v, i) - rateBackw(beta, v_ref, v, i)
        elif (j == i-1):
            mat[idx] = rateForw(alpha, v_ref, v, j)
        elif (j == i+1):
            mat[idx] = rateBackw(beta, v_ref, v, j)
    return mat

def updateState(previous, v, time_step):
    previous = previous.reshape((5, 1))
    transition_matrix = np.zeros((5, 5))
    delta_S = setTransition(transition_matrix, v)@previous
    new_val = previous + delta_S*time_step
    return new_val.reshape((5,))

def influx(open_prob, v, noise, time_step):
    Ca = Ca_in(v)
    par_fitting = 0.62
    if (noise["active"]):
        if (np.random.rand() < 0.5):
            xi = -np.abs(np.random.normal(0, noise["an"]*(open_prob*Ca)**0.5))
        else:
            xi = np.abs(np.random.logistic(0, (3**0.5)/np.pi*noise["ap"]*(open_prob*Ca)**0.5))
    else:
        xi = 0
    return par_fitting*(n_ch*open_prob*Ca*time_step + xi)

def Ca_in(v):
    base_voltage = 55e-3 #V
    z = 2
    e = 1.602e-19 #C
    Ca_molecules = conductance*(base_voltage - v)/(z*e)*1e-12 #pS*V=pA=10^(-12)A
    return Ca_molecules

def CalciumMoleculesToConcentration(molecules):
    nA = 6.022e23 
    Vbtn = 0.13e-15 #L
    return molecules/nA/Vbtn
