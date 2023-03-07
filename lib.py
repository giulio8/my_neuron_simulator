import numpy as np

def sigmoidalNonLinearity(v):
    a = 1/0.003 #1/0.029
    v12 = 0.036 #0.036
    lambdaMax = 36.03#36.03 #Hz
    return lambdaMax/(1+np.exp(-a*(v-v12)))

def poisson(rate, time_step):
    p = 1 - np.exp(-rate*time_step)
    return p

def entropy(probability):
    probability = np.array(probability)
    return -np.sum(probability*np.log2(probability, out=np.zeros_like(probability), where=(probability!=0)))

def mutualInformation2(spike_probability, release_prob_during_AP, release_prob_no_AP):
    v_prob = spike_probability*release_prob_during_AP + (1 - spike_probability)*release_prob_no_AP  # P(V[n]=1)  Total probability theorem
    entropy_v = entropy([v_prob, 1-v_prob]) if v_prob > 0 else 0  # H(V[n])
    joint_prob = np.array([spike_probability*release_prob_during_AP, # P(S[n]=1, V[n]=1)
                    spike_probability*(1 - release_prob_during_AP), # P(S[n]=1, V[n]=0)
                    (1 - spike_probability)*release_prob_no_AP, # P(S[n]=0, V[n]=1)
                    (1 - spike_probability)*(1 - release_prob_no_AP)]) # P(S[n]=0, V[n]=0)
    joint_entropy = entropy(joint_prob[joint_prob > 0])# if np.all(joint_prob > 0) else 0 # H(V[n], S[n])
    entropy_s = entropy([spike_probability, 1-spike_probability])   #  H(S[n])
    conditional_entropy = joint_entropy - entropy_s    #  H(V[n] | S[n]) = H(V[n], S[n]) - H(S[n])

    return entropy_v - conditional_entropy    #  I(S[n] ; V[n]) = H(V[n]) - H(V[n] | S[n])

def mutualInformation(spike_probability, release_prob_during_AP, release_prob_no_AP):
    v_prob = spike_probability*release_prob_during_AP + (1 - spike_probability)*release_prob_no_AP  # P(V[n]=1)  Total probability theorem
    entropy_v = entropy([v_prob, 1-v_prob]) if v_prob > 1e-7 and v_prob < 1 else 0  # H(V[n])
    conditional_prob_S1 = np.array([
                    release_prob_during_AP, # P(V[n]=1 | S[n]=1)
                    1 - release_prob_during_AP # P(V[n]=0 | S[n]=1)
                    ]) 
    conditional_prob_S0 = np.array([
                    release_prob_no_AP, # P(V[n]=1 | S[n]=0)
                    1 - release_prob_no_AP # P(V[n]=0 | S[n]=0)
                    ])
    if (np.any(conditional_prob_S0<1e-7)==False):
        conditional_entropy = entropy(conditional_prob_S0)*(1-spike_probability)   #  H(V[n] | S[n])
    else:
        conditional_entropy = 0
    if (np.any(conditional_prob_S1<1e-7)==False):
        conditional_entropy += entropy(conditional_prob_S1)*spike_probability

    return entropy_v - conditional_entropy    #  I(S[n] ; V[n]) = H(V[n]) - H(V[n] | S[n])

def updateVoltage(previous, current, C, Urest, membrane_tau, time_step, thermal_noise, temperature, R):
    delta_v = current/C - (previous - Urest)/membrane_tau
    if (thermal_noise == True):
        delta_v += thermalNoise(R, temperature, 1/time_step)
    return time_step*delta_v + previous

def thermalNoise(R, T, bandwidth):
    kB = 1.38e-23
    return np.random.normal(0, (4*kB*R*T*bandwidth)**0.5)

def updateSitesProbabilities(probabilities, Ca_pre, time_step):
    k_plus = np.array([3.75e-3, 2.5e-3, 5e-4, 7.5e-3]) # (uM*ms)^(-1)
    k_minus = np.array([4e-4, 1e-3, 0.1, 10])  # ms^(-1)
    taus = 1/(k_plus*Ca_pre + k_minus)
    O = time_step*1000*(k_plus*Ca_pre - probabilities/taus) + probabilities

    return O

def sitesProbabilities(index, Ca_pre, time_step):
    k_plus = np.array([3.75e-3, 2.5e-3, 5e-4, 7.5e-3]) # (uM*ms)^(-1)
    k_minus = np.array([4e-4, 1e-3, 0.1, 10])  # ms^(-1)
    taus = 1/(k_plus*Ca_pre + k_minus)
    O = taus*k_plus*Ca_pre*(1 - np.exp(-index*time_step*1000/taus))

    return O


def vesicleReplenishment(previous, replenishment_time=0, time_step=0.02e-3):
    if (replenishment_time == 0):
        return previous + 1

    rand = np.random.random()
    prob = poisson(1/replenishment_time, time_step)
    if (prob > rand):
        return previous + 1
    else:
        return previous

def spontaneousRate(Ca_pre):
    a = np.array([7181, 606, 100e3]) # uM, uM, s^(-1)

    return a[2]/(1 + np.exp((a[0] - Ca_pre)/a[1]))

def updateStoredCa(previous, Ca_Astro, time_step):
    gamma = 0.02 #s^(-1)
    alpha = 101 #s^(-1)
    Ca_thresh = 196.4e-3 #uM

    delta_Ca = -gamma*previous
    if (Ca_Astro > Ca_thresh):
        delta_Ca += alpha*Ca_Astro
    return time_step*delta_Ca + previous



def J_channel(inhibition_parameter, Ca, CaER, IP3):
    c1 = 0.185
    v1 = 6 #s^(-1)
    m_infty = IP3/(IP3 + 0.13)
    n_infty = Ca/(Ca + 0.08234)
    return c1*v1*((m_infty*n_infty*inhibition_parameter)**3)*(Ca-CaER)

def J_leak(Ca, CaER):
    c1 = 0.185
    v2 = 0.11 #s^(-1)
    return c1*v2*(Ca - CaER)

def J_pump(Ca):
    v3 = 0.9 #uM/s
    k3 = 0.1 #uM
    return v3*(Ca**2)/((k3**2) + (Ca**2))

def Ca_ER(Ca):
    c0 = 2.0 #uM
    c1 = 0.19 #volume ratio
    return (c0 - Ca)/c1

def updateInhibitionParameter (previous, IP3, Ca_Astro, time_step):
    N_IP3Rs = 20 #IP3Rs cluster size (Swillens et al., 1999)
    a2 = 0.2 #(uM s)^(-1)
    d2 = 1.049 #uM
    d1 = 0.13 #uM
    d3 = 0.9434 #uM
    alpha = a2*d2*(IP3 + d1)/(IP3 + d3)
    beta = a2*Ca_Astro
    delta_q = alpha*(1-previous)-beta*previous
    strength = alpha*(1-previous)+beta*previous
    delta_q += np.random.normal(0, (strength/N_IP3Rs)**0.5) # Langevin approach

    return delta_q*time_step + previous


"""def updateAstro (Ca_Astro, IP3, inhibition_parameter, glu, time_step, glu_active=False):
    new_inhibition_parameter = updateInhibitionParameter(inhibition_parameter, IP3, Ca_Astro, time_step)
    new_IP3 = updateIP3(IP3, Ca_Astro, glu, time_step, glu_active)
    new_Ca_Astro = updateCaAstro(Ca_Astro, inhibition_parameter, IP3, time_step)

    return new_IP3, new_Ca_Astro, new_inhibition_parameter"""

def updateCaAstro (previous, inhibition_parameter, IP3, time_step):
    CaER = Ca_ER(previous)
    delta_Ca = -J_channel(inhibition_parameter, previous, CaER, IP3) - J_pump(previous) - J_leak(previous, CaER)

    return delta_Ca*time_step + previous

def updateIP3 (previous, Ca_Astro, glu, time_step, glu_active=False):
    one_over_tau = 0.140 #s^(-1)
    equilibrium_conc = 160e-3 #uM
    vIP3 = 0.13 #uM/s
    kIP3 = 1.1 #uM
    alpha = 0.8
    v_glu = 0.062 #uM/s
    k_glu = 785e-3 #uM
    n_glu = 0.45 #cooperativity

    delta_IP3 = one_over_tau*(-previous+equilibrium_conc) + vIP3*(Ca_Astro + (1-alpha)*kIP3)/(kIP3+Ca_Astro)
    if (glu_active == True):
        delta_IP3 += v_glu*(glu**n_glu)/(k_glu**n_glu + glu**n_glu)

    return delta_IP3*time_step + previous


def updateProbSpikeActive(previous, p_init, p_init_prev):
    return 1 - ((1 - previous)*(1 - p_init)/(1 - p_init_prev))

def updateProbSpikeReady(previous, p_init, p_init_prev):
    return previous*(1 - p_init)/(1 - p_init_prev)