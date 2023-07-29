import numpy as np
import matplotlib.pyplot as plt
import lib
import pickle
import copy
from AP_VGCC_dynamics import AP, VGCC
import importlib
import axon

units = {
        "membrane_tau": "s",
        "spike_duration": "s",
        "t_ref_s": "s",
        "t_ref_v": "s",
        "C": "uF",
        "R": "ohm",
        "threshold": "V",
        "Urest": "V",
        "hyperp_v": "V",
        "hyperp_tau": "s",
        "vesicles_replenishment_time": "s",
        "glutamate_clearance_time": "s",
        "time_step": "s",
        "simulation_time": "s",
        "temperature": "K",
        "lambda_fixed": "",
        "current": "A",
        "f_current": "Hz",
        "lambda_": "Hz"
    }

class Properties:
    def __init__(self, membrane_tau, ap: AP.Spike, t_ref_s, t_ref_v, C,
    LTP_active = True, STD_active = True, N_v_max = 10, glutamate_clearance_time=2):

        self.membrane_tau = membrane_tau
        self.spike_duration = ap.t_under
        self.C = C,
        self.R = membrane_tau/C
        self.t_ref_s = t_ref_s # absolute refractary period after one AP
        self.t_ref_v = t_ref_v # absolute refractary period of release vesicle machinery
        self.LTP_active = LTP_active   # set to False to inactivate astrocytic feedback
        self.STD_active = STD_active # set to False to have vesicle immediately recover
        self.Urest = ap.v_rest
        self.threshold = ap.v_thresh
        self.hyperp_v = ap.v_under
        self.N_v_max = N_v_max
        self.vesicles_replenishment_time = 0.6/N_v_max #s, 0.6/N_v_max   active if STD_active==True
        self.glutamate_clearance_time = glutamate_clearance_time

    def __str__(self) -> str:
        string = ""
        for k, v in self.__dict__.items():
            string += k + " = " + str(v) + " " + (units[k] if k in units.keys() else "") + "\n"
        return string

class SimulationParameters:
    units = {
        "time_step": "s",
        "simulation_time": "s",
        "temperature": "K",
        "N": "",
        "lambda_fixed": "",
        "current": "A",
        "f_current": "Hz",
        "lambda_": "Hz",
        "N_iterations": "",
        "init_25s": ""
    }
    def __init__(self, time_step, simulation_time, temperature, DiffEq=False,
    current_amplitude=0, current_type="DC", f_current=0, init_25s=False, init_N=0, N_iterations=1) -> None:
        self.time_step = time_step
        self.DiffEq = DiffEq
        self.simulation_time = simulation_time
        self.temperature = temperature
        self.init_25s = init_25s
        self.init_N = init_N
        self.N = int(np.ceil(simulation_time/time_step))
        self.lambda_fixed = False
        if (current_type == "DC"): # Use a DC current of 1.5uA
            self.current = current_amplitude*np.ones(self.N)
        elif (current_type == "random"):# Use a random current stimulus
            fmax_sample = f_current #Hz
            N_sample = int(fmax_sample*simulation_time)
            sample = np.random.normal(0, current_amplitude, size=N_sample) # white gaussian noise
            self.current = np.interp(np.linspace(0, 1, self.N), np.linspace(0, 1, N_sample), sample)
        #Use a pulse train current
        elif (current_type == "pulse"):
            self.f_current = f_current
            self.current = np.zeros(self.N)
            for i in range(0, self.N, int(1/time_step/f_current)):
                self.current[i] = 1.5e-6/time_step
        elif (current_type == "fixed_rate"):
            self.lambda_fixed = True
            self.lambda_ = f_current

        self.N_iterations = N_iterations

    def __str__(self) -> str:
        string = ""
        for k, v in self.__dict__.items():
            string += k + " = " + str(v) + " " + (units[k] if k in units.keys() else "") + "\n"
        return string

class Noise:
    def __init__(self, thermal=False, axonal=None, AP_width_CV=0, escape=True, spontaneous_release=True,
                 h_init_random=True, VGCC=False, Astro=False, astro_release_rate=0) -> None:
        self.thermal = thermal
        self.AP_width_CV = AP_width_CV
        self.axonal = axonal
        self.escape = escape
        self.spontaneous_release = spontaneous_release
        self.h_init_random = h_init_random
        self.VGCC = VGCC
        self.Astro = Astro
        self.astro_release_rate = astro_release_rate

    def __str__(self) -> str:
        string = ""
        for k, v in self.__dict__.items():
            string += k + " = " + str(v) + "\n"
        return string

class Variable:
    def __init__(self, title, unit="", value=0, init=True, array=False, N=0, dtype=np.float64):
        self.title = title
        if (init == True and array == True):
            self.value = np.zeros(N, dtype=dtype)
        else:
            self.value = value
        self.unit = unit
        self.array = array
        self.n_plots = 1

    def get(self, index=None):
        if (self.array == True):
            return self.value[index]
        else:
            return self.value

    def set(self, value, index=None):
        if (self.array == True):
            self.value[index] = value
        else:
            self.value = value

    def setAllTo(self, shape, value):
        if(self.array):
            self.value = np.full(shape, fill_value=value)
        else:
            self.value = value

    def initialize(self, value):
        if (self.array == True):
            self.value[0] = value
        else:
            self.value = value
    
    def update(self, idx, updateFunction): #shape
        if (self.array == True):
            self.value[idx] = updateFunction(self.value[idx - 1])
        else:
            self.value = updateFunction(self.value)

    def group(self, other):
        if (self.n_plots == 1):
            self.value = [self.value, other.value]
        else:
            self.value.append(other.value)
        self.n_plots += 1

    def restrict(self, start, end):
        if (self.n_plots == 1):
            self.value = self.value[start:end]
        else:
            for i in range(self.n_plots):
                self.value[i] = self.value[i][start:end]

    def averageEach(self, n):
        if (self.n_plots == 1):
            self.value = averageEach(self.value, n)
        else:
            for i in range(self.n_plots):
                self.value[i] = averageEach(self.value[i], n)

    def __add__(self, other):
        sum = copy.copy(self)
        sum.value = self.value + other.value
        return sum

    def __truediv__(self, n):
        div = copy.copy(self)
        div.value = self.value / n
        return div




class Variables:

    
    def __init__(self, N, track_variables=None, init=True, u=0, u_pre=0, spike_rate=0, spike_probability=0, open_prob=0, S=0, S_pre=0, release_rate=0,
    release_vector=0, spont_release=0, N_v=0, ap_duration_count=0, Ca_pre=0, Ca_Astro=0, Ca_stored=0, site_probabilities=np.zeros(4),
    release_prob=0, release_prob_a_posteriori=0, glu=0, IP3=0, h=0, mutual_information=0) -> None:
        self.track_variables = track_variables
        self.N = N
        self.u_pre = Variable("Membrane voltage before axon", "V", u_pre, init, True, N)
        self.u = Variable("Membrane voltage", "V", u, init, True, N)
        self.spike_rate = Variable("Spike rate", "Hz", spike_rate, init, "spike_rate" in track_variables, N)
        self.spike_probability = Variable("Spike probability", "", spike_probability, init, "spike_probability" in track_variables, N)
        self.open_prob = Variable("Open probability", "", open_prob, init, "open_prob" in track_variables, N)
        self.S_pre = Variable("Action potentials before axon", "", S_pre, init, True, N, dtype=np.int16)
        self.S = Variable("Action potentials", "", S, init, True, N, dtype=np.int16)
        self.release_rate = Variable("Release rate", "Hz", release_rate, init, "release_rate" in track_variables, N)
        self.release_vector = Variable("Vesicles released", "", release_vector, init, "release_vector" in track_variables, N, dtype=np.int16)
        self.spont_release = Variable("Vesicles released", "", spont_release, init, "spont_release" in track_variables, N, dtype=np.int16)
        self.N_v = Variable("Vesicle in Ready pool", "", N_v, init, "N_v" in track_variables, N, dtype=np.int16)
        self.ap_duration_count = Variable("AP width, normalized w.r.t. deltaT", "", ap_duration_count, init, "ap_duration_count" in track_variables, N, dtype=np.int16)
        self.Ca_pre = Variable("Presynaptic calcium", "uM", Ca_pre, init, "Ca_pre" in track_variables, N)
        self.Ca_Astro = Variable("Astrocytic calcium", "uM", Ca_Astro, init, "Ca_Astro" in track_variables, N)
        self.Ca_stored = Variable("Stored calcium", "uM", Ca_stored, init, "Ca_stored" in track_variables, N)
        self.site_probabilities = []
        for i in range(4):
           self.site_probabilities.append(Variable("Active site channel " + str(i) + " opening probability", "", site_probabilities[i], init, "site_probabilities" in track_variables, N))
        self.release_prob = Variable("Vesicle release probability a priori", "", release_prob, init, "release_prob" in track_variables, N)
        self.release_prob_a_posteriori = Variable("Vesicle release probability", "", release_prob_a_posteriori, init, "release_prob_a_posteriori" in track_variables, N)
        self.glu = Variable("Glutamate available for Astrocytic mGlur binding", "uM", glu, init, "glu" in track_variables, N)
        self.IP3 = Variable("IP3 concentration (Astrocyte)", "uM", IP3, init, "IP3" in track_variables, N)
        self.h = Variable("IP3R gating variable", "", h, init, "h" in track_variables, N)
        self.mutual_information = Variable("Mutual information", "bits/sec", mutual_information, init, "mutual_information" in track_variables, N)
        #self.entropy_v = Variable("Entropy of vesicle release", "bits/sec", entropy_v, init, "entropy_v" in track_variables, N)
        #self.conditional_entropy = Variable("Conditional Entropy of vesicle release given AP", "bits/sec", cond_entropy, init, "conditional_entropy" in track_variables, N)

    def initializeArrays(self, N):
        for k, v in self:
            self.__dict__[k] = np.zeros(N)

    def __iter__(self):
        v_fil = {k: self.__dict__[k] for k in self.track_variables}
        return iter(v_fil.items())
    

    def group(self, other):
        group = copy.copy(self)
        for k, v in group:
            group.__dict__[k].group(other.__dict__[k])
        return group
    
    def restrict(self, start, end):
        restriction = copy.deepcopy(self)
        for k, v in restriction:
            restriction.__dict__[k].restrict(start, end)
        return restriction
 

    def __add__(self, other):
        sum = copy.copy(self)
        for k, v in sum:
            sum.__dict__[k] += other.__dict__[k]
        return sum

    def __truediv__(self, n):
        if (n == 1):
            return self
        
        div = copy.copy(self)
        for k, v in div:
            div.__dict__[k] /= n
        return div


class Simulator:
    def __init__(self, properties: Properties, simulation_parameters: SimulationParameters, noise: Noise):
        self.p = properties
        self.s = simulation_parameters
        self.noise = noise

    def iterateVariables(self, track_variables, valueRand) -> Variables:
        N = self.s.N
        time_step = self.s.time_step
        var = Variables(N, track_variables)
        # states "init", "AP", "Spike-AfterPotential"
        state = "init"
        var.u.setAllTo(N, self.p.Urest)
        var.u_pre.setAllTo(N, self.p.Urest)
        var.ap_duration_count.initialize(int(self.p.spike_duration/time_step))
        #ap_duration_count_sum = 0
        ap_duration_count = var.ap_duration_count.get(0)
        spike_active = False #we'll set this to True during an AP
        last_spike = -np.Inf
        last_release = -np.Inf
        if (self.noise.Astro):
            last_astro_release = -np.Inf
            astro_release_prob = lib.poisson(1000*self.noise.astro_release_rate, time_step)
        s_ref_count = int(self.p.t_ref_s/time_step)
        v_ref_count = int(self.p.t_ref_v/time_step)
        var.N_v.initialize(self.s.init_N)
        var.IP3.initialize(160e-3 if self.s.init_25s == False else 0.421021) #uM (equilibrium concentration) #(0.421021) 
        var.Ca_Astro.initialize(0 if self.s.init_25s == False else 0.2)
        var.h.initialize(valueRand if self.noise.h_init_random else (0.705339 if self.s.init_25s else 0)) #0.705339 valore di salto

        T_info = int(0.02/time_step)
            
        ap_obj = AP.Spike(
                total_width=4e-3,
                t_peak=0.5e-3,
                t_under=2e-3,
                v_thresh=-34.3e-3,
                v_rest=-65e-3,
                v_peak=40e-3,
                v_under=-80e-3
            )
        fst_Ca_obj = VGCC.fastCalcium(ap_obj, time_step)
        if (self.noise.VGCC):
            Ca_AP_avg, Ca_AP_std = fst_Ca_obj.simulateStatistics(N_ITER=100)
            """f = open("AP_VGCC_dynamics/calcium_data", "rb")
            Ca_data = pickle.load(f)
            Ca_AP_avg = Ca_data["mean"]
            Ca_AP_std = Ca_data["std"]"""


        glutamate_clearance_time = 2e-3 #s
        glu_duration_count = int(glutamate_clearance_time/(time_step))
        glu_active = False
        m_info_delta_T_sum = 0

        if (self.s.lambda_fixed):
            var.spike_rate.setAllTo(N, self.s.lambda_)
            var.spike_probability.setAllTo(N, lib.poisson(var.spike_rate.get(0), time_step))

        # SRM
        for i in range(1, N):
            # Voltage
            #if ((state == "init" or state == "Spike-AfterPotential") and self.s.lambda_fixed == False):
            if (self.s.lambda_fixed == False):
                func = lambda u: lib.updateVoltage(u, self.s.current[i], self.p.C, self.p.Urest, self.p.membrane_tau, time_step, self.noise.thermal, self.s.temperature, self.p.R)
                var.u_pre.update(i, func)
            #elif (state == "AP"):
                #state = "Spike-AfterPotential"
            """elif (state == "Spike-AfterPotential"):
                u[i] = time_step*(current[i]/C - (u[i-1] - Urest)/membrane_tau) + u[i-1]
                #u[i] -= time_step*(u[i-1] - Urest)/hyperp_tau"""

            if (self.s.lambda_fixed == False):
                if (self.noise.escape):
                    var.spike_rate.set(lib.sigmoidalNonLinearity(var.u_pre.get(i) - self.p.Urest), i)
                    var.spike_probability.set(lib.poisson(var.spike_rate.get(i), time_step), i)
                else:
                    var.spike_probability.set(int(var.u_pre.get(i) >= self.p.threshold), i)

            rand = np.random.random()
            if (i - last_spike >= s_ref_count and var.spike_probability.get(i) > rand):
                var.S_pre.set(1, i)
                # AP waveform pasted (formatter impulse)
                if (self.noise.axonal["active"]):
                    fst_Ca_obj.setAPShapeRandom(self.p.spike_duration, self.noise.axonal["AP_width_CV"])
                for j, voltage in enumerate(fst_Ca_obj.v):
                    if (i+j < N):
                        var.u_pre.set(voltage, i+j)

        if (self.noise.axonal["active"]):
            if (var.u.array == False or var.u_pre.array == False):
                raise Exception()
            var.S.value, var.u.value = axon.getModifiedSpikeTrain(var.S_pre.value, var.u_pre.value, self.p.t_ref_v, self.p.Urest, self.noise.axonal, time_step)
        else:
            var.S.value = var.S_pre.value
            var.u.value = var.u_pre.value

        for i in range(1, N):  
            if (var.S.get(i) == 1): 
                last_spike = i   
                if (self.noise.axonal["active"]):
                    #ap_duration_count = int(np.round((self.p.spike_duration/time_step)*(1 + np.random.normal(0, self.noise.AP_width_CV))))
                    fst_Ca_obj.setAPShape(explicit_waveform = var.u.value[i:i+fst_Ca_obj.N])
                if (self.noise.VGCC):
                    fst_Ca_obj.simulateAP(unit="µM")
                #var.u.set(self.p.hyperp_v, i)


            # Presynaptic Calcium stays fixed to 300uM during an AP when VGCC noise is turned off#
            if (i - last_spike < ap_duration_count):
                if (self.noise.VGCC):
                    #Ca_AP = 1e6*np.random.normal(Ca_AP_avg[i - last_spike], Ca_AP_std[i - last_spike])
                    Ca_AP = fst_Ca_obj.Ca_conc[i - last_spike]
                else:
                    Ca_AP = 300
                spike_active = True
            else:
                Ca_AP = 0
                spike_active = False

            # glutamate stays fixed to 200uM for the period following a vesicle release
            if (i - last_release < glu_duration_count):
                var.glu.set(200, i)
                glu_active = True
            else:
                var.glu.set(0, i)
                glu_active = False

            # Vesicle replenishment
            if (var.N_v.get(i-1) < self.p.N_v_max):
                replenishment_time = self.p.vesicles_replenishment_time if self.p.STD_active == True else 0
                func = lambda N: lib.vesicleReplenishment(N, replenishment_time, time_step)
                var.N_v.update(i, func)
            else:
                var.N_v.set(self.p.N_v_max, i)

            #Calcium dynamics
            if (self.p.LTP_active == True):
                if (self.noise.Astro and i - last_astro_release >= v_ref_count and np.random.rand() < astro_release_prob*var.Ca_Astro.get(i-1)):
                    last_astro_release = i
                if ((self.noise.Astro == False) or i - last_astro_release < glu_duration_count):
                    func = lambda Ca: lib.updateStoredCa(Ca, var.Ca_Astro.get(i-1), time_step)
                    var.Ca_stored.update(i, func)
                else:
                    func = lambda Ca: lib.updateStoredCa(Ca, 0, time_step)
                    var.Ca_stored.update(i, func)
                funcCa_Astro = lambda Ca_Astro: lib.updateCaAstro(Ca_Astro, var.h.get(i-1), var.IP3.get(i-1), time_step)
                var.Ca_Astro.update(i, funcCa_Astro)
                funcIP3 = lambda IP3: lib.updateIP3(IP3, var.Ca_Astro.get(i-1), var.glu.get(i), time_step, glu_active)
                var.IP3.update(i, funcIP3)
                funch = lambda h: lib.updateIP3RGating(h, var.IP3.get(i-1), var.Ca_Astro.get(i-1), time_step)
                var.h.update(i, funch)

            Ca_pre_tot = Ca_AP+var.Ca_stored.get(i) if Ca_AP+var.Ca_stored.get(i) > 0 else 0
            var.Ca_pre.set(Ca_pre_tot, i)

            #Vesicle release dynamics
            open_prob_temp = 1
            if (self.s.DiffEq==True):
                new_probs = lib.sitesProbabilities(i, var.Ca_pre.get(i), time_step)
            else:
                probs = np.array([var.site_probabilities[0].get(i), var.site_probabilities[1].get(i), var.site_probabilities[2].get(i), var.site_probabilities[3].get(i)])
                new_probs = lib.updateSitesProbabilities(probs, var.Ca_pre.get(i), time_step)
            for j in range(4):
                var.site_probabilities[j].set(new_probs[j] if new_probs[j] < 1 else 1, i)
                open_prob_temp *= var.site_probabilities[j].get(i)
            var.open_prob.set(open_prob_temp if open_prob_temp > 0 else 0, i)
            
            if (i - last_release >= v_ref_count): # Check if vesicle release machinery is ready
                var.release_rate.set(var.spike_rate.get(i)*var.open_prob.get(i), i)
                # open_prob => Probability of having one release during the AP, with N=1
                release_prob_1_c1 = (1 - var.open_prob.get(i))**(1/ap_duration_count)
                release_prob_during_AP = 1 - release_prob_1_c1**var.N_v.get(i)  # P(V[n]=1 | S[n]=1)
                # probability of having one release without an AP
                spontaneous_rate = lib.spontaneousRate(var.Ca_pre.get(i))
                release_prob_no_AP = lib.poisson(var.N_v.get(i)*spontaneous_rate, time_step) if self.noise.spontaneous_release and var.N_v.get(i) > 0 else 0   # P(V[n]=1 | S[n]=0)
                var.release_prob.set(var.spike_probability.get(i)*release_prob_during_AP + (1 - var.spike_probability.get(i))*release_prob_no_AP, i)  # P(V[n]=1)  Total probability theorem)
                if (spike_active == True):
                    var.release_prob_a_posteriori.set(release_prob_during_AP, i)
                else:
                    var.release_prob_a_posteriori.set(release_prob_no_AP, i)
                rand = np.random.random()
                if (var.release_prob_a_posteriori.get(i) > rand): #1 vesicle is released
                    var.N_v.set(var.N_v.get(i) - 1, i)
                    var.release_vector.set(1, i)
                    last_release = i
                    if (spike_active == False):
                        var.spont_release.set(1, i)
            else:
                var.release_rate.set(0, i)
                var.release_prob.set(0, i)
                var.release_prob_a_posteriori.set(0, i)
                release_prob_during_AP = 0
                release_prob_no_AP = 0

            # Mutual information
            mutual_information = lib.mutualInformation(var.spike_probability.get(i), release_prob_during_AP, release_prob_no_AP) #bits in one single time step
            #var.mutual_information.set(lib.mutualInformation(var.spike_probability.get(i), release_prob_during_AP, release_prob_no_AP), i)
            if (i % T_info == 0):
                prev = var.mutual_information.get(i-T_info)
                succ = m_info_delta_T_sum/T_info/time_step
                for j, value in enumerate(np.linspace(prev, succ, T_info)):
                    var.mutual_information.set(value, i-T_info+j+1) # bits/sec
                m_info_delta_T_sum = 0
            m_info_delta_T_sum += mutual_information

            # Variable set to state variables
            var.ap_duration_count.set(ap_duration_count, i)
            """# Sampling
            if (i % T_info == 0):
                for j in range(T_info):
                    var.ap_duration_count.set(ap_duration_count_sum/T_info/time_step, i-j) # bits/sec
                ap_duration_count_sum = 0
            ap_duration_count_sum += ap_duration_count

        max = np.argmax(np.diff(var.Ca_pre.value, n=2))
        print("stacchi: Ca_pre: %f, Ca_Astro: %f, Ca_stored: %f, IP3: %f, glu: %f, h: %f"
        "al tempo %fs"
        % (var.Ca_pre.value[max], var.Ca_Astro.value[max], var.Ca_stored.value[max], var.IP3.value[max], var.glu.value[max], var.h.value[max],
        time_step*max))"""
        return var

    def simulate(self, track_variables, N_iterations = None, save=True, save_each=5, name="") -> Variables:
        importlib.reload(axon)
        importlib.reload(VGCC)
        if (N_iterations == None):
            N_iterations = self.s.N_iterations
        
        N = self.s.N
        var_sum = Variables(N, track_variables) 

        f = open("random", "rb")
        list = pickle.load(f)
        f.close()

        for i in range(1, N_iterations + 1):
            var_sum += self.iterateVariables(track_variables, list[i-1])
            if (save == True and (i%save_each==0 or i == N_iterations)):
                saveResults(self.s, self.p, var_sum/i, i, name)

        return var_sum/N_iterations

            
def saveResults(parameters: SimulationParameters, properties: Properties, results: Variables, actual_iterations=0, name=""):
    root = "risultati/"+name
    tmp = parameters.N_iterations
    parameters.N_iterations = actual_iterations
    f1 = open(root+"parameters", "wb")
    pickle.dump(parameters, f1)
    parameters.N_iterations = tmp
    f1.close()
    f2 = open(root+"properties", "wb")
    pickle.dump(properties, f2)
    f2.close()
    f3 = open(root+"values", "wb")
    pickle.dump(results, f3)
    f3.close()

def averageEach(array, n):
    sum = 0
    for i, val in enumerate(array, 1):
        if (i % n == 0):
            for j in range(n):
                prev = array[i-n]
                succ = sum/n
                array[i-j] = prev + (succ - prev)*(n-j)/n
            sum = 0
        sum += val
    return array