import numpy as np
from . import AP
from . import calciumDynamics as CD

class fastCalcium:
    def __init__(self, ap: AP.Spike, time_step, N_ITER) -> None:
        self.spike_pulse = AP.SpikePulse(ap, time_step)
        self.N = self.spike_pulse.N
        self.v = self.spike_pulse.getValues()

        self.ap_noise = {
            "active": True,
            "an": 250,
            "ap": 350
        }

        self.N_ITER = N_ITER

        S = np.zeros((5, self.N))
        S[0, 0] = 0.1
        influx = np.zeros(self.N)
        Ca_VGCCs = np.zeros((self.N_ITER, self.N))

        decaying_k = 0.8

        for j in range(self.N_ITER):
            for i in range(1, self.N):
                S[:, i] = CD.updateState(S[:, i-1], self.v[i], time_step)
                influx[i] = CD.influx(S[4, i-1], self.v[i], self.ap_noise, time_step)
                Ca_VGCCs[j, i] = Ca_VGCCs[j, i-1]*decaying_k + influx[i]

        Ca_conc = CD.CalciumMoleculesToConcentration(Ca_VGCCs)
        self.Ca_conc_avg = np.mean(Ca_conc, axis=0) 
        self.Ca_conc_std = np.std(Ca_conc, axis=0)

    def getCalcium(self):
        return self.Ca_conc_avg, self.Ca_conc_std