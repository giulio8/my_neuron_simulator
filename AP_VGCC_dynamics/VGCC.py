import numpy as np
from . import AP
from . import calciumDynamics as CD

class fastCalcium:
    def __init__(self, ap: AP.Spike, time_step) -> None:
        self.time_step = time_step
        self.spike_pulse = AP.SpikePulse(ap, time_step)
        self.N = self.spike_pulse.N
        self.v = self.spike_pulse.getValues()

        self.ap_noise = {
            "active": True,
            "an": 250,
            "ap": 350
        }

    def simulateAP(self, unit="M"):
        site_probs = np.zeros((5, self.N))
        site_probs[0, 0] = 0.1
        influx = np.zeros(self.N)
        Ca_VGCCs = np.zeros(self.N)

        decaying_k = 0.8
        for i in range(1, self.N):
            site_probs[:, i] = CD.updateState(site_probs[:, i-1], self.v[i], self.time_step)
            influx[i] = CD.influx(site_probs[4, i-1], self.v[i], self.ap_noise, self.time_step)
            Ca_VGCCs[i] = Ca_VGCCs[i-1]*decaying_k + influx[i]

        if (unit == "M"):
            return CD.CalciumMoleculesToConcentration(Ca_VGCCs)
        elif (unit == "µM"):
            return 1e6*CD.CalciumMoleculesToConcentration(Ca_VGCCs)
        return Ca_VGCCs

    def simulateStatistics(self, N_ITER):
        Ca_VGCCs = np.zeros((N_ITER, self.N))

        for j in range(N_ITER):
            Ca_VGCCs[j, :] = self.simulateAP(unit="molecules")

        Ca_conc = CD.CalciumMoleculesToConcentration(Ca_VGCCs)
        self.Ca_conc_avg = np.mean(Ca_conc, axis=0) 
        self.Ca_conc_std = np.std(Ca_conc, axis=0)

        return self.Ca_conc_avg, self.Ca_conc_std

    def getCalcium(self):
        return self.Ca_conc_avg, self.Ca_conc_std