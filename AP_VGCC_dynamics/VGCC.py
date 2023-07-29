import numpy as np
from . import AP
from . import calciumDynamics as CD
import importlib
import matplotlib.pyplot as plt

class fastCalcium:
    def __init__(self, ap: AP.Spike, time_step) -> None:
        importlib.reload(AP)
        self.time_step = time_step
        self.updateSpikeShape(ap)
        self.N = int(self.spike.total_width // time_step)
        

        self.ap_noise = {
            "active": True,
            "an": 250,
            "ap": 350
        }

    def updateSpikeShape(self, ap_obj):
            self.spike = ap_obj
            self.v = self.spike.getPulseValues(self.time_step)

    def setAPShape(self, obj=False, ap_obj: AP.Spike=None, explicit_waveform=None):
        if (obj):
            self.updateSpikeShape(ap_obj)
        else:
            self.v = explicit_waveform

    def setAPShapeRandom(self, base_width, CV_width):
        new_duration = -1
        while (new_duration < self.spike.t_peak or new_duration > self.spike.total_width):
            new_duration = np.random.normal(base_width, CV_width*base_width)
        self.spike.t_under = new_duration
        self.updateSpikeShape(self.spike)

    def simulateAP(self, unit="M"):

        #importlib.reload(CD)

        site_probs = np.zeros((5, self.N))
        site_probs[0, 0] = 0.1
        influx = np.zeros(self.N)
        self.Ca_VGCCs = np.zeros(self.N)

        decaying_k = 0.8
        try:
            for i in range(1, self.N):
                site_probs[:, i] = CD.updateState(site_probs[:, i-1], self.v[i], self.time_step)
                influx[i] = CD.influx(site_probs[4, i-1], self.v[i], self.ap_noise, self.time_step)
                tmp = self.Ca_VGCCs[i-1]*decaying_k + influx[i]
                if (tmp > 0):
                    self.Ca_VGCCs[i] = tmp
                else:
                    self.Ca_VGCCs[i] = 0
        except:
            print("exception ciclo")

        """debugif (np.any(site_probs[4, :] < 0)):
            for i in range(5):
                plt.plot(site_probs[i, :])
            plt.show()"""

        if (unit == "M"):
            self.Ca_conc = CD.CalciumMoleculesToConcentration(self.Ca_VGCCs)
            return self.Ca_conc
        elif (unit == "µM"):
            self.Ca_conc = 1e6*CD.CalciumMoleculesToConcentration(self.Ca_VGCCs)
            return self.Ca_conc
        else:
            return self.Ca_VGCCs

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