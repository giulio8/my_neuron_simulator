
import numpy as np

class Spike:
    def __init__(self, total_width, t_peak, t_under, v_rest, v_peak, v_under):
        self.total_width = total_width
        self.t_peak = t_peak
        self.t_under = t_under
        self.v_rest = v_rest
        self.v_peak = v_peak
        self.v_under = v_under

class SpikePulse:
    def __init__(self, spike: Spike, time_step):
        self.p = spike
        self.N = int(self.p.total_width // time_step)
        self.peak = int(self.p.t_peak // time_step)
        self.under = int(self.p.t_under // time_step)
        self.val = np.zeros(self.N)
        self.val[:self.peak] = np.linspace(self.p.v_rest, self.p.v_peak, self.peak)
        self.val[self.peak:self.under] = np.linspace(self.p.v_peak, self.p.v_under, self.under - self.peak)
        self.val[self.under:] = np.linspace(self.p.v_under, self.p.v_rest, self.N - self.under)

    def getValues(self):
        return self.val