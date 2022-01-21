import math
import torch
import numpy as np
import slayerSNN


class SlayerLayer(slayerSNN.layer):
    def calculateSrmKernel(self):
        if self.neuron["type"] == "SRMALPHA":
            return super().calculateSrmKernel()
        elif self.neuron["type"] == "LIF":
            tau = self.neuron["tauSr"]
            return self._calculateLIFKernel(tau=tau, mult=1.0 / tau)

    def calculateRefKernel(self):
        if self.neuron["type"] == "SRMALPHA":
            return super().calculateSrmKernel()
        elif self.neuron["type"] == "LIF":
            mult = -self.neuron["scaleRef"] * self.neuron["theta"]
            return self._calculateLIFKernel(tau=self.neuron["tauSr"], mult=mult)

    def _calculateLIFKernel(self, tau, mult=1, eps=1e-5):
        kernel = []
        for t in np.arange(0, self.simulation["tSample"], self.simulation["Ts"]):
            kernel.append(mult * math.exp(-t / tau))
            if abs(kernel[-1]) < eps and t > tau:
                break

        return torch.FloatTensor(kernel)
