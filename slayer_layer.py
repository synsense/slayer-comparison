import math
import torch
import numpy as np
import slayerSNN


class SlayerLayer(slayerSNN.layer):
    def calculateSrmKernel(self):
        if self.neuron["type"] == "SRMALPHA":
            return super().calculateSrmKernel()
        if self.neuron["type"] == "CUBALIF":
            tau = self.neuron["tauSr"]
            return self._calculateAlphaKernel(tau=tau, mult=1.0)
        elif self.neuron["type"] == "LIF":
            tau = self.neuron["tauSr"]
            return self._calculateLIFKernel(tau=tau, mult=1.0)

    def calculateRefKernel(self):
        if self.neuron["type"] == "SRMALPHA":
            return super().calculateRefKernel()
        elif self.neuron["type"] in ("CUBALIF", "LIF"):
            mult = -self.neuron["scaleRef"] * self.neuron["theta"]
            return self._calculateLIFKernel(tau=self.neuron["tauSr"], mult=mult)

    def _calculateLIFKernel(self, tau, mult=1, eps=1e-5):
        kernel = []
        for t in np.arange(0, self.simulation["tSample"], self.simulation["Ts"]):
            kernel.append(mult * math.exp(-t / tau))
            if abs(kernel[-1]) < eps and t > tau:
                break

        return torch.FloatTensor(kernel)

    def _calculateAlphaKernel(self, tau, mult=1, EPSILON=1e-5):
        # - Preserve original behavior for SRMALPHA
        if self.neuron["type"] == "SRMALPHA":
            return super()._calculateAlphaKernel(tau, mult, EPSILON)

        # - This is similar to original alpha kernel, but shifted by one timestep and with different scaling
        eps = []
        for t in np.arange(0, self.simulation["tSample"], self.simulation["Ts"]):
            epsVal = mult * (t + 1) * math.exp(-t / tau)
            if abs(epsVal) < EPSILON and t > tau:
                break
            eps.append(epsVal)
        return torch.FloatTensor(eps)
