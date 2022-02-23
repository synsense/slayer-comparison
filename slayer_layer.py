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
        elif self.neuron["type"] == "IAF":
            return self._calculateIAFKernel(mult=1.0)
        else:
            raise ValueError(f"Neuron type `{self.neuron['type']}` not recognized.")

    def calculateRefKernel(self):
        if self.neuron["type"] == "SRMALPHA":
            return super().calculateRefKernel()
        elif self.neuron["type"] in ("CUBALIF", "LIF"):
            mult = -self.neuron["scaleRef"] * self.neuron["theta"]
            return self._calculateLIFKernel(tau=self.neuron["tauSr"], mult=mult)
        elif self.neuron["type"] == "IAF":
            mult = -self.neuron["scaleRef"] * self.neuron["theta"]
            return self._calculateIAFKernel(mult=mult)
        else:
            raise ValueError(f"Neuron type `{self.neuron['type']}` not recognized.")

    def _calculateLIFKernel(self, tau, mult=1, eps=1e-5):
        kernel = []
        for t in np.arange(0, self.simulation["tSample"], self.simulation["Ts"]):
            kernel.append(mult * math.exp(-t / tau))
            if abs(kernel[-1]) < eps and t > tau:
                break

        return torch.FloatTensor(kernel)

    def _calculateIAFKernel(self, mult=1):
        return mult * torch.ones(self.simulation["tSample"]).float()

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
