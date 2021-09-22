# Author: Alban Gossard
# Last modification: 2021/22/09

import torch
import numpy as np

class baseNUFFT:
    def set_dims(self, K, dims, device, Nb=1, doublePrecision=False):
        self.K = K
        self.ndim = len(dims)
        self.dims = dims
        self.device = device
        self.Nbatch = Nb
        if doublePrecision:
            self.torch_dtype = torch.float64
            self.np_dtype = np.float64
            self.eps = 1e-12
        else:
            self.torch_dtype = torch.float32
            self.np_dtype = np.float32
            self.eps = 1e-6

        self.precomputedTrig = False

        if self.ndim not in [2, 3]:
            raise Exception("Only NUFFT in dimension 2 and 3 are implemented")

        self.nx = self.dims[0]
        self.xx = torch.arange(self.nx, device=self.device, dtype=self.torch_dtype)-self.nx/2.
        if self.ndim>=2:
            self.ny = self.dims[1]
            self.xy = torch.arange(self.ny, device=self.device, dtype=self.torch_dtype)-self.ny/2.
        if self.ndim==3:
            self.nz = self.dims[2]
            self.xz = torch.arange(self.nz, device=self.device, dtype=self.torch_dtype)-self.nz/2.

        self._set_dims()
    def test_xi(self, xi):
      if not xi.shape == (self.K, self.ndim):
          raise Exception("xi does not have shape "+str((self.K, self.ndim))+ '!='+ str(xi.shape))
      elif not self.precomputedTrig:
          raise Exception("Precomputing has not been applied yet")
    def test_f(self, f):
        if f.shape[0] != self.Nbatch and f.shape[0]!=1:
            raise Exception("The batch size does not correspond to the one indicated in set_dims. Expected the first dimension to be of size "+str(self.Nbatch)+" but got shape[0]="+str(f.shape[0]))
    def forward(self, f, xi):
        if self.ndim==2:
            return self._forward2D(f, xi)
        elif self.ndim==3:
            return self._forward3D(f, xi)
    def adjoint(self, y, xi):
        if self.ndim==2:
            return self._adjoint2D(y, xi)
        elif self.ndim==3:
            return self._adjoint3D(y, xi)
    def backward_forward(self, f, g, xi):
        if self.ndim==2:
            return self._backward_forward2D(f, g, xi)
        elif self.ndim==3:
            return self._backward_forward3D(f, g, xi)
    def backward_adjoint(self, y, g, xi):
        if self.ndim==2:
            return self._backward_adjoint2D(y, g, xi)
        elif self.ndim==3:
            return self._backward_adjoint3D(y, g, xi)
