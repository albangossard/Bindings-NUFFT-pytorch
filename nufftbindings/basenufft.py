import torch
import numpy as np

class baseNUFFT:
    def set_dims(self, K, dims, device, Nb=1, doublePrecision=False):
        self.K = K
        self.ndim = len(dims)
        if self.ndim!=2:
            raise Exception("Only 2D NUFFT is available yet")
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

        self.nx = self.dims[0]
        self.xx = torch.arange(self.nx, device=self.device, dtype=self.torch_dtype)-self.nx/2.
        if self.ndim==2:
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
