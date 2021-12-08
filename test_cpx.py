# Author: Alban Gossard
# Last modification: 2021/08/12

import torch
import numpy as np
import nufftbindings.pykeops as nufft
# import nufftbindings.nfft as nufft

nx = ny = 320
K = int(nx*ny)
Nb = 3

device = torch.device("cuda:0")
# device = torch.device("cpu")

xi = torch.rand(K, 2, device=device)*2*np.pi-np.pi
xi.requires_grad = True

nufft.nufft.set_dims(K, (nx, ny), device, Nb=Nb)
nufft.nufft.precompute(xi)

f = torch.zeros(Nb, nx, ny, device=device, dtype=torch.cfloat)
import matplotlib.pyplot as plt
f[0,:nx//2,:ny//2]=1
print(f.dtype)
y = nufft.forward(xi, f)
g = nufft.adjoint(xi, y)
plt.figure(1)
plt.imshow(f[0].real.detach().cpu().numpy())
plt.figure(2)
plt.imshow(g[0].real.detach().cpu().numpy())
plt.show()
l = g.abs().pow(2).sum()
print(l)
l.backward()
