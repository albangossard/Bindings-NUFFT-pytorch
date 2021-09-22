import torch
import numpy as np
# import nufftbindings.cufinufft as nufft
import nufftbindings.pykeops as nufft

nx = ny = 320
K = int(nx*ny)
Nb = 3

device = torch.device("cuda:0")

xi = torch.rand(K, 2, device=device)*2*np.pi-np.pi
xi.requires_grad = True

nufft.nufft.set_dims(K, (nx, ny), device, Nb=Nb)
nufft.nufft.precompute(xi)

f = torch.randn(Nb, nx, ny, 2, device=device)
y = nufft.forward(xi, f)
g = nufft.adjoint(xi, y)
l = g.abs().pow(2).sum()
l.backward()
