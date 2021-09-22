# Author: Alban Gossard
# Last modification: 2021/22/09

import torch, time
import numpy as np
import nufftbindings.kbnufft as nufftkb
import nufftbindings.cufinufft as nufftcu
import nufftbindings.nfft as nufftnf
import nufftbindings.pykeops as nufftko

nx = ny = nz = 32
K = int(nx*ny*nz)
Nb = 3

list_modulenames = ["pyKeOps", "kbnufft", "cuFINUFFT", "NFFT"]
list_device = ["cuda:0", "cuda:0", "cuda:0", "cpu"]
list_module = [nufftko, nufftkb, nufftcu, nufftnf]

fnp = np.random.randn(Nb, nx, ny, nz, 2)
xinp = np.random.uniform(-np.pi, np.pi, (K, 3))

for d, module, modulename in zip(list_device, list_module, list_modulenames):
    print((" "+modulename+" ").center(30,"#"))

    device = torch.device(d)

    torch.manual_seed(1)
    f = torch.tensor(fnp, device=device)
    xi = torch.tensor(xinp, device=device)
    xi.requires_grad = True

    module.nufft.set_dims(K, (nx, ny, nz), device, Nb=Nb)
    module.nufft.precompute(xi)

    tic = time.time()
    out = module.forward(xi, module.adjoint(xi, module.forward(xi, f)))
    l = out.abs().pow(2).sum()
    toc = time.time()
    print("Elapsed time forward: %1.3f s"%(toc-tic))
    tic = time.time()
    l.backward()
    toc = time.time()
    print("Elapsed time backward: %1.3f s"%(toc-tic))
    grad = xi.grad.detach().cpu().clone()
    xi.grad.zero_()

    if modulename=="pyKeOps":
        with torch.no_grad():
            outref = out.cpu().clone()
            gradref = grad.clone()

    with torch.no_grad():
        out = out.cpu()
        print('  relative error = %1.3e'%(torch.norm(out-outref)/torch.norm(out)).item())
        print('  correlation = %1.3e'%((out*outref).sum()/(torch.norm(out)*torch.norm(outref))).item())
        print('  ratio = %1.3e'%(out.norm()/outref.norm()).item())
        print('')
