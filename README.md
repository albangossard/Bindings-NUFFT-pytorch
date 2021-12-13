# Bindings-NUFFT-pytorch

Bindings to use a Non-uniform Fourier Transform in pytorch with differentiation with respect to the input and to the positions.

Bindings with automatic differentiation are implemented for the following libraries:
- [torchkbnufft](https://github.com/mmuckley/torchkbnufft): works on GPU
- [cufinufft](https://github.com/flatironinstitute/cufinufft/): works on GPU
- [NFFT](https://www-user.tu-chemnitz.de/~potts/nfft/): works on CPU and it uses the [PyNFFT](https://github.com/pyNFFT/pyNFFT) library
- [pykeops](https://www.kernel-operations.io/keops/python/installation.html): works both on CPU and on GPU, a NUFFT is implemented in `nufftbindings/pykeops.py` as a reduction operation

This repository was initially created for personal use. Feel free to create a pull request and to propose any improvement.

## Example

Import the nufft with the backend you want to use by executing one of the following lines:
```python
import nufftbindings.kbnufft as nufft
import nufftbindings.cufinufft as nufft
import nufftbindings.nfft as nufft
import nufftbindings.pykeops as nufft
```
```python
nx = ny = 320       # size of the image
K = int(nx*ny)      # number of points in the Fourier domain
Nb = 3              # size of the batch

# CUDA device if you use kbnufft, cufinufft or pykeops
device = torch.device("cuda:0")
# otherwise use
# device = torch.device("cuda:0")

dtype = torch.complex64

# define the positions in the Fourier domain
xi = torch.rand(K, 2, device=device)*2*np.pi-np.pi
xi.requires_grad = True

# precomputations
nufft.nufft.set_dims(K, (nx, ny), device, Nb=3)
nufft.nufft.precompute(xi)

f = torch.randn(Nb, nx, ny, device=device, dtype=dtype)
y = nufft.forward(xi, f)
g = nufft.adjoint(xi, y)
# you can define whatever cost function you want and compute the gradient with respect to the image f or to xi
loss = g.abs().pow(2).sum()
loss.backward()
```

## Requirements

The codes were tested with the following configuration:

- CUDA 11.2
- Python 3.6.9 and 3.8.11
- PyTorch 1.3.0 and 1.9.0
- Numpy 1.17.2 and 1.20.3
- PyNFFT 1.3.2
- PyKeOps 1.2 and 1.5
- cuFINUFFT 1.2
- torchkbnufft 1.2.0

**Note**: The cuFINUFFT library is designed to work with `pycuda`. It can be easily modified to support pytorch. Just replace the following lines in the `cufinufft.py` file:
```python
if not c.dtype == fk.dtype == self.complex_dtype:
    raise TypeError("cufinufft execute expects {} dtype arguments "
                    "for this plan. Check plan and arguments.".format(
                        self.complex_dtype))

ier = self._exec_plan(c.ptr, fk.ptr, self.plan)
```
by
```python
ier = self._exec_plan(c, fk, self.plan)
```

## License

See the [LICENSE file](https://github.com/albangossard/Bindings-NUFFT-pytorch)
