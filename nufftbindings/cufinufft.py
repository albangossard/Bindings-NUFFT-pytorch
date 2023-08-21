# Author: Alban Gossard
# Last modification: 2023/16/08

import numpy as np

# Use pycuda.autoprimaryctx to have the same CUDA context as for pytorch
# and avoid the trick that consists in changing the sources of cuFINUFFT
# import pycuda.autoinit
import pycuda.autoprimaryctx
import pycuda.driver as drv

import torch
from cufinufft import cufinufft
from .basenufft import *


class Holder(drv.PointerHolderBase):
    # Refer to https://gist.github.com/szagoruyko/440c561f7fce5f1b20e6154d801e6033
    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()
        self.dtype = {
                torch.float32: np.float32,
                torch.float64: np.float64,
                torch.complex64: np.complex64,
                torch.complex128: np.complex128,
            }[t.dtype]
        self.size = np.prod(t.shape)
        self.ptr = self.gpudata
    def get_pointer(self):
        return self.gpudata


class Nufft(baseNUFFT):
    def _set_dims(self):
        xx = np.arange(self.nx)-self.nx/2.
        xy = np.arange(self.ny)-self.ny/2.
        if self.ndim==2:
            self.XX, self.XY = np.meshgrid(xx, xy)
        if self.ndim==3:
            xz = np.arange(self.nz)-self.nz/2.
            self.XX, self.XY, self.XZ = np.meshgrid(xx, xy, xz)
        self.XX = torch.tensor(self.XX.T, device=self.device)
        self.XY = torch.tensor(self.XY.T, device=self.device)
        if self.ndim==3:
            self.XZ = torch.tensor(self.XZ.T, device=self.device)

        if self.Nbatch is None:
            raise Exception("The batch size should be specified in set_dims")

        self.plan_forward = None
        self.plan_adjoint = None
        self.plan_forward_batch = None
        self.plan_adjoint_batch = None

    def precompute(self, xi):
        xic = xi.detach().contiguous()
        xix = Holder(xic[:,0].type(self.torch_dtype).contiguous())
        xiy = Holder(xic[:,1].type(self.torch_dtype).contiguous())
        if self.ndim==3:
            xiz = Holder(xic[:,2].type(self.torch_dtype).contiguous())

        if self.plan_forward is not None:
            del self.plan_forward
            del self.plan_adjoint
            del self.plan_forward_batch
            del self.plan_adjoint_batch
        if self.ndim==2:
            self.plan_forward = cufinufft(2, (self.nx, self.ny), 1, eps=self.eps, dtype=self.np_dtype)
            self.plan_adjoint = cufinufft(1, (self.nx, self.ny), 1, eps=self.eps, dtype=self.np_dtype)
            self.plan_forward_batch = cufinufft(2, (self.nx, self.ny), self.Nbatch, eps=self.eps, dtype=self.np_dtype)
            self.plan_adjoint_batch = cufinufft(1, (self.nx, self.ny), self.Nbatch, eps=self.eps, dtype=self.np_dtype)

            self.plan_forward.set_pts(xix, xiy)
            self.plan_adjoint.set_pts(xix, xiy)
            self.plan_forward_batch.set_pts(xix, xiy)
            self.plan_adjoint_batch.set_pts(xix, xiy)
        elif self.ndim==3:
            self.plan_forward = cufinufft(2, (self.nx, self.ny, self.nz), 1, eps=self.eps, dtype=self.torch_dtype)
            self.plan_adjoint = cufinufft(1, (self.nx, self.ny, self.nz), 1, eps=self.eps, dtype=self.torch_dtype)
            self.plan_forward_batch = cufinufft(2, (self.nx, self.ny, self.nz), self.Nbatch, eps=self.eps, dtype=self.torch_dtype)
            self.plan_adjoint_batch = cufinufft(1, (self.nx, self.ny, self.nz), self.Nbatch, eps=self.eps, dtype=self.torch_dtype)

            self.plan_forward.set_pts(xix, xiy, xiz)
            self.plan_adjoint.set_pts(xix, xiy, xiz)
            self.plan_forward_batch.set_pts(xix, xiy, xiz)
            self.plan_adjoint_batch.set_pts(xix, xiy, xiz)

        self.xiprecomputed = xi.clone()

        self.precomputedTrig = True

    def _forward2D(self, f, xi):
        self.test_xi(xi)
        self.test_f(f)
        ndim = len(f.shape)
        iscpx = f.is_complex()
        if ndim==4 and not iscpx or ndim==3 and iscpx:
            Nbatch = f.shape[0]
            if iscpx:
                y = torch.zeros(Nbatch, self.K, device=self.device, dtype=self.torch_cpxdtype)
                fcpx = f.type(self.torch_cpxdtype).contiguous()
            else:
                y = torch.zeros(Nbatch, self.K, 2, device=self.device, dtype=self.torch_dtype)
                fcpx = f.type(self.torch_dtype).contiguous()
            if Nbatch==1:
                self.plan_forward.execute(Holder(y), Holder(fcpx))
            else:
                self.plan_forward_batch.execute(Holder(y), Holder(fcpx))
            return y
        else:
            raise Exception("Error: f should have 4 dimensions (one axis for real/imaginary parts) or 3 dimensions (complex)")
    def _adjoint2D(self, y, xi):
        self.test_xi(xi)
        self.test_f(y)
        ndim = len(y.shape)
        iscpx = y.is_complex()
        if ndim==3 and not iscpx or ndim==2 and iscpx:
            Nbatch = y.shape[0]
            if iscpx:
                f = torch.zeros(Nbatch, self.nx, self.ny, device=self.device, dtype=self.torch_cpxdtype)
                ycpx = y.type(self.torch_cpxdtype).contiguous()
            else:
                f = torch.zeros(Nbatch, self.nx, self.ny, 2, device=self.device, dtype=self.torch_dtype)
                ycpx = y.type(self.torch_dtype).contiguous()
            if Nbatch==1:
                self.plan_adjoint.execute(Holder(ycpx), Holder(f))
            else:
                self.plan_adjoint_batch.execute(Holder(ycpx), Holder(f))
            return f
        else:
            raise Exception("Error: y should have 3 dimensions (one axis for real/imaginary parts) or 2 dimensions (complex)")
    def _backward_forward2D(self, f, g, xi):
        self.test_xi(xi)
        ndim = len(f.shape)
        iscpx = f.is_complex()
        if ndim==4 and not iscpx or ndim==3 and iscpx:
            Nbatch = f.shape[0]

            if iscpx:
                vec_fx = torch.mul(self.XX[None,:,:].contiguous(), f.contiguous())
                vec_fy = torch.mul(self.XY[None,:,:].contiguous(), f.contiguous())
            else:
                vec_fx = torch.mul(self.XX[None,:,:,None].contiguous(), f.contiguous())
                vec_fy = torch.mul(self.XY[None,:,:,None].contiguous(), f.contiguous())

            grad = torch.zeros_like(xi)

            if iscpx:
                tmp = torch.zeros(Nbatch, self.K, device=self.device, dtype=self.torch_cpxdtype)
                vec_fx = vec_fx.type(self.torch_cpxdtype).contiguous()
            else:
                tmp = torch.zeros(Nbatch, self.K, 2, device=self.device, dtype=self.torch_dtype)
                vec_fx = vec_fx.type(self.torch_dtype).contiguous()
            if Nbatch==1:
                self.plan_forward.execute(Holder(tmp), Holder(vec_fx))
            else:
                self.plan_forward_batch.execute(Holder(tmp), Holder(vec_fx))

            if iscpx:
                grad[:,0] = ( torch.mul(tmp.imag, g.real) - torch.mul(tmp.real, g.imag) ).sum(axis=0)
            else:
                grad[:,0] = ( torch.mul(tmp[...,1], g[...,0]) - torch.mul(tmp[...,0], g[...,1]) ).sum(axis=0)

            if iscpx:
                vec_fy = vec_fy.type(self.torch_cpxdtype).contiguous()
            else:
                vec_fy = vec_fy.type(self.torch_dtype).contiguous()
            if Nbatch==1:
                self.plan_forward.execute(Holder(tmp), Holder(vec_fy))
            else:
                self.plan_forward_batch.execute(Holder(tmp), Holder(vec_fy))

            if iscpx:
                grad[:,1] = ( torch.mul(tmp.imag, g.real) - torch.mul(tmp.real, g.imag) ).sum(axis=0)
            else:
                grad[:,1] = ( torch.mul(tmp[...,1], g[...,0]) - torch.mul(tmp[...,0], g[...,1]) ).sum(axis=0)

            return grad
        else:
            raise Exception("Error: f should have 4 dimensions (one axis for real/imaginary parts) or 3 dimensions (complex)")

    def _backward_adjoint2D(self, y, g, xi):
        self.test_xi(xi)
        ndim = len(y.shape)
        iscpx = y.is_complex()
        if ndim==3 and not iscpx or ndim==2 and iscpx:
            Nbatch = y.shape[0]

            if iscpx:
                vecx_grad_output = torch.mul(self.XX[None,:,:].contiguous(), g.contiguous())
                vecy_grad_output = torch.mul(self.XY[None,:,:].contiguous(), g.contiguous())
            else:
                vecx_grad_output = torch.mul(self.XX[None,:,:,None].contiguous(), g.contiguous())
                vecy_grad_output = torch.mul(self.XY[None,:,:,None].contiguous(), g.contiguous())

            grad = torch.zeros_like(xi)

            if iscpx:
                tmp = torch.zeros(Nbatch, self.K, device=self.device, dtype=self.torch_cpxdtype)
                vecx_grad_output = vecx_grad_output.type(self.torch_cpxdtype).contiguous()
            else:
                tmp = torch.zeros(Nbatch, self.K, 2, device=self.device, dtype=self.torch_dtype)
                vecx_grad_output = vecx_grad_output.type(self.torch_dtype).contiguous()
            if Nbatch==1:
                self.plan_forward.execute(Holder(tmp), Holder(vecx_grad_output))
            else:
                self.plan_forward_batch.execute(Holder(tmp), Holder(vecx_grad_output))

            if iscpx:
                grad[:,0] = ( torch.mul(tmp.imag, y.real) - torch.mul(tmp.real, y.imag) ).sum(axis=0)
            else:
                grad[:,0] = ( torch.mul(tmp[...,1], y[...,0]) - torch.mul(tmp[...,0], y[...,1]) ).sum(axis=0)

            if iscpx:
                vecy_grad_output = vecy_grad_output.type(self.torch_cpxdtype).contiguous()
            else:
                vecy_grad_output = vecy_grad_output.type(self.torch_dtype).contiguous()
            if Nbatch==1:
                self.plan_forward.execute(Holder(tmp), Holder(vecy_grad_output))
            else:
                self.plan_forward_batch.execute(Holder(tmp), Holder(vecy_grad_output))

            if iscpx:
                grad[:,1] = ( torch.mul(tmp.imag, y.real) - torch.mul(tmp.real, y.imag) ).sum(axis=0)
            else:
                grad[:,1] = ( torch.mul(tmp[...,1], y[...,0]) - torch.mul(tmp[...,0], y[...,1]) ).sum(axis=0)

            return grad
        else:
            raise Exception("Error: y should have 3 dimensions (one axis for real/imaginary parts) or 2 dimensions (complex)")

    def _forward3D(self, f, xi):
        self.test_xi(xi)
        self.test_f(f)
        ndim = len(f.shape)
        if ndim==5:
            Nbatch = f.shape[0]
            y = torch.zeros(Nbatch, self.K, 2, device=self.device, dtype=self.torch_dtype)
            fcpx = f.type(self.torch_dtype).contiguous()
            if Nbatch==1:
                self.plan_forward.execute(Holder(y), Holder(fcpx))
            else:
                self.plan_forward_batch.execute(Holder(y), Holder(fcpx))
            return y
        else:
            raise Exception("Error: f should have 5 dimensions")
    def _adjoint3D(self, y, xi):
        self.test_xi(xi)
        self.test_f(y)
        ndim = len(y.shape)
        if ndim==3:
            Nbatch = y.shape[0]
            f = torch.zeros(Nbatch, self.nx, self.ny, self.nz, 2, device=self.device, dtype=self.torch_dtype)
            ycpx = y.type(self.torch_dtype).contiguous()
            if Nbatch==1:
                self.plan_adjoint.execute(Holder(ycpx), Holder(f))
            else:
                self.plan_adjoint_batch.execute(Holder(ycpx), Holder(f))
            return f
        else:
            raise Exception("Error: y should have 3 dimensions")
    def _backward_forward3D(self, f, g, xi):
        self.test_xi(xi)
        ndim = len(f.shape)
        if ndim==5:
            Nbatch = f.shape[0]

            vec_fx = torch.mul(self.XX[None,:,:,:,None].contiguous(), f.contiguous())
            vec_fy = torch.mul(self.XY[None,:,:,:,None].contiguous(), f.contiguous())
            vec_fz = torch.mul(self.XZ[None,:,:,:,None].contiguous(), f.contiguous())

            grad = torch.zeros_like(xi)

            tmp = torch.zeros(Nbatch, self.K, 3, device=self.device, dtype=self.torch_dtype)
            vec_fx = vec_fx.type(self.torch_dtype).contiguous()
            if Nbatch==1:
                self.plan_forward.execute(Holder(tmp), Holder(vec_fx))
            else:
                self.plan_forward_batch.execute(Holder(tmp), Holder(vec_fx))

            grad[:,0] = ( torch.mul(tmp[...,1], g[...,0]) - torch.mul(tmp[...,0], g[...,1]) ).sum(axis=0)

            vec_fy = vec_fy.type(self.torch_dtype).contiguous()
            if Nbatch==1:
                self.plan_forward.execute(Holder(tmp), Holder(vec_fy))
            else:
                self.plan_forward_batch.execute(Holder(tmp), Holder(vec_fy))

            grad[:,1] = ( torch.mul(tmp[...,1], g[...,0]) - torch.mul(tmp[...,0], g[...,1]) ).sum(axis=0)

            vec_fz = vec_fz.type(self.torch_dtype).contiguous()
            if Nbatch==1:
                self.plan_forward.execute(Holder(tmp), Holder(vec_fz))
            else:
                self.plan_forward_batch.execute(Holder(tmp), Holder(vec_fz))

            grad[:,2] = ( torch.mul(tmp[...,1], g[...,0]) - torch.mul(tmp[...,0], g[...,1]) ).sum(axis=0)

            return grad
        else:
            raise Exception("Error: f should have 5 dimensions")

    def _backward_adjoint3D(self, y, g, xi):
        self.test_xi(xi)
        ndim = len(y.shape)
        if ndim==3:
            Nbatch = y.shape[0]

            vecx_grad_output = torch.mul(self.XX[None,:,:,:,None].contiguous(), g.contiguous())
            vecy_grad_output = torch.mul(self.XY[None,:,:,:,None].contiguous(), g.contiguous())
            vecz_grad_output = torch.mul(self.XZ[None,:,:,:,None].contiguous(), g.contiguous())

            grad = torch.zeros_like(xi)

            tmp = torch.zeros(Nbatch, self.K, 3, device=self.device, dtype=self.torch_dtype)
            vecx_grad_output = vecx_grad_output.type(self.torch_dtype).contiguous()
            if Nbatch==1:
                self.plan_forward.execute(Holder(tmp), Holder(vecx_grad_output))
            else:
                self.plan_forward_batch.execute(Holder(tmp), Holder(vecx_grad_output))

            grad[:,0] = ( torch.mul(tmp[...,1], y[...,0]) - torch.mul(tmp[...,0], y[...,1]) ).sum(axis=0)

            vecy_grad_output = vecy_grad_output.type(self.torch_dtype).contiguous()
            if Nbatch==1:
                self.plan_forward.execute(Holder(tmp), Holder(vecy_grad_output))
            else:
                self.plan_forward_batch.execute(Holder(tmp), Holder(vecy_grad_output))

            grad[:,1] = ( torch.mul(tmp[...,1], y[...,0]) - torch.mul(tmp[...,0], y[...,1]) ).sum(axis=0)

            vecz_grad_output = vecz_grad_output.type(self.torch_dtype).contiguous()
            if Nbatch==1:
                self.plan_forward.execute(Holder(tmp), Holder(vecz_grad_output))
            else:
                self.plan_forward_batch.execute(Holder(tmp), Holder(vecz_grad_output))

            grad[:,2] = ( torch.mul(tmp[...,1], y[...,0]) - torch.mul(tmp[...,0], y[...,1]) ).sum(axis=0)

            return grad
        else:
            raise Exception("Error: y should have 3 dimensions")


nufft=Nufft()



class FClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xi, f):
        ctx.save_for_backward(xi, f)
        output = nufft.forward(f, xi)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        xi, f = ctx.saved_tensors
        grad_input = nufft.backward_forward(f, grad_output, xi)
        grad_input_f = nufft.adjoint(grad_output, xi)
        return grad_input, grad_input_f

class FtClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xi, y):
        ctx.save_for_backward(xi, y)
        output = nufft.adjoint(y, xi)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        xi, y = ctx.saved_tensors
        grad_input = nufft.backward_adjoint(y, grad_output, xi)
        grad_input_y = nufft.forward(grad_output, xi)
        return grad_input, grad_input_y

forward = FClass.apply
adjoint = FtClass.apply
