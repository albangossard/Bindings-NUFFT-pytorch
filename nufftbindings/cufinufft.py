# Author: Alban Gossard
# Last modification: 2021/22/09

import numpy as np
import torch
import pycuda.autoinit
from pycuda.gpuarray import to_gpu
from cufinufft import cufinufft
from nufftbindings.basenufft import *

# Note: in order for cuFINUFFT to work with pytorch tensor (and not GPUArray from pycuda), we have to do the following changes in the 'execute' method in lib/python3.XX/site-packages/cufinufft/cufinufft.py:
# Comment the test ```if not c.dtype == fk.dtype == self.complex_dtype:```
# Replace c.ptr by c and fk.ptr by fk in ier = self._exec_plan(c.ptr, fk.ptr, self.plan)


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
        xinp = xi.detach().cpu().numpy()

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

            self.plan_forward.set_pts(to_gpu(xinp[:,0].astype(self.np_dtype)), to_gpu(xinp[:,1].astype(self.np_dtype)))
            self.plan_adjoint.set_pts(to_gpu(xinp[:,0].astype(self.np_dtype)), to_gpu(xinp[:,1].astype(self.np_dtype)))
            self.plan_forward_batch.set_pts(to_gpu(xinp[:,0].astype(self.np_dtype)), to_gpu(xinp[:,1].astype(self.np_dtype)))
            self.plan_adjoint_batch.set_pts(to_gpu(xinp[:,0].astype(self.np_dtype)), to_gpu(xinp[:,1].astype(self.np_dtype)))
        elif self.ndim==3:
            self.plan_forward = cufinufft(2, (self.nx, self.ny, self.nz), 1, eps=self.eps, dtype=self.np_dtype)
            self.plan_adjoint = cufinufft(1, (self.nx, self.ny, self.nz), 1, eps=self.eps, dtype=self.np_dtype)
            self.plan_forward_batch = cufinufft(2, (self.nx, self.ny, self.nz), self.Nbatch, eps=self.eps, dtype=self.np_dtype)
            self.plan_adjoint_batch = cufinufft(1, (self.nx, self.ny, self.nz), self.Nbatch, eps=self.eps, dtype=self.np_dtype)

            self.plan_forward.set_pts(to_gpu(xinp[:,0].astype(self.np_dtype)), to_gpu(xinp[:,1].astype(self.np_dtype)), to_gpu(xinp[:,2].astype(self.np_dtype)))
            self.plan_adjoint.set_pts(to_gpu(xinp[:,0].astype(self.np_dtype)), to_gpu(xinp[:,1].astype(self.np_dtype)), to_gpu(xinp[:,2].astype(self.np_dtype)))
            self.plan_forward_batch.set_pts(to_gpu(xinp[:,0].astype(self.np_dtype)), to_gpu(xinp[:,1].astype(self.np_dtype)), to_gpu(xinp[:,2].astype(self.np_dtype)))
            self.plan_adjoint_batch.set_pts(to_gpu(xinp[:,0].astype(self.np_dtype)), to_gpu(xinp[:,1].astype(self.np_dtype)), to_gpu(xinp[:,2].astype(self.np_dtype)))

        self.precomputedTrig = True

    def _forward2D(self, f, xi):
        self.test_xi(xi)
        self.test_f(f)
        ndim = len(f.shape)
        if ndim==4:
            Nbatch = f.shape[0]
            y = torch.zeros(Nbatch, self.K, 2, device=self.device, dtype=self.torch_dtype)
            fcpx = f.type(self.torch_dtype).contiguous()
            if Nbatch==1:
                self.plan_forward.execute(y.data_ptr(), fcpx.data_ptr())
            else:
                self.plan_forward_batch.execute(y.data_ptr(), fcpx.data_ptr())
            return y
        else:
            raise Exception("Error: f should have 4 dimensions")
    def _adjoint2D(self, y, xi):
        self.test_xi(xi)
        self.test_f(y)
        ndim = len(y.shape)
        if ndim==3:
            Nbatch = y.shape[0]
            f = torch.zeros(Nbatch, self.nx, self.ny, 2, device=self.device, dtype=self.torch_dtype)
            ycpx = y.type(self.torch_dtype).contiguous()
            if Nbatch==1:
                self.plan_adjoint.execute(ycpx.data_ptr(), f.data_ptr())
            else:
                self.plan_adjoint_batch.execute(ycpx.data_ptr(), f.data_ptr())
            return f
        else:
            raise Exception("Error: y should have 3 dimensions")
    def _backward_forward2D(self, f, g, xi):
        self.test_xi(xi)
        ndim = len(f.shape)
        if ndim==4:
            Nbatch = f.shape[0]

            vec_fx = torch.mul(self.XX[None,:,:,None].contiguous(), f.contiguous())
            vec_fy = torch.mul(self.XY[None,:,:,None].contiguous(), f.contiguous())

            grad = torch.zeros_like(xi)

            tmp = torch.zeros(Nbatch, self.K, 2, device=self.device, dtype=self.torch_dtype)
            vec_fx = vec_fx.type(self.torch_dtype).contiguous()
            if Nbatch==1:
                self.plan_forward.execute(tmp.data_ptr(), vec_fx.data_ptr())
            else:
                self.plan_forward_batch.execute(tmp.data_ptr(), vec_fx.data_ptr())

            grad[:,0] = ( torch.mul(tmp[...,1], g[...,0]) - torch.mul(tmp[...,0], g[...,1]) ).sum(axis=0)

            vec_fy = vec_fy.type(self.torch_dtype).contiguous()
            if Nbatch==1:
                self.plan_forward.execute(tmp.data_ptr(), vec_fy.data_ptr())
            else:
                self.plan_forward_batch.execute(tmp.data_ptr(), vec_fy.data_ptr())

            grad[:,1] = ( torch.mul(tmp[...,1], g[...,0]) - torch.mul(tmp[...,0], g[...,1]) ).sum(axis=0)

            return grad
        else:
            raise Exception("Error: f should have 4 dimensions")

    def _backward_adjoint2D(self, y, g, xi):
        self.test_xi(xi)
        ndim = len(y.shape)
        if ndim==3:
            Nbatch = y.shape[0]

            vecx_grad_output = torch.mul(self.XX[None,:,:,None].contiguous(), g.contiguous())
            vecy_grad_output = torch.mul(self.XY[None,:,:,None].contiguous(), g.contiguous())

            grad = torch.zeros_like(xi)

            tmp = torch.zeros(Nbatch, self.K, 2, device=self.device, dtype=self.torch_dtype)
            vecx_grad_output = vecx_grad_output.type(self.torch_dtype).contiguous()
            if Nbatch==1:
                self.plan_forward.execute(tmp.data_ptr(), vecx_grad_output.data_ptr())
            else:
                self.plan_forward_batch.execute(tmp.data_ptr(), vecx_grad_output.data_ptr())

            grad[:,0] = ( torch.mul(tmp[...,1], y[...,0]) - torch.mul(tmp[...,0], y[...,1]) ).sum(axis=0)

            vecy_grad_output = vecy_grad_output.type(self.torch_dtype).contiguous()
            if Nbatch==1:
                self.plan_forward.execute(tmp.data_ptr(), vecy_grad_output.data_ptr())
            else:
                self.plan_forward_batch.execute(tmp.data_ptr(), vecy_grad_output.data_ptr())

            grad[:,1] = ( torch.mul(tmp[...,1], y[...,0]) - torch.mul(tmp[...,0], y[...,1]) ).sum(axis=0)

            return grad
        else:
            raise Exception("Error: y should have 3 dimensions")

    def _forward3D(self, f, xi):
        self.test_xi(xi)
        self.test_f(f)
        ndim = len(f.shape)
        if ndim==5:
            Nbatch = f.shape[0]
            y = torch.zeros(Nbatch, self.K, 2, device=self.device, dtype=self.torch_dtype)
            fcpx = f.type(self.torch_dtype).contiguous()
            if Nbatch==1:
                self.plan_forward.execute(y.data_ptr(), fcpx.data_ptr())
            else:
                self.plan_forward_batch.execute(y.data_ptr(), fcpx.data_ptr())
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
                self.plan_adjoint.execute(ycpx.data_ptr(), f.data_ptr())
            else:
                self.plan_adjoint_batch.execute(ycpx.data_ptr(), f.data_ptr())
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
                self.plan_forward.execute(tmp.data_ptr(), vec_fx.data_ptr())
            else:
                self.plan_forward_batch.execute(tmp.data_ptr(), vec_fx.data_ptr())

            grad[:,0] = ( torch.mul(tmp[...,1], g[...,0]) - torch.mul(tmp[...,0], g[...,1]) ).sum(axis=0)

            vec_fy = vec_fy.type(self.torch_dtype).contiguous()
            if Nbatch==1:
                self.plan_forward.execute(tmp.data_ptr(), vec_fy.data_ptr())
            else:
                self.plan_forward_batch.execute(tmp.data_ptr(), vec_fy.data_ptr())

            grad[:,1] = ( torch.mul(tmp[...,1], g[...,0]) - torch.mul(tmp[...,0], g[...,1]) ).sum(axis=0)

            vec_fz = vec_fz.type(self.torch_dtype).contiguous()
            if Nbatch==1:
                self.plan_forward.execute(tmp.data_ptr(), vec_fz.data_ptr())
            else:
                self.plan_forward_batch.execute(tmp.data_ptr(), vec_fz.data_ptr())

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
                self.plan_forward.execute(tmp.data_ptr(), vecx_grad_output.data_ptr())
            else:
                self.plan_forward_batch.execute(tmp.data_ptr(), vecx_grad_output.data_ptr())

            grad[:,0] = ( torch.mul(tmp[...,1], y[...,0]) - torch.mul(tmp[...,0], y[...,1]) ).sum(axis=0)

            vecy_grad_output = vecy_grad_output.type(self.torch_dtype).contiguous()
            if Nbatch==1:
                self.plan_forward.execute(tmp.data_ptr(), vecy_grad_output.data_ptr())
            else:
                self.plan_forward_batch.execute(tmp.data_ptr(), vecy_grad_output.data_ptr())

            grad[:,1] = ( torch.mul(tmp[...,1], y[...,0]) - torch.mul(tmp[...,0], y[...,1]) ).sum(axis=0)

            vecz_grad_output = vecz_grad_output.type(self.torch_dtype).contiguous()
            if Nbatch==1:
                self.plan_forward.execute(tmp.data_ptr(), vecz_grad_output.data_ptr())
            else:
                self.plan_forward_batch.execute(tmp.data_ptr(), vecz_grad_output.data_ptr())

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
