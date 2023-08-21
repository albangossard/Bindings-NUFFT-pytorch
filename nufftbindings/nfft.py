# Author: Alban Gossard
# Last modification: 2021/22/09

import numpy as np
import torch
from pynfft.nfft import NFFT
from .basenufft import *


class Nufft(baseNUFFT):
    def _set_dims(self):
        xx = np.arange(self.nx)-self.nx/2.
        xy = np.arange(self.ny)-self.ny/2.
        if self.ndim==2:
            self.XX, self.XY = np.meshgrid(xx, xy)
        if self.ndim==3:
            xz = np.arange(self.nz)-self.nz/2.
            self.XX, self.XY, self.XZ = np.meshgrid(xx, xy, xz)
            self.XZ=self.XZ.T
        self.XX=self.XX.T
        self.XY=self.XY.T
        if self.ndim==2:
            self.plan = NFFT([self.nx, self.ny], self.K)
        elif self.ndim==3:
            self.plan = NFFT([self.nx, self.ny, self.nz], self.K)

    def precompute(self, xi):
        self.plan.x = (xi.data.cpu().numpy()/(2*np.pi)+0.5)%1-0.5
        self.plan.precompute()
        self.xiprecomputed = xi.clone()
        self.precomputedTrig = True
    def _forward_np(self, f):
        self.plan.f_hat = f
        y = self.plan.trafo()
        return y

    def _forward_simple2D(self, f, xi):
        self.test_xi(xi)
        iscpx = f.is_complex()
        if iscpx:
            fnp = f.data.cpu().numpy()
        else:
            fnp = f[:,:,0].data.cpu().numpy() + 1j*f[:,:,1].data.cpu().numpy()

        self.plan.f_hat = fnp
        ynp = self.plan.trafo()

        if iscpx:
            y = torch.zeros(self.K, dtype=self.torch_cpxdtype, device=self.device)
            y[:] = torch.tensor(ynp, dtype=self.torch_cpxdtype, device=self.device)
        else:
            y = torch.zeros(self.K, 2, dtype=self.torch_dtype, device=self.device)
            y[:,0] = torch.tensor(ynp.real, dtype=self.torch_dtype, device=self.device)
            y[:,1] = torch.tensor(ynp.imag, dtype=self.torch_dtype, device=self.device)
        return y
    def _adjoint_simple2D(self, y, xi):
        self.test_xi(xi)
        iscpx = y.is_complex()
        if iscpx:
            ynp = y.data.cpu().numpy()
        else:
            ynp = y[:,0].data.cpu().numpy() + 1j*y[:,1].data.cpu().numpy()

        self.plan.f = ynp
        fnp = self.plan.adjoint()

        if iscpx:
            f = torch.zeros(self.nx, self.ny, dtype=self.torch_cpxdtype, device=self.device)
            f[:,:] = torch.tensor(fnp, dtype=self.torch_cpxdtype, device=self.device)
        else:
            f = torch.zeros(self.nx, self.ny, 2, dtype=self.torch_dtype, device=self.device)
            f[:,:,0] = torch.tensor(fnp.real, dtype=self.torch_dtype, device=self.device)
            f[:,:,1] = torch.tensor(fnp.imag, dtype=self.torch_dtype, device=self.device)
        return f
    def _backward_forward_simple2D(self, f, g, xi):
        self.test_xi(xi)
        iscpx = f.is_complex()
        if iscpx:
            gnp = g.data.cpu().numpy()
            fnp = f.data.cpu().numpy()
        else:
            gnp = g[:,0].data.cpu().numpy() + 1j*g[:,1].data.cpu().numpy()
            fnp = f[:,:,0].data.cpu().numpy() + 1j*f[:,:,1].data.cpu().numpy()

        vec_fx = np.multiply(self.XX, fnp)
        vec_fy = np.multiply(self.XY, fnp)

        gradnp = np.zeros(xi.shape)
        tmp = self._forward_np(vec_fx)
        gradnp[:,0] = np.multiply(tmp.imag, gnp.real) - np.multiply(tmp.real, gnp.imag)
        tmp = self._forward_np(vec_fy)
        gradnp[:,1] = np.multiply(tmp.imag, gnp.real) - np.multiply(tmp.real, gnp.imag)

        grad = torch.tensor(gradnp, dtype=self.torch_dtype, device=self.device)
        return grad
    def _backward_adjoint_simple2D(self, y, g, xi):
        self.test_xi(xi)
        iscpx = y.is_complex()
        if iscpx:
            gnp = g.data.cpu().numpy()
            ynp = y.data.cpu().numpy()
        else:
            gnp = g[:,:,0].data.cpu().numpy() + 1j*g[:,:,1].data.cpu().numpy()
            ynp = y[:,0].data.cpu().numpy() + 1j*y[:,1].data.cpu().numpy()

        vecx_grad_output = np.multiply(self.XX, gnp)
        vecy_grad_output = np.multiply(self.XY, gnp)

        gradnp = np.zeros(xi.shape)
        tmp = self._forward_np(vecx_grad_output)
        gradnp[:,0] = np.multiply(tmp.imag, ynp.real) - np.multiply(tmp.real, ynp.imag)
        tmp = self._forward_np(vecy_grad_output)
        gradnp[:,1] = np.multiply(tmp.imag, ynp.real) - np.multiply(tmp.real, ynp.imag)

        grad = torch.tensor(gradnp, dtype=self.torch_dtype, device=self.device)
        return grad

    def _forward2D(self, f, xi):
        self.test_xi(xi)
        ndim = len(f.shape)
        iscpx = f.is_complex()
        if ndim==4 and not iscpx or ndim==3 and iscpx:
            Nbatch = f.shape[0]
            if iscpx:
                y = torch.zeros(Nbatch, self.K, dtype=self.torch_cpxdtype, device=self.device)
            else:
                y = torch.zeros(Nbatch, self.K, 2, dtype=self.torch_dtype, device=self.device)
            for n in range(Nbatch):
                if iscpx:
                    fnp = f[n].data.cpu().numpy()
                else:
                    fnp = f[n,:,:,0].data.cpu().numpy() + 1j*f[n,:,:,1].data.cpu().numpy()

                self.plan.f_hat = fnp
                ynp = self.plan.trafo()

                if iscpx:
                    y[n] = torch.tensor(ynp, dtype=self.torch_cpxdtype, device=self.device)
                else:
                    y[n,:,0] = torch.tensor(ynp.real, dtype=self.torch_dtype, device=self.device)
                    y[n,:,1] = torch.tensor(ynp.imag, dtype=self.torch_dtype, device=self.device)
            return y
        elif ndim==3 and not iscpx or ndim==2 and iscpx:
            return self._forward_simple2D(f, xi)
        else:
            raise Exception("Error: f should have 2, 3 or 4 dimensions (batch mode)")
    def _adjoint2D(self, y, xi):
        self.test_xi(xi)
        ndim = len(y.shape)
        iscpx = y.is_complex()
        if ndim==3 and not iscpx or ndim==2 and iscpx:
            Nbatch = y.shape[0]
            if iscpx:
                f = torch.zeros(Nbatch, self.nx, self.ny, dtype=self.torch_cpxdtype, device=self.device)
            else:
                f = torch.zeros(Nbatch, self.nx, self.ny, 2, dtype=self.torch_dtype, device=self.device)
            for n in range(Nbatch):
                if iscpx:
                    ynp = y[n].data.cpu().numpy()
                else:
                    ynp = y[n,:,0].data.cpu().numpy() + 1j*y[n,:,1].data.cpu().numpy()

                self.plan.f = ynp
                fnp = self.plan.adjoint()

                if iscpx:
                    f[n] = torch.tensor(fnp, dtype=self.torch_cpxdtype, device=self.device)
                else:
                    f[n,:,:,0] = torch.tensor(fnp.real, dtype=self.torch_dtype, device=self.device)
                    f[n,:,:,1] = torch.tensor(fnp.imag, dtype=self.torch_dtype, device=self.device)
            return f
        elif ndim==2 and not iscpx or ndim==1 and iscpx:
            return self._adjoint_simple2D(y, xi)
        else:
            raise Exception("Error: y should have 1, 2 or 3 dimensions (batch mode)")
    def _backward_forward2D(self, f, g, xi):
        self.test_xi(xi)
        ndim = len(f.shape)
        iscpx = f.is_complex()
        if ndim==4 and not iscpx or ndim==3 and iscpx:
            Nbatch = f.shape[0]
            gradnp = np.zeros(xi.shape)
            grad = torch.zeros(self.K, 2, dtype=self.torch_dtype, device=self.device)
            for n in range(Nbatch):
                if iscpx:
                    gnp = g[n].data.cpu().numpy()
                    fnp = f[n].data.cpu().numpy()
                else:
                    gnp = g[n,:,0].data.cpu().numpy() + 1j*g[n,:,1].data.cpu().numpy()
                    fnp = f[n,:,:,0].data.cpu().numpy() + 1j*f[n,:,:,1].data.cpu().numpy()

                vec_fx = np.multiply(self.XX, fnp)
                vec_fy = np.multiply(self.XY, fnp)

                tmp = self._forward_np(vec_fx)
                gradnp[:,0] = np.multiply(tmp.imag, gnp.real) - np.multiply(tmp.real, gnp.imag)
                tmp = self._forward_np(vec_fy)
                gradnp[:,1] = np.multiply(tmp.imag, gnp.real) - np.multiply(tmp.real, gnp.imag)

                grad += torch.tensor(gradnp, dtype=self.torch_dtype, device=self.device)
            return grad
        elif ndim==3 and not iscpx or ndim==2 and iscpx:
            return self._backward_forward_simple2D(f, g, xi)
        else:
            raise Exception("Error: f should have 2, 3 or 4 dimensions (batch mode)")
    def _backward_adjoint2D(self, y, g, xi):
        self.test_xi(xi)
        ndim = len(y.shape)
        iscpx = y.is_complex()
        if ndim==3 and not iscpx or ndim==2 and iscpx:
            Nbatch = y.shape[0]
            gradnp = np.zeros(xi.shape)
            grad = torch.zeros(self.K, 2, dtype=self.torch_dtype, device=self.device)
            for n in range(Nbatch):
                if iscpx:
                    gnp = g[n].data.cpu().numpy()
                    ynp = y[n].data.cpu().numpy()
                else:
                    gnp = g[n,:,:,0].data.cpu().numpy() + 1j*g[n,:,:,1].data.cpu().numpy()
                    ynp = y[n,:,0].data.cpu().numpy() + 1j*y[n,:,1].data.cpu().numpy()

                vecx_grad_output = np.multiply(self.XX, gnp)
                vecy_grad_output = np.multiply(self.XY, gnp)

                tmp = self._forward_np(vecx_grad_output)
                gradnp[:,0] = np.multiply(tmp.imag, ynp.real) - np.multiply(tmp.real, ynp.imag)
                tmp = self._forward_np(vecy_grad_output)
                gradnp[:,1] = np.multiply(tmp.imag, ynp.real) - np.multiply(tmp.real, ynp.imag)

                grad += torch.tensor(gradnp, dtype=self.torch_dtype, device=self.device)
            return grad
        elif ndim==2 and not iscpx or ndim==1 and iscpx:
            return self._backward_adjoint_simple2D(y, g, xi)
        else:
            raise Exception("Error: y should have 1, 2 or 3 dimensions (batch mode)")

    def _forward_simple3D(self, f, xi):
        self.test_xi(xi)
        fnp = f[:,:,:,0].data.cpu().numpy() + 1j*f[:,:,:,1].data.cpu().numpy()

        self.plan.f_hat = fnp
        ynp = self.plan.trafo()

        y = torch.zeros(self.K, 2, dtype=self.torch_dtype, device=self.device)
        y[:,0] = torch.tensor(ynp.real, dtype=self.torch_dtype, device=self.device)
        y[:,1] = torch.tensor(ynp.imag, dtype=self.torch_dtype, device=self.device)
        return y
    def _adjoint_simple_3D(self, y, xi):
        self.test_xi(xi)
        ynp = y[:,0].data.cpu().numpy() + 1j*y[:,1].data.cpu().numpy()

        self.plan.f = ynp
        fnp = self.plan.adjoint()

        f = torch.zeros(self.nx, self.ny, self.nz, 2, dtype=self.torch_dtype, device=self.device)
        f[:,:,:,0] = torch.tensor(fnp.real, dtype=self.torch_dtype, device=self.device)
        f[:,:,:,1] = torch.tensor(fnp.imag, dtype=self.torch_dtype, device=self.device)
        return f
    def _backward_forward_simple3D(self, f, g, xi):
        self.test_xi(xi)
        gnp = g[:,0].data.cpu().numpy() + 1j*g[:,1].data.cpu().numpy()
        fnp = f[:,:,:,0].data.cpu().numpy() + 1j*f[:,:,:,1].data.cpu().numpy()

        vec_fx = np.multiply(self.XX, fnp)
        vec_fy = np.multiply(self.XY, fnp)
        vec_fz = np.multiply(self.XZ, fnp)

        gradnp = np.zeros(xi.shape)
        tmp = self._forward_np(vec_fx)
        gradnp[:,0] = np.multiply(tmp.imag, gnp.real) - np.multiply(tmp.real, gnp.imag)
        tmp = self._forward_np(vec_fy)
        gradnp[:,1] = np.multiply(tmp.imag, gnp.real) - np.multiply(tmp.real, gnp.imag)
        tmp = self._forward_np(vec_fz)
        gradnp[:,2] = np.multiply(tmp.imag, gnp.real) - np.multiply(tmp.real, gnp.imag)

        grad = torch.tensor(gradnp, dtype=self.torch_dtype, device=self.device)
        return grad
    def _backward_adjoint_simple3D(self, y, g, xi):
        self.test_xi(xi)
        gnp = g[:,:,:,0].data.cpu().numpy() + 1j*g[:,:,:,1].data.cpu().numpy()
        ynp = y[:,0].data.cpu().numpy() + 1j*y[:,1].data.cpu().numpy()

        vecx_grad_output = np.multiply(self.XX, gnp)
        vecy_grad_output = np.multiply(self.XY, gnp)
        vecz_grad_output = np.multiply(self.XZ, gnp)

        gradnp = np.zeros(xi.shape)
        tmp = self._forward_np(vecx_grad_output)
        gradnp[:,0] = np.multiply(tmp.imag, ynp.real) - np.multiply(tmp.real, ynp.imag)
        tmp = self._forward_np(vecy_grad_output)
        gradnp[:,1] = np.multiply(tmp.imag, ynp.real) - np.multiply(tmp.real, ynp.imag)
        tmp = self._forward_np(vecz_grad_output)
        gradnp[:,2] = np.multiply(tmp.imag, ynp.real) - np.multiply(tmp.real, ynp.imag)

        grad = torch.tensor(gradnp, dtype=self.torch_dtype, device=self.device)
        return grad

    def _forward3D(self, f, xi):
        self.test_xi(xi)
        ndim = len(f.shape)
        if ndim==5:
            Nbatch = f.shape[0]
            y = torch.zeros(Nbatch, self.K, 2, dtype=self.torch_dtype, device=self.device)
            for n in range(Nbatch):
                fnp = f[n,:,:,:,0].data.cpu().numpy() + 1j*f[n,:,:,:,1].data.cpu().numpy()

                self.plan.f_hat = fnp
                ynp = self.plan.trafo()

                y[n,:,0] = torch.tensor(ynp.real, dtype=self.torch_dtype, device=self.device)
                y[n,:,1] = torch.tensor(ynp.imag, dtype=self.torch_dtype, device=self.device)
            return y
        elif ndim==4:
            return self._forward_simple3D(f, xi)
        else:
            raise Exception("Error: f should have 4 or 5 dimensions (batch mode)")
    def _adjoint3D(self, y, xi):
        self.test_xi(xi)
        ndim = len(y.shape)
        if ndim==3:
            Nbatch = y.shape[0]
            f = torch.zeros(Nbatch, self.nx, self.ny, self.nz, 2, dtype=self.torch_dtype, device=self.device)
            for n in range(Nbatch):
                ynp = y[n,:,0].data.cpu().numpy() + 1j*y[n,:,1].data.cpu().numpy()

                self.plan.f = ynp
                fnp = self.plan.adjoint()

                f[n,:,:,:,0] = torch.tensor(fnp.real, dtype=self.torch_dtype, device=self.device)
                f[n,:,:,:,1] = torch.tensor(fnp.imag, dtype=self.torch_dtype, device=self.device)
            return f
        elif ndim==2:
            return self._adjoint_simple3D(y, xi)
        else:
            raise Exception("Error: y should have 2 or 3 dimensions (batch mode)")
    def _backward_forward3D(self, f, g, xi):
        self.test_xi(xi)
        ndim = len(f.shape)
        if ndim==5:
            Nbatch = f.shape[0]
            gradnp = np.zeros(xi.shape)
            grad = torch.zeros(self.K, 3, dtype=self.torch_dtype, device=self.device)
            for n in range(Nbatch):
                gnp = g[n,:,0].data.cpu().numpy() + 1j*g[n,:,1].data.cpu().numpy()
                fnp = f[n,:,:,:,0].data.cpu().numpy() + 1j*f[n,:,:,:,1].data.cpu().numpy()

                vec_fx = np.multiply(self.XX, fnp)
                vec_fy = np.multiply(self.XY, fnp)
                vec_fz = np.multiply(self.XZ, fnp)

                tmp = self._forward_np(vec_fx)
                gradnp[:,0] = np.multiply(tmp.imag, gnp.real) - np.multiply(tmp.real, gnp.imag)
                tmp = self._forward_np(vec_fy)
                gradnp[:,1] = np.multiply(tmp.imag, gnp.real) - np.multiply(tmp.real, gnp.imag)
                tmp = self._forward_np(vec_fz)
                gradnp[:,2] = np.multiply(tmp.imag, gnp.real) - np.multiply(tmp.real, gnp.imag)

                grad += torch.tensor(gradnp, dtype=self.torch_dtype, device=self.device)
            return grad
        elif ndim==4:
            return self._backward_forward_simple3D(f, g, xi)
        else:
            raise Exception("Error: f should have 3 or 4 dimensions (batch mode)")
    def _backward_adjoint3D(self, y, g, xi):
        self.test_xi(xi)
        ndim = len(y.shape)
        if ndim==3:
            Nbatch = y.shape[0]
            gradnp = np.zeros(xi.shape)
            grad = torch.zeros(self.K, 3, dtype=self.torch_dtype, device=self.device)
            for n in range(Nbatch):
                gnp = g[n,:,:,:,0].data.cpu().numpy() + 1j*g[n,:,:,:,1].data.cpu().numpy()
                ynp = y[n,:,0].data.cpu().numpy() + 1j*y[n,:,1].data.cpu().numpy()

                vecx_grad_output = np.multiply(self.XX, gnp)
                vecy_grad_output = np.multiply(self.XY, gnp)
                vecz_grad_output = np.multiply(self.XZ, gnp)

                tmp = self._forward_np(vecx_grad_output)
                gradnp[:,0] = np.multiply(tmp.imag, ynp.real) - np.multiply(tmp.real, ynp.imag)
                tmp = self._forward_np(vecy_grad_output)
                gradnp[:,1] = np.multiply(tmp.imag, ynp.real) - np.multiply(tmp.real, ynp.imag)
                tmp = self._forward_np(vecz_grad_output)
                gradnp[:,2] = np.multiply(tmp.imag, ynp.real) - np.multiply(tmp.real, ynp.imag)

                grad += torch.tensor(gradnp, dtype=self.torch_dtype, device=self.device)
            return grad
        elif ndim==2:
            return self._backward_adjoint_simple3D(y, g, xi)
        else:
            raise Exception("Error: y should have 2 or 3 dimensions (batch mode)")


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
