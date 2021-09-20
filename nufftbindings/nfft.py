import numpy as np
import torch
from pynfft.nfft import NFFT
from nufftbindings.basenufft import *


class Nufft(baseNUFFT):
    def _set_dims(self):
        xx = np.arange(self.nx)-self.nx/2.
        xy = np.arange(self.ny)-self.ny/2.
        self.XX, self.XY = np.meshgrid(xx, xy)
        self.XX=self.XX.T
        self.XY=self.XY.T
        self.plan = NFFT([self.nx, self.ny], self.K)

    def precompute(self, xi):
        self.plan.x = (xi.data.cpu().numpy()/(2*np.pi)+0.5)%1-0.5
        self.plan.precompute()
        self.precomputedTrig = True
    def _forward_np(self, f):
        self.plan.f_hat = f
        y = self.plan.trafo()
        return y
    def _forward_simple(self, f, xi):
        self.test_xi(xi)
        fnp = f[:,:,0].data.cpu().numpy() + 1j*f[:,:,1].data.cpu().numpy()

        self.plan.f_hat = fnp
        ynp = self.plan.trafo()

        y = torch.zeros(self.K, 2, dtype=self.torch_dtype, device=self.device)
        y[:,0] = torch.tensor(ynp.real, dtype=self.torch_dtype, device=self.device)
        y[:,1] = torch.tensor(ynp.imag, dtype=self.torch_dtype, device=self.device)
        return y
    def _adjoint_simple(self, y, xi):
        self.test_xi(xi)
        ynp = y[:,0].data.cpu().numpy() + 1j*y[:,1].data.cpu().numpy()

        self.plan.f = ynp
        fnp = self.plan.adjoint()

        f = torch.zeros(self.nx, self.ny, 2, dtype=self.torch_dtype, device=self.device)
        f[:,:,0] = torch.tensor(fnp.real, dtype=self.torch_dtype, device=self.device)
        f[:,:,1] = torch.tensor(fnp.imag, dtype=self.torch_dtype, device=self.device)
        return f
    def _backward_forward_simple(self, f, g, xi):
        self.test_xi(xi)
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
    def _backward_adjoint_simple(self, y, g, xi):
        self.test_xi(xi)
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

    def forward(self, f, xi):
        self.test_xi(xi)
        ndim = len(f.shape)
        if ndim==4:
            Nbatch = f.shape[0]
            y = torch.zeros(Nbatch, self.K, 2, dtype=self.torch_dtype, device=self.device)
            for n in range(Nbatch):
                fnp = f[n,:,:,0].data.cpu().numpy() + 1j*f[n,:,:,1].data.cpu().numpy()

                self.plan.f_hat = fnp
                ynp = self.plan.trafo()

                y[n,:,0] = torch.tensor(ynp.real, dtype=self.torch_dtype, device=self.device)
                y[n,:,1] = torch.tensor(ynp.imag, dtype=self.torch_dtype, device=self.device)
            return y
        elif ndim==3:
            return self._forward_simple(f, xi)
        else:
            raise Exception("Error: f should have 3 or 4 dimensions (batch mode)")
    def adjoint(self, y, xi):
        self.test_xi(xi)
        ndim = len(y.shape)
        if ndim==3:
            Nbatch = y.shape[0]
            f = torch.zeros(Nbatch, self.nx, self.ny, 2, dtype=self.torch_dtype, device=self.device)
            for n in range(Nbatch):
                ynp = y[n,:,0].data.cpu().numpy() + 1j*y[n,:,1].data.cpu().numpy()

                self.plan.f = ynp
                fnp = self.plan.adjoint()

                f[n,:,:,0] = torch.tensor(fnp.real, dtype=self.torch_dtype, device=self.device)
                f[n,:,:,1] = torch.tensor(fnp.imag, dtype=self.torch_dtype, device=self.device)
            return f
        elif ndim==2:
            return self._adjoint_simple(y, xi)
        else:
            raise Exception("Error: y should have 2 or 3 dimensions (batch mode)")
    def backward_forward(self, f, g, xi):
        self.test_xi(xi)
        ndim = len(f.shape)
        if ndim==4:
            Nbatch = f.shape[0]
            gradnp = np.zeros(xi.shape)
            grad = torch.zeros(self.K, 2, dtype=self.torch_dtype, device=self.device)
            for n in range(Nbatch):
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
        elif ndim==3:
            return self._backward_forward_simple(f, g, xi)
        else:
            raise Exception("Error: f should have 3 or 4 dimensions (batch mode)")
    def backward_adjoint(self, y, g, xi):
        self.test_xi(xi)
        ndim = len(y.shape)
        if ndim==3:
            Nbatch = y.shape[0]
            gradnp = np.zeros(xi.shape)
            grad = torch.zeros(self.K, 2, dtype=self.torch_dtype, device=self.device)
            for n in range(Nbatch):
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
        elif ndim==2:
            return self._backward_adjoint_simple(y, g, xi)
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
