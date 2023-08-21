# Author: Alban Gossard
# Last modification: 2022/23/08

import torch
import torchkbnufft as tkbn
from .basenufft import *


class Nufft(baseNUFFT):
    def _set_dims(self):
        if self.ndim==2:
            self.XX, self.XY = torch.meshgrid(self.xx, self.xy)
            self.zx = self.xx.view(-1,1).repeat(1, self.nx).view(-1,1)
            self.zy = self.xy.view(1,-1).repeat(self.ny, 1).view(-1,1)
            self.nufft_ob = tkbn.KbNufft((self.nx,self.ny)).to(self.device)
            self.nufft_adj_ob = tkbn.KbNufftAdjoint((self.nx,self.ny)).to(self.device)
        elif self.ndim==3:
            self.XX, self.XY, self.XZ = torch.meshgrid(self.xx, self.xy, self.xz)
            self.zx = self.xx.view(-1,1,1).repeat(1, self.nx, self.nx).view(-1,1)
            self.zy = self.xy.view(1,-1,1).repeat(self.ny, 1, self.ny).view(-1,1)
            self.zz = self.xy.view(1,1,-1).repeat(self.nz, self.nz, 1).view(-1,1)
            self.nufft_ob = tkbn.KbNufft((self.nx,self.ny,self.nz)).to(self.device)
            self.nufft_adj_ob = tkbn.KbNufftAdjoint((self.nx,self.ny,self.nz)).to(self.device)

    def precompute(self, xi):
        self.xiprecomputed = xi.clone()
        self.precomputedTrig = True

    def _forward2D(self, f, xi):
        self.test_xi(xi)
        ndim = len(f.shape)
        iscpx = f.is_complex()
        if ndim != 4 and not iscpx or ndim != 3 and iscpx:
            raise Exception("Error: f should have 4 dimensions: batch, nx, ny, r/i or 3 dimensions: batch, nx, ny (complex dtype)")
        if iscpx:
            f = f[:,None].type(self.torch_cpxdtype) # batch,nx,ny
        else:
            f = f[:,None].type(self.torch_dtype) # batch,nx,ny,r/i
        xi = xi.permute(1,0).type(self.torch_dtype)
        y = self.nufft_ob(f, xi)[:,0]
        return y
    def _adjoint2D(self, y, xi):
        self.test_xi(xi)
        ndim = len(y.shape)
        iscpx = y.is_complex()
        if ndim != 3 and not iscpx or ndim != 2 and iscpx:
            raise Exception("Error: y should have 3 dimensions: batch, K, r/i or 2 dimensions: batch, K (complex dtype)")
        if iscpx:
            y = y[:,None].type(self.torch_cpxdtype) # batch,K
        else:
            y = y[:,None].type(self.torch_dtype) # batch,K,r/i
        xi = xi.permute(1,0).type(self.torch_dtype)
        f = self.nufft_adj_ob(y, xi)[:,0]
        return f
    def _backward_forward2D(self, f, g, xi):
        self.test_xi(xi)
        ndim = len(f.shape)
        iscpx = f.is_complex()
        grad = torch.zeros(xi.shape, dtype=self.torch_dtype, device=self.device)
        if ndim != 4 and not iscpx or ndim != 3 and iscpx:
            raise Exception("Error: f should have 4 dimensions: batch, nx, ny, r/i or 3 dimensions: batch, nx, ny (complex dtype)")
        if iscpx:
            f = f[:,None].type(self.torch_cpxdtype) # batch,nx,ny
            xi = xi.permute(1,0).type(self.torch_dtype)
            g = g[:,None].type(self.torch_cpxdtype) # batch,K
            #                          batch,coil,nx,ny
            vec_fx = torch.mul(self.XX[None,None], f)
            vec_fy = torch.mul(self.XY[None,None], f)

            tmp = self.nufft_ob(vec_fx, xi)[:,0]
            grad[:,0] = ( torch.mul(tmp.imag, g[:,0].real) - torch.mul(tmp.real, g[:,0].imag) ).sum(axis=0)
            tmp = self.nufft_ob(vec_fy, xi)[:,0]
            grad[:,1] = ( torch.mul(tmp.imag, g[:,0].real) - torch.mul(tmp.real, g[:,0].imag) ).sum(axis=0)
        else:
            f = f[:,None].type(self.torch_dtype) # batch,nx,ny,r/i
            xi = xi.permute(1,0).type(self.torch_dtype)
            g = g[:,None].type(self.torch_dtype) # batch,K,r/i
            #                          batch,coil,nx,ny,r/i
            vec_fx = torch.mul(self.XX[None,None,...,None], f)
            vec_fy = torch.mul(self.XY[None,None,...,None], f)

            tmp = self.nufft_ob(vec_fx, xi)[:,0]
            grad[:,0] = ( torch.mul(tmp[...,1], g[:,0,...,0]) - torch.mul(tmp[...,0], g[:,0,...,1]) ).sum(axis=0)
            tmp = self.nufft_ob(vec_fy, xi)[:,0]
            grad[:,1] = ( torch.mul(tmp[...,1], g[:,0,...,0]) - torch.mul(tmp[...,0], g[:,0,...,1]) ).sum(axis=0)

        return grad
    def _backward_adjoint2D(self, y, g, xi):
        self.test_xi(xi)
        ndim = len(y.shape)
        iscpx = y.is_complex()
        grad = torch.zeros(xi.shape, dtype=self.torch_dtype, device=self.device)
        if ndim != 3 and not iscpx or ndim != 2 and iscpx:
            raise Exception("Error: y should have 3 dimensions: batch, K, r/i or 2 dimensions: batch, K (complex dtype)")
        if iscpx:
            y = y[:,None].type(self.torch_cpxdtype) # batch,K
            xi = xi.permute(1,0).type(self.torch_dtype)
            g = g[:,None].type(self.torch_cpxdtype) # batch,nx,ny
            #                          batch,coil,nx,ny
            vecx_grad_output = torch.mul(self.XX[None,None], g)
            vecy_grad_output = torch.mul(self.XY[None,None], g)

            tmp = self.nufft_ob(vecx_grad_output, xi)[:,0]
            grad[:,0] = ( torch.mul(tmp.imag, y[:,0].real) - torch.mul(tmp.real, y[:,0].imag) ).sum(axis=0)
            tmp = self.nufft_ob(vecy_grad_output, xi)[:,0]
            grad[:,1] = ( torch.mul(tmp.imag, y[:,0].real) - torch.mul(tmp.real, y[:,0].imag) ).sum(axis=0)
        else:
            y = y[:,None].type(self.torch_dtype) # batch,K,r/i
            xi = xi.permute(1,0).type(self.torch_dtype)
            g = g[:,None].type(self.torch_dtype) # batch,nx,ny,r/i
            #                          batch,coil,nx,ny,r/i
            vecx_grad_output = torch.mul(self.XX[None,None,...,None], g)
            vecy_grad_output = torch.mul(self.XY[None,None,...,None], g)

            tmp = self.nufft_ob(vecx_grad_output, xi)[:,0]
            grad[:,0] = ( torch.mul(tmp[...,1], y[:,0,...,0]) - torch.mul(tmp[...,0], y[:,0,...,1]) ).sum(axis=0)
            tmp = self.nufft_ob(vecy_grad_output, xi)[:,0]
            grad[:,1] = ( torch.mul(tmp[...,1], y[:,0,...,0]) - torch.mul(tmp[...,0], y[:,0,...,1]) ).sum(axis=0)

        return grad

    def _forward3D(self, f, xi):
        self.test_xi(xi)
        ndim = len(f.shape)
        if ndim != 5:
            raise Exception("Error: f should have 5 dimensions: batch, nx, ny, nz, r/i")
        f = f[:,None].type(self.torch_dtype) # batch,nx,ny,nz,r/i
        xi = xi.permute(1,0).type(self.torch_dtype)
        y = self.nufft_ob(f, xi)[:,0]
        return y
    def _adjoint3D(self, y, xi):
        self.test_xi(xi)
        ndim = len(y.shape)
        if ndim != 3:
            raise Exception("Error: y should have 3 dimensions: batch, K, r/i")
        y = y[:,None].type(self.torch_dtype) # batch,K,r/i
        xi = xi.permute(1,0).type(self.torch_dtype)
        f = self.nufft_adj_ob(y, xi)[:,0]
        return f
    def _backward_forward3D(self, f, g, xi):
        self.test_xi(xi)
        ndim = len(f.shape)
        grad = torch.zeros(xi.shape, dtype=self.torch_dtype, device=self.device)
        if ndim != 5:
            raise Exception("Error: f should have 5 dimensions: batch, nx, ny, nz, r/i")
        f = f[:,None].type(self.torch_dtype) # batch,nx,ny,nz,r/i
        xi = xi.permute(1,0).type(self.torch_dtype)
        g = g[:,None].type(self.torch_dtype) # batch,K,r/i
        #                          batch,coil,nx,ny,nz,r/i
        vec_fx = torch.mul(self.XX[None,None,...,None], f)
        vec_fy = torch.mul(self.XY[None,None,...,None], f)
        vec_fz = torch.mul(self.XZ[None,None,...,None], f)

        tmp = self.nufft_ob(vec_fx, xi)[:,0]
        grad[:,0] = ( torch.mul(tmp[...,1], g[:,0,...,0]) - torch.mul(tmp[...,0], g[:,0,...,1]) ).sum(axis=0)
        tmp = self.nufft_ob(vec_fy, xi)[:,0]
        grad[:,1] = ( torch.mul(tmp[...,1], g[:,0,...,0]) - torch.mul(tmp[...,0], g[:,0,...,1]) ).sum(axis=0)
        tmp = self.nufft_ob(vec_fz, xi)[:,0]
        grad[:,2] = ( torch.mul(tmp[...,1], g[:,0,...,0]) - torch.mul(tmp[...,0], g[:,0,...,1]) ).sum(axis=0)

        return grad
    def _backward_adjoint3D(self, y, g, xi):
        self.test_xi(xi)
        ndim = len(y.shape)
        grad = torch.zeros(xi.shape, dtype=self.torch_dtype, device=self.device)
        if ndim != 3:
            raise Exception("Error: y should have 3 dimensions: batch, K, r/i")
        y = y[:,None].type(self.torch_dtype) # batch,K,r/i
        xi = xi.permute(1,0).type(self.torch_dtype)
        g = g[:,None].type(self.torch_dtype) # batch,nx,ny,nz,r/i
        #                          batch,coil,nx,ny,nz,r/i
        vecx_grad_output = torch.mul(self.XX[None,None,...,None], g)
        vecy_grad_output = torch.mul(self.XY[None,None,...,None], g)
        vecz_grad_output = torch.mul(self.XY[None,None,...,None], g)

        tmp = self.nufft_ob(vecx_grad_output, xi)[:,0]
        grad[:,0] = ( torch.mul(tmp[...,1], y[:,0,...,0]) - torch.mul(tmp[...,0], y[:,0,...,1]) ).sum(axis=0)
        tmp = self.nufft_ob(vecy_grad_output, xi)[:,0]
        grad[:,1] = ( torch.mul(tmp[...,1], y[:,0,...,0]) - torch.mul(tmp[...,0], y[:,0,...,1]) ).sum(axis=0)
        tmp = self.nufft_ob(vecz_grad_output, xi)[:,0]
        grad[:,1] = ( torch.mul(tmp[...,1], y[:,0,...,0]) - torch.mul(tmp[...,0], y[:,0,...,1]) ).sum(axis=0)

        return grad


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
