import torch
import torchkbnufft as tkbn
from nufftbindings.basenufft import *


class Nufft(baseNUFFT):
    def _set_dims(self):
        self.XX, self.XY = torch.meshgrid(self.xx, self.xy)
        self.XX=self.XX
        self.XY=self.XY

        self.zx = self.xx.view(-1,1).repeat(1, self.nx).view(-1,1)
        self.zy = self.xy.view(-1,1).view(1,-1).repeat(self.ny, 1).view(-1,1)

        self.nufft_ob = tkbn.KbNufft((self.nx,self.ny)).to(self.device)
        self.nufft_adj_ob = tkbn.KbNufftAdjoint((self.nx,self.ny)).to(self.device)

    def precompute(self, xi):
        self.precomputedTrig = True

    def forward(self, f, xi):
        self.test_xi(xi)
        ndim = len(f.shape)
        if ndim != 4:
            raise Exception("Error: f should have 4 dimensions: batch, nx, ny, r/i")
        f = f[:,None].type(torch.double) # batch,nx,ny,r/i
        xi = xi.permute(1,0).type(torch.double)
        y = self.nufft_ob(f, xi)[:,0]
        return y
    def adjoint(self, y, xi):
        self.test_xi(xi)
        ndim = len(y.shape)
        if ndim != 3:
            raise Exception("Error: y should have 3 dimensions: batch, K, r/i")
        y = y[:,None].type(torch.double) # batch,K,r/i
        xi = xi.permute(1,0).type(torch.double)
        f = self.nufft_adj_ob(y, xi)[:,0]
        return f
    def backward_forward(self, f, g, xi):
        self.test_xi(xi)
        ndim = len(f.shape)
        grad = torch.zeros(xi.shape, dtype=self.torch_dtype, device=self.device)
        if ndim != 4:
            raise Exception("Error: f should have 4 dimensions: batch, nx, ny, r/i")
        f = f[:,None].type(torch.double) # batch,nx,ny,r/i
        xi = xi.permute(1,0).type(torch.double)
        g = g[:,None].type(torch.double) # batch,K,r/i
        #                          batch,coil,nx,ny,r/i
        vec_fx = torch.mul(self.XX[None,None,...,None], f)
        vec_fy = torch.mul(self.XY[None,None,...,None], f)

        tmp = self.nufft_ob(vec_fx, xi)[:,0]
        grad[:,0] = ( torch.mul(tmp[...,1], g[:,0,...,0]) - torch.mul(tmp[...,0], g[:,0,...,1]) ).sum(axis=0)
        tmp = self.nufft_ob(vec_fy, xi)[:,0]
        grad[:,1] = ( torch.mul(tmp[...,1], g[:,0,...,0]) - torch.mul(tmp[...,0], g[:,0,...,1]) ).sum(axis=0)

        return grad
    def backward_adjoint(self, y, g, xi):
        self.test_xi(xi)
        ndim = len(y.shape)
        grad = torch.zeros(xi.shape, dtype=self.torch_dtype, device=self.device)
        if ndim != 3:
            raise Exception("Error: y should have 3 dimensions: batch, K, r/i")
        y = y[:,None].type(torch.double) # batch,K,r/i
        xi = xi.permute(1,0).type(torch.double)
        g = g[:,None].type(torch.double) # batch,nx,ny,r/i
        #                          batch,coil,nx,ny,r/i
        vecx_grad_output = torch.mul(self.XX[None,None,...,None], g)
        vecy_grad_output = torch.mul(self.XY[None,None,...,None], g)

        tmp = self.nufft_ob(vecx_grad_output, xi)[:,0]
        grad[:,0] = ( torch.mul(tmp[...,1], y[:,0,...,0]) - torch.mul(tmp[...,0], y[:,0,...,1]) ).sum(axis=0)
        tmp = self.nufft_ob(vecy_grad_output, xi)[:,0]
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
