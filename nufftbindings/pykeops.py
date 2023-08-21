# Author: Alban Gossard
# Last modification: 2021/22/09

import torch
from pykeops.torch import LazyTensor, ComplexLazyTensor
# from pykeops import IntCst, Imag2Complex
from .basenufft import *


class Nufft(baseNUFFT):
    def _set_dims(self):
        self.cos_coupling = None
        self.sin_coupling = None

        if self.ndim==2:
            self.zx = self.xx.view(-1,1).repeat(1, self.nx).view(-1,1)
            self.zy = self.xy.view(1,-1).repeat(self.ny, 1).view(-1,1)
        elif self.ndim==3:
            self.zx = self.xx.view(-1,1,1).repeat(1, self.nx, self.nx).view(-1,1)
            self.zy = self.xy.view(1,-1,1).repeat(self.ny, 1, self.ny).view(-1,1)
            self.zz = self.xz.view(1,1,-1).repeat(self.nz, self.nz, 1).view(-1,1)

        self.zx = LazyTensor( self.zx[None,:,:] )
        self.zy = LazyTensor( self.zy[None,:,:] )
        if self.ndim==3:
            self.zz = LazyTensor( self.zz[None,:,:] )

    def precompute(self, xi):
        xix = xi[:,0].type(self.torch_dtype).view(-1,1).contiguous()
        xiy = xi[:,1].type(self.torch_dtype).view(-1,1).contiguous()
        if self.ndim==3:
            xiz = xi[:,2].type(self.torch_dtype).view(-1,1).contiguous()
        xix = LazyTensor( xix[:,None,:] )
        xiy = LazyTensor( xiy[:,None,:] )
        if self.ndim==3:
            xiz = LazyTensor( xiz[:,None,:] )

        if self.ndim==2:
            coupling = xix*self.zx+xiy*self.zy
        elif self.ndim==3:
            coupling = xix*self.zx+xiy*self.zy+xiz*self.zz
        self.cos_coupling = coupling.cos()
        self.sin_coupling = coupling.sin()

        self.xiprecomputed = xi.clone()

        self.precomputedTrig = True

    def _forward2D(self, f, xi):
        self.test_xi(xi)
        ndim = len(f.shape)
        iscpx = f.is_complex()
        if ndim==4 and not iscpx or ndim==3 and iscpx:
            Nbatch = f.shape[0]
            if iscpx:
                f = f.permute(1,2,0) # nx,ny,batch
                fr = f.real.type(self.torch_dtype).view(-1,Nbatch).contiguous()
                fi = f.imag.type(self.torch_dtype).view(-1,Nbatch).contiguous()
            else:
                f = f.permute(1,2,3,0) # nx,ny,r/i,batch
                fr = f[:,:,0,:].type(self.torch_dtype).view(-1,Nbatch).contiguous()
                fi = f[:,:,1,:].type(self.torch_dtype).view(-1,Nbatch).contiguous()

            fr = LazyTensor( fr[None,:,:] )
            fi = LazyTensor( fi[None,:,:] )
            f_cos = fr.concat(fi)
            f_sin = fi.concat(-fr)
            y_concat = (f_cos*self.cos_coupling+f_sin*self.sin_coupling).sum(dim=1)
            if iscpx:
                y = torch.zeros(self.K, Nbatch, dtype=self.torch_cpxdtype, device=self.device)
                y.real = y_concat[:,:Nbatch]
                y.imag = y_concat[:,Nbatch:]
                y = y.permute(1,0) # batch,K
            else:
                y = torch.zeros(self.K, 2, Nbatch, dtype=self.torch_dtype, device=self.device)
                y[:,0,:] = y_concat[:,:Nbatch]
                y[:,1,:] = y_concat[:,Nbatch:]
                y = y.permute(2,0,1) # batch,K,r/i

            return y
        elif ndim==3 and not iscpx or ndim==2 and iscpx:
            if iscpx:
                fr = f.real.type(self.torch_dtype).view(-1,1).contiguous()
                fi = f.imag.type(self.torch_dtype).view(-1,1).contiguous()
            else:
                fr = f[:,:,0].type(self.torch_dtype).view(-1,1).contiguous()
                fi = f[:,:,1].type(self.torch_dtype).view(-1,1).contiguous()
            fr = LazyTensor( fr[None,:,:] )
            fi = LazyTensor( fi[None,:,:] )
            f_cos = fr.concat(fi)
            f_sin = fi.concat(-fr)
            y_concat = (f_cos*self.cos_coupling+f_sin*self.sin_coupling).sum(dim=1)
            if iscpx:
                y = torch.zeros(self.K, dtype=self.torch_cpxdtype, device=self.device)
                y.real = y_concat[:,0].view(-1)
                y.imag = y_concat[:,1].view(-1)
            else:
                y = torch.zeros(self.K, 2, dtype=self.torch_dtype, device=self.device)
                y[:,0] = y_concat[:,0].view(-1)
                y[:,1] = y_concat[:,1].view(-1)
            return y
        else:
            raise Exception("Error: f should have 2, 3 or 4 dimensions (batch mode)")
    def _adjoint2D(self, y, xi):
        self.test_xi(xi)
        ndim = len(y.shape)
        iscpx = y.is_complex()
        if ndim==3 and not iscpx or ndim==2 and iscpx:
            Nbatch = y.shape[0]
            if iscpx:
                y = y.permute(1,0) # K,batch
                yr = y.real.type(self.torch_dtype).view(-1,Nbatch).contiguous()
                yi = y.imag.type(self.torch_dtype).view(-1,Nbatch).contiguous()
            else:
                y = y.permute(1,2,0) # K,r/i,batch
                yr = y[:,0,:].type(self.torch_dtype).view(-1,Nbatch).contiguous()
                yi = y[:,1,:].type(self.torch_dtype).view(-1,Nbatch).contiguous()
            yr = LazyTensor( yr[:,None,:] )
            yi = LazyTensor( yi[:,None,:] )
            y_cos=yr.concat(yi)
            y_sin=(-yi).concat(yr)
            f_concat = (y_cos*self.cos_coupling+y_sin*self.sin_coupling).sum(dim=0)

            if iscpx:
                f = torch.zeros(self.nx,self.ny, Nbatch, dtype=self.torch_cpxdtype, device=self.device)
                f.real = f_concat[:,:Nbatch].view(self.nx,self.ny,Nbatch)
                f.imag = f_concat[:,Nbatch:].view(self.nx,self.ny,Nbatch)
                f = f.permute(2,0,1) # batch,nx,ny
            else:
                f = torch.zeros(self.nx,self.ny, 2, Nbatch, dtype=self.torch_dtype, device=self.device)
                f[:,:,0,:] = f_concat[:,:Nbatch].view(self.nx,self.ny,Nbatch)
                f[:,:,1,:] = f_concat[:,Nbatch:].view(self.nx,self.ny,Nbatch)
                f = f.permute(3,0,1,2) # batch,nx,ny,r/i
            return f
        elif ndim==2 and not iscpx or ndim==1 and iscpx:

            if iscpx:
                yr = y.real.type(self.torch_dtype).view(-1,1).contiguous()
                yi = y.imag.type(self.torch_dtype).view(-1,1).contiguous()
            else:
                yr = y[:,0].type(self.torch_dtype).view(-1,1).contiguous()
                yi = y[:,1].type(self.torch_dtype).view(-1,1).contiguous()
            yr = LazyTensor( yr[:,None,:] )
            yi = LazyTensor( yi[:,None,:] )
            y_cos=yr.concat(yi)
            y_sin=(-yi).concat(yr)
            f_concat = (y_cos*self.cos_coupling+y_sin*self.sin_coupling).sum(dim=0)

            if iscpx:
                f = torch.zeros(self.nx,self.ny, dtype=self.torch_cpxdtype, device=self.device)
                f.real = f_concat[:,0].view(self.nx,self.ny)
                f.imag = f_concat[:,1].view(self.nx,self.ny)
            else:
                f = torch.zeros(self.nx,self.ny, 2, dtype=self.torch_dtype, device=self.device)
                f[:,:,0] = f_concat[:,0].view(self.nx,self.ny)
                f[:,:,1] = f_concat[:,1].view(self.nx,self.ny)

            return f
        else:
            raise Exception("Error: y should have 2 or 3 dimensions (batch mode)")
    def _backward_forward2D(self, f, g, xi):
        self.test_xi(xi)
        ndim = len(f.shape)
        iscpx = f.is_complex()
        if ndim==4 and not iscpx or ndim==3 and iscpx:
            Nbatch = f.shape[0]
            if iscpx:
                f = f.permute(1,2,0) # nx,ny,batch
                g = g.permute(1,0) # K,batch
                fr = f.real.type(self.torch_dtype).view(-1,Nbatch).contiguous()
                fi = f.imag.type(self.torch_dtype).view(-1,Nbatch).contiguous()
            else:
                f = f.permute(1,2,3,0) # nx,ny,r/i,batch
                g = g.permute(1,2,0) # K,r/i,batch
                fr = f[:,:,0,:].type(self.torch_dtype).view(-1,Nbatch).contiguous()
                fi = f[:,:,1,:].type(self.torch_dtype).view(-1,Nbatch).contiguous()

            fr = LazyTensor( fr[None,:,:] )
            fi = LazyTensor( fi[None,:,:] )

            vec_frx = self.zx*fr
            vec_fix = self.zx*fi
            vec_fry = self.zy*fr
            vec_fiy = self.zy*fi

            grad = torch.zeros(xi.shape, dtype=self.torch_dtype, device=self.device)

            vec_cos = vec_frx.concat(vec_fix).concat(vec_fry).concat(vec_fiy)
            vec_sin = vec_fix.concat(-vec_frx).concat(vec_fiy).concat(-vec_fry)
            tmp_concat = (vec_cos*self.cos_coupling+vec_sin*self.sin_coupling).sum(dim=1).view(-1,4*Nbatch)
            tmp1r = tmp_concat[:,:Nbatch]
            tmp1i = tmp_concat[:,Nbatch:2*Nbatch]
            if iscpx:
                grad[:,0] = (torch.mul(tmp1i, g.real) - torch.mul(tmp1r, g.imag)).sum(dim=1)
            else:
                grad[:,0] = (torch.mul(tmp1i, g[:,0,:]) - torch.mul(tmp1r, g[:,1,:])).sum(dim=1)

            tmp2r = tmp_concat[:,2*Nbatch:3*Nbatch]
            tmp2i = tmp_concat[:,3*Nbatch:]
            if iscpx:
                grad[:,1] = (torch.mul(tmp2i, g.real) - torch.mul(tmp2r, g.imag)).sum(dim=1)
            else:
                grad[:,1] = (torch.mul(tmp2i, g[:,0,:]) - torch.mul(tmp2r, g[:,1,:])).sum(dim=1)


            return grad
        elif ndim==3 and not iscpx or ndim==2 and iscpx:
            if iscpx:
                fr = f.real.type(self.torch_dtype).view(-1,1).contiguous()
                fi = f.imag.type(self.torch_dtype).view(-1,1).contiguous()
            else:
                fr = f[:,:,0].type(self.torch_dtype).view(-1,1).contiguous()
                fi = f[:,:,1].type(self.torch_dtype).view(-1,1).contiguous()
            fr = LazyTensor( fr[None,:,:] )
            fi = LazyTensor( fi[None,:,:] )

            vec_frx = self.zx*fr
            vec_fix = self.zx*fi
            vec_fry = self.zy*fr
            vec_fiy = self.zy*fi

            grad = torch.zeros(xi.shape, dtype=self.torch_dtype, device=self.device)

            vec_cos = vec_frx.concat(vec_fix).concat(vec_fry).concat(vec_fiy)
            vec_sin = vec_fix.concat(-vec_frx).concat(vec_fiy).concat(-vec_fry)
            tmp_concat = (vec_cos*self.cos_coupling+vec_sin*self.sin_coupling).sum(dim=1).view(-1,4)
            tmp1r = tmp_concat[:,0]
            tmp1i = tmp_concat[:,1]
            if iscpx:
                grad[:,0] = torch.mul(tmp1i, g.real) - torch.mul(tmp1r, g.imag)
            else:
                grad[:,0] = torch.mul(tmp1i, g[:,0]) - torch.mul(tmp1r, g[:,1])

            tmp2r = tmp_concat[:,2]
            tmp2i = tmp_concat[:,3]
            if iscpx:
                grad[:,1] = torch.mul(tmp2i, g.real) - torch.mul(tmp2r, g.imag)
            else:
                grad[:,1] = torch.mul(tmp2i, g[:,0]) - torch.mul(tmp2r, g[:,1])

            return grad
        else:
            raise Exception("Error: f should have 3 or 4 dimensions (batch mode)")

    def _backward_adjoint2D(self, y, g, xi):
        self.test_xi(xi)
        ndim = len(y.shape)
        iscpx = y.is_complex()
        if ndim==3 and not iscpx or ndim==2 and iscpx:
            Nbatch = y.shape[0]
            if iscpx:
                y = y.permute(1,0) # K,batch
                g = g.permute(1,2,0) # nx,ny,batch
                gr = g.real.type(self.torch_dtype).view(-1,Nbatch).contiguous()
                gi = g.imag.type(self.torch_dtype).view(-1,Nbatch).contiguous()
            else:
                y = y.permute(1,2,0) # K,r/i,batch
                g = g.permute(1,2,3,0) # nx,ny,r/i,batch
                gr = g[:,:,0,:].type(self.torch_dtype).view(-1,Nbatch).contiguous()
                gi = g[:,:,1,:].type(self.torch_dtype).view(-1,Nbatch).contiguous()

            gr = LazyTensor( gr[None,:,:] )
            gi = LazyTensor( gi[None,:,:] )

            vecx_grad_outputr = self.zx*gr
            vecx_grad_outputi = self.zx*gi
            vecy_grad_outputr = self.zy*gr
            vecy_grad_outputi = self.zy*gi

            grad = torch.zeros(xi.shape, dtype=self.torch_dtype, device=self.device)

            vec_grad_output_cos = vecx_grad_outputr.concat(vecx_grad_outputi).concat(vecy_grad_outputr).concat(vecy_grad_outputi)
            vec_grad_output_sin = vecx_grad_outputi.concat(-vecx_grad_outputr).concat(vecy_grad_outputi).concat(-vecy_grad_outputr)
            tmp_concat = (vec_grad_output_cos*self.cos_coupling+vec_grad_output_sin*self.sin_coupling).sum(dim=1).view(-1,4*Nbatch)
            tmp1r = tmp_concat[:,:Nbatch]
            tmp1i = tmp_concat[:,Nbatch:2*Nbatch]
            if iscpx:
                grad[:,0] = (torch.mul(tmp1i, y.real) - torch.mul(tmp1r, y.imag)).sum(dim=1)
            else:
                grad[:,0] = (torch.mul(tmp1i, y[:,0,:]) - torch.mul(tmp1r, y[:,1,:])).sum(dim=1)

            tmp2r = tmp_concat[:,2*Nbatch:3*Nbatch]
            tmp2i = tmp_concat[:,3*Nbatch:]
            if iscpx:
                grad[:,1] = (torch.mul(tmp2i, y.real) - torch.mul(tmp2r, y.imag)).sum(dim=1)
            else:
                grad[:,1] = (torch.mul(tmp2i, y[:,0,:]) - torch.mul(tmp2r, y[:,1,:])).sum(dim=1)


            return grad
        elif ndim==2 and not iscpx or ndim==1 and iscpx:
            if iscpx:
                gr = g.real.type(self.torch_dtype).view(-1,1).contiguous()
                gi = g.imag.type(self.torch_dtype).view(-1,1).contiguous()
            else:
                gr = g[:,:,0].type(self.torch_dtype).view(-1,1).contiguous()
                gi = g[:,:,1].type(self.torch_dtype).view(-1,1).contiguous()
            gr = LazyTensor( gr[None,:,:] )
            gi = LazyTensor( gi[None,:,:] )

            vecx_grad_outputr = self.zx*gr
            vecx_grad_outputi = self.zx*gi
            vecy_grad_outputr = self.zy*gr
            vecy_grad_outputi = self.zy*gi

            grad = torch.zeros(xi.shape, dtype=self.torch_dtype, device=self.device)

            vec_grad_output_cos = vecx_grad_outputr.concat(vecx_grad_outputi).concat(vecy_grad_outputr).concat(vecy_grad_outputi)
            vec_grad_output_sin = vecx_grad_outputi.concat(-vecx_grad_outputr).concat(vecy_grad_outputi).concat(-vecy_grad_outputr)
            tmp_concat = (vec_grad_output_cos*self.cos_coupling+vec_grad_output_sin*self.sin_coupling).sum(dim=1).view(-1,4)
            tmp1r = tmp_concat[:,0]
            tmp1i = tmp_concat[:,1]
            if iscpx:
                grad[:,0] = torch.mul(tmp1i, y.real) - torch.mul(tmp1r, y.imag)
            else:
                grad[:,0] = torch.mul(tmp1i, y[:,0]) - torch.mul(tmp1r, y[:,1])

            tmp2r = tmp_concat[:,2]
            tmp2i = tmp_concat[:,3]
            if iscpx:
                grad[:,1] = torch.mul(tmp2i, y.real) - torch.mul(tmp2r, y.imag)
            else:
                grad[:,1] = torch.mul(tmp2i, y[:,0]) - torch.mul(tmp2r, y[:,1])

            return grad
        else:
            raise Exception("Error: y should have 2 or 3 dimensions (batch mode)")

    def _forward3D(self, f, xi):
        self.test_xi(xi)
        ndim = len(f.shape)
        if ndim==5:
            Nbatch = f.shape[0]
            f = f.permute(1,2,3,4,0) # nx,ny,nz,r/i,batch

            fr = f[:,:,:,0,:].type(self.torch_dtype).view(-1,Nbatch).contiguous()
            fi = f[:,:,:,1,:].type(self.torch_dtype).view(-1,Nbatch).contiguous()
            fr = LazyTensor( fr[None,:,:] )
            fi = LazyTensor( fi[None,:,:] )
            f_cos = fr.concat(fi)
            f_sin = fi.concat(-fr)
            y_concat = (f_cos*self.cos_coupling+f_sin*self.sin_coupling).sum(dim=1)
            y = torch.zeros(self.K, 2, Nbatch, dtype=self.torch_dtype, device=self.device)
            y[:,0,:] = y_concat[:,:Nbatch]
            y[:,1,:] = y_concat[:,Nbatch:]

            y = y.permute(2,0,1) # batch,K,r/i
            return y
        else:
            raise Exception("Error: f should have 5 dimensions (batch size, nx, ny, nz, 2)")
    def _adjoint3D(self, y, xi):
        self.test_xi(xi)
        ndim = len(y.shape)
        if ndim==3:
            Nbatch = y.shape[0]
            y = y.permute(1,2,0) # K,r/i,batch

            yr = y[:,0,:].type(self.torch_dtype).view(-1,Nbatch).contiguous()
            yi = y[:,1,:].type(self.torch_dtype).view(-1,Nbatch).contiguous()
            yr = LazyTensor( yr[:,None,:] )
            yi = LazyTensor( yi[:,None,:] )
            y_cos=yr.concat(yi)
            y_sin=(-yi).concat(yr)
            f_concat = (y_cos*self.cos_coupling+y_sin*self.sin_coupling).sum(dim=0)

            f = torch.zeros(self.nx,self.ny,self.nz, 2, Nbatch, dtype=self.torch_dtype, device=self.device)
            f[:,:,:,0,:] = f_concat[:,:Nbatch].view(self.nx,self.ny,self.nz,Nbatch)
            f[:,:,:,1,:] = f_concat[:,Nbatch:].view(self.nx,self.ny,self.nz,Nbatch)

            f = f.permute(4,0,1,2,3) # batch,nx,ny,nz,r/i
            return f
        else:
            raise Exception("Error: y should have 3 dimensions (batch size, K, 2)")
    def _backward_forward3D(self, f, g, xi):
        self.test_xi(xi)
        ndim = len(f.shape)
        if ndim==5:
            Nbatch = f.shape[0]
            f = f.permute(1,2,3,4,0) # nx,ny,nz,r/i,batch
            g = g.permute(1,2,0) # K,r/i,batch

            fr = f[:,:,:,0,:].type(self.torch_dtype).view(-1,Nbatch).contiguous()
            fi = f[:,:,:,1,:].type(self.torch_dtype).view(-1,Nbatch).contiguous()
            fr = LazyTensor( fr[None,:,:] )
            fi = LazyTensor( fi[None,:,:] )

            vec_frx = self.zx*fr
            vec_fix = self.zx*fi
            vec_fry = self.zy*fr
            vec_fiy = self.zy*fi
            vec_frz = self.zz*fr
            vec_fiz = self.zz*fi

            grad = torch.zeros(xi.shape, dtype=self.torch_dtype, device=self.device)

            vec_cos = vec_frx.concat(vec_fix).concat(vec_fry).concat(vec_fiy).concat(vec_frz).concat(vec_fiz)
            vec_sin = vec_fix.concat(-vec_frx).concat(vec_fiy).concat(-vec_fry).concat(vec_fiz).concat(-vec_frz)
            tmp_concat = (vec_cos*self.cos_coupling+vec_sin*self.sin_coupling).sum(dim=1).view(-1,6*Nbatch)
            tmp1r = tmp_concat[:,:Nbatch]
            tmp1i = tmp_concat[:,Nbatch:2*Nbatch]
            grad[:,0] = (torch.mul(tmp1i, g[:,0,:]) - torch.mul(tmp1r, g[:,1,:])).sum(dim=1)

            tmp2r = tmp_concat[:,2*Nbatch:3*Nbatch]
            tmp2i = tmp_concat[:,3*Nbatch:4*Nbatch]
            grad[:,1] = (torch.mul(tmp2i, g[:,0,:]) - torch.mul(tmp2r, g[:,1,:])).sum(dim=1)

            tmp3r = tmp_concat[:,4*Nbatch:5*Nbatch]
            tmp3i = tmp_concat[:,5*Nbatch:]
            grad[:,2] = (torch.mul(tmp3i, g[:,0,:]) - torch.mul(tmp3r, g[:,1,:])).sum(dim=1)


            return grad
        else:
            raise Exception("Error: f should have 5 dimensions (batch size, nx, ny, nz, 2)")

    def _backward_adjoint3D(self, y, g, xi):
        self.test_xi(xi)
        ndim = len(y.shape)
        if ndim==3:
            Nbatch = y.shape[0]
            y = y.permute(1,2,0) # K,r/i,batch
            g = g.permute(1,2,3,4,0) # nx,ny,r/i,batch

            gr = g[:,:,:,0,:].type(self.torch_dtype).view(-1,Nbatch).contiguous()
            gi = g[:,:,:,1,:].type(self.torch_dtype).view(-1,Nbatch).contiguous()
            gr = LazyTensor( gr[None,:,:] )
            gi = LazyTensor( gi[None,:,:] )

            vecx_grad_outputr = self.zx*gr
            vecx_grad_outputi = self.zx*gi
            vecy_grad_outputr = self.zy*gr
            vecy_grad_outputi = self.zy*gi
            vecz_grad_outputr = self.zz*gr
            vecz_grad_outputi = self.zz*gi

            grad = torch.zeros(xi.shape, dtype=self.torch_dtype, device=self.device)

            vec_grad_output_cos = vecx_grad_outputr.concat(vecx_grad_outputi).concat(vecy_grad_outputr).concat(vecy_grad_outputi).concat(vecz_grad_outputr).concat(vecz_grad_outputi)
            vec_grad_output_sin = vecx_grad_outputi.concat(-vecx_grad_outputr).concat(vecy_grad_outputi).concat(-vecy_grad_outputr).concat(vecz_grad_outputi).concat(-vecz_grad_outputr)
            tmp_concat = (vec_grad_output_cos*self.cos_coupling+vec_grad_output_sin*self.sin_coupling).sum(dim=1).view(-1,6*Nbatch)
            tmp1r = tmp_concat[:,:Nbatch]
            tmp1i = tmp_concat[:,Nbatch:2*Nbatch]
            grad[:,0] = (torch.mul(tmp1i, y[:,0,:]) - torch.mul(tmp1r, y[:,1,:])).sum(dim=1)

            tmp2r = tmp_concat[:,2*Nbatch:3*Nbatch]
            tmp2i = tmp_concat[:,3*Nbatch:4*Nbatch]
            grad[:,1] = (torch.mul(tmp2i, y[:,0,:]) - torch.mul(tmp2r, y[:,1,:])).sum(dim=1)

            tmp3r = tmp_concat[:,4*Nbatch:5*Nbatch]
            tmp3i = tmp_concat[:,5*Nbatch:]
            grad[:,2] = (torch.mul(tmp3i, y[:,0,:]) - torch.mul(tmp3r, y[:,1,:])).sum(dim=1)


            return grad
        else:
            raise Exception("Error: y should have 3 dimensions (batch size, K, 2)")



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
