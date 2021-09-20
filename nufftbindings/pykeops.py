import torch
from pykeops.torch import LazyTensor
from nufftbindings.basenufft import *


class Nufft(baseNUFFT):
    def _set_dims(self):
        self.cos_coupling = None
        self.sin_coupling = None

        self.zx = self.xx.view(-1,1).repeat(1, self.nx).view(-1,1)
        self.zy = self.xy.view(-1,1).view(1,-1).repeat(self.ny, 1).view(-1,1)

        self.zx = LazyTensor( self.zx[None,:,:] )
        self.zy = LazyTensor( self.zy[None,:,:] )

    def precompute(self, xi):
        xix = xi[:,0].type(self.torch_dtype).view(-1,1).contiguous()
        xiy = xi[:,1].type(self.torch_dtype).view(-1,1).contiguous()
        xix = LazyTensor( xix[:,None,:] )
        xiy = LazyTensor( xiy[:,None,:] )

        coupling = xix*self.zx+xiy*self.zy
        self.cos_coupling = coupling.cos()
        self.sin_coupling = coupling.sin()

        self.precomputedTrig = True

    def _forward_real(self, fr, fi):
        return (fr*self.cos_coupling+fi*self.sin_coupling).sum(dim=1)
    def _forward_imag(self, fr, fi):
        return (fi*self.cos_coupling-fr*self.sin_coupling).sum(dim=1)
    def _adjoint_real(self, yr, yi):
        return (yr*self.cos_coupling-yi*self.sin_coupling).sum(dim=0)
    def _adjoint_imag(self, yr, yi):
        return (yi*self.cos_coupling+yr*self.sin_coupling).sum(dim=0)

    def forward(self, f, xi):
        self.test_xi(xi)
        ndim = len(f.shape)
        if ndim==4:
            Nbatch = f.shape[0]
            f = f.permute(1,2,3,0) # nx,ny,r/i,batch

            fr = f[:,:,0,:].type(self.torch_dtype).view(-1,Nbatch).contiguous()
            fi = f[:,:,1,:].type(self.torch_dtype).view(-1,Nbatch).contiguous()
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
        elif ndim==3:
            fr = f[:,:,0].type(self.torch_dtype).view(-1,1).contiguous()
            fi = f[:,:,1].type(self.torch_dtype).view(-1,1).contiguous()
            fr = LazyTensor( fr[None,:,:] )
            fi = LazyTensor( fi[None,:,:] )
            f_cos = fr.concat(fi)
            f_sin = fi.concat(-fr)
            y_concat = (f_cos*self.cos_coupling+f_sin*self.sin_coupling).sum(dim=1)
            y = torch.zeros(self.K, 2, dtype=self.torch_dtype, device=self.device)
            y[:,0] = y_concat[:,0].view(-1)
            y[:,1] = y_concat[:,1].view(-1)
            return y
        else:
            raise Exception("Error: f should have 3 or 4 dimensions (batch mode)")
    def adjoint(self, y, xi):
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

            f = torch.zeros(self.nx,self.ny, 2, Nbatch, dtype=self.torch_dtype, device=self.device)
            f[:,:,0,:] = f_concat[:,:Nbatch].view(self.nx,self.ny,Nbatch)
            f[:,:,1,:] = f_concat[:,Nbatch:].view(self.nx,self.ny,Nbatch)

            f = f.permute(3,0,1,2) # batch,nx,ny,r/i
            return f
        elif ndim==2:

            yr = y[:,0].type(self.torch_dtype).view(-1,1).contiguous()
            yi = y[:,1].type(self.torch_dtype).view(-1,1).contiguous()
            yr = LazyTensor( yr[:,None,:] )
            yi = LazyTensor( yi[:,None,:] )
            y_cos=yr.concat(yi)
            y_sin=(-yi).concat(yr)
            f_concat = (y_cos*self.cos_coupling+y_sin*self.sin_coupling).sum(dim=0)

            f = torch.zeros(self.nx,self.ny, 2, dtype=self.torch_dtype, device=self.device)
            f[:,:,0] = f_concat[:,0].view(self.nx,self.ny)
            f[:,:,1] = f_concat[:,1].view(self.nx,self.ny)

            return f
        else:
            raise Exception("Error: y should have 2 or 3 dimensions (batch mode)")
    def backward_forward(self, f, g, xi):
        self.test_xi(xi)
        ndim = len(f.shape)
        if ndim==4:
            Nbatch = f.shape[0]
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
            grad[:,0] = (torch.mul(tmp1i, g[:,0,:]) - torch.mul(tmp1r, g[:,1,:])).sum(dim=1)

            tmp2r = tmp_concat[:,2*Nbatch:3*Nbatch]
            tmp2i = tmp_concat[:,3*Nbatch:]
            grad[:,1] = (torch.mul(tmp2i, g[:,0,:]) - torch.mul(tmp2r, g[:,1,:])).sum(dim=1)


            return grad
        elif ndim==3:
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
            grad[:,0] = torch.mul(tmp1i, g[:,0]) - torch.mul(tmp1r, g[:,1])

            tmp2r = tmp_concat[:,2]
            tmp2i = tmp_concat[:,3]
            grad[:,1] = torch.mul(tmp2i, g[:,0]) - torch.mul(tmp2r, g[:,1])

            return grad
        else:
            raise Exception("Error: f should have 3 or 4 dimensions (batch mode)")

    def backward_adjoint(self, y, g, xi):
        self.test_xi(xi)
        ndim = len(y.shape)
        if ndim==3:
            Nbatch = y.shape[0]
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
            grad[:,0] = (torch.mul(tmp1i, y[:,0,:]) - torch.mul(tmp1r, y[:,1,:])).sum(dim=1)

            tmp2r = tmp_concat[:,2*Nbatch:3*Nbatch]
            tmp2i = tmp_concat[:,3*Nbatch:]
            grad[:,1] = (torch.mul(tmp2i, y[:,0,:]) - torch.mul(tmp2r, y[:,1,:])).sum(dim=1)


            return grad
        elif ndim==2:
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
            grad[:,0] = torch.mul(tmp1i, y[:,0]) - torch.mul(tmp1r, y[:,1])

            tmp2r = tmp_concat[:,2]
            tmp2i = tmp_concat[:,3]
            grad[:,1] = torch.mul(tmp2i, y[:,0]) - torch.mul(tmp2r, y[:,1])

            return grad
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
