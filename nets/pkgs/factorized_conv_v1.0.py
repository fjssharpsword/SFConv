# encoding: utf-8
"""
Factorization Convolutional Layer with Spectral Norm Regularization.
Author: Jason.Fang
Update time: 18/08/2021
"""
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.modules import conv

class FactorizedConv(nn.Module):

    def __init__(self, conv, rank_scale=1.0, spec=True):
        super(FactorizedConv, self).__init__()

        self.shape = conv.weight.shape
        a, b, c, d = self.shape
        dim1, dim2 = a * c, b * d
        self.rank = max(int(round(rank_scale * min(a, b))), 1)
        self.P = nn.Parameter(torch.zeros(dim1, self.rank))
        self.Q = nn.Parameter(torch.zeros(self.rank, dim2))

        self.spec = spec
        if self.spec: #spectral norm regularization
            nn.init.kaiming_normal_(self.P.data)
            nn.init.kaiming_normal_(self.Q.data)
            #the right largest singular value of W of power iteration (PI)
            self.register_buffer('u', torch.Tensor(1, dim1).normal_())
        else: # without Frobenius norm regularization for fair comparison methods of weight decay
            #weight = conv.weight.data.reshape(dim1, dim2)
            #P, Q = self._spectral_init(weight, self.rank)
            #self.P.data[:,:P.shape[1]] = P
            #self.Q.data[:Q.shape[0],:] = Q
            nn.init.kaiming_normal_(self.P.data)
            nn.init.kaiming_normal_(self.Q.data)

        self.kwargs = {}
        for name in ['bias', 'stride', 'padding', 'dilation', 'groups']:
            attr = getattr(conv, name)
            setattr(self, name, attr)
            self.kwargs[name] = attr
        delattr(conv, 'weight')
        delattr(conv, 'bias')

    def _spectral_init(self, weight, rank):
    
        U, S, V = torch.svd(weight)
        sqrtS = torch.diag(torch.sqrt(S[:rank]))

        return torch.matmul(U[:,:rank], sqrtS), torch.matmul(V[:,:rank], sqrtS).T

    def forward(self, x):

        out_channels, in_channels, ks1, ks2 = self.shape 
                     
        W_ = self.Q.T.reshape(in_channels, ks2, 1, self.rank).permute(3, 0, 2, 1)
        x = F.conv2d(x, 
                     W_, 
                     None,
                     stride=(1, self.stride[1]),
                     padding=(0, self.padding[1]),
                     dilation=(1, self.dilation[1]),
                     groups=self.groups).contiguous()

        W_ = self.P.reshape(out_channels, ks1, self.rank, 1).permute(0, 2, 1, 3)
        return F.conv2d(x, 
                        W_,
                        self.bias,
                        stride=(self.stride[0], 1),
                        padding=(self.padding[0], 0),
                        dilation=(self.dilation[0], 1),
                        groups=self.groups).contiguous()

    def _frobgrad(self, matrices):

        assert type(matrices) == list
        assert len(matrices) == 2
        output = [None for _ in matrices]
            
        P, Q = matrices
        PM, MQ = matrices
            
        output[0] = torch.chain_matmul(P, MQ, MQ.T)
        output[-1] = torch.chain_matmul(PM.T, PM, Q)

        return output
    
    #https://www.ucg.ac.me/skladiste/blog_10701/objava_23569/fajlovi/power.pdf
    def _l2normalize(self, v, eps=1e-12):
        return v / (torch.norm(v) + eps)
    def _power_iteration(self, W, u=None, Ip=1):
        """
        power iteration for max_singular_value
        """
        #xp = W.data
        if not Ip >= 1:
            raise ValueError("Power iteration should be a positive integer")
        if u is None:
            u = torch.FloatTensor(1, W.size(0)).normal_(0, 1).cuda()
 
        for _ in range(Ip):
            v = self._l2normalize(torch.matmul(u, W.data), eps=1e-12)
            u = self._l2normalize(torch.matmul(v, torch.transpose(W.data, 0, 1)), eps=1e-12)
        S = (F.linear(u, torch.transpose(W.data, 0, 1)) * v).squeeze()

        return S, u

    def _specgrad(self, matrices):
    
        assert type(matrices) == list
        assert len(matrices) == 2
        output = [None for _ in matrices]
        P, Q = matrices

        W_hat = torch.matmul(P, Q)
        #_, S, _ = torch.svd(W_hat)
        #_, S, _ =np.linalg.svd(np.dot(P.cpu().numpy(), Q.cpu().numpy()), full_matrices=True)
        S, u = self._power_iteration(W_hat, self.u)
        self.u.copy_(u)

        #Nuclear norm: torch.sum(abs(S)) = torch.norm(S, p=1) <==> L1 
        #Frobenius norm: torch.norm(S,p=2) <==> L2 
        #Spectral norm: torch.max(S) = torch.norm(S,float('inf'))
        output[0] = torch.max(S)*P 
        output[-1] = torch.max(S)*Q

        return output

    def updategrad(self, coef=1E-4):

        if self.spec: #spectral norm regularization
            Pgrad, Qgrad = self._specgrad([self.P.data, self.Q.data])
        else: # Frobenius norm regularization
            Pgrad, Qgrad = self._frobgrad([self.P.data, self.Q.data])

        if self.P.grad is None:
            self.P.grad = coef * Pgrad
        else:
            self.P.grad += coef * Pgrad

        if self.Q.grad is None:
            self.Q.grad = coef * Qgrad
        else:
            self.Q.grad += coef * Qgrad

#backward
def weightdecay(model, coef=1E-4, skiplist=[]):

    def _apply_weightdecay(name, module, skiplist=[]):
        if hasattr(module, 'updategrad'):
            return not any(name[-len(entry):] == entry for entry in skiplist)
        return False

    module_list = enumerate(module for name, module in model.named_modules() if _apply_weightdecay(name, module, skiplist=skiplist))
    for i, module in module_list:
        module.updategrad(coef=coef)

if __name__ == "__main__":
    #for debug  
    x =  torch.rand(2, 3, 16, 16).cuda()
    fconv = FactorizedConv(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding =1, stride=2, bias=False), rank_scale=1.0, spec=False).cuda()
    out = fconv(x)
    print(out.shape)

