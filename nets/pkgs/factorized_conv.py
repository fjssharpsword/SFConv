# encoding: utf-8
"""
Factorization Convolutional Layer with Spectral Decay.
Author: Jason.Fang
Update time: 26/08/2021
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
        nn.init.kaiming_normal_(self.P.data)
        nn.init.kaiming_normal_(self.Q.data)

        #convolutional parameters
        self.kwargs = {}
        for name in ['bias', 'stride', 'padding', 'dilation', 'groups']:
            attr = getattr(conv, name)
            setattr(self, name, attr)
            self.kwargs[name] = attr
        delattr(conv, 'weight')
        delattr(conv, 'bias')

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

    #approximated SVD
    #https://jeremykun.com/2016/05/16/singular-value-decomposition-part-2-theorem-proof-algorithm/
    def _power_iteration(self, W, eps=1e-10, Ip=2):
        """
        power iteration for max_singular_value
        """
        v = torch.FloatTensor(W.size(1), 1).normal_(0, 1).cuda()
        W_s = torch.matmul(W.T, W)
        #while True:
        for _ in range(Ip):
            v_t = v
            v = torch.matmul(W_s, v_t)
            v = v/torch.norm(v)
            #if abs(torch.dot(v.squeeze(), v_t.squeeze())) > 1 - eps: #converged
            #    break

        u = torch.matmul(W, v)
        s = torch.norm(u)
        u = u/s
        #return left vector, sigma, right vector
        return u, s, v

    def _specgrad(self, matrices):
    
        assert type(matrices) == list
        assert len(matrices) == 2
        output = [None for _ in matrices]
        P, Q = matrices

        #SVD approximated solve
        #W_hat = torch.matmul(P, Q)
        u_p, s_p, v_p = self._power_iteration(P) 
        u_q, s_q, v_q = self._power_iteration(Q)

        #calculate gradient
        #Nuclear norm: torch.sum(abs(S)) = torch.norm(S, p=1) <==> L1 
        #Frobenius norm: torch.norm(S,p=2) <==> L2 
        #Spectral norm: torch.max(S) = torch.norm(S,float('inf'))
        output[0] = torch.matmul(u_p, v_p.T) # *s_p 
        output[-1] = torch.matmul(u_q, v_q.T) # * s_q

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
