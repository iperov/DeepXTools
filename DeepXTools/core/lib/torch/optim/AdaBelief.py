import torch
from .Optimizer import Optimizer

class AdaBelief(Optimizer):
    def __init__(self, params, betas=(0.9, 0.999), eps=1e-16,):
        self._betas = betas
        super().__init__(params)
        
        # for group in self.param_groups:
        #     for p in group['params']:
        #         if p.grad is None:
        #             continue

        #         grad = p.grad.data
        #         beta1, beta2 = self._betas

        #         state = self.state[p]
        #         state['m_t'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
        #         state['v_t'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    
        # import code
        # code.interact(local=dict(globals(), **locals()))
    
    def step(self, iteration : int = None, grad_mult : float = 1.0, lr=1e-3, lr_dropout : float = 1.0):

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                beta1, beta2 = self._betas

                state = self.state[p]
                if len(state) == 0:
                    m_t = state['m_t'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    v_t = state['v_t'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                else:
                    m_t = state['m_t']
                    v_t = state['v_t']
                
                if grad_mult != 1.0:
                    grad = grad * grad_mult
                    
                m_t.mul_(beta1).add_(  grad           , alpha=1 - beta1)
                v_t.mul_(beta2).add_( (grad - m_t)**2 , alpha=1 - beta2)

                v_diff = (-lr * m_t).div_( v_t.sqrt().add_(1e-16) )

                if lr_dropout != 1.0:
                    lrd = torch.full(p.data.size(), lr_dropout, device=p.device)
                    torch.bernoulli(lrd, out=lrd)
                    v_diff *= lrd

                p.data.add_(v_diff)
