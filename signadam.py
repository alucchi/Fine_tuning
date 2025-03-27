import torch
import math
from torch.optim import Optimizer

class SignAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(SignAdam, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                # Use sign of gradient instead of gradient itself
                grad_sign = torch.sign(grad)
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Update first moment estimate using gradient sign
                exp_avg.mul_(beta1).add_(grad_sign, alpha=(1 - beta1))
                
                # Update second moment estimate using gradient sign squared
                exp_avg_sq.mul_(beta2).addcmul_(grad_sign, grad_sign, value=1 - beta2)
                
                # Compute denominator with bias correction
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                
                # Compute step
                step_size = lr / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss
