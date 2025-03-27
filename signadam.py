import torch
import math
from torch.optim import Optimizer

import torch
import math
from torch.optim import Optimizer

class LowPrecisionAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
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
                
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                state = self.state[p]
                
                # Robust state initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Use bfloat16 instead of float16
                    state['exp_avg'] = torch.zeros_like(p, dtype=torch.bfloat16)
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.bfloat16)
                
                state['step'] += 1
                
                # Convert to float32 for calculations
                exp_avg = state['exp_avg'].float()
                exp_avg_sq = state['exp_avg_sq'].float()
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Perform updates with additional numerical stability checks
                grad_scaled = grad.float()
                
                # Clip gradients to prevent extreme values
                grad_scaled = torch.clamp(
                    grad_scaled, 
                    min=-1e4, 
                    max=1e4
                )
                
                # Update first moment estimate with stability
                exp_avg.mul_(beta1).add_(grad_scaled, alpha=(1 - beta1))
                
                # Update second moment estimate with stability
                exp_avg_sq.mul_(beta2).addcmul_(grad_scaled, grad_scaled, value=1 - beta2)
                
                # Convert back to bfloat16 for storage
                state['exp_avg'].copy_(exp_avg.bfloat16())
                state['exp_avg_sq'].copy_(exp_avg_sq.bfloat16())
                
                # Compute denominator with bias correction and additional stability
                denom = (
                    (exp_avg_sq.sqrt() / math.sqrt(bias_correction2))
                    .add_(eps)
                    .clamp(min=eps, max=1e4)
                )
                
                # Compute step with additional checks
                step_size = lr / bias_correction1
                update = exp_avg / denom * step_size
                
                # Clip update to prevent extreme changes
                update = torch.clamp(
                    update, 
                    min=-1, 
                    max=1
                )
                
                p.add_(-update)
        
        return loss

# Alternative approach: Quantized Adam
class QuantizedAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, bits=8):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.bits = bits
        super().__init__(params, defaults)
    
    def _quantize(self, tensor, bits=8):
        """Quantize tensor to specified number of bits"""
        max_val = tensor.abs().max()
        scale = max_val / (2 ** (bits - 1) - 1)
        quantized = (tensor / scale).round().clamp(-2**(bits-1), 2**(bits-1)-1)
        return quantized, scale
    
    def _dequantize(self, quantized, scale):
        """Dequantize tensor"""
        return quantized.float() * scale
    
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
                
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                state = self.state[p]
                
                # State initialization with quantization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg_quantized'] = torch.zeros_like(p, dtype=torch.int8)
                    state['exp_avg_sq_quantized'] = torch.zeros_like(p, dtype=torch.int8)
                    state['exp_avg_scale'] = torch.ones_like(p, dtype=torch.float32)
                    state['exp_avg_sq_scale'] = torch.ones_like(p, dtype=torch.float32)
                
                state['step'] += 1
                
                # Dequantize moment estimates
                exp_avg = self._dequantize(state['exp_avg_quantized'], state['exp_avg_scale'])
                exp_avg_sq = self._dequantize(state['exp_avg_sq_quantized'], state['exp_avg_sq_scale'])
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Update first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1))
                
                # Update second moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Requantize and update scales
                exp_avg_quantized, exp_avg_scale = self._quantize(exp_avg, self.bits)
                exp_avg_sq_quantized, exp_avg_sq_scale = self._quantize(exp_avg_sq, self.bits)
                
                # Store quantized values and scales
                state['exp_avg_quantized'].copy_(exp_avg_quantized)
                state['exp_avg_sq_quantized'].copy_(exp_avg_sq_quantized)
                state['exp_avg_scale'].copy_(exp_avg_scale)
                state['exp_avg_sq_scale'].copy_(exp_avg_sq_scale)
                
                # Compute denominator with bias correction
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                
                # Compute step
                step_size = lr / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss
