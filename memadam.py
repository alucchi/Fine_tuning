import torch
import math
from torch.optim import Optimizer

class Float8Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def adaptive_quantize(self, tensor, bits=8):
        """
        Adaptive quantization with controlled precision reduction
        """
        # Prevent NaNs and infinities
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e-4, neginf=-1e-4)
        
        # Compute robust scale
        max_val = tensor.abs().max()
        
        # Determine quantization range based on bit depth
        max_range = 2 ** (bits - 1) - 1
        
        # Compute scale to map to integer range
        scale = max_val / max_range if max_val > 0 else 1.0
        
        # Quantize with careful clipping
        quantized = torch.clamp(
            torch.round(tensor / (scale + 1e-8)),
            min=-max_range, 
            max=max_range
        ).to(torch.int8)
        
        return quantized, scale
    
    def adaptive_dequantize(self, quantized, scale):
        """
        Safe dequantization with minimal information loss
        """
        return quantized.float() * (scale + 1e-8)
    
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
                
                # Gradient preprocessing
                grad = torch.nan_to_num(grad, nan=0.0, posinf=1e-4, neginf=-1e-4)
                
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                state = self.state[p]
                
                # State initialization with adaptive quantization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg_quantized'] = torch.zeros_like(p, dtype=torch.int8)
                    state['exp_avg_sq_quantized'] = torch.zeros_like(p, dtype=torch.int8)
                    state['exp_avg_scale'] = torch.ones_like(p, dtype=torch.float32)
                    state['exp_avg_sq_scale'] = torch.ones_like(p, dtype=torch.float32)
                
                state['step'] += 1
                
                # Dequantize moment estimates
                exp_avg = self.adaptive_dequantize(
                    state['exp_avg_quantized'], 
                    state['exp_avg_scale']
                )
                exp_avg_sq = self.adaptive_dequantize(
                    state['exp_avg_sq_quantized'], 
                    state['exp_avg_sq_scale']
                )
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Moment estimate updates with stability
                exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Adaptive quantization
                exp_avg_quantized, exp_avg_scale = self.adaptive_quantize(exp_avg)
                exp_avg_sq_quantized, exp_avg_sq_scale = self.adaptive_quantize(exp_avg_sq)
                
                # Store quantized values and scales
                state['exp_avg_quantized'].copy_(exp_avg_quantized)
                state['exp_avg_sq_quantized'].copy_(exp_avg_sq_quantized)
                state['exp_avg_scale'].copy_(exp_avg_scale)
                state['exp_avg_sq_scale'].copy_(exp_avg_sq_scale)
                
                # Stable denominator computation
                denom = (
                    exp_avg_sq.sqrt() / math.sqrt(bias_correction2 + 1e-8)
                ).clamp(min=eps).add_(eps)
                
                # Compute adaptive step
                step_size = lr / (bias_correction1 + 1e-8)
                update = exp_avg / denom * step_size
                
                # Conservative update
                p.add_(-update)
        
        return loss

class Float16Adam(Optimizer):
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
