import torch
import math
from torch.optim import Optimizer

class WramOptim(Optimizer):
    """
    Optimiseur adaptatif avec déport séquentiel des états vers la RAM système.
    Permet de s'affranchir des limitations de mémoire VRAM en streamant 
    les tenseurs de moments via le bus PCIe.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if not 0.0 <= lr: raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps: raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0: raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0: raise ValueError(f"Invalid beta2: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(WramOptim, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    # Allocation des états sur CPU (RAM système) avec verrouillage mémoire
                    state['exp_avg'] = torch.zeros_like(p, device='cpu', pin_memory=True)
                    state['exp_avg_sq'] = torch.zeros_like(p, device='cpu', pin_memory=True)
                    state['step'] = torch.tensor(0.0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            decay = group['weight_decay']
            eps = group['eps']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None: continue
                
                grad = p.grad
                state = self.state[p]
                
                # Mise à jour du compteur de pas
                state['step'] += 1
                step_t = state['step']

                # Application du Weight Decay
                if decay != 0:
                    p.mul_(1 - lr * decay)

                # Transfert séquentiel CPU -> GPU
                m = state['exp_avg'].to(p.device, non_blocking=True)
                v = state['exp_avg_sq'].to(p.device, non_blocking=True)

                # Calcul Adam
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step_t
                bias_correction2 = 1 - beta2 ** step_t
                
                step_size = lr / bias_correction1
                denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # Mise à jour des poids
                p.addcdiv_(m, denom, value=-step_size)

                # Retour séquentiel GPU -> CPU
                state['exp_avg'].copy_(m, non_blocking=True)
                state['exp_avg_sq'].copy_(v, non_blocking=True)
                
                # Nettoyage immédiat des buffers temporaires
                del m, v

        return loss