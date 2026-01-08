# sam.py
import torch


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer wrapper.

    Usage:
        base_opt = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_opt, lr=1e-3, momentum=0.9)
    """
    def __init__(self, params, base_optimizer, rho: float = 0.05, **kwargs):
        if rho <= 0:
            raise ValueError("Invalid rho, should be > 0")
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2) for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm.to(shared_device)

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        scale = self.param_groups[0]["rho"] / (grad_norm + 1e-12)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()

    @torch.no_grad()
    def zero_grad(self):
        self.base_optimizer.zero_grad()
