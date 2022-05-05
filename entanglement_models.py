import torch

eps = 1e-6
t_eps = torch.Tensor([eps]).cuda()

t_eps_cpu = torch.Tensor([eps])
def signed_lp_norm(inp, p):
    sign = torch.sign(inp)
    inp = torch.maximum(t_eps_cpu, inp * sign)

    lse, signs = signed_logsumexp(torch.log(inp) * p, sign, dim=0)
    return signs * torch.exp(lse / p)

def signed_logsumexp(inp, sign, dim=1):
    maxes = torch.max(inp, dim=dim, keepdim=True).values
    expsum = torch.sum(sign * torch.exp(inp - maxes), dim=dim)
    signs = torch.sign(expsum)
    result = torch.log(torch.maximum(t_eps, expsum * signs)) + maxes.squeeze()
    return result, signs

class PnormEntanglementModel(torch.nn.Module):
    def __init__(self):
        super(PnormEntanglementModel, self).__init__()
        self.p = torch.nn.parameter.Parameter(torch.Tensor([1]))

    def forward(self, r, a, b):
        inp = torch.stack([
            a - r,
            b - r
        ], dim=1)
        sign = torch.sign(inp)
        inp = torch.maximum(t_eps, inp * sign)

        lse, signs = signed_logsumexp(torch.log(inp) * self.p, sign)
        return r + signs * torch.exp(lse / self.p)

class AvgScalingFeatureEntanglementModel(torch.nn.Module):
    def __init__(self):
        super(AvgScalingFeatureEntanglementModel, self).__init__()
        self.alpha = torch.nn.parameter.Parameter(data=torch.randn(1))

    def forward(self, r, a, b):
        da = a - r
        db = b - r

        return r + da + db - (da * db) / (torch.max(da, db) * torch.exp(self.alpha))

class ScalingFeatureEntanglementModel(torch.nn.Module):
    def __init__(self):
        super(ScalingFeatureEntanglementModel, self).__init__()
        self.alpha = torch.nn.parameter.Parameter(data=torch.randn(1))

    def forward(self, r, a, b):
        da = a - r
        db = b - r

        return r + da + db - (da * db) / (r * torch.exp(self.alpha))

class UniformFeatureEntanglementModel(torch.nn.Module):
    def __init__(self):
        super(UniformFeatureEntanglementModel, self).__init__()
        self.alpha = torch.nn.parameter.Parameter(data=torch.randn(1))

    def forward(self, r, a, b):
        da = a - r
        db = b - r

        return r + da + db - da * db / torch.exp(self.alpha)

class AdditiveBaselineEntanglementModel(torch.nn.Module):
    def __init__(self):
        super(AdditiveBaselineEntanglementModel, self).__init__()

    def forward(self, r, a, b):
        return a + b - r

class LinearBaselineEntanglementModel(torch.nn.Module):
    def __init__(self):
        super(LinearBaselineEntanglementModel, self).__init__()
        self.m = torch.nn.parameter.Parameter(data=torch.randn(1))

    def forward(self, r, a, b):
        return r + ((a - r) + (b - r)) * self.m
