import torch

'''
class FullNeuralAssociativeEntanglementModel(torch.nn.Module):
    def __init__(self, hiddens = 1024):
        super(FullNeuralAssociativeEntanglementModel).__init__()
        self.linear1 = torch.nn.Linear(1, hiddens)
        self.linear2 = torch.nn.Linear(hiddens, 1)

    def single_f(self, x):
        return self.linear2(torch.nn.functional.relu(self.linear1(x)))

    def forward(self, r, a, b):
        da = a - r
        db = b - r

        fda = self.single_f(

        dc = self.linear1(
'''

eps = 1e-6
class LogLogEntanglementModel(torch.nn.Module):
    def __init__(self):
        super(LogLogEntanglementModel, self).__init__()
        self.alpha = torch.nn.parameter.Parameter(data=torch.randn(1))

    def forward(self, r, a, b):
        da = a - r
        db = b - r

        combined = torch.exp(torch.logsumexp(
            torch.stack([
                torch.log(abs(da) + eps) * self.alpha,
                torch.log(abs(db) + eps) * self.alpha
            ]), 0, keepdim=False
        ) / self.alpha)

        return r + combined

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
