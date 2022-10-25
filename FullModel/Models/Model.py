from torch import nn
class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB


    def forward(self, x1, x2):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)

        return