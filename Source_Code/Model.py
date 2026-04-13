import torch
from torch import nn

class LinearRegressionModel(nn.Module): 

    def __init__(self):
        super().__init__() 
        """
        Start Coding Here
        """
        self.weights = nn.Parameter(torch.rand(1))
        self.bias = nn.Parameter(torch.rand(1))

    def forward(self, x):
        """
        Start Coding Here
        """
        y = self.weights * x +self.bias
        return y


class PolynomialRegressionModel(nn.Module):
    def __init__(self, degree):
        super().__init__()
        self.degree = degree
        """
        Start Coding Here
        """
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.randn(1)) for i in range(self.degree + 1)])
    
    def forward(self, x):
        x = x.squeeze()
        y = self.weights[0] * torch.ones_like(x)
        for i in range(1, self.degree + 1):
            y = y + self.weights[i] * x.pow(i)
        return y
    

class LogisticRegression(nn.Module):
    def __init__(self,num_input):
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(num_input,1)

    def forward(self,x):
        y = torch.sigmoid(self.linear(x))
        return y
