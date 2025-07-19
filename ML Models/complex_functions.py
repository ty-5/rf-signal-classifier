import torch
import torch.nn.functional as F
'''
This file contains commonly used PyTorch functions that can support complex128
'''

#Theoretically better at preserving phase but requires dynamic bias calculation
def magnitude_complex_ReLU(tensor: torch.complex, bias = 0):
    magnitude = ((tensor.imag ** 2) + (tensor.real ** 2)) ** (1/2)
    angle = torch.atan((tensor.imag) / (tensor.real))
    
    magnitude = F.relu(magnitude + bias)
    
    return magnitude * torch.exp(1j * angle)


#Consistent but naive due to splitting the complex tensor into 2 float tensors
def naive_complex_ReLU(tensor: torch.complex):
    real = tensor.real
    imag = tensor.imag 
    
    return torch.complex(F.relu(real), F.relu(imag))
    
x = torch.tensor([-2 + -3j], dtype=torch.complex128)
y = torch.tensor(2 + 1j, dtype=torch.complex128)

print(magnitude_complex_ReLU(y, -5))