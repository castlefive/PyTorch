import torch
from torch import optim
import math

avengers_series = torch.FloatTensor([[1], [2], [3]])
attendance_in_UBD = torch.FloatTensor([[42], [62], [66]])

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=0.01)

for i in range(5001):
    hypothesis = W * avengers_series + b
    cost = torch.mean((hypothesis - attendance_in_UBD) ** 2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if i % 1000 == 0:
        print(i, W.item(), b.item(), cost.item())

expected_audiences = float((W.item()) * 4 + float(b.item()))
print('어벤져스4 예상 관객 수(UBD):', math.floor(expected_audiences))
