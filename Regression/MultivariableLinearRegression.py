import torch
from torch import optim
import math

avengers_data = torch.FloatTensor([
    [56, 14, 18],
    [94, 29, 39],
    [159, 35, 50]
])

attendance_in_UBD = torch.FloatTensor([[42], [62], [66]])

W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=1e-5)

cost = 15
while int(cost) > 14:
    hypothesis = avengers_data.matmul(W) + b
    cost = torch.mean((hypothesis - attendance_in_UBD) ** 2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

avengers4_data = torch.FloatTensor([
    [214, 44, 66]
])

print('Cost:', cost.item())
print('어벤져스4 예상 관객 수(UBD):', math.floor(avengers4_data.matmul(W) + b))
