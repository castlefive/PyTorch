import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


score_train_data = [[3, 2],
                    [0, 1],
                    [1, 0],
                    [3, 4],
                    [4, 0],
                    [1, 2]]

result_train_data = [[1],
                     [0],
                     [1],
                     [0],
                     [1],
                     [0]]

score_train = torch.FloatTensor(score_train_data)
result_train = torch.FloatTensor(result_train_data)

model = BinaryClassifier()
optimizer = optim.SGD(model.parameters(), lr=1)

for step in range(1000):
    hypothesis = model(score_train)
    cost = F.binary_cross_entropy(hypothesis, result_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if step % 100 == 0:
        print(step, cost.item())

score_test_data = [[2, 1]]
result_test_data = [1]

score_test = torch.FloatTensor(score_test_data)
result_test = torch.FloatTensor(result_test_data)

hypothesis = model(score_test)
prediction = hypothesis >= torch.FloatTensor([0.5])
correct_prediction = prediction.float() == result_test
print(correct_prediction)
