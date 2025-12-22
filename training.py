import torch.optim as optim
from network import model, train_loader
import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


for epoch in range(3):
    model.train()
    correct, total = 0, 0

    for X, y in train_loader:
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(logits, 1)
        correct += (preds == y).sum().item()
        total += y.size(0)




