import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim

from data import load_dataloaders
from network import SignCNN
from training import train_one_epoch, evaluate
from matplotlib import pyplot as plt

def test_single_example(model, test_loader, device):
    model.eval()

    X, y = next(iter(test_loader))   # ein Batch

    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)

    img = X[0].cpu().squeeze()
    true_label = y[0].item()
    pred_label = preds[0].item()

    print("True label:", true_label)
    print("Pred label:", pred_label)

    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()

def main():
    path = kagglehub.dataset_download("datamunge/sign-language-mnist")
    print(path)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    train_loader, test_loader = load_dataloaders(path)

    model = SignCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(15):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}: accuracy={acc:.3f}")
        test_single_example(model, test_loader, device)



if __name__ == "__main__":
    main()

