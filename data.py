import kagglehub
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


def load_dataloaders(path):


    train_df = pd.read_csv(path + "/sign_mnist_test.csv")
    test_df = pd.read_csv(path + "/sign_mnist_train.csv")

    print(train_df.head()) # wir haben 28x28 Bilder (784 Pixel) und immer ein label

    X_train = train_df.iloc[:, 1:].values / 255.0
    y_train = train_df.iloc[:, 0].values

    X_test  = test_df.iloc[:, 1:].values / 255.0
    y_test  = test_df.iloc[:, 0].values


    X_train = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, 28, 28)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_test  = torch.tensor(X_test, dtype=torch.float32).view(-1, 1, 28, 28)
    y_test  = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    test_ds  = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=128)

    return train_loader, test_loader




