import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchcrf import CRF  # Install with pip install pytorch-crf
import numpy as np
from Classificaltion_Evaluation import ClassificationEvaluation


class SimpleADCRF(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SimpleADCRF, self).__init__()
        # Adaptive Deep component: a small MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        # CRF layer
        self.crf = CRF(num_classes, batch_first=True)

    def forward(self, x):
        # x: [batch, seq_len, features] (for 1D data, seq_len=1)
        emissions = self.fc1(x)
        emissions = self.relu(emissions)
        emissions = self.fc2(emissions)  # [batch, seq_len, num_classes]
        return emissions


def Model_ADCRF(train_data, train_target, test_data, test_target, sol=None):
    print("Adaptive Deep Conditional Random Field (ADCRF)")
    if sol is None:
        sol = [5, 5, 0.01]
    HN = int(sol[0])
    lr = sol[2]
    epochs = int(sol[1])

    # Convert data to fixed-size input
    IMG_SIZE = 10

    def preprocess(data):
        temp = np.zeros((data.shape[0], IMG_SIZE))
        for i in range(data.shape[0]):
            temp[i, :] = np.resize(data[i], IMG_SIZE)
        return temp

    train_data = preprocess(train_data)
    test_data = preprocess(test_data)

    # Convert to PyTorch tensors
    X_train = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1)  # [batch, seq_len=1, features]
    X_test = torch.tensor(test_data, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(train_target, dtype=torch.long)
    y_test = torch.tensor(test_target, dtype=torch.long)

    num_classes = y_train.shape[1]

    # Create dataset and loader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = SimpleADCRF(input_dim=IMG_SIZE, hidden_dim=HN, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            emissions = model(x_batch)  # [batch, seq_len, num_classes]
            # CRF expects [batch, seq_len, num_classes] and [batch, seq_len]
            loss = -model.crf(emissions, y_batch)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        emissions = model(X_test)
        pred = model.crf.decode(emissions)  # Returns list of sequences
        pred = np.array([p[0] for p in pred]).reshape(y_test.shape)

    avg = np.mean(pred)
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = ClassificationEvaluation(y_test.numpy(), pred)
    return Eval
