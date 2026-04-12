import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms
from timeit import default_timer as timer

# ========================
# DEVICE SETUP (MPS)
# ========================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# ========================
# DATA LOADING
# ========================
transform = transforms.ToTensor()

train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

BATCH_SIZE = 32

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# ========================
# MODEL
# ========================
class FashionMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(10, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10 * 7 * 7, 10)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.classifier(x)
        return x

model = FashionMNISTModel().to(device)

# ========================
# LOSS + OPTIMIZER
# ========================
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# ========================
# ACCURACY FUNCTION
# ========================
def accuracy_fn(y_true, y_pred):
    correct = (y_true == y_pred).sum().item()
    return (correct / len(y_pred)) * 100

# ========================
# TRAIN STEP
# ========================
def train_step(model, dataloader):
    model.train()
    train_loss, train_acc = 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        train_loss += loss.item()
        train_acc += accuracy_fn(y, y_pred.argmax(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc

# ========================
# TEST STEP
# ========================
def test_step(model, dataloader):
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            test_pred = model(X)

            test_loss += loss_fn(test_pred, y).item()
            test_acc += accuracy_fn(y, test_pred.argmax(dim=1))

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc

# ========================
# TRAINING LOOP
# ========================
EPOCHS = 3

train_time_start_on_gpu = timer()

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 30)

    train_loss, train_acc = train_step(model, train_loader)
    test_loss, test_acc = test_step(model, test_loader)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%")

train_time_end_on_gpu = timer()
total_train_time_model = train_time_end_on_gpu - train_time_start_on_gpu
print(f"\nTotal training time: {total_train_time_model:.3f} seconds")