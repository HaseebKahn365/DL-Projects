import random
import numpy as np
from micrograd.engine import Value
from micrograd.nn import MLP
from sklearn.datasets import make_moons

# Set seeds for reproducibility
np.random.seed(1337)
random.seed(1337)

# Create dataset
X_raw, y_raw = make_moons(n_samples=100, noise=0.1)
y_raw = y_raw * 2 - 1 # make y -1 or 1

# Initialize model with different sizes to prove it works dynamically
model = MLP(2, [16, 8, 1])
print(model)
print("Number of parameters:", len(model.parameters()))

# Training parameters
steps = 50
learning_rate = 0.5

def loss_fn():
    # Batch sampling simulation
    batch_size = 20
    ri = np.random.permutation(X_raw.shape[0])[:batch_size]
    Xb, yb = X_raw[ri], y_raw[ri]
    
    inputs = [list(map(Value, xrow)) for xrow in Xb]
    scores = list(map(model, inputs))
    
    # Margin loss
    losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(yb, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    
    # L2 Regularization
    alpha = 1e-4
    reg_loss = alpha * sum((p*p) for p in model.parameters())
    total_loss = data_loss + reg_loss
    
    # Accuracy
    acc = sum((yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)) / len(yb)
    
    return total_loss, acc, ri

# Training loop
print("Starting training...")
for k in range(steps):
    # Forward and metrics
    total_loss, acc, batch_indices = loss_fn()
    
    # Backward
    model.zero_grad()
    total_loss.backward()
    
    # SGD Update
    lr = learning_rate * (1 - 0.5 * k / steps)
    for p in model.parameters():
        p.data -= lr * p.grad
        
    # RECORD telemetry
    model._debugger.record(k, total_loss, acc, batch_indices)
    
    if k % 10 == 0:
        print(f"step {k} loss {total_loss.data:.4f}, accuracy {acc*100:.1f}%")

print("Training finished. Launching debugger...")
model.show()
