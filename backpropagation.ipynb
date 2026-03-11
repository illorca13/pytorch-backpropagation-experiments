# Question 1
# a
import torch
a = torch.tensor(list(range(9)))

print(a)

# b
b = a.view(3, 3)
print(b)

# check 
a[0] = 99
print(b) # changed therefore share memory

# c
c = b[1:,1:]
print(c)

# d
d = torch.sqrt_(a.float())
print(d)

# Question 2
# a) scalar-valued function

import torch
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = torch.tensor(1.0)
loss = (x * y + z) ** 2



# b) loss
loss.backward() 
print(x.grad, y.grad) # shows the gradient of the leaf tensors

# c) detatching
u = (x*y).detach




# Question 3 - Linear regression with manual gradients
import torch
torch.manual_seed(0)
# dataset
# y = 2x + 0.5 + noise
t_c = torch.linspace(-1, 1, 20) # inputs x evenly spaced 20 numbers from -1 to 1
t_u = 2.0 * t_c + 0.5 + 0.1 * torch.randn_like(t_c)   # targets y



# a) manual gradient model
# y_hat = w*x + b

def model(t_c, w, b):
    return w * t_c + b

# Loss (scalar objective): MSE
def loss_fn(t_u, t_p):
    return ((t_p - t_u) ** 2).mean() # t_p predicted values



# b) Manual gradients of MSE with respect w and b

# L = mean((w*x + b - y)^2)
# dL/dw = (2/n) * sum( (w*x + b - y) * x )
# dL/db = (2/n) * sum( (w*x + b - y) )

def grad_fn(t_c, t_u, t_p):
    n = t_c.numel()
    d_loss_d_tp = 2.0 * (t_p - t_u) / n  # derivative of mean squared error  predictions
    d_tp_d_w = t_c  # derivative of (w*x + b) with respect w
    d_tp_d_b = 1.0  # derivative of (w*x + b) with respect b

    grad_w = (d_loss_d_tp * d_tp_d_w).sum()
    grad_b = (d_loss_d_tp * d_tp_d_b).sum()
    return grad_w, grad_b

# c) Gradient descent loop

w = torch.tensor(0.0)   # initial slope
b = torch.tensor(0.0)   # initial intercept
learning_rate = 1e-2 
n_epochs = 200

for epoch in range(1, n_epochs + 1):
    # Forward pass
    t_p = model(t_c, w, b) # predictions
    loss = loss_fn(t_u, t_p) # loss achieved

    # Manual backward pass (gradients)
    grad_w, grad_b = grad_fn(t_c, t_u, t_p)

    # Gradient descent update
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

    # Observe learning
    if epoch == 1 or epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss {loss.item():.6f} | w {w.item():.4f} | b {b.item():.4f}")

print("\nFinal parameters:")
print("w =", w.item())
print("b =", b.item())


# question 4 
# a)
import torch.nn as nn

model = nn.Linear(1, 1) # creates w and b automatically

# b) 
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD (model.parameters(), lr=0.1)

n_epochs = 200

for epoch in range(1, n_epochs + 1):
    # Forward pass (same idea as Q3): predict
    t_p = model(t_c)

    # Compute scalar loss (same idea as Q3): MSE
    loss = loss_fn(t_p, t_u)

    # Backward pass + update (now automated)
    optimizer.zero_grad()   # important because gradients accumulate
    loss.backward()         # compute gradients automatically
    optimizer.step()        # update parameters

    if epoch == 1 or epoch % 20 == 0:
        print(
            f"Epoch {epoch:3d} | Loss {loss.item():.6f} | "
            f"w {model.weight.item():.4f} | b {model.bias.item():.4f}"
        )

# c)
for name, param in model.named_parameters():
    print(name, param.data, param.grad)


# Question 5
# a)
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(0)

# dataset (same style as before)
x = torch.linspace(-1, 1, 20).unsqueeze(1)               # (20,1)
y = 2.0 * x + 0.5 + 0.1 * torch.randn_like(x)            # (20,1)

loss_fn = nn.MSELoss()

def make_model(hidden_units, activation):
    return nn.Sequential(
        nn.Linear(1, hidden_units),
        activation,
        nn.Linear(hidden_units, 1)
    )

def grad_stats(model):
    # simple gradient magnitude summary
    mags = []
    for p in model.parameters():
        if p.grad is not None:
            mags.append(p.grad.abs().mean().item())
    return sum(mags) / len(mags) if mags else 0.0

for H in [1, 4, 16, 64]:
    model = make_model(H, nn.Tanh())
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(1, 201):
        y_hat = model(x)
        loss = loss_fn(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        g = grad_stats(model)
        optimizer.step()

        if epoch in [1, 20, 50, 100, 200]:
            print(f"H={H:2d} | epoch={epoch:3d} | loss={loss.item():.6f} | grad_mean={g:.6e}")
