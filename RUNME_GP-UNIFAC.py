import gpytorch
import torch


# 1. Define a pure Multi-Task GP for 1 binary system (Input: x1, Outputs: [g^E/RT, h^E/RT])
class PureBinaryThermodynamicGP(gpytorch.models.ExactGP):

  def __init__(self, train_x, train_y, likelihood):
    super().__init__(train_x, train_y, likelihood)
    self.mean_module = gpytorch.means.MultitaskMean(
        gpytorch.means.ConstantMean(), num_tasks=2
    )
    base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    self.covar_module = gpytorch.kernels.MultitaskKernel(
        base_kernel, num_tasks=2, rank=1
    )

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


# --- SETUP DATA ---
# x: composition (mole fraction x1), shape (N, 1)
x_train = torch.linspace(0.01, 0.99, 40).unsqueeze(-1)
# y: Task 0 = g^E/RT, Task 1 = h^E/RT
y_train = torch.cat(
    [2.0 * x_train * (1 - x_train), 3.0 * x_train * (1 - x_train)], dim=-1
)

# --- INITIALIZE ---
likelihood = gpytorch.likelihoods.MultitaskLikelihood(
    num_tasks=2, likelihood=gpytorch.likelihoods.GaussianLikelihood()
)
model = PureBinaryThermodynamicGP(x_train, y_train, likelihood)

# --- THE TRAINING / OPTIMIZATION LOOP ---
model.train()
likelihood.train()
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(likelihood.parameters()), lr=0.1
)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

print("Optimizing GP kernel hyperparameters...")
for epoch in range(50):
  optimizer.zero_grad()
  output = model(x_train)
  loss = -mll(
      output, y_train
  )  # MLL loss optimizes the GP fit without a neural network
  loss.backward()
  optimizer.step()

# --- GETTING THE ANALYTICAL DERIVATIVE FOR THERMODYNAMICS ---
model.eval()
likelihood.eval()

# Pick a query composition where you want the derivative
x_query = torch.tensor([[0.4]], dtype=torch.float32)
x_query.requires_grad_(True)  # Turn on autograd tracking for the input

# Forward pass through the trained GP
predictions = likelihood(model(x_query))
mean_pred = predictions.mean  # Shape: (1, 2) -> [g^E/RT, h^E/RT]
gE_RT_pred = mean_pred[:, 0:1]

# Because this is in PyTorch, we can instantly compute the exact derivative
# of the GP output with respect to x_query!
(dgE_dx,) = torch.autograd.grad(
    outputs=gE_RT_pred,
    inputs=x_query,
    grad_outputs=torch.ones_like(gE_RT_pred),
    create_graph=False,
)

print(f"\nAt composition x1 = 0.4:")
print(f"Predicted g^E/RT: {gE_RT_pred.item():.4f}")
print(f"Predicted h^E/RT: {mean_pred[:, 1:2].item():.4f}")
print(
    f"Exact Analytical Derivative (dg^E/dx1 from GP): {dgE_dx.item():.4f}"
)  # <--- This is your thermodynamic link!