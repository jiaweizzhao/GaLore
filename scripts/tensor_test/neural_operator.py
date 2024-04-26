"""
Training a neural operator on Darcy-Flow - Author Robert Joseph
========================================
In this example, we demonstrate how to use the small Darcy-Flow example we ship with the package on Incremental FNO and Incremental Resolution as well as using Galore tensor decomposition.

Assuming one installs the neuraloperator library: Instructions can be found here: https://github.com/NeuralOperator/neuraloperator
"""

# %%
#
import torch
import matplotlib.pyplot as plt
import sys
from neuralop.training.callbacks import BasicLoggerCallback
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop.training.callbacks import IncrementalCallback
from neuralop.datasets import data_transforms
from neuralop import LpLoss, H1Loss
from neuralop.training import AdamW
from neuralop.utils import count_model_params


# %%
# Loading the Darcy flow dataset
train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=1000, batch_size=32, 
        test_resolutions=[16, 32], n_tests=[100, 50],
        test_batch_sizes=[32, 32],
        positional_encoding=True
)
# %%
# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Set up the incremental FNO model
# We start with 2 modes in each dimension
# We choose to update the modes by the incremental gradient explained algorithm

starting_modes = (10, 10)
incremental = False

model = FNO(
    max_n_modes=(20, 20),
    n_modes=starting_modes,
    hidden_channels=64,
    in_channels=1,
    out_channels=1,
    n_layers=4
)
callbacks = [
    IncrementalCallback(
        incremental_loss_gap=True,
        incremental_grad=False,
        incremental_grad_eps=0.9999,
        incremental_buffer=5,
        incremental_max_iter=1,
        incremental_grad_max_iter=2,
    )
]     
model = model.to(device)
n_params = count_model_params(model)
galore_params = []
galore_params.extend(list(model.fno_blocks.convs.parameters()))
print(galore_params[0].shape, galore_params[1].shape, galore_params[2].shape, galore_params[3].shape)
galore_params.pop(0)
id_galore_params = [id(p) for p in galore_params]
# make parameters without "rank" to another group
regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
# then call galore_adamw
# In this case we have a 5d tensor representing the weights in the spectral layers of the FNO
# A good rule of thumb for tensor decomposition is that we should limit the rank to atmost 0.75, and increase the epochs and tune the lr accordingly compared to the baseline.
# Low rank decomposition takes longer to converge, but it is more memory efficient.
param_groups = [{'params': regular_params}, 
                {'params': galore_params, 'rank': 0.2 , 'update_proj_gap': 10, 'scale': 0.25, 'proj_type': "std", 'dim': 5}]
optimizer = AdamW(param_groups, lr=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
data_transform = data_transforms.IncrementalDataProcessor(
    in_normalizer=None,
    out_normalizer=None,
    positional_encoding=None,
    device=device,
    dataset_sublist=[2, 1],
    dataset_resolution=16,
    dataset_indices=[2, 3],
    epoch_gap=10,
    verbose=True,
)

data_transform = data_transform.to(device)
# %%
# Set up the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
train_loss = h1loss
eval_losses = {"h1": h1loss, "l2": l2loss}
print("\n### OPTIMIZER rank ###\n", i, optimizer)
sys.stdout.flush()

# Finally pass all of these to the Trainer
trainer = Trainer(
    model=model,
    n_epochs=100,
    data_processor=data_transform,
    callbacks=callbacks,
    device=device,
    verbose=True,
)

# %%
# Train the model
trainer.train(
    train_loader,
    test_loaders,
    optimizer,
    scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
)

# %%
# Plot the prediction, and compare with the ground-truth
# Note that we trained on a very small resolution for
# a very small number of epochs
# In practice, we would train at larger resolution, on many more samples.
#
# However, for practicity, we created a minimal example that
# i) fits in just a few Mb of memory
# ii) can be trained quickly on CPU
#
# In practice we would train a Neural Operator on one or multiple GPUs

test_samples = test_loaders[32].dataset

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index]
    # Input x
    x = data["x"].to(device)
    # Ground-truth
    y = data["y"].to(device)
    # Model prediction
    out = model(x.unsqueeze(0))
    ax = fig.add_subplot(3, 3, index * 3 + 1)
    x = x.cpu().squeeze().detach().numpy()
    y = y.cpu().squeeze().detach().numpy()
    ax.imshow(x, cmap="gray")
    if index == 0:
        ax.set_title("Input x")
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index * 3 + 2)
    ax.imshow(y.squeeze())
    if index == 0:
        ax.set_title("Ground-truth y")
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index * 3 + 3)
    ax.imshow(out.cpu().squeeze().detach().numpy())
    if index == 0:
        ax.set_title("Model prediction")
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle("Inputs, ground-truth output and prediction.", y=0.98)
plt.tight_layout()
fig.show()