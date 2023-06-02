import wandb
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_NAME = 'pytorch-resume-run'
CHECKPOINT_PATH = 'checkpoint.pt'
N_EPOCHS = 100

# Dummy data
X = torch.randn(64, 8, requires_grad=True)
Y = torch.empty(64, 1).random_(2)
model = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)
metric = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
epoch = 0

#run_type = "new"
run_type = "resuming"
if run_type == "new":
    wandb_id = wandb.util.generate_id()
elif run_type == "resuming":
    wandb_id = "xm30rlyx"

run = wandb.init(project=PROJECT_NAME, id=wandb_id)
print(f"RESUMING: {bool(wandb.run.resumed)}")
if wandb.run.resumed:
    checkpoint = torch.load(wandb.restore(CHECKPOINT_PATH))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"RESUMING: {wandb_id}; EPOCH: {epoch}; LOSS: {loss}")

model.train()
while epoch < N_EPOCHS:
    optimizer.zero_grad()
    output = model(X)
    loss = metric(output, Y)
    wandb.log({'loss': loss.item()}, step=epoch)
    loss.backward()
    optimizer.step()
    print(epoch)
    if ((epoch+1) % int(N_EPOCHS/2) == 0) and run_type=="new":
        raise Exception("ceasing training...")
    save_dict = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}
    torch.save(save_dict, CHECKPOINT_PATH)
    wandb.save(CHECKPOINT_PATH) # saves checkpoint to wandb
    epoch += 1