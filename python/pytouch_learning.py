import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device on pytoch version {torch.__version__}")

data_dir = "~/dl4j-examples-data/dl4j-examples/flower_photos"

# Define the batch size
batch_size = 16

# Define the number of epochs to train for
num_epochs = 3

# Define the transformations for the data
transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# Load the dataset and split it into training and validation sets
dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
num_train = int(0.8 * len(dataset))
num_val = len(dataset) - num_train
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train, num_val])

# Create data loaders for the training and validation sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the ResNet50 model and the optimizer
model = torchvision.models.resnet50(weights=None)
model.fc = nn.Linear(2048, len(dataset.classes))
model = model.to(device)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Train the model for the specified number of epochs
start_time = time.time()

for epoch in range(3):
    train_loss = 0.0
    train_correct = 0.0
    train_total = 0.0
    val_loss = 0.0
    val_correct = 0.0
    val_total = 0.0

    # Train the model on the training set
    model.train()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        # Print the training progress
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} Batch {batch_idx+1}/{len(train_loader)} "
                  f"Loss: {train_loss/train_total:.4f} Acc: {100*train_correct/train_total:.2f}%")

    print("Starting validation")
    # Evaluate the model on the validation set
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    # Print the validation progress
    print(f"Epoch {epoch+1}/{num_epochs} "
          f"Val Loss: {val_loss/val_total:.4f} Val Acc: {100*val_correct/val_total:.2f}%")

end_time = time.time()
# Track the total training time
total_time = end_time - start_time
print(f"Total training time: {total_time:.2f} seconds")
