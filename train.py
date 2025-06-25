import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from AlexNet_pytorch import AlexNetTorch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Check for MPS (Metal Performance Shaders) availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS for training")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA for training")
else:
    device = torch.device("cpu")
    print("Using CPU for training")

def get_train_valid_loader(data_dir, batch_size, augment, random_seed, valid_size=0.1, shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    valid_transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transform
    transform = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return data_loader

def download_and_prepare_dataset(data_dir='./data'):
    """Download and prepare CIFAR-10 dataset with progress tracking"""
    print(f"Downloading CIFAR-10 dataset to {data_dir}...")
    print("Dataset size: ~170MB (compressed) / ~1.6GB (uncompressed)")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Download training set
    print("\nDownloading training set...")
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    
    # Download test set
    print("\nDownloading test set...")
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    print(f"\nDataset downloaded and prepared in {data_dir}")
    
    return train_dataset, test_dataset

# Download dataset
print("\nInitializing training...")
train_dataset, test_dataset = download_and_prepare_dataset()

# CIFAR10 dataset 
print("\nPreparing data loaders...")
train_loader, valid_loader = get_train_valid_loader(data_dir = './data',batch_size = 64,augment = False,random_seed = 1)
test_loader = get_test_loader(data_dir = './data', batch_size = 64)

num_classes = 10
num_epochs = 20
batch_size = 64
learning_rate = 0.005

print("\nInitializing model...")
model = AlexNetTorch(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  

print("\nStarting training...")
print(f"Training on device: {device}")
print(f"Number of epochs: {num_epochs}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {learning_rate}")

# Train the model
total_step = len(train_loader)
validation_frequency = 100  # Validate every 100 steps

# Lists to store metrics for plotting
train_losses = []
val_losses = []
val_accuracies = []
epochs = []

# Create plots directory
os.makedirs('plots', exist_ok=True)

def save_training_plots(epoch, train_loss, val_loss, val_acc):
    """Save training plots after each epoch"""
    epochs.append(epoch)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/training_progress_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training plots saved for epoch {epoch}")

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        # Validation during training
        if (i + 1) % validation_frequency == 0:
            model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for val_images, val_labels in valid_loader:
                    val_images = val_images.to(device)
                    val_labels = val_labels.to(device)
                    val_outputs = model(val_images)
                    val_loss += criterion(val_outputs, val_labels).item()
                    _, predicted = torch.max(val_outputs.data, 1)
                    total += val_labels.size(0)
                    correct += (predicted == val_labels).sum().item()
                    del val_images, val_labels, val_outputs
            
            avg_val_loss = val_loss / len(valid_loader)
            val_accuracy = 100 * correct / total
            
            print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.2f}%'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item(), avg_val_loss, val_accuracy))
            
            model.train()  # Set model back to training mode

    # End of epoch validation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        val_loss = 0.0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        avg_val_loss = val_loss / len(valid_loader)
        print('Epoch [{}/{}] completed - Train Loss: {:.4f}, Val Loss: {:.4f}, Val Accuracy: {:.2f}%'
              .format(epoch+1, num_epochs, running_loss/len(train_loader), avg_val_loss, 100 * correct / total))

    # Save training plots after each epoch
    save_training_plots(epoch+1, running_loss/len(train_loader), avg_val_loss, 100 * correct / total)

# Final comprehensive plot
def create_final_summary_plot():
    """Create a comprehensive final plot with all training metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Training and Validation Loss
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Accuracy
    ax2.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy', linewidth=2, marker='^')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Loss Difference (Overfitting indicator)
    loss_diff = [abs(t - v) for t, v in zip(train_losses, val_losses)]
    ax3.plot(epochs, loss_diff, 'purple', label='|Train Loss - Val Loss|', linewidth=2, marker='d')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss Difference')
    ax3.set_title('Overfitting Indicator (Loss Difference)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Training Summary Table
    ax4.axis('off')
    summary_text = f"""
Training Summary:
================
Total Epochs: {len(epochs)}
Final Training Loss: {train_losses[-1]:.4f}
Final Validation Loss: {val_losses[-1]:.4f}
Final Validation Accuracy: {val_accuracies[-1]:.2f}%
Best Validation Accuracy: {max(val_accuracies):.2f}%
Best Epoch: {epochs[val_accuracies.index(max(val_accuracies))]}

Model Configuration:
===================
Learning Rate: {learning_rate}
Batch Size: {batch_size}
Optimizer: SGD with momentum
Weight Decay: 0.005
Momentum: 0.9
    """
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('plots/final_training_summary.png', dpi=300, bbox_inches='tight')
    plt.show()  # Display the final plot
    print("Final training summary plot saved as 'plots/final_training_summary.png'")

# Create final summary plot
create_final_summary_plot()