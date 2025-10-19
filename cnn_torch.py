import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from multiprocessing import freeze_support

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool  = nn.MaxPool2d(2, 2)

        # After two 2x2 pools: 64x64 -> 32x32 -> 16x16, with 32 channels
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    basic_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # Datasets
    train_dataset = datasets.ImageFolder(root='GSDogsAndCats', transform=basic_transform)
    eval_dataset  = datasets.ImageFolder(root='Eval',           transform=basic_transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    eval_loader  = DataLoader(eval_dataset,  batch_size=64, shuffle=False)

    # Model, loss, optimizer (no AMP, no scheduler)
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch}/{num_epochs}]  Loss: {epoch_loss:.4f}")

    # Evaluation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in eval_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = 100. * correct / total
    print(f"Evaluation Accuracy: {accuracy:.2f}%")

    # Save the model
    # save_path = 'simple_cnn_model_minimal.pth'
    # torch.save(model.state_dict(), save_path)
    # print(f"Model saved to {save_path}")

if __name__ == "__main__":
    freeze_support()
    main()
