import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.onnx

from NN import ReflectionProbeCNN


class ReflectionDataset(Dataset):
    def __init__(self, img_dir, coord_file):
        self.img_dir = img_dir
        self.images = []
        self.coords = []

        # Load coordinates
        with open(coord_file, 'r') as f:
            for line in f:
                parts = line.strip().split(':')
                name = parts[0].strip()
                coords = parts[1].strip().split(',')

                # Ensure we have exactly three coordinate values
                if len(coords) != 3:
                    print(f"Skipping invalid coordinate entry for {name}: {coords}")
                    continue

                try:
                    coords = list(map(float, coords))
                except ValueError as e:
                    print(f"Skipping invalid coordinate values for {name}: {coords} - Error: {e}")
                    continue

                self.coords.append(coords)

                # Load images
                img_faces = []
                for i in range(6):
                    img_path = os.path.join(img_dir, f"{name}_face{i}.png")
                    if not os.path.exists(img_path):
                        print(f"Skipping missing image file for {name}_face{i}")
                        break
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((128, 128))  # Ensure all images are the same size
                    img_faces.append(np.array(img))

                # Ensure all six faces are loaded
                if len(img_faces) == 6:
                    self.images.append(np.stack(img_faces, axis=0))  # Shape: (6, H, W, 3)
                else:
                    print(f"Skipping incomplete image set for {name}")

        self.images = np.array(self.images) / 255.0  # Normalize images
        self.coords = np.array(self.coords)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Return coordinates and images, reshape images to match model output shape
        return torch.tensor(self.coords[idx], dtype=torch.float32), torch.tensor(self.images[idx], dtype=torch.float32).permute(0, 3, 1, 2)


# Paths
img_dir = "EnvironmentMaps"  # Update this path to your environment maps folder
coord_file = os.path.join(img_dir, "coordinates.txt")

# Create dataset and dataloader
dataset = ReflectionDataset(img_dir, coord_file)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss, and optimizer
model = ReflectionProbeCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

min_loss = float('inf')
min_loss_epoch = -1

# Training loop
num_epochs = 5000
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for coords, images in dataloader:
        optimizer.zero_grad()
        outputs = model(coords)
        # Reshape images to match model output shape
        images = images.permute(0, 1, 2, 3, 4)  # From [batch_size, 6, 128, 128, 3] to [batch_size, 6, 3, 128, 128]
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * coords.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.12f}')
    if epoch_loss < min_loss:
        min_loss = epoch_loss
        min_loss_epoch = epoch + 1

print(f'The epoch with the minimum loss is {min_loss_epoch} with a loss of {min_loss:.12f}')

# Save the trained model
torch.save(model.state_dict(), "model2.pth")

# Load the trained model
model.load_state_dict(torch.load("model2.pth"))
model.eval()

# Dummy input for the model
dummy_input = torch.randn(1, 3)

# Export the model
torch.onnx.export(model, dummy_input, "model2.onnx",
                  input_names=['input'], output_names=['output'])
