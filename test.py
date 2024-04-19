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
        return torch.tensor(self.coords[idx], dtype=torch.float32), torch.tensor(self.images[idx],
                                                                                 dtype=torch.float32).permute(0, 3, 1,
                                                                                                              2)


def load_model(model_path):
    model = ReflectionProbeCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def generate_reflection_textures(model, coords):
    with torch.no_grad():
        coords_tensor = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        outputs = model(coords_tensor)
        outputs = outputs.squeeze(0).permute(0, 2, 3, 1)  # Remove batch dimension and permute to (6, 128, 128, 3)
        return outputs.numpy()


def save_textures_as_png(textures, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    faces = ['PositiveX', 'NegativeX', 'PositiveY', 'NegativeY', 'PositiveZ', 'NegativeZ']
    for i, face in enumerate(faces):
        img = (textures[i] * 255).astype(np.uint8)  # Convert to 8-bit pixel values
        img = Image.fromarray(img)
        img.save(os.path.join(output_dir, f'reflection_{face}.png'))


def main():
    model_path = "model2.pth"
    output_dir = "output"
    model = load_model(model_path)

    # Generate random coordinates (example: [x, y, z] within the range of your scene)
    random_coords = [0.0, 1.0, 0.0]  # Adjust this as needed

    textures = generate_reflection_textures(model, random_coords)
    save_textures_as_png(textures, output_dir)
    print(f"Reflection textures saved to {output_dir}")


if __name__ == "__main__":
    main()
