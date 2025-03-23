import os
import requests
import zipfile
from io import BytesIO
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

# Define dataset download and extraction function
def download_and_unzip(url="http://cs231n.stanford.edu/tiny-imagenet-200.zip", extract_path="data/", verbose=True):
    """Downloads and extracts dataset if not already present."""
    dataset_folder = os.path.join(extract_path, "tiny-imagenet-200")
    
    if not os.path.exists(dataset_folder):
        print("Downloading dataset...")
        
        # Use requests to stream the download
        response = requests.get(url, stream=True)
        
        # Check if the request was successful
        if response.status_code == 200:
            total_size_in_bytes = int(response.headers.get('content-length', 0))  # Get the total size of the content
            chunk_size = 1024  # 1k chunk size for the download
            
            # Use tqdm to show download progress if verbose is True
            if verbose:
                with tqdm(total=total_size_in_bytes, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    data = BytesIO()
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        data.write(chunk)
                        pbar.update(len(chunk))  # Update the progress bar by the chunk size
            else:
                data = BytesIO()
                for chunk in response.iter_content(chunk_size=chunk_size):
                    data.write(chunk)
                
            data.seek(0)  # Move the cursor to the beginning of the BytesIO object
            with zipfile.ZipFile(data, 'r') as zip_ref:
                if verbose:
                    for file in tqdm(zip_ref.infolist(), desc="Extracting"):
                        zip_ref.extract(file, extract_path)
                else:
                    zip_ref.extractall(extract_path)
            print(f"Dataset extracted to {extract_path}")
        else:
            print("Failed to download dataset.")
    else:
        print("Dataset already exists, skipping download.")


def get_default_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to fit the input dimensions of the network
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def get_dataloaders(data_dir="data/tiny-imagenet-200", transform=get_default_transforms(), batch_size_train=32, batch_size_val=16, num_train_workers=8):
    
    train_dataset = ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)
    val_dataset = ImageFolder(root=os.path.join(data_dir, "val"), transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=num_train_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)

    return train_loader, val_loader

def denormalize(image):
    """Denormalizes an ImageNet tensor image for visualization."""
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    return np.clip(image, 0, 1)

def visualize_samples(dataloader, num_classes=10):
    """Visualizes a few samples from the dataset."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    sampled_classes = set()
    
    for images, labels in dataloader:
        for img, label in zip(images, labels):
            if label.item() not in sampled_classes:
                row, col = divmod(len(sampled_classes), 5)
                axes[row, col].imshow(denormalize(img))
                axes[row, col].axis("off")
                sampled_classes.add(label.item())
                
            if len(sampled_classes) >= num_classes:
                break
        if len(sampled_classes) >= num_classes:
            break
            
    plt.show()


if __name__ == "__main__":
    user_input = input("Do you want to download and extract the dataset? (Y/[N]): ").strip().lower()
    if user_input == 'y':
        download_and_unzip()
    
    train_loader, test_loader = get_dataloaders()
    num_classes = len(train_loader.dataset.classes)

    print(f"Number of classes: {num_classes}")
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of test samples: {len(test_loader.dataset)}")

    visualize_samples(train_loader)