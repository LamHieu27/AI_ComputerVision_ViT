import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.config import ProjectConfig

class NIHChestXrayDataset(Dataset):
    """
    NIH Chest X-Ray 14 PyTorch Dataset supporting multi-label classification.
    """
    def __init__(self, split="train", transform=None, limit_samples=None, random_state=42):
        self.image_dir = ProjectConfig.IMAGE_DIR
        self.class_names = ProjectConfig.CLASS_NAMES
        self.transform = transform
        
        # Load dataset CSV metadata
        if not os.path.exists(ProjectConfig.CSV_PATH):
            raise FileNotFoundError(f"Không tìm thấy file CSV tại {ProjectConfig.CSV_PATH}")
        self.df = pd.read_csv(ProjectConfig.CSV_PATH)
        
        # Load the list of image filenames for the current split
        if split in ["train", "val"]:
            split_file = ProjectConfig.TRAIN_VAL_LIST
        elif split == "test":
            split_file = ProjectConfig.TEST_LIST
        else:
            raise ValueError("Split must be one of 'train', 'val', or 'test'")
            
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Không tìm thấy file danh sách phân chia tại {split_file}")
            
        with open(split_file, "r") as f:
            split_images = set(line.strip() for line in f if line.strip())
            
        # Filter metadata for this split
        self.df = self.df[self.df["Image Index"].isin(split_images)].reset_index(drop=True)
        
        # Create a split for train and validation if train_val is requested
        if split in ["train", "val"]:
            # Perform a stable patient-level or index-level split (80% train, 20% val)
            # Patient-level split prevents data leakage (same patient in both train and val)
            # NIH dataset has Patient ID column
            unique_patients = self.df["Patient ID"].unique()
            np.random.seed(random_state)
            np.random.shuffle(unique_patients)
            
            train_size = int(0.8 * len(unique_patients))
            train_patients = set(unique_patients[:train_size])
            
            if split == "train":
                self.df = self.df[self.df["Patient ID"].isin(train_patients)].reset_index(drop=True)
            else:
                self.df = self.df[~self.df["Patient ID"].isin(train_patients)].reset_index(drop=True)
        
        # Parse labels to binary multi-hot vectors
        # For each row, labels are split by '|'
        # e.g., 'Cardiomegaly|Effusion' -> [1, 0, 1, 0...]
        self.labels = []
        for raw_label in self.df["Finding Labels"]:
            label_vec = np.zeros(len(self.class_names), dtype=np.float32)
            if raw_label != "No Finding":
                for item in raw_label.split("|"):
                    item = item.strip()
                    if item in self.class_names:
                        idx = self.class_names.index(item)
                        label_vec[idx] = 1.0
            self.labels.append(label_vec)
            
        self.labels = np.array(self.labels)
        
        # Optionally limit samples for faster debugging / testing during course grading
        if limit_samples is not None and limit_samples < len(self.df):
            np.random.seed(random_state)
            indices = np.random.choice(len(self.df), limit_samples, replace=False)
            self.df = self.df.iloc[indices].reset_index(drop=True)
            self.labels = self.labels[indices]
            
        print(f"Đã tải split '{split}': {len(self.df)} mẫu ảnh.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image Index"]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback if image is missing or corrupted: return a blank gray image
            # to prevent the training script from crashing
            image = Image.fromarray(np.ones((ProjectConfig.IMAGE_SIZE, ProjectConfig.IMAGE_SIZE, 3), dtype=np.uint8) * 128)
            
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32), img_name

def get_transforms():
    """
    Returns PyTorch image transforms for training, validation and testing.
    """
    # Image Net standard normalization values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.Resize((ProjectConfig.IMAGE_SIZE, ProjectConfig.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((ProjectConfig.IMAGE_SIZE, ProjectConfig.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, val_test_transform

def get_dataloaders(batch_size=32, limit_samples=None, pin_memory=True, num_workers=0):
    """
    Creates train, validation and test PyTorch DataLoaders.
    """
    train_transform, val_test_transform = get_transforms()
    
    train_dataset = NIHChestXrayDataset(
        split="train", 
        transform=train_transform, 
        limit_samples=limit_samples
    )
    val_dataset = NIHChestXrayDataset(
        split="val", 
        transform=val_test_transform, 
        limit_samples=limit_samples // 4 if limit_samples else None
    )
    test_dataset = NIHChestXrayDataset(
        split="test", 
        transform=val_test_transform, 
        limit_samples=limit_samples // 4 if limit_samples else None
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader
