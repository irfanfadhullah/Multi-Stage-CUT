import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class XYDataset(Dataset):
    def __init__(self, root_A, root_A_mask, root_B, root_B_mask=None, transform=None):
        """
        Dataset for training with masks.
        
        Args:
            root_A: Directory containing source images
            root_A_mask: Directory containing source masks
            root_B: Directory containing target images
            root_B_mask: Optional directory containing target masks
            transform: Optional transform to be applied
        """
        self.root_A = root_A
        self.root_A_mask = root_A_mask
        self.root_B = root_B
        self.root_B_mask = root_B_mask
        self.transform = transform
        
        # Get image lists
        self.A_images = self.listdir(root_A)
        self.A_masks = self.listdir(root_A_mask)
        self.B_images = self.listdir(root_B)
        self.B_masks = self.listdir(root_B_mask) if root_B_mask else None
        
        # Verify A and A_mask have same number of files
        assert len(self.A_images) == len(self.A_masks), \
            f"Mismatch between A images ({len(self.A_images)}) and A masks ({len(self.A_masks)})"
        
        # Calculate dataset length
        self.A_len = len(self.A_images)
        self.B_len = len(self.B_images)
        self.length_dataset = max(self.A_len, self.B_len)

    def listdir(self, path):
        """List all files in directory, excluding hidden files."""
        if path is None:
            return []
        files = []
        for f in os.listdir(path):
            if not f.startswith('.'):
                files.append(f)
        return sorted(files)  # Sort for consistency
    
    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        # Get cyclic index for both A and B
        A_idx = index % self.A_len
        B_idx = index % self.B_len
        
        # Get file names
        A_img_name = self.A_images[A_idx]
        A_mask_name = self.A_masks[A_idx]
        B_img_name = self.B_images[B_idx]
        B_mask_name = self.B_masks[B_idx] if self.B_masks else None

        # Construct paths
        A_path = os.path.join(self.root_A, A_img_name)
        A_mask_path = os.path.join(self.root_A_mask, A_mask_name)
        B_path = os.path.join(self.root_B, B_img_name)
        B_mask_path = os.path.join(self.root_B_mask, B_mask_name) if self.B_masks else None

        # Load images
        A_img = np.array(Image.open(A_path).convert("RGB"))
        A_mask = np.array(Image.open(A_mask_path).convert("RGB"))
        B_img = np.array(Image.open(B_path).convert("RGB"))
        B_mask = np.array(Image.open(B_mask_path).convert("RGB")) if B_mask_path else None

        # Apply transformations
        if self.transform:
            if B_mask is not None:
                augmentations = self.transform(
                    image_a=A_img,
                    mask_a=A_mask,
                    image_b=B_img,
                    mask_b=B_mask
                )
                A_img = augmentations['image_a']
                A_mask = augmentations['mask_a']
                B_img = augmentations['image_b']
                B_mask = augmentations['mask_b']
            else:
                augmentations = self.transform(
                    image_a=A_img,
                    mask_a=A_mask,
                    image_b=B_img
                )
                A_img = augmentations['image_a']
                A_mask = augmentations['mask_a']
                B_img = augmentations['image_b']
                B_mask = None

        if B_mask is not None:
            return A_img, A_mask, B_img, B_mask
        return A_img, A_mask, B_img

class XInferenceDataset(Dataset):
    def __init__(self, root_A, root_A_mask, transform=None):
        """
        Dataset for inference with masks.
        
        Args:
            root_A: Directory containing source images
            root_A_mask: Directory containing source masks
            transform: Optional transform to be applied
        """
        self.root_A = root_A
        self.root_A_mask = root_A_mask
        self.transform = transform
        
        # Get image lists
        self.A_images = self.listdir(root_A)
        self.A_masks = self.listdir(root_A_mask)
        
        # Verify A and A_mask have same number of files
        assert len(self.A_images) == len(self.A_masks), \
            f"Mismatch between A images ({len(self.A_images)}) and A masks ({len(self.A_masks)})"
        
        self.length_dataset = len(self.A_images)

    def listdir(self, path):
        """List all files in directory, excluding hidden files."""
        files = []
        for f in os.listdir(path):
            if not f.startswith('.'):
                files.append(f)
        return sorted(files)  # Sort for consistency

    def __len__(self):
        return self.length_dataset
        
    def __getitem__(self, index):
        # Get file names
        A_img_name = self.A_images[index]
        A_mask_name = self.A_masks[index]

        # Construct paths
        A_path = os.path.join(self.root_A, A_img_name)
        A_mask_path = os.path.join(self.root_A_mask, A_mask_name)

        # Load images
        A_img = np.array(Image.open(A_path).convert("RGB"))
        A_mask = np.array(Image.open(A_mask_path).convert("RGB"))

        # Apply transformations
        if self.transform:
            augmentations = self.transform(image=A_img, mask=A_mask)
            A_img = augmentations['image']
            A_mask = augmentations['mask']

        return A_img, A_mask, A_path

# def get_loaders(
#     root_A,
#     root_A_mask,
#     root_B,
#     root_B_mask=None,
#     batch_size=1,
#     train_transform=None,
#     val_transform=None,
#     num_workers=4,
#     pin_memory=True,
# ):
#     """Helper function to create training and validation data loaders."""
#     train_ds = XYDataset(
#         root_A=root_A,
#         root_A_mask=root_A_mask,
#         root_B=root_B,
#         root_B_mask=root_B_mask,
#         transform=train_transform,
#     )
    
#     train_loader = DataLoader(
#         train_ds,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         shuffle=True,
#     )
    
#     val_ds = XYDataset(
#         root_A=root_A,
#         root_A_mask=root_A_mask,
#         root_B=root_B,
#         root_B_mask=root_B_mask,
#         transform=val_transform,
#     )
    
#     val_loader = DataLoader(
#         val_ds,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         shuffle=False,
#     )
    
#     return train_loader, val_loader