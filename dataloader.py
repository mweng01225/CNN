import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class Loader(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: Path to directory that contains 'cats' and 'dogs' folders.
        transform: Optional torchvision transforms to apply.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Build lists of file paths and labels
        self.image_paths = []
        self.labels = []
        
        cat_dir = os.path.join(root_dir, 'cat')
        dog_dir = os.path.join(root_dir, 'dog')
        
        for fname in os.listdir(cat_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(cat_dir, fname))
                self.labels.append(0)  # label 0 for cats
                
        for fname in os.listdir(dog_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(dog_dir, fname))
                self.labels.append(1)  # label 1 for dogs

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)  # no .convert('L') and no resize
        
        if self.transform:
            image = self.transform(image)
        else:
            image = (np.array(image, dtype=np.float32) / 255.0) - 0.5
            
        label = self.labels[idx]
        return image, label
    

