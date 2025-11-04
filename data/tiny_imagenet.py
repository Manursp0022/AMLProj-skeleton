import os
import shutil
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import subprocess

class TinyImageNetDataModule:
    def __init__(self, data_root='./dataset', batch_size=64, num_workers=4):
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_path = os.path.join(data_root, 'tiny-imagenet-200')
        
    def download_and_prepare(self):
        """Download e prepara il dataset Tiny ImageNet"""
        if os.path.exists(self.dataset_path):
            print("Dataset già presente, skip download")
            return
            
        os.makedirs(self.data_root, exist_ok=True)
        
        # Download
        print("Download Tiny ImageNet...")
        zip_path = os.path.join(self.data_root, 'tiny-imagenet-200.zip')
        subprocess.run([
            'wget', 
            'http://cs231n.stanford.edu/tiny-imagenet-200.zip',
            '-O', zip_path
        ])
        
        # Unzip
        print("Estrazione...")
        subprocess.run(['unzip', '-q', zip_path, '-d', self.data_root])
        os.remove(zip_path)
        
        # Riorganizza validation set
        print("Riorganizzazione validation set...")
        self._reorganize_val_folder()
        
    def _reorganize_val_folder(self):
        """Riorganizza la cartella val in sottocartelle per classe"""
        val_dir = os.path.join(self.dataset_path, 'val')
        annotations_file = os.path.join(val_dir, 'val_annotations.txt')
        
        with open(annotations_file) as f:
            for line in f:
                fn, cls, *_ = line.split('\t')
                class_dir = os.path.join(val_dir, cls)
                os.makedirs(class_dir, exist_ok=True)
                
                src = os.path.join(val_dir, 'images', fn)
                dst = os.path.join(class_dir, fn)
                shutil.copyfile(src, dst)
        
        # Rimuovi cartella images
        shutil.rmtree(os.path.join(val_dir, 'images'))
        print("Dataset preparato con successo!")
        
    def get_dataloaders(self, train_transform, val_transform):
        """Ritorna i DataLoader per train e validation"""
        train_path = os.path.join(self.dataset_path, 'train')
        val_path = os.path.join(self.dataset_path, 'val')
        
        train_dataset = ImageFolder(root=train_path, transform=train_transform)
        val_dataset = ImageFolder(root=val_path, transform=val_transform)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader