import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
import wandb

from models.customnet import CustomNet
from data.tiny_imagenet import TinyImageNetDataModule
from utils.transforms import get_train_transforms, get_val_transforms
from utils.training import train, validate

def main():
    config = {
        'batch_size': 64,
        'num_epochs': 10,
        'learning_rate': 0.001,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    wandb.init(
        project="tiny-imagenet-custom",
        config=config
    )
    
    device = config['device']
    print(f"Training su: {device}")
    
    print("\n📦 Preparazione dataset...")
    data_module = TinyImageNetDataModule(
        data_root='./dataset',
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    #Only the first time    
    data_module.download_and_prepare()
    
    train_loader, val_loader = data_module.get_dataloaders(
        train_transform=get_train_transforms(),
        val_transform=get_val_transforms()
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    print("\n🧠 Creating model...")
    model = CustomNet().to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {num_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scaler = GradScaler()  # Per mixed precision
    
    # TRAINING LOOP 
    print("\n🚀 starting training...\n")
    best_val_acc = 0.0
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch}/{config['num_epochs']}")
        print(f"{'='*50}")
        
        # Training
        train_loss, train_acc = train(
            epoch, model, train_loader, 
            criterion, optimizer, scaler, device
        )
        
        # Validation
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Log su Wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"\n✅ Nuovo best model! Val Acc: {val_acc:.2f}%")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'checkpoints/best_model.pth')
    
    print(f"\n🎉 Training completato! Best Val Acc: {best_val_acc:.2f}%")
    wandb.finish()

if __name__ == '__main__':
    main()