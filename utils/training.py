import torch
from torch.cuda.amp import autocast, GradScaler

def train(epoch, model, train_loader, criterion, optimizer, scaler, device='cuda'):
    """
    Funzione di training per una epoch
    
    Args:
        epoch: numero epoch corrente
        model: modello PyTorch
        train_loader: DataLoader per training
        criterion: loss function
        optimizer: ottimizzatore
        scaler: GradScaler per mixed precision
        device: 'cuda' o 'cpu'
    
    Returns:
        train_loss: loss media dell'epoca
        train_accuracy: accuracy dell'epoca
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Mixed Precision Training
        optimizer.zero_grad()  # ESSENZIALE!
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # Backward con scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Print ogni 100 batch
        if batch_idx % 100 == 0:
            print(f'Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')
    
    return train_loss, train_accuracy


def validate(model, val_loader, criterion, device='cuda'):
    """
    Funzione di validazione
    
    Args:
        model: modello PyTorch
        val_loader: DataLoader per validation
        criterion: loss function
        device: 'cuda' o 'cpu'
    
    Returns:
        val_loss: loss media di validazione
        val_accuracy: accuracy di validazione
    """
    model.eval()
    val_loss = 0
    correct, total = 0, 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total
    
    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    
    return val_loss, val_accuracy