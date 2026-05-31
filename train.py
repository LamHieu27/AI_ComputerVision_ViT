import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from src.config import ProjectConfig
from src.dataset import get_dataloaders
from src.models.vit import get_vit_model
from src.utils import calculate_metrics, plot_training_history

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    # Progress bar in Vietnamese for premium feel
    pbar = tqdm(dataloader, desc="  Đang huấn luyện")
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Backward pass & Optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=loss.item())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    all_targets = []
    all_preds = []
    
    pbar = tqdm(dataloader, desc="  Đang đánh giá")
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        running_loss += loss.item() * images.size(0)
        
        # Sigmoid activation to get probabilities [0, 1]
        probs = torch.sigmoid(logits)
        
        all_targets.append(labels.cpu().numpy())
        all_preds.append(probs.cpu().numpy())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    
    # Concatenate all targets and predictions
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    
    # Calculate medical metrics
    metrics = calculate_metrics(all_targets, all_preds)
    
    return epoch_loss, metrics

def main():
    parser = argparse.ArgumentParser(description="Chương trình huấn luyện Vision Transformer (ViT) cho phổi NIH")
    parser.add_argument("--model", type=str, default="scratch", choices=["pretrained", "scratch"],
                        help="Loại model ViT: 'pretrained' (ImageNet) hoặc 'scratch' (tự viết)")
    parser.add_argument("--epochs", type=int, default=5, help="Số lượng epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=16, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--limit", type=int, default=1000, 
                        help="Giới hạn số lượng mẫu ảnh để chạy demo nhanh (mặc định: 1000 mẫu, đặt -1 để chạy hết)")
    parser.add_argument("--resume", action="store_true", 
                        help="Tiếp tục huấn luyện từ checkpoint tốt nhất trước đó")
    
    args = parser.parse_args()
    
    # Setup device
    device = ProjectConfig.get_device()
    print("=" * 60)
    print(f"BẮT ĐẦU HUẤN LUYỆN HỆ THỐNG NHẬN DIỆN TỔN THƯƠNG PHỔI - VIT")
    print(f"Thiết bị đang sử dụng: {device.upper()}")
    print(f"Kiểu kiến trúc model: {args.model.upper()}")
    print("=" * 60)
    
    # Prepare checkpoits folder
    os.makedirs(ProjectConfig.CHECKPOINT_DIR, exist_ok=True)
    
    # Limit samples logic
    sample_limit = args.limit if args.limit > 0 else None
    
    # Load Dataloaders
    print("Đang tải dữ liệu và khởi tạo dataloaders...")
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=args.batch_size, 
        limit_samples=sample_limit
    )
    
    # Load Model
    model = get_vit_model(args.model, num_classes=ProjectConfig.NUM_CLASSES)
    model = model.to(device)
    
    # Multi-label Criterion and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=ProjectConfig.WEIGHT_DECAY)
    
    # History logs for plotting
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": []
    }
    
    best_val_auc = 0.0
    start_epoch = 1
    
    checkpoint_name = f"best_vit_{args.model}.pth"
    checkpoint_path = os.path.join(ProjectConfig.CHECKPOINT_DIR, checkpoint_name)
    
    # Resume training logic
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Đang tải checkpoint để tiếp tục huấn luyện từ: {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_auc = checkpoint.get('val_auc', 0.0)
        print(f"--> Tiếp tục huấn luyện thành công từ Epoch {start_epoch} (AUC tốt nhất trước đó: {best_val_auc:.4f})")
    
    # Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        val_auc = val_metrics["mean_auc"]
        val_f1 = val_metrics["macro_f1"]
        
        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        
        print(f"  Kết quả Epoch {epoch}:")
        print(f"    Train Loss     : {train_loss:.4f}")
        print(f"    Val Loss       : {val_loss:.4f}")
        print(f"    Val Mean AUC   : {val_auc:.4f} (Độ chính xác chuẩn đoán y học)")
        print(f"    Val Macro F1   : {val_f1:.4f}")
        
        # Checkpoint saving if AUC improves
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            checkpoint_name = f"best_vit_{args.model}.pth"
            checkpoint_path = os.path.join(ProjectConfig.CHECKPOINT_DIR, checkpoint_name)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'model_type': args.model
            }, checkpoint_path)
            print(f"  --> [LƯU MODEL] Đã lưu model tốt nhất mới với AUC = {val_auc:.4f} tại: {checkpoint_path}")
            
    print("\n" + "=" * 60)
    print("HUẤN LUYỆN HOÀN THÀNH!")
    print(f"Điểm AUC cao nhất đạt được trên tập Validation: {best_val_auc:.4f}")
    
    # Plot and save training history
    history_plot_path = os.path.join(ProjectConfig.BASE_DIR, "reports", f"training_history_{args.model}.png")
    plot_training_history(history, history_plot_path)
    print("=" * 60)

if __name__ == "__main__":
    main()
