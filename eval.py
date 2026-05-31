import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import ProjectConfig
from src.dataset import get_dataloaders
from src.models.vit import get_vit_model
from src.utils import calculate_metrics, plot_roc_curves

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    
    all_targets = []
    all_preds = []
    all_filenames = []
    
    pbar = tqdm(dataloader, desc="  Đang dự đoán tập Test")
    for images, labels, filenames in pbar:
        images = images.to(device)
        
        logits = model(images)
        probs = torch.sigmoid(logits)
        
        all_targets.append(labels.numpy())
        all_preds.append(probs.cpu().numpy())
        all_filenames.extend(filenames)
        
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    
    return all_targets, all_preds, all_filenames

def main():
    parser = argparse.ArgumentParser(description="Chương trình đánh giá hiệu năng Vision Transformer (ViT) trên tập Test")
    parser.add_argument("--model", type=str, default="scratch", choices=["pretrained", "scratch"],
                        help="Loại model ViT: 'pretrained' (ImageNet) hoặc 'scratch' (tự viết)")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Đường dẫn đến file trọng số .pth (Nếu bỏ trống sẽ tự động lấy file best trong thư mục checkpoints)")
    parser.add_argument("--limit", type=int, default=500, 
                        help="Giới hạn số lượng mẫu tập test để chạy nhanh (mặc định: 500 mẫu, đặt -1 để chạy hết)")
    
    args = parser.parse_args()
    
    device = ProjectConfig.get_device()
    print("=" * 60)
    print("BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST")
    print(f"Thiết bị đang sử dụng: {device.upper()}")
    print(f"Kiểu kiến trúc model: {args.model.upper()}")
    print("=" * 60)
    
    # Auto-resolve checkpoint path if not provided
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(ProjectConfig.CHECKPOINT_DIR, f"best_vit_{args.model}.pth")
        
    # Check if checkpoint exists
    has_checkpoint = os.path.exists(checkpoint_path)
    
    # Load Model
    model = get_vit_model(args.model, num_classes=ProjectConfig.NUM_CLASSES)
    
    if has_checkpoint:
        print(f"Đang tải trọng số checkpoint từ: {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"--> Đã tải thành công checkpoint tốt nhất tại epoch {checkpoint.get('epoch', 'N/A')} với AUC = {checkpoint.get('val_auc', 0.0):.4f}")
    else:
        print(f"CẢNH BÁO: Không tìm thấy file checkpoint tại {checkpoint_path}.")
        print("Tiến hành đánh giá bằng trọng số khởi tạo ngẫu nhiên (chỉ để demo pipeline).")
        
    model = model.to(device)
    
    # Load test dataloader
    sample_limit = args.limit if args.limit > 0 else None
    print("Đang tải dữ liệu tập Test...")
    _, _, test_loader = get_dataloaders(batch_size=16, limit_samples=sample_limit)
    
    # Run evaluation
    y_true, y_pred, filenames = evaluate_model(model, test_loader, device)
    
    # Calculate global & class metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Create reports directory
    os.makedirs(os.path.join(ProjectConfig.BASE_DIR, "reports"), exist_ok=True)
    
    print("\n" + "=" * 50)
    print("BÁO CÁO KẾT QUẢ ĐÁNH GIÁ CHI TIẾT (TEST REPORT)")
    print("=" * 50)
    print(f"Tổng số ảnh đánh giá   : {len(y_true)}")
    print(f"Độ chính xác Mean AUC  : {metrics['mean_auc']:.4f}")
    print(f"Macro F1-Score         : {metrics['macro_f1']:.4f}")
    print(f"Macro Precision        : {metrics['macro_precision']:.4f}")
    print(f"Macro Recall           : {metrics['macro_recall']:.4f}")
    print("-" * 50)
    
    # Print table of individual pathology AUCs
    print(f"{'Bệnh lý phổi (Pathology)':<30} | {'AUC Score':<10} | {'Tên tiếng Việt':<25}")
    print("-" * 70)
    
    class_results = []
    for name in ProjectConfig.CLASS_NAMES:
        auc_score = metrics["class_aucs"].get(name, 0.5)
        vn_name = ProjectConfig.CLASS_NAMES_VN.get(name, name)
        print(f"{name:<30} | {auc_score:<10.4f} | {vn_name:<25}")
        class_results.append({
            "Pathology": name,
            "AUC": auc_score,
            "Vietnamese_Name": vn_name
        })
        
    print("=" * 70)
    
    # Save test results to a CSV report
    report_df = pd.DataFrame(class_results)
    report_csv_path = os.path.join(ProjectConfig.BASE_DIR, "reports", f"evaluation_report_{args.model}.csv")
    report_df.to_csv(report_csv_path, index=False)
    print(f"Báo cáo chi tiết đã được lưu dưới dạng CSV tại: {report_csv_path}")
    
    # Plot and save ROC Curves
    roc_plot_path = os.path.join(ProjectConfig.BASE_DIR, "reports", f"roc_curves_{args.model}.png")
    plot_roc_curves(y_true, y_pred, roc_plot_path)
    print("=" * 70)

if __name__ == "__main__":
    main()
