import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score
from src.config import ProjectConfig

def calculate_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Calculates classification metrics for multi-label NIH Chest X-ray.
    
    y_true: numpy array of shape [N, 14]
    y_pred_probs: numpy array of shape [N, 14]
    """
    num_classes = len(ProjectConfig.CLASS_NAMES)
    metrics_dict = {}
    
    # Calculate overall metrics
    # Using threshold-based predictions
    y_pred_bin = (y_pred_probs >= threshold).astype(int)
    
    try:
        macro_f1 = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
        macro_precision = precision_score(y_true, y_pred_bin, average="macro", zero_division=0)
        macro_recall = recall_score(y_true, y_pred_bin, average="macro", zero_division=0)
    except Exception as e:
        macro_f1, macro_precision, macro_recall = 0.0, 0.0, 0.0
        print(f"Lỗi khi tính toán metric tổng quát: {e}")

    # Calculate ROC-AUC for each class and mean ROC-AUC
    auc_scores = []
    class_aucs = {}
    
    for i, name in enumerate(ProjectConfig.CLASS_NAMES):
        try:
            # Check if both classes are present in y_true for this label
            if len(np.unique(y_true[:, i])) > 1:
                class_auc = roc_auc_score(y_true[:, i], y_pred_probs[:, i])
                class_aucs[name] = class_auc
                auc_scores.append(class_auc)
            else:
                # Fallback if split has only 0s or only 1s
                class_aucs[name] = 0.5
        except Exception as e:
            class_aucs[name] = 0.5
            print(f"Lỗi khi tính AUC cho {name}: {e}")
            
    mean_auc = np.mean(auc_scores) if auc_scores else 0.5
    
    metrics_dict["mean_auc"] = mean_auc
    metrics_dict["macro_f1"] = macro_f1
    metrics_dict["macro_precision"] = macro_precision
    metrics_dict["macro_recall"] = macro_recall
    metrics_dict["class_aucs"] = class_aucs
    
    return metrics_dict

def plot_roc_curves(y_true, y_pred_probs, save_path="roc_curves.png"):
    """
    Plots ROC curves for all 14 diseases and saves the figure.
    """
    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, 14))
    
    auc_scores = []
    
    for i, (name, color) in enumerate(zip(ProjectConfig.CLASS_NAMES, colors)):
        vn_name = ProjectConfig.CLASS_NAMES_VN.get(name, name)
        try:
            if len(np.unique(y_true[:, i])) > 1:
                fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_probs[:, i])
                roc_auc = auc(fpr, tpr)
                auc_scores.append(roc_auc)
                
                plt.plot(
                    fpr, tpr, color=color, lw=2,
                    label=f'{vn_name} ({name}) (AUC = {roc_auc:.3f})'
                )
        except Exception as e:
            continue
            
    mean_auc = np.mean(auc_scores) if auc_scores else 0.5
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tỷ lệ Dương tính giả (False Positive Rate)', fontsize=12)
    plt.ylabel('Tỷ lệ Dương tính thật (True Positive Rate)', fontsize=12)
    plt.title(f'Đồ thị ROC Curves - Mean AUC = {mean_auc:.3f}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9, bbox_to_anchor=(1.0, 0.0), ncol=1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Đã vẽ và lưu biểu đồ ROC Curves tại: {save_path}")

def plot_training_history(history, save_path="training_history.png"):
    """
    Plots training loss and validation AUC curves and saves the figure.
    
    history: dict containing 'train_loss', 'val_loss', and optionally 'val_auc' lists.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot loss on left y-axis
    color = 'tab:red'
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', color=color, fontsize=12)
    ax1.plot(epochs, history["train_loss"], 'o-', color='tab:red', label='Train Loss')
    if "val_loss" in history:
        ax1.plot(epochs, history["val_loss"], 's--', color='tab:orange', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Plot AUC on right y-axis if present
    if "val_auc" in history:
        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('Validation Mean AUC', color=color, fontsize=12)
        ax2.plot(epochs, history["val_auc"], '^--', color='tab:blue', label='Val Mean AUC')
        ax2.tick_params(axis='y', labelcolor=color)
        
    plt.title('Lịch sử Huấn luyện Model Vision Transformer (ViT)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Đã lưu biểu đồ lịch sử huấn luyện tại: {save_path}")
