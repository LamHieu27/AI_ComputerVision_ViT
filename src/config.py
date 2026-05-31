import os

class ProjectConfig:
    # --- Paths ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")
    IMAGE_DIR = os.path.join(DATASET_DIR, "images-224", "images-224")
    
    CSV_PATH = os.path.join(DATASET_DIR, "Data_Entry_2017.csv")
    TRAIN_VAL_LIST = os.path.join(DATASET_DIR, "train_val_list_NIH.txt")
    TEST_LIST = os.path.join(DATASET_DIR, "test_list_NIH.txt")
    
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    PRETRAINED_DENSENET_PATH = os.path.join(DATASET_DIR, "pretrained_model.h5")
    
    # --- Dataset Config ---
    IMAGE_SIZE = 224
    NUM_CLASSES = 14
    
    # Standard 14 classes of NIH Chest X-ray 14
    CLASS_NAMES = [
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia"
    ]
    
    # Vietnamese translations for the 14 classes to look amazing in report/web UI
    CLASS_NAMES_VN = {
        "Atelectasis": "Xẹp phổi",
        "Cardiomegaly": "Tim to / Phì đại cơ tim",
        "Effusion": "Tràn dịch màng phổi",
        "Infiltration": "Thâm nhiễm phổi",
        "Mass": "U phổi",
        "Nodule": "Nốt mờ phổi",
        "Pneumonia": "Viêm phổi",
        "Pneumothorax": "Tràn khí màng phổi",
        "Consolidation": "Đông đặc phổi",
        "Edema": "Phù phổi",
        "Emphysema": "Khí phế thũng",
        "Fibrosis": "Xơ hóa phổi",
        "Pleural_Thickening": "Dày màng phổi",
        "Hernia": "Thoát vị cơ hoành"
    }

    # --- Vision Transformer (ViT) Hyperparameters ---
    PATCH_SIZE = 16          # 16x16 pixel patches
    EMBED_DIM = 384          # Embedding dimension (small/medium ViT for educational trainability)
    NUM_HEADS = 8            # Multi-head attention heads
    NUM_LAYERS = 6           # Encoder Transformer blocks
    MLP_RATIO = 4            # MLP hidden dimension ratio
    DROP_RATE = 0.1          # Dropout probability
    ATTN_DROP_RATE = 0.1     # Attention dropout probability

    # --- Training Hyperparameters ---
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 10              # Default epochs for training
    DEVICE = "cpu"           # Will automatically check and update to GPU 'cuda' if available
    
    @classmethod
    def get_device(cls):
        import torch
        if torch.cuda.is_available():
            cls.DEVICE = "cuda"
        return cls.DEVICE
