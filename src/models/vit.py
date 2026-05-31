import torch
import torch.nn as nn
import torchvision.models as models
from src.config import ProjectConfig

# =====================================================================
# 1. VISION TRANSFORMER IMPLEMENTED FROM SCRATCH (FOR ACADEMIC EXPLANATION)
# =====================================================================

class PatchEmbedding(nn.Module):
    """
    Splits an image into patches, flattens them, and projects them into embedding space.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=384):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Linear projection is implemented as a 2D convolution for computational efficiency
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x):
        # Input shape: [Batch, Channels, Height, Width]
        x = self.proj(x)  # [Batch, EmbedDim, GridHeight, GridWidth]
        x = x.flatten(2)  # [Batch, EmbedDim, NumPatches]
        x = x.transpose(1, 2)  # [Batch, NumPatches, EmbedDim]
        return x

class MultiHeadSelfAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention (MHSA) module.
    """
    def __init__(self, embed_dim=384, num_heads=8, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query, Key, Value joint projection matrix
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape
        
        # 1. Project to Q, K, V
        qkv = self.qkv(x)  # [B, N, 3 * C]
        qkv = qkv.reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, Heads, N, HeadDim]
        q, k, v = qkv[0], qkv[1], qkv[2]   # Each is [B, Heads, N, HeadDim]

        # 2. Compute Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, Heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 3. Aggregate Values
        out = (attn @ v).transpose(1, 2)  # [B, N, Heads, HeadDim]
        out = out.reshape(batch_size, num_patches, embed_dim)  # [B, N, C]
        
        # 4. Final linear projection and dropout
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class MLPBlock(nn.Module):
    """
    Feed-Forward Network (MLP) within the Transformer block.
    """
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerEncoderBlock(nn.Module):
    """
    A single Transformer Encoder Block containing Self-Attention and MLP.
    """
    def __init__(self, embed_dim=384, num_heads=8, mlp_ratio=4, drop=0.1, attn_drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            attn_drop=attn_drop, 
            proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(
            in_features=embed_dim, 
            hidden_features=embed_dim * mlp_ratio, 
            drop=drop
        )

    def forward(self, x):
        # Self-Attention + Skip Connection
        x = x + self.attn(self.norm1(x))
        # MLP + Skip Connection
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformerFromScratch(nn.Module):
    """
    Full Vision Transformer implementation built from scratch.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=14,
                 embed_dim=384, depth=6, num_heads=8, mlp_ratio=4, drop_rate=0.1, attn_drop_rate=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size=img_size, 
            patch_size=patch_size, 
            in_channels=in_channels, 
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Learnable Class (CLS) Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Learnable Positional Embedding (+1 for the CLS token)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stacked Transformer Encoder Blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            ) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Final classification head projecting to the 14 labels
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize positional embeddings and cls token weight
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # 1. Patchify and embed images
        x = self.patch_embed(x)  # [B, NumPatches, EmbedDim]
        
        # 2. Append learnable CLS token to front of sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, EmbedDim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, NumPatches + 1, EmbedDim]
        
        # 3. Add positional embeddings and apply dropout
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 4. Pass through Encoder Blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        # 5. Extract only the CLS token output [B, 0, EmbedDim] and project to predictions
        cls_token_out = x[:, 0]
        logits = self.head(cls_token_out)  # [B, 14]
        
        return logits


# =====================================================================
# 2. PRE-TRAINED VIT MODEL (TORCHVISION) FINE-TUNED FOR THE TASK
# =====================================================================

class FineTunedPretrainedViT(nn.Module):
    """
    Standard Vision Transformer (ViT-B-16) pre-trained on ImageNet-1k,
    with the head updated to classify the 14 pulmonary conditions.
    """
    def __init__(self, num_classes=14, pretrained=True):
        super().__init__()
        if pretrained:
            # Load pre-trained ViT-B-16 from torchvision
            weights = models.ViT_B_16_Weights.DEFAULT
            self.model = models.vit_b_16(weights=weights)
        else:
            self.model = models.vit_b_16()
            
        # The classification head of vit_b_16 is typically named `heads` and is a Sequential block.
        # We replace the head with a new linear classifier projecting to 14 classes.
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# =====================================================================
# 3. FACTORY FUNCTION
# =====================================================================

def get_vit_model(model_type="pretrained", num_classes=14):
    """
    Factory function to retrieve either the pre-trained or custom scratch ViT.
    """
    if model_type == "pretrained":
        print("Đang khởi tạo model Pretrained Vision Transformer (ViT-B/16)...")
        return FineTunedPretrainedViT(num_classes=num_classes, pretrained=True)
    elif model_type == "scratch":
        print("Đang khởi tạo model Custom Vision Transformer (tự viết từ scratch)...")
        return VisionTransformerFromScratch(
            img_size=ProjectConfig.IMAGE_SIZE,
            patch_size=ProjectConfig.PATCH_SIZE,
            in_channels=3,
            num_classes=num_classes,
            embed_dim=ProjectConfig.EMBED_DIM,
            depth=ProjectConfig.NUM_LAYERS,
            num_heads=ProjectConfig.NUM_HEADS,
            mlp_ratio=ProjectConfig.MLP_RATIO,
            drop_rate=ProjectConfig.DROP_RATE,
            attn_drop_rate=ProjectConfig.ATTN_DROP_RATE
        )
    else:
        raise ValueError("model_type must be either 'pretrained' or 'scratch'")

if __name__ == "__main__":
    # Quick sanity check code
    x = torch.randn(2, 3, 224, 224)
    model = get_vit_model("scratch")
    out = model(x)
    print("Scratch ViT output shape:", out.shape)
    
    model_pre = get_vit_model("pretrained")
    out_pre = model_pre(x)
    print("Pretrained ViT output shape:", out_pre.shape)
