import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from typing import Dict, Tuple, Optional, Union


# ==================== ResEmoteNet Components ====================


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        z = self.avg_pool(x).view(b, c)
        s = self.fc1(z)
        s = self.relu(s)
        s = self.fc2(s)
        s = self.sigmoid(s)
        s = s.view(b, c, 1, 1)
        return x * s.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out


class ResEmoteNetBackbone(nn.Module):
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            64, 128, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(
            128, 256, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.se_block = SqueezeExcitationBlock(256, reduction=16)
        self.res_block1 = ResidualBlock(256, 512, stride=2)
        self.res_block2 = ResidualBlock(512, 1024, stride=2)
        self.res_block3 = ResidualBlock(1024, 2048, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool3(x)

        x = self.se_block(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)  # (batch, 2048)
        return x


# ==================== Multi-Head Attention Fusion Module ====================


class MultiHeadAttentionFusion(nn.Module):
    """
    Adaptive fusion module using Multi-Head Attention.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        ffn_expansion: FFN expansion ratio.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.2,
        ffn_expansion: int = 2
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ffn_expansion, embed_dim),
            nn.Dropout(dropout)
        )

        self.residual_dropout = nn.Dropout(dropout)

        self.agg_weights = nn.Parameter(torch.ones(2) / 2.0)

    def forward(
        self,
        x: torch.Tensor,
        aggregation: str = 'mean'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, 2, embed_dim)
            aggregation: 'mean', 'weighted', 'cnn', or 'transformer'.

        Returns:
            fused: (batch_size, embed_dim)
            attn_weights: (batch_size, num_heads, 2, 2)
        """
        attn_output, attn_weights = self.mha(x, x, x)
        x = self.norm1(self.residual_dropout(x) + attn_output)

        ffn_output = self.ffn(x)
        x = self.norm2(self.residual_dropout(x) + ffn_output)

        if aggregation == 'mean':
            fused = torch.mean(x, dim=1)
        elif aggregation == 'weighted':
            weights = F.softmax(self.agg_weights, dim=0)
            fused = (x * weights.view(1, 2, 1)).sum(dim=1)
        elif aggregation == 'cnn':
            fused = x[:, 0, :]
        elif aggregation == 'transformer':
            fused = x[:, 1, :]
        else:
            raise ValueError(f"Unknown aggregation strategy: {aggregation}")

        return fused, attn_weights


# ==================== Hybrid Model ====================


class HybridEmotionRecognition(nn.Module):
    """
    Hybrid CNN-Transformer model for facial emotion recognition.

    - ResEmoteNet branch (CNN)
    - Swin Transformer Tiny branch
    - Projection + normalization
    - MHA fusion
    - Main classifier + auxiliary heads
    """
    def __init__(
        self,
        num_classes: int = 7,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.2,
        pretrained_swin: bool = True,
        use_gradient_checkpointing: bool = False,
        aggregation: str = 'mean'
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.aggregation = aggregation

        # ============ Branch 1: ResEmoteNet (CNN) ============
        self.resemotenet = ResEmoteNetBackbone(input_channels=3)
        resemotenet_dim = 2048

        # ============ Branch 2: Swin Transformer Tiny ============
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained_swin,
            num_classes=0
        )
        swin_dim = self.swin.num_features  # 768

        if use_gradient_checkpointing and hasattr(self.swin, "set_grad_checkpointing"):
            self.swin.set_grad_checkpointing(True)

        # ============ Projection Layers ============
        self.cnn_projection = nn.Sequential(
            nn.Linear(resemotenet_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.transformer_projection = nn.Sequential(
            nn.Linear(swin_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # ============ Per-Branch Normalization ============
        self.cnn_norm = nn.LayerNorm(embed_dim)
        self.transformer_norm = nn.LayerNorm(embed_dim)

        # ============ Fusion Module ============
        self.fusion = MultiHeadAttentionFusion(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            ffn_expansion=2
        )

        # ============ Main Classification Head ============
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

        # ============ Auxiliary Heads ============
        # CNN-only head
        self.cnn_aux_head = nn.Sequential(
            nn.Linear(resemotenet_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

        # Transformer-only head
        self.transformer_aux_head = nn.Sequential(
            nn.Linear(swin_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize projection, classifier, and aux heads."""
        for module in [
            self.cnn_projection,
            self.transformer_projection,
            self.classifier,
            self.cnn_aux_head,
            self.transformer_aux_head,
        ]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = False,
        return_attention: bool = False,
        return_aux: bool = False
    ) -> Union[torch.Tensor, Tuple]:
        """
        Args:
            x: (batch_size, 3, 224, 224)
            return_embeddings: if True, returns intermediate embeddings dict.
            return_attention: if True, returns attention weights.
            return_aux: if True, returns auxiliary logits (cnn_aux, tr_aux).

        Returns (depending on flags):
            logits_main
            + embeddings (optional)
            + attention_weights (optional)
            + logits_cnn_aux, logits_tr_aux (if return_aux=True)
        """
        # ============ Extract features ============
        cnn_features = self.resemotenet(x)          # (B, 2048)
        transformer_features = self.swin(x)         # (B, 768)

        # Aux logits from raw features
        logits_cnn_aux = self.cnn_aux_head(cnn_features)          # (B, C)
        logits_tr_aux = self.transformer_aux_head(transformer_features)

        # Project + normalize for fusion
        cnn_embed = self.cnn_projection(cnn_features)
        cnn_embed = self.cnn_norm(cnn_embed)

        transformer_embed = self.transformer_projection(transformer_features)
        transformer_embed = self.transformer_norm(transformer_embed)

        embeddings_seq = torch.stack([cnn_embed, transformer_embed], dim=1)

        # ============ Fusion ============
        fused_embed, attention_weights = self.fusion(
            embeddings_seq,
            aggregation=self.aggregation
        )

        # ============ Main Classification ============
        logits_main = self.classifier(fused_embed)

        # Prepare outputs
        outputs: list = [logits_main]

        if return_embeddings:
            embeddings = {
                'cnn_raw': cnn_features,
                'transformer_raw': transformer_features,
                'cnn_projected': cnn_embed,
                'transformer_projected': transformer_embed,
                'fused': fused_embed
            }
            outputs.append(embeddings)

        if return_attention:
            outputs.append(attention_weights)

        if return_aux:
            outputs.append(logits_cnn_aux)
            outputs.append(logits_tr_aux)

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns average attention matrix per sample: (batch, 2, 2).
        """
        with torch.no_grad():
            cnn_features = self.resemotenet(x)
            transformer_features = self.swin(x)

            cnn_embed = self.cnn_projection(cnn_features)
            cnn_embed = self.cnn_norm(cnn_embed)

            transformer_embed = self.transformer_projection(transformer_features)
            transformer_embed = self.transformer_norm(transformer_embed)

            embeddings_seq = torch.stack([cnn_embed, transformer_embed], dim=1)
            _, attention_weights = self.fusion(
                embeddings_seq, aggregation=self.aggregation
            )
            avg_attention = attention_weights.mean(dim=1)
        return avg_attention

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_params_by_component(self) -> Dict[str, int]:
        return {
            'resemotenet': sum(p.numel() for p in self.resemotenet.parameters()),
            'swin_transformer': sum(p.numel() for p in self.swin.parameters()),
            'cnn_projection': sum(p.numel() for p in self.cnn_projection.parameters()),
            'transformer_projection': sum(p.numel() for p in self.transformer_projection.parameters()),
            'cnn_norm': sum(p.numel() for p in self.cnn_norm.parameters()),
            'transformer_norm': sum(p.numel() for p in self.transformer_norm.parameters()),
            'fusion_module': sum(p.numel() for p in self.fusion.parameters()),
            'classifier': sum(p.numel() for p in self.classifier.parameters()),
            'cnn_aux_head': sum(p.numel() for p in self.cnn_aux_head.parameters()),
            'transformer_aux_head': sum(p.numel() for p in self.transformer_aux_head.parameters()),
            'total': self.get_num_params(),
            'trainable': self.get_trainable_params()
        }

    def freeze_backbone(
        self,
        freeze_cnn: bool = False,
        freeze_transformer: bool = False
    ) -> None:
        if freeze_cnn:
            for p in self.resemotenet.parameters():
                p.requires_grad = False
            print("✓ Froze ResEmoteNet backbone")

        if freeze_transformer:
            for p in self.swin.parameters():
                p.requires_grad = False
            print("✓ Froze Swin Transformer backbone")


# ==================== Model Factory Functions ====================


def create_hybrid_model(
    num_classes: int = 7,
    embed_dim: int = 512,
    num_heads: int = 8,
    dropout: float = 0.2,
    pretrained_swin: bool = True,
    use_gradient_checkpointing: bool = False,
    aggregation: str = 'mean'
) -> HybridEmotionRecognition:
    model = HybridEmotionRecognition(
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        pretrained_swin=pretrained_swin,
        use_gradient_checkpointing=use_gradient_checkpointing,
        aggregation=aggregation
    )
    return model


def create_ablation_models(num_classes: int = 7) -> Dict[str, nn.Module]:
    models: Dict[str, nn.Module] = {}

    models['hybrid_full'] = create_hybrid_model(
        num_classes=num_classes,
        aggregation='mean'
    )

    print(f"Created {len(models)} models for ablation studies")
    return models


# ==================== Model Summary ====================


def print_model_summary(model: HybridEmotionRecognition) -> None:
    print("=" * 70)
    print("HYBRID EMOTION RECOGNITION MODEL SUMMARY")
    print("=" * 70)

    params = model.get_params_by_component()

    print("\nComponent-wise Parameter Count:")
    print(f"  ResEmoteNet (CNN Branch):      {params['resemotenet']:>12,} parameters")
    print(f"  Swin Transformer Tiny:         {params['swin_transformer']:>12,} parameters")
    print(f"  CNN Projection Layer:          {params['cnn_projection']:>12,} parameters")
    print(f"  Transformer Projection Layer:  {params['transformer_projection']:>12,} parameters")
    print(f"  CNN Normalization:             {params['cnn_norm']:>12,} parameters")
    print(f"  Transformer Normalization:     {params['transformer_norm']:>12,} parameters")
    print(f"  Fusion Module (MHA):           {params['fusion_module']:>12,} parameters")
    print(f"  Main Classification Head:      {params['classifier']:>12,} parameters")
    print(f"  CNN Auxiliary Head:            {params['cnn_aux_head']:>12,} parameters")
    print(f"  Transformer Auxiliary Head:    {params['transformer_aux_head']:>12,} parameters")
    print("  " + "-" * 50)
    print(f"  Total Parameters:              {params['total']:>12,} parameters")
    print(f"  Trainable Parameters:          {params['trainable']:>12,} parameters")

    print("\nModel Configuration:")
    print(f"  Embedding Dimension:           {model.embed_dim}")
    print(f"  Number of Attention Heads:     {model.fusion.num_heads}")
    print(f"  Number of Classes:             {model.num_classes}")
    print(f"  Aggregation Strategy:          {model.aggregation}")

    print("=" * 70)
