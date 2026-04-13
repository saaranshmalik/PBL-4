from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FusionConfig:
    """Configuration for the multimodal transformer demo model."""

    NUM_CLASSES: int = 7
    EMOTION_LABELS: tuple[str, ...] = (
        "Neutral",
        "Happy",
        "Sad",
        "Angry",
        "Fear",
        "Surprise",
        "Disgust",
    )

    VISUAL_BACKBONE: str = "vit_base"
    VISUAL_INPUT_DIM: int = 768
    VISUAL_PROJ_DIM: int = 256
    VISUAL_SEQ_LEN: int = 197
    VISUAL_DROPOUT: float = 0.1

    AUDIO_BACKBONE: str = "wav2vec2"
    AUDIO_INPUT_DIM: int = 768
    AUDIO_PROJ_DIM: int = 256
    AUDIO_SEQ_LEN: int = 128
    AUDIO_DROPOUT: float = 0.1

    D_MODEL: int = 256
    N_HEADS: int = 8
    N_ENCODER_LAYERS: int = 2
    N_FUSION_LAYERS: int = 1
    FFN_DIM: int = 1024
    DROPOUT: float = 0.1
    MAX_SEQ_LEN: int = 512

    CLASSIFIER_HIDDEN: int = 256
    CLASSIFIER_DROPOUT: float = 0.3

    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-2
    LABEL_SMOOTHING: float = 0.1
    BATCH_SIZE: int = 32
    MAX_EPOCHS: int = 50
    WARMUP_STEPS: int = 1000


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class VisualFeatureProjector(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class AudioFeatureProjector(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class VisualTransformerEncoder(nn.Module):
    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.projector = VisualFeatureProjector(cfg.VISUAL_INPUT_DIM, cfg.D_MODEL, cfg.VISUAL_DROPOUT)
        self.pos_encoding = PositionalEncoding(cfg.D_MODEL, cfg.MAX_SEQ_LEN, cfg.DROPOUT)
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.D_MODEL) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.D_MODEL,
            nhead=cfg.N_HEADS,
            dim_feedforward=cfg.FFN_DIM,
            dropout=cfg.DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.N_ENCODER_LAYERS,
            norm=nn.LayerNorm(cfg.D_MODEL),
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = visual_features.size(0)
        x = self.projector(visual_features)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_encoding(x)

        if src_key_padding_mask is not None:
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
            src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x[:, 0, :], x


class AudioTransformerEncoder(nn.Module):
    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.projector = AudioFeatureProjector(cfg.AUDIO_INPUT_DIM, cfg.D_MODEL, cfg.AUDIO_DROPOUT)
        self.pos_encoding = PositionalEncoding(cfg.D_MODEL, cfg.MAX_SEQ_LEN, cfg.DROPOUT)
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.D_MODEL) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.D_MODEL,
            nhead=cfg.N_HEADS,
            dim_feedforward=cfg.FFN_DIM,
            dropout=cfg.DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.N_ENCODER_LAYERS,
            norm=nn.LayerNorm(cfg.D_MODEL),
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = audio_features.size(0)
        x = self.projector(audio_features)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_encoding(x)

        if src_key_padding_mask is not None:
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
            src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x[:, 0, :], x


class CrossModalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.v2a_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.a2v_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.norm_v = nn.LayerNorm(d_model)
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_v2 = nn.LayerNorm(d_model)
        self.norm_a2 = nn.LayerNorm(d_model)

        self.ffn_v = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_a = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

        self.visual_gate = nn.Linear(d_model, 1)
        self.audio_gate = nn.Linear(d_model, 1)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
        visual_cls: torch.Tensor,
        audio_cls: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        v_gate = torch.sigmoid(self.visual_gate(visual_cls))
        a_gate = torch.sigmoid(self.audio_gate(audio_cls))

        v_norm = self.norm_v(visual_tokens)
        v_cross, _ = self.v2a_attn(
            query=v_norm,
            key=audio_tokens,
            value=audio_tokens,
            key_padding_mask=audio_mask,
            need_weights=False,
        )
        v_cross = visual_tokens + a_gate.unsqueeze(1) * v_cross
        v_out = v_cross + self.ffn_v(self.norm_v2(v_cross))

        a_norm = self.norm_a(audio_tokens)
        a_cross, _ = self.a2v_attn(
            query=a_norm,
            key=visual_tokens,
            value=visual_tokens,
            key_padding_mask=visual_mask,
            need_weights=False,
        )
        a_cross = audio_tokens + v_gate.unsqueeze(1) * a_cross
        a_out = a_cross + self.ffn_a(self.norm_a2(a_cross))

        return v_out, a_out, v_gate, a_gate


class MultimodalFusionTransformer(nn.Module):
    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossModalAttention(cfg.D_MODEL, cfg.N_HEADS, cfg.DROPOUT) for _ in range(cfg.N_FUSION_LAYERS)]
        )
        self.norm_v = nn.LayerNorm(cfg.D_MODEL)
        self.norm_a = nn.LayerNorm(cfg.D_MODEL)

        joint_layer = nn.TransformerEncoderLayer(
            d_model=cfg.D_MODEL * 2,
            nhead=cfg.N_HEADS,
            dim_feedforward=cfg.FFN_DIM,
            dropout=cfg.DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.joint_encoder = nn.TransformerEncoder(
            joint_layer,
            num_layers=2,
            norm=nn.LayerNorm(cfg.D_MODEL * 2),
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
        visual_cls: torch.Tensor,
        audio_cls: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        v_tokens = visual_tokens
        a_tokens = audio_tokens
        v_cls = visual_cls
        a_cls = audio_cls

        v_gates = []
        a_gates = []
        for layer in self.layers:
            v_tokens, a_tokens, vg, ag = layer(v_tokens, a_tokens, v_cls, a_cls, visual_mask, audio_mask)
            v_cls = v_tokens.mean(dim=1)
            a_cls = a_tokens.mean(dim=1)
            v_gates.append(vg)
            a_gates.append(ag)

        v_repr = self.norm_v(v_cls)
        a_repr = self.norm_a(a_cls)
        joint_flat = torch.cat([v_repr, a_repr], dim=-1).unsqueeze(1)
        fused_repr = self.joint_encoder(joint_flat).squeeze(1)

        info = {
            "visual_gates": torch.stack(v_gates, dim=1).mean(dim=1),
            "audio_gates": torch.stack(a_gates, dim=1).mean(dim=1),
        }
        return fused_repr, info


class EmotionClassifier(nn.Module):
    def __init__(self, cfg: FusionConfig):
        super().__init__()
        input_dim = cfg.D_MODEL * 2
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, cfg.CLASSIFIER_HIDDEN),
            nn.GELU(),
            nn.Dropout(cfg.CLASSIFIER_DROPOUT),
            nn.Linear(cfg.CLASSIFIER_HIDDEN, cfg.CLASSIFIER_HIDDEN // 2),
            nn.GELU(),
            nn.Dropout(cfg.CLASSIFIER_DROPOUT / 2),
            nn.Linear(cfg.CLASSIFIER_HIDDEN // 2, cfg.NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class TransformerFusionEmotionModel(nn.Module):
    """Complete multimodal transformer model."""

    def __init__(self, cfg: FusionConfig = FusionConfig()):
        super().__init__()
        self.cfg = cfg
        self.visual_encoder = VisualTransformerEncoder(cfg)
        self.audio_encoder = AudioTransformerEncoder(cfg)
        self.fusion = MultimodalFusionTransformer(cfg)
        self.classifier = EmotionClassifier(cfg)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            elif "cls_token" in name:
                nn.init.trunc_normal_(param, std=0.02)

    def forward(
        self,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        visual_cls, visual_tokens = self.visual_encoder(visual_features, visual_mask)
        audio_cls, audio_tokens = self.audio_encoder(audio_features, audio_mask)

        fused, gate_info = self.fusion(
            visual_tokens,
            audio_tokens,
            visual_cls,
            audio_cls,
            visual_mask,
            audio_mask,
        )
        logits = self.classifier(fused)

        return {
            "logits": logits,
            "probs": F.softmax(logits, dim=-1),
            "prediction": logits.argmax(dim=-1),
            "visual_gate": gate_info["visual_gates"],
            "audio_gate": gate_info["audio_gates"],
        }

    def predict_emotion(self, visual_features: torch.Tensor, audio_features: torch.Tensor) -> Dict[str, object]:
        self.eval()
        with torch.no_grad():
            out = self.forward(visual_features, audio_features)

        probs = out["probs"][0]
        pred_idx = out["prediction"][0].item()
        v_gate = out["visual_gate"][0].item()
        a_gate = out["audio_gate"][0].item()

        return {
            "label": FusionConfig.EMOTION_LABELS[pred_idx],
            "confidence": probs[pred_idx].item(),
            "all_probs": {
                FusionConfig.EMOTION_LABELS[i]: probs[i].item()
                for i in range(len(FusionConfig.EMOTION_LABELS))
            },
            "visual_gate": v_gate,
            "audio_gate": a_gate,
            "dominant_modality": "visual" if v_gate > a_gate else "audio",
        }


class EmotionRecognitionLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        label_smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets)


class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        super().__init__(optimizer, lr_lambda)


class EmotionTrainer:
    def __init__(self, model: TransformerFusionEmotionModel, cfg: FusionConfig):
        self.model = model
        self.cfg = cfg
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY,
            betas=(0.9, 0.98),
        )
        self.criterion = EmotionRecognitionLoss(
            num_classes=cfg.NUM_CLASSES,
            label_smoothing=cfg.LABEL_SMOOTHING,
        )

    def train_step(
        self,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(visual_features, audio_features)
        loss = self.criterion(output["logits"], labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        acc = (output["prediction"] == labels).float().mean().item()
        return {"loss": loss.item(), "accuracy": acc}

    def eval_step(
        self,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            output = self.model(visual_features, audio_features)
            loss = self.criterion(output["logits"], labels)
            acc = (output["prediction"] == labels).float().mean().item()
        return {"loss": loss.item(), "accuracy": acc}


def demo() -> None:
    print("=" * 65)
    print("  Transformer-Based Multimodal Fusion - Emotion Recognition")
    print("  Demo forward pass with random tensors")
    print("=" * 65)

    cfg = FusionConfig()
    model = TransformerFusionEmotionModel(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nModel Parameters:")
    print(f"  Total:     {total_params:,}")
    print(f"  Trainable: {trainable:,}")

    batch_size = 2
    visual_feats = torch.randn(batch_size, cfg.VISUAL_SEQ_LEN, cfg.VISUAL_INPUT_DIM)
    audio_feats = torch.randn(batch_size, cfg.AUDIO_SEQ_LEN, cfg.AUDIO_INPUT_DIM)
    labels = torch.randint(0, cfg.NUM_CLASSES, (batch_size,))

    print("\nInput shapes:")
    print(f"  Visual: {tuple(visual_feats.shape)}")
    print(f"  Audio:  {tuple(audio_feats.shape)}")

    model.eval()
    with torch.no_grad():
        output = model(visual_feats, audio_feats)

    print("\nOutput shapes:")
    print(f"  Logits:      {tuple(output['logits'].shape)}")
    print(f"  Probs:       {tuple(output['probs'].shape)}")
    print(f"  Predictions: {tuple(output['prediction'].shape)}")

    result = model.predict_emotion(visual_feats[0:1], audio_feats[0:1])
    print("\nSample Prediction:")
    print(f"  Emotion:            {result['label']}")
    print(f"  Confidence:         {result['confidence']:.3f}")
    print(f"  Dominant modality:  {result['dominant_modality']}")
    print(f"  Visual gate:        {result['visual_gate']:.3f}")
    print(f"  Audio gate:         {result['audio_gate']:.3f}")

    criterion = EmotionRecognitionLoss(
        num_classes=cfg.NUM_CLASSES,
        label_smoothing=cfg.LABEL_SMOOTHING,
    )
    model.train()
    output_train = model(visual_feats, audio_feats)
    loss = criterion(output_train["logits"], labels)
    print(f"\nTraining loss (random labels): {loss.item():.4f}")
    print("=" * 65)


if __name__ == "__main__":
    demo()
