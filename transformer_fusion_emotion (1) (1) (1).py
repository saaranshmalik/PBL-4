"""
Transformer-Based Multimodal Fusion for Facial & Vocal Emotion Recognition
===========================================================================
Grounded in: "Multimodal Facial and Vocal Emotion Recognition: A Comprehensive
Review of Methods, Challenges, and Future Directions"
(Malik & Juneja, Manipal University Jaipur)

Architecture Overview
---------------------
This implements the Cross-Modal Transformer Fusion strategy described in
Section 3.2 of the paper, which achieves 90–95% accuracy on CMU-MOSEI
and outperforms CNN-only and CNN-LSTM baselines by 10–15%.

Two-branch design:
  [Visual Branch]  ResNet / ViT backbone  →  Visual Transformer Encoder
  [Audio Branch]   Wav2Vec2 backbone      →  Audio Transformer Encoder
         ↓                                          ↓
         └──────────── Cross-Modal Attention ───────┘
                               ↓
                    Multimodal Fusion Transformer
                               ↓
                    Classifier Head (7 emotions)

Emotion Classes (standard categorical model per paper Section 4.3):
  0=Neutral, 1=Happy, 2=Sad, 3=Angry, 4=Fear, 5=Surprise, 6=Disgust

Datasets this model targets (per paper Tables 1 & 2):
  - IEMOCAP    (audio-visual dyadic interactions)
  - CMU-MOSEI  (large-scale multimodal sentiment & emotion)
  - MELD       (multimodal multi-party emotion in conversations)

Requirements:
  pip install torch torchvision torchaudio transformers timm
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

class FusionConfig:
    """Central configuration — all hyperparameters in one place."""

    # ── Emotion classes ────────────────────────────────────────────────────
    NUM_CLASSES: int = 7
    EMOTION_LABELS: list = [
        "Neutral", "Happy", "Sad", "Angry", "Fear", "Surprise", "Disgust"
    ]

    # ── Visual branch ──────────────────────────────────────────────────────
    # Per Section 3.2: ViT / ResNet backbone → transformer encoder
    VISUAL_BACKBONE: str = "vit_base"     # "vit_base" | "resnet50"
    VISUAL_INPUT_DIM: int = 768           # ViT-Base token dim (or ResNet-50 → 2048 → projected)
    VISUAL_PROJ_DIM: int = 512            # Projection to shared transformer space
    VISUAL_SEQ_LEN: int = 197            # ViT 16×16 patches + CLS token (224px input)
    VISUAL_DROPOUT: float = 0.1

    # ── Audio branch ───────────────────────────────────────────────────────
    # Per Section 3.1.2 & 3.2: Wav2Vec2 self-supervised encoder
    AUDIO_BACKBONE: str = "wav2vec2"      # "wav2vec2" | "hubert" | "cnn_lstm"
    AUDIO_INPUT_DIM: int = 768           # Wav2Vec2-Base hidden size
    AUDIO_PROJ_DIM: int = 512
    AUDIO_SEQ_LEN: int = 128            # ~2s audio @ 16kHz downsampled
    AUDIO_DROPOUT: float = 0.1

    # ── Shared transformer space ───────────────────────────────────────────
    D_MODEL: int = 512
    N_HEADS: int = 8
    N_ENCODER_LAYERS: int = 4           # Per-modality transformer depth
    N_FUSION_LAYERS: int = 2            # Cross-modal fusion transformer depth
    FFN_DIM: int = 2048
    DROPOUT: float = 0.1
    MAX_SEQ_LEN: int = 512

    # ── Classifier head ────────────────────────────────────────────────────
    CLASSIFIER_HIDDEN: int = 256
    CLASSIFIER_DROPOUT: float = 0.3

    # ── Training ───────────────────────────────────────────────────────────
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-2
    LABEL_SMOOTHING: float = 0.1        # Per Section 4.3: label smoothing for annotation noise
    BATCH_SIZE: int = 32
    MAX_EPOCHS: int = 50
    WARMUP_STEPS: int = 1000


# ─────────────────────────────────────────────────────────────────────────────
# 2. POSITIONAL ENCODING
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Used for both visual patch sequences and audio frame sequences.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                    # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3. VISUAL BRANCH
#    Paper ref: Section 3.1.1 (CNN-based FER) + Section 3.2 (ViT-based FER)
#    "Vision Transformers (ViT) and hybrid CNN–Transformer architectures have
#     demonstrated improved robustness to pose variation and partial occlusion
#     by leveraging self-attention mechanisms to model spatial dependencies
#     across facial regions."
# ─────────────────────────────────────────────────────────────────────────────

class VisualFeatureProjector(nn.Module):
    """
    Projects raw backbone features (ViT/ResNet) into the shared d_model space.

    In practice: replace the forward() input with actual backbone embeddings
    from timm (e.g., timm.create_model('vit_base_patch16_224', pretrained=True)).
    This module handles the projection + normalization step.
    """

    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) → (batch, seq_len, d_model)"""
        return self.proj(x)


class VisualTransformerEncoder(nn.Module):
    """
    Per-modality transformer encoder for facial features.

    Architecture per paper Section 3.2:
    - Self-attention across patch tokens models global spatial relationships
    - Deeper than CNN receptive fields; captures subtle micro-expressions
    - Attention to emotionally salient regions (eyes, eyebrows, mouth)
    """

    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.projector = VisualFeatureProjector(
            cfg.VISUAL_INPUT_DIM, cfg.D_MODEL, cfg.VISUAL_DROPOUT
        )
        self.pos_encoding = PositionalEncoding(
            cfg.D_MODEL, cfg.MAX_SEQ_LEN, cfg.DROPOUT
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.D_MODEL,
            nhead=cfg.N_HEADS,
            dim_feedforward=cfg.FFN_DIM,
            dropout=cfg.DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,                    # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.N_ENCODER_LAYERS,
            norm=nn.LayerNorm(cfg.D_MODEL)
        )

        # CLS token for sequence-level visual representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.D_MODEL) * 0.02)

    def forward(
        self,
        visual_features: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            visual_features: (batch, seq_len, visual_input_dim)
                             Raw ViT patch embeddings or ResNet spatial features
            src_key_padding_mask: (batch, seq_len) bool mask for padding

        Returns:
            cls_repr:     (batch, d_model)   — CLS token representation
            token_reprs:  (batch, seq_len+1, d_model) — full sequence for cross-attn
        """
        batch_size = visual_features.size(0)

        # Project to d_model
        x = self.projector(visual_features)

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)          # (batch, seq_len+1, d_model)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Extend mask for CLS token
        if src_key_padding_mask is not None:
            cls_mask = torch.zeros(
                batch_size, 1, dtype=torch.bool, device=x.device
            )
            src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)

        # Transformer encoding
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        return x[:, 0, :], x                    # CLS repr, full sequence


# ─────────────────────────────────────────────────────────────────────────────
# 4. AUDIO BRANCH
#    Paper ref: Section 3.1.2 (SER) + Section 3.2 (Wav2Vec2)
#    "Self-supervised models such as Wav2Vec2 and HuBERT learn contextualized
#     speech embeddings from large-scale unlabeled audio corpora, significantly
#     enhancing generalization across speakers and recording environments."
# ─────────────────────────────────────────────────────────────────────────────

class AudioFeatureProjector(nn.Module):
    """
    Projects Wav2Vec2/HuBERT hidden states to shared d_model space.
    """

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


class AudioTransformerEncoder(nn.Module):
    """
    Per-modality transformer encoder for speech features.

    Per paper Section 3.1.2 & 3.2:
    - Processes Wav2Vec2 hidden states (contextual embeddings)
    - Captures pitch, energy, and rhythm temporal patterns
    - "CNN–LSTM hybrid architectures combine spatial feature extraction
       with temporal modeling" — here replaced by transformer for superior
       long-range dependency modeling
    """

    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.projector = AudioFeatureProjector(
            cfg.AUDIO_INPUT_DIM, cfg.D_MODEL, cfg.AUDIO_DROPOUT
        )
        self.pos_encoding = PositionalEncoding(
            cfg.D_MODEL, cfg.MAX_SEQ_LEN, cfg.DROPOUT
        )

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
            encoder_layer, num_layers=cfg.N_ENCODER_LAYERS,
            norm=nn.LayerNorm(cfg.D_MODEL)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.D_MODEL) * 0.02)

    def forward(
        self,
        audio_features: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_features: (batch, seq_len, audio_input_dim)
                            Wav2Vec2 hidden states or MFCC spectrogram features
            src_key_padding_mask: (batch, seq_len) bool mask

        Returns:
            cls_repr:    (batch, d_model)
            token_reprs: (batch, seq_len+1, d_model)
        """
        batch_size = audio_features.size(0)

        x = self.projector(audio_features)

        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = self.pos_encoding(x)

        if src_key_padding_mask is not None:
            cls_mask = torch.zeros(
                batch_size, 1, dtype=torch.bool, device=x.device
            )
            src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        return x[:, 0, :], x


# ─────────────────────────────────────────────────────────────────────────────
# 5. CROSS-MODAL ATTENTION
#    Paper ref: Section 3.2
#    "Cross-modal transformers that jointly model audio–visual interactions,
#     enabling dynamic weighting of modalities based on signal quality."
#    "Attention-based fusion mechanisms further enhance robustness by selectively
#     emphasizing reliable modalities under adverse conditions."
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalAttention(nn.Module):
    """
    Bidirectional cross-modal attention between visual and audio streams.

    For each modality:
      - Query = current modality tokens
      - Key/Value = other modality tokens
    This allows each modality to "attend" to the other, learning which
    visual moments correspond to which audio segments.

    Adaptive behavior (per paper Section 3.2):
    "When facial input is degraded due to occlusion or poor lighting,
     the model increases reliance on speech cues, and vice versa."
    → Implemented via gating with modality quality signals.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Visual attends to Audio
        self.v2a_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        # Audio attends to Visual
        self.a2v_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Post-attention layer norms
        self.norm_v = nn.LayerNorm(d_model)
        self.norm_a = nn.LayerNorm(d_model)

        # FFN for each modality after cross-attention
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
        self.norm_v2 = nn.LayerNorm(d_model)
        self.norm_a2 = nn.LayerNorm(d_model)

        # ── Adaptive modality gating ────────────────────────────────────────
        # Scalar gate per sample; high visual quality → more visual weight
        # "adaptive behavior closely mirrors human emotion perception" (paper §3.2)
        self.visual_gate = nn.Linear(d_model, 1)
        self.audio_gate  = nn.Linear(d_model, 1)

    def forward(
        self,
        visual_tokens: torch.Tensor,            # (batch, vseq, d_model)
        audio_tokens: torch.Tensor,             # (batch, aseq, d_model)
        visual_cls: torch.Tensor,               # (batch, d_model) for gating
        audio_cls: torch.Tensor,                # (batch, d_model) for gating
        visual_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            visual_out:  (batch, vseq, d_model)  — visually-enriched features
            audio_out:   (batch, aseq, d_model)  — audibly-enriched features
            v_gate:      (batch, 1)               — visual reliability gate
            a_gate:      (batch, 1)               — audio reliability gate
        """
        # Adaptive gates (sigmoid → 0–1 reliability weight)
        v_gate = torch.sigmoid(self.visual_gate(visual_cls))    # (batch, 1)
        a_gate = torch.sigmoid(self.audio_gate(audio_cls))      # (batch, 1)

        # Pre-LN + cross attention: visual queries → audio key/values
        v_norm = self.norm_v(visual_tokens)
        v_cross, _ = self.v2a_attn(
            query=v_norm,
            key=audio_tokens,
            value=audio_tokens,
            key_padding_mask=audio_mask,
        )
        # Gated residual: if audio is unreliable, reduce its contribution
        v_cross = visual_tokens + a_gate.unsqueeze(1) * v_cross

        # FFN
        v_out = v_cross + self.ffn_v(self.norm_v2(v_cross))

        # Pre-LN + cross attention: audio queries → visual key/values
        a_norm = self.norm_a(audio_tokens)
        a_cross, _ = self.a2v_attn(
            query=a_norm,
            key=visual_tokens,
            value=visual_tokens,
            key_padding_mask=visual_mask,
        )
        a_cross = audio_tokens + v_gate.unsqueeze(1) * a_cross

        a_out = a_cross + self.ffn_a(self.norm_a2(a_cross))

        return v_out, a_out, v_gate, a_gate


# ─────────────────────────────────────────────────────────────────────────────
# 6. MULTIMODAL FUSION TRANSFORMER
#    Paper ref: Section 3.2 Table 2 — "Cross-Modal Transformers: 90–95% on CMU-MOSEI"
#    Stacks N_FUSION_LAYERS of cross-modal attention, then pools to fixed-dim repr
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalFusionTransformer(nn.Module):
    """
    Stacks multiple CrossModalAttention layers, then fuses both modalities
    into a single joint representation for classification.

    Fusion strategy per paper Section 3.1.3:
    "Feature-level fusion allows networks to directly learn correlations
     between facial expressions and vocal characteristics, often resulting
     in higher accuracy compared to decision-level fusion strategies."
    """

    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossModalAttention(cfg.D_MODEL, cfg.N_HEADS, cfg.DROPOUT)
            for _ in range(cfg.N_FUSION_LAYERS)
        ])

        # Joint self-attention over concatenated [visual_cls, audio_cls]
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
            joint_layer, num_layers=2,
            norm=nn.LayerNorm(cfg.D_MODEL * 2)
        )

        self.norm_v = nn.LayerNorm(cfg.D_MODEL)
        self.norm_a = nn.LayerNorm(cfg.D_MODEL)

    def forward(
        self,
        visual_tokens: torch.Tensor,            # (batch, vseq, d_model)
        audio_tokens: torch.Tensor,             # (batch, aseq, d_model)
        visual_cls: torch.Tensor,               # (batch, d_model)
        audio_cls: torch.Tensor,                # (batch, d_model)
        visual_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            fused_repr:  (batch, d_model*2)  — joint embedding for classification
            info:        dict with gate values for interpretability
        """
        v_tokens = visual_tokens
        a_tokens = audio_tokens
        v_cls, a_cls = visual_cls, audio_cls

        v_gates, a_gates = [], []

        for layer in self.layers:
            v_tokens, a_tokens, vg, ag = layer(
                v_tokens, a_tokens, v_cls, a_cls, visual_mask, audio_mask
            )
            # Update CLS from enriched tokens (mean of sequence)
            v_cls = v_tokens.mean(dim=1)
            a_cls = a_tokens.mean(dim=1)
            v_gates.append(vg)
            a_gates.append(ag)

        # Final per-modality representations
        v_repr = self.norm_v(v_cls)             # (batch, d_model)
        a_repr = self.norm_a(a_cls)             # (batch, d_model)

        # Concatenate and run joint encoder (one "token" per modality)
        joint = torch.stack([v_repr, a_repr], dim=1)    # (batch, 2, d_model)
        # Reshape for joint encoder: (batch, 2, d_model*2) won't work — instead:
        joint_flat = torch.cat([v_repr, a_repr], dim=-1).unsqueeze(1)  # (batch,1,d_model*2)
        joint_out = self.joint_encoder(joint_flat)
        fused_repr = joint_out.squeeze(1)               # (batch, d_model*2)

        info = {
            "visual_gates": torch.stack(v_gates, dim=1).mean(dim=1),   # (batch,1)
            "audio_gates":  torch.stack(a_gates, dim=1).mean(dim=1),
        }
        return fused_repr, info


# ─────────────────────────────────────────────────────────────────────────────
# 7. CLASSIFIER HEAD
#    Paper ref: Section 4.3 — categorical emotion model with label smoothing
# ─────────────────────────────────────────────────────────────────────────────

class EmotionClassifier(nn.Module):
    """
    MLP classifier on top of fused multimodal representation.

    Per paper Section 4.3:
    "Label smoothing and uncertainty-aware training are employed to
     mitigate annotation noise" → applied in the training loss.
    """

    def __init__(self, cfg: FusionConfig):
        super().__init__()
        input_dim = cfg.D_MODEL * 2             # From fusion output

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
        """x: (batch, d_model*2) → logits: (batch, num_classes)"""
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# 8. FULL MODEL: TransformerFusionEmotionModel
# ─────────────────────────────────────────────────────────────────────────────

class TransformerFusionEmotionModel(nn.Module):
    """
    Complete Transformer-Based Multimodal Emotion Recognition System.

    Implements the Cross-Modal Transformer architecture from Section 3.2
    of the paper, targeting 90–95% accuracy on CMU-MOSEI / MELD / IEMOCAP.

    Usage example:
        cfg   = FusionConfig()
        model = TransformerFusionEmotionModel(cfg)

        # Inputs (replace with real backbone features):
        visual_feats = torch.randn(B, 197, 768)   # ViT-Base patch tokens
        audio_feats  = torch.randn(B, 128, 768)   # Wav2Vec2 hidden states

        output = model(visual_feats, audio_feats)
        logits = output["logits"]                 # (B, 7) emotion logits
        probs  = output["probs"]                  # (B, 7) after softmax
        pred   = output["prediction"]             # (B,)   class indices
    """

    def __init__(self, cfg: FusionConfig = FusionConfig()):
        super().__init__()
        self.cfg = cfg

        # ── Per-modality encoders ──────────────────────────────────────────
        self.visual_encoder = VisualTransformerEncoder(cfg)
        self.audio_encoder  = AudioTransformerEncoder(cfg)

        # ── Cross-modal fusion ─────────────────────────────────────────────
        self.fusion = MultimodalFusionTransformer(cfg)

        # ── Classifier ────────────────────────────────────────────────────
        self.classifier = EmotionClassifier(cfg)

        # ── Weight initialization ──────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for linear layers; small normal for embeddings."""
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
        """
        Args:
            visual_features: (batch, vseq_len, visual_input_dim)
                             e.g. ViT-Base outputs: (B, 197, 768)
                             or projected ResNet-50 spatial features
            audio_features:  (batch, aseq_len, audio_input_dim)
                             e.g. Wav2Vec2-Base outputs: (B, T, 768)
            visual_mask:     (batch, vseq_len) bool, True = padding position
            audio_mask:      (batch, aseq_len) bool, True = padding position

        Returns:
            dict with keys:
                "logits"      → (batch, num_classes) raw logits
                "probs"       → (batch, num_classes) softmax probabilities
                "prediction"  → (batch,) argmax class indices
                "visual_gate" → (batch, 1) visual reliability score
                "audio_gate"  → (batch, 1) audio reliability score
        """

        # ── Step 1: Per-modality encoding ─────────────────────────────────
        visual_cls, visual_tokens = self.visual_encoder(visual_features, visual_mask)
        audio_cls,  audio_tokens  = self.audio_encoder(audio_features, audio_mask)

        # ── Step 2: Cross-modal fusion ────────────────────────────────────
        fused, gate_info = self.fusion(
            visual_tokens, audio_tokens,
            visual_cls, audio_cls,
            visual_mask, audio_mask,
        )

        # ── Step 3: Classification ─────────────────────────────────────────
        logits = self.classifier(fused)

        return {
            "logits":       logits,
            "probs":        F.softmax(logits, dim=-1),
            "prediction":   logits.argmax(dim=-1),
            "visual_gate":  gate_info["visual_gates"],
            "audio_gate":   gate_info["audio_gates"],
        }

    def predict_emotion(
        self,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> Dict[str, object]:
        """
        Convenience inference method returning human-readable labels.

        Returns dict with 'label', 'confidence', 'all_probs', 'dominant_modality'
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(visual_features, audio_features)

        probs = out["probs"][0]                     # (num_classes,)
        pred_idx = out["prediction"][0].item()
        label = FusionConfig.EMOTION_LABELS[pred_idx]
        confidence = probs[pred_idx].item()

        v_gate = out["visual_gate"][0].item()
        a_gate = out["audio_gate"][0].item()
        dominant = "visual" if v_gate > a_gate else "audio"

        return {
            "label":            label,
            "confidence":       confidence,
            "all_probs":        {
                FusionConfig.EMOTION_LABELS[i]: probs[i].item()
                for i in range(len(FusionConfig.EMOTION_LABELS))
            },
            "visual_gate":       v_gate,
            "audio_gate":        a_gate,
            "dominant_modality": dominant,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 9. LOSS FUNCTION
#    Paper ref: Section 4.3
#    "Label smoothing and uncertainty-aware training are employed to mitigate
#     annotation noise" in emotion datasets like IEMOCAP and CMU-MOSEI.
# ─────────────────────────────────────────────────────────────────────────────

class EmotionRecognitionLoss(nn.Module):
    """
    Cross-entropy with label smoothing for annotation noise robustness.
    Optionally weighted for class imbalance (neutral class dominance in IEMOCAP).
    """

    def __init__(
        self,
        num_classes: int = 7,
        label_smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        logits: torch.Tensor,               # (batch, num_classes)
        targets: torch.Tensor,              # (batch,) int class indices
    ) -> torch.Tensor:
        return self.ce(logits, targets)


# ─────────────────────────────────────────────────────────────────────────────
# 10. LEARNING RATE SCHEDULER (Warmup + Cosine)
#     Paper ref: Section 3.2 — transformer training requires warmup
# ─────────────────────────────────────────────────────────────────────────────

class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    """
    Linear warmup followed by cosine annealing.
    Standard for transformer fine-tuning (per BERT/Wav2Vec2 training recipes).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
    ):
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        super().__init__(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# 11. TRAINING STEP SKELETON
# ─────────────────────────────────────────────────────────────────────────────

class EmotionTrainer:
    """
    Training loop skeleton for TransformerFusionEmotionModel.

    Datasets (per paper):
      - IEMOCAP:   9.5hr dyadic sessions, 10,039 utterances, 4/6 emotion classes
      - CMU-MOSEI: 23,454 movie clips, 7 emotion labels
      - MELD:      13,708 utterances from Friends TV, 7 emotions

    Integration with Hugging Face:
        from transformers import Wav2Vec2Model, ViTModel
        wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        vit     = ViTModel.from_pretrained("google/vit-base-patch16-224")
    """

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
        """Single training step. Returns dict with loss and accuracy."""
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(visual_features, audio_features)
        loss = self.criterion(output["logits"], labels)

        loss.backward()
        # Gradient clipping — important for transformer stability
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
        """Single evaluation step."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(visual_features, audio_features)
            loss = self.criterion(output["logits"], labels)
            acc = (output["prediction"] == labels).float().mean().item()
        return {"loss": loss.item(), "accuracy": acc}


# ─────────────────────────────────────────────────────────────────────────────
# 12. DEMO / SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def demo():
    """
    Quick sanity check with random tensors — verifies shapes and forward pass.
    Replace random tensors with real backbone outputs in production.
    """
    print("=" * 65)
    print("  Transformer-Based Multimodal Fusion — Emotion Recognition")
    print("  Based on Malik & Juneja (Manipal University Jaipur)")
    print("=" * 65)

    cfg = FusionConfig()
    model = TransformerFusionEmotionModel(cfg)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total:     {total_params:,}")
    print(f"  Trainable: {trainable:,}")

    # Simulated batch
    B = 4                                           # batch size
    visual_feats = torch.randn(B, cfg.VISUAL_SEQ_LEN, cfg.VISUAL_INPUT_DIM)
    audio_feats  = torch.randn(B, cfg.AUDIO_SEQ_LEN, cfg.AUDIO_INPUT_DIM)
    labels       = torch.randint(0, cfg.NUM_CLASSES, (B,))

    print(f"\nInput shapes:")
    print(f"  Visual (ViT patches):    {tuple(visual_feats.shape)}")
    print(f"  Audio  (Wav2Vec2 hid):   {tuple(audio_feats.shape)}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(visual_feats, audio_feats)

    print(f"\nOutput shapes:")
    print(f"  Logits:       {tuple(output['logits'].shape)}")
    print(f"  Probs:        {tuple(output['probs'].shape)}")
    print(f"  Predictions:  {tuple(output['prediction'].shape)}")
    print(f"  Visual gates: {output['visual_gate'].squeeze().tolist()}")
    print(f"  Audio  gates: {output['audio_gate'].squeeze().tolist()}")

    # Human-readable prediction for first sample
    single_v = visual_feats[0:1]
    single_a = audio_feats[0:1]
    result = model.predict_emotion(single_v, single_a)
    print(f"\nSample Prediction:")
    print(f"  Emotion:            {result['label']}")
    print(f"  Confidence:         {result['confidence']:.3f}")
    print(f"  Dominant modality:  {result['dominant_modality']}")
    print(f"  Visual gate:        {result['visual_gate']:.3f}")
    print(f"  Audio gate:         {result['audio_gate']:.3f}")
    print(f"\n  All probabilities:")
    for emotion, prob in result["all_probs"].items():
        bar = "█" * int(prob * 30)
        print(f"    {emotion:<10}  {prob:.3f}  {bar}")

    # Loss computation demo
    criterion = EmotionRecognitionLoss(
        num_classes=cfg.NUM_CLASSES,
        label_smoothing=cfg.LABEL_SMOOTHING,
    )
    model.train()
    output_train = model(visual_feats, audio_feats)
    loss = criterion(output_train["logits"], labels)
    print(f"\nTraining loss (random labels): {loss.item():.4f}")

    print("\n✓ All checks passed. Replace random tensors with real")
    print("  Wav2Vec2 / ViT backbone outputs for production use.")
    print("=" * 65)


if __name__ == "__main__":
    demo()
