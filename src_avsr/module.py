# sub_model/modules.py
import torch
import torch.nn as nn
from fairseq.models import FairseqEncoder

__all__ = [
    "WhisperEncoderWrapper",
    "Projector",
    "UniCrossAttn",
    "AdapterBlock",
    "build_adapter",
    "build_fusion_module",
]

# =========================
# Encoders & Wrappers
# =========================

class WhisperEncoderWrapper(FairseqEncoder):
    def __init__(self, whisper_encoder: nn.Module):
        
        # Wraps a HuggingFace Whisper encoder to be compatible with Fairseq interfaces.

        super().__init__(None)
        self.whisper = whisper_encoder

    def forward(self, source):
        """
        Forward pass for the encoder. Expects a dictionary with 'audio' or a raw tensor.
        Returns the last hidden state of the Whisper encoder.
        """
        x = source["audio"] if isinstance(source, dict) else source
        return self.whisper(x).last_hidden_state

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.
        Required for Fairseq beam search (though usually not used in this specific pipeline).
        """
        return encoder_out

class Projector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        A simple MLP projector to map fused features to the LLM embedding dimension.
        Structure: Linear -> ReLU -> Linear.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# =========================
# Attention Blocks
# =========================

class UniCrossAttn(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        """
        Unidirectional Cross-Attention Block.
        Standard Transformer decoder layer architecture without self-attention.
        Used for aligning modalities (e.g., Audio-to-Video alignment).
        """
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor):
        """
        q: Query tensor (target modality)
        kv: Key/Value tensor (source modality)
        """
        x, _ = self.mha(q, kv, kv, need_weights=False)
        x = self.ln(x + q)
        y = self.ffn(x)
        y = self.ln2(y + x)
        return y

# =========================
# Adapters
# =========================

class AdapterBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride):
        """
        Convolutional Adapter Block for downsampling and dimension projection.
        Structure: Conv1d (Downsample) -> Linear (Projection).
        """
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_dim, out_channels=in_dim, 
            kernel_size=kernel_size, stride=stride, padding=0
        )
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.proj(x)
        return x

def build_adapter(cfg, modality):
    """
    Factory function to build adapters based on configuration and modality.
    Handles dimension matching logic for different fusion strategies.
    """
    fusion_mode = cfg.modality_fuse
    
    if modality == "audio":
        # Audio usually keeps dimension (1280 -> 1280), handled by stride if needed.
        out_dim = cfg.whisper_embed_dim
        return AdapterBlock(
            in_dim=cfg.whisper_embed_dim,
            out_dim=out_dim,
            kernel_size=4, stride=4
        )
        
    elif modality == "video":
        if fusion_mode == "concat":
            # In Concat mode, Video output keeps 768 to match the checkpoint logic.
            out_dim = cfg.avhubert_embed_dim
        else:
            # Other modes align to 1280.
            out_dim = cfg.whisper_embed_dim
            
        return AdapterBlock(
            in_dim=cfg.avhubert_embed_dim,
            out_dim=out_dim,
            kernel_size=2, stride=2
        )
    else:
        raise ValueError(f"Unknown modality: {modality}")

# =========================
# Fusion Strategies
# =========================

class FusionConcat(nn.Module):
    def __init__(self, input_dim_a, input_dim_v):
        """
        Concatenation Fusion.
        Output dimension is sum of input dimensions.
        """
        super().__init__()
        self.out_dim = input_dim_a + input_dim_v

    def forward(self, a, v):
        # Handle missing modalities during inference by padding with zeros.
        # This assumes batch-wise consistency or is handled by masking downstream.
        
        if a is None and v is not None:
            # Infer audio dimension from self.out_dim and v.size(-1)
            dim_a = self.out_dim - v.size(-1)
            a = v.new_zeros(v.size(0), v.size(1), dim_a)
            
        if v is None and a is not None:
            dim_v = self.out_dim - a.size(-1)
            v = a.new_zeros(a.size(0), a.size(1), dim_v)

        return torch.cat([a, v], dim=2)

class FusionAdd(nn.Module):
    def __init__(self, common_dim):
        """
        Element-wise Addition Fusion.
        Requires input dimensions to be identical.
        """
        super().__init__()
        self.out_dim = common_dim

    def forward(self, a, v):
        if a is None: return v
        if v is None: return a
        return a + v

class FusionGLU(nn.Module):
    def __init__(self, common_dim):
        """
        Gated Linear Unit (GLU) Fusion.
        Learns a sigmoid gate to weigh the contribution of audio vs video.
        """
        super().__init__()
        self.glu_gate = nn.Linear(2 * common_dim, common_dim)
        self.out_dim = common_dim

    def forward(self, a, v):
        if a is None and v is not None: a = torch.zeros_like(v)
        if v is None and a is not None: v = torch.zeros_like(a)
        gate = torch.sigmoid(self.glu_gate(torch.cat([a, v], dim=-1)))
        return gate * a + (1.0 - gate) * v

class FusionMLP(nn.Module):
    def __init__(self, common_dim):
        """
        MLP Fusion.
        Concatenates inputs and passes through an MLP to fuse features.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * common_dim, 2 * common_dim), 
            nn.GELU(), 
            nn.Linear(2 * common_dim, common_dim)
        )
        self.out_dim = common_dim

    def forward(self, a, v):
        if a is None and v is not None: a = torch.zeros_like(v)
        if v is None and a is not None: v = torch.zeros_like(a)
        return self.mlp(torch.cat([a, v], dim=-1))

class FusionCrossAttn(nn.Module):
    def __init__(self, mode, common_dim, num_heads, num_layers, dropout):
        """
        Cross-Attention Fusion Module.
        Supports:
        - align-v2a: Video aligns to Audio
        - align-a2v: Audio aligns to Video
        - cross-att: Bi-directional alignment
        """
        super().__init__()
        self.mode = mode
        self.out_dim = common_dim
        self.layers = num_layers
        self.xblocks_a2v = None
        self.xblocks_v2a = None

        if "a2v" in mode or "cross" in mode:
            self.xblocks_a2v = nn.ModuleList([
                UniCrossAttn(common_dim, num_heads, dropout) for _ in range(num_layers)
            ])
        if "v2a" in mode or "cross" in mode:
            self.xblocks_v2a = nn.ModuleList([
                UniCrossAttn(common_dim, num_heads, dropout) for _ in range(num_layers)
            ])

    def forward(self, a, v):
        if a is None: return v
        if v is None: return a

        a_cur, b_cur = a, v
        if self.mode == "align-v2a":
            for blk in self.xblocks_v2a:
                b_cur = blk(q=b_cur, kv=a_cur) 
            return b_cur
        elif self.mode == "align-a2v":
            for blk in self.xblocks_a2v:
                a_cur = blk(q=a_cur, kv=b_cur)
            return a_cur
        elif self.mode == "cross-att":
            for blk_a2v, blk_v2a in zip(self.xblocks_a2v, self.xblocks_v2a):
                y1 = blk_a2v(q=a_cur, kv=b_cur)
                y2 = blk_v2a(q=b_cur, kv=a_cur)
                a_cur, b_cur = y1, y2
            return 0.5 * (a_cur + b_cur)
        return a + v

def build_fusion_module(cfg):
    """
    Factory function to instantiate the appropriate fusion module based on config.
    """
    mode = cfg.modality_fuse
    common_dim = cfg.whisper_embed_dim 
    
    if mode == "concat":
        # Concat Dimension = Audio(1280) + Video(adapter_out)
        # Note: Video Adapter returns 768 in concat mode.
        return FusionConcat(cfg.whisper_embed_dim, cfg.avhubert_embed_dim)
    elif mode == "add":
        return FusionAdd(common_dim)
    elif mode == "glu":
        return FusionGLU(common_dim)
    elif mode == "mlp":
        return FusionMLP(common_dim)
    elif mode in ["cross-att", "align-v2a", "align-a2v"]:
        return FusionCrossAttn(
            mode=mode, 
            common_dim=common_dim, 
            num_heads=cfg.fuse_heads, 
            num_layers=cfg.fuse_layers, 
            dropout=cfg.align_dropout
        )
    else:
        raise ValueError(f"Unknown fusion mode: {mode}")