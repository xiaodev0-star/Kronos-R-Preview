"""Kronos-R-Preview model package."""
from model.tokenizer import HierarchicalQuantizer, build_tokenizer_kwargs
from model.kronos_preview import KronosPreview, KronosPreviewWithReasoning
from model.kronos_bert import KronosBert

import torch


def load_tokenizer(path, device):
    """Load a frozen BSQ tokenizer from checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    tok = HierarchicalQuantizer(**build_tokenizer_kwargs(ckpt.get("config", {})))
    tok.load_state_dict(ckpt["model_state_dict"])
    tok.to(device).eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    return tok
