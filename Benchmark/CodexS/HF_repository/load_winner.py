import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

def load_winner_model(kge_path: str = r"kge", device: str = "cpu"):
    """
    Load the CoDEx-S ComplEx winner model from Hugging Face.

    Args:
        kge_path : absolute path to your local codex/kge directory
        device   : "cpu" or "cuda"

    Returns:
        winner_model : KgeModel ready for inference
    """
    sys.path.insert(0, kge_path)
    from kge.model import KgeModel
    from kge.util.io import load_checkpoint

    print("Downloading winner_model from Hugging Face...")
    path = hf_hub_download(
        repo_id="aaryaupadhya20/codex-s-complex-winner",
        filename="winner_model.pt"
    )

    print("Loading checkpoint...")
    checkpoint   = load_checkpoint(path, device=device)
    winner_model = KgeModel.create_from(checkpoint)
    winner_model.eval()
    print("winner_model loaded and ready!")
    return winner_model


if __name__ == "__main__":
    import torch

    model = load_winner_model()

    # Score a test triple (integer indices from CoDEx-S)
    s = torch.tensor([0])
    p = torch.tensor([1])
    o = torch.tensor([2])

    score = model.score_spo(s, p, o, direction="o")
    print(f"Test triple score: {score.item():.4f}")
