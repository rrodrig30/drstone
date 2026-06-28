"""Phase 4b — self-supervised 3D CNN features + fusion (Dr Stone).

Right-sized deep learning for small N: a 3D convolutional autoencoder is
pretrained self-supervised (reconstruction, NO labels) on the stone ROI patches,
the encoder is frozen, and its bottleneck is used as deep features. We then test
whether deep features add anything over the handcrafted HU+pH+labs model:

    handcrafted-only  (Phase-3 baseline, ~0.825 AUC)
    deep-only         (frozen CNN features -> UA)
    fused             (deep + handcrafted -> UA)

SSL pretraining is label-free, so pretraining on all patches then cross-
validating the supervised probe does not leak labels (standard SSL protocol).

Run:
    python -m drstone.cnn
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from drstone import config as C
from drstone.modeling import build_features, ALL_FEATS, cv_auc, make_ua_model

warnings.filterwarnings("ignore")

HU_LO, HU_HI = -200.0, 1500.0     # stone-relevant window for the CNN input
BOTTLENECK = 64
EPOCHS = 120


def _norm(patch):
    return np.clip((patch.astype(np.float32) - HU_LO) / (HU_HI - HU_LO), 0, 1)


def load_patches():
    man = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_patches.csv"),
                      dtype={"canonical_mrn": str})
    man = man[man["found"] == True].copy()
    X, ids = [], []
    for _, r in man.iterrows():
        p = os.path.join(C.OUTPUT_DIR, "patches", f"{r['canonical_mrn']}.npy")
        if os.path.exists(p):
            X.append(_norm(np.load(p)))
            ids.append(r["canonical_mrn"])
    X = np.stack(X)[:, None]                       # (N,1,D,D,D)
    return X.astype(np.float32), ids


def build_ae(D):
    import torch.nn as nn

    class AE(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv3d(1, 16, 3, 2, 1), nn.BatchNorm3d(16), nn.ReLU(),
                nn.Conv3d(16, 32, 3, 2, 1), nn.BatchNorm3d(32), nn.ReLU(),
                nn.Conv3d(32, 64, 3, 2, 1), nn.BatchNorm3d(64), nn.ReLU())
            self.fd = d // 8 if d % 8 == 0 else (d + 7) // 8
            self.flat = 64 * self.fd ** 3
            self.fc_enc = nn.Linear(self.flat, BOTTLENECK)
            self.fc_dec = nn.Linear(BOTTLENECK, self.flat)
            self.dec = nn.Sequential(
                nn.ConvTranspose3d(64, 32, 4, 2, 1), nn.BatchNorm3d(32), nn.ReLU(),
                nn.ConvTranspose3d(32, 16, 4, 2, 1), nn.BatchNorm3d(16), nn.ReLU(),
                nn.ConvTranspose3d(16, 1, 4, 2, 1), nn.Sigmoid())

        def encode(self, x):
            h = self.enc(x).flatten(1)
            return self.fc_enc(h)

        def forward(self, x):
            z = self.encode(x)
            h = self.fc_dec(z).view(-1, 64, self.fd, self.fd, self.fd)
            return self.dec(h)
    return AE(D)


def pretrain_and_extract(X):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = X.shape[-1]
    ae = build_ae(D).to(dev)
    xt = torch.tensor(X)
    dl = DataLoader(TensorDataset(xt), batch_size=16, shuffle=True)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = torch.nn.MSELoss()
    ae.train()
    for ep in range(EPOCHS):
        tot = 0.0
        for (xb,) in dl:
            xb = xb.to(dev)
            # SSL augmentation: random axis flips
            for ax in (2, 3, 4):
                if torch.rand(1).item() < 0.5:
                    xb = torch.flip(xb, dims=[ax])
            rec = ae(xb)
            rec = rec[..., :D, :D, :D]
            loss = lossf(rec, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * xb.size(0)
        if (ep + 1) % 30 == 0:
            print(f"    AE epoch {ep + 1}/{EPOCHS} recon MSE={tot / len(xt):.4f}")
    ae.eval()
    with torch.no_grad():
        feats = ae.encode(xt.to(dev)).cpu().numpy()
    return feats


def main():
    print("Loading stone patches...")
    X, ids = load_patches()
    print(f"  patches: {X.shape}")
    print("Self-supervised pretraining (3D autoencoder)...")
    feats = pretrain_and_extract(X)
    deep = pd.DataFrame(feats, columns=[f"deep_{i}" for i in range(feats.shape[1])])
    deep.insert(0, "canonical_mrn", ids)
    deep.to_csv(os.path.join(C.OUTPUT_DIR, "drstone_deep_features.csv"), index=False)

    # Merge with handcrafted features + labels
    hc = build_features()
    df = hc.merge(deep, on="canonical_mrn", how="inner")
    y = df["y_ua"]
    deep_cols = [c for c in df.columns if c.startswith("deep_")]
    print(f"\n==== UA-vs-non-UA: does the CNN add over handcrafted? (n={len(df)}, UA={int(y.sum())}) ====")
    sets = {
        "handcrafted (HU+pH+labs)": ALL_FEATS,
        "deep-only (frozen CNN)": deep_cols,
        "fused (deep + handcrafted)": ALL_FEATS + deep_cols,
    }
    print(f"  {'model':30s} {'CV AUC':>8s}")
    for name, feats_ in sets.items():
        auc, _ = cv_auc(lambda f=feats_: make_ua_model(f), df[feats_], y, df["canonical_mrn"])
        print(f"  {name:30s} {auc:8.3f}")
    print("\nVerdict: fused > handcrafted -> CNN texture adds signal; "
          "else -> peak HU already captures it (report honestly).")
    print(f"deep features -> {os.path.join(C.OUTPUT_DIR, 'drstone_deep_features.csv')}")

    # t-SNE of the deep features, colored by composition (does SSL cluster by chemistry?)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        Z = StandardScaler().fit_transform(df[deep_cols].values)
        perp = max(5, min(30, len(df) // 4))
        emb = TSNE(n_components=2, perplexity=perp, init="pca", random_state=0).fit_transform(Z)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        palette = {"CaOx": "#2c7fb8", "CaP": "#7fcdbb", "UA": "#d95f0e",
                   "Struvite": "#756bb1", "Cystine": "#e7298a", "Other": "#999999"}
        for comp, g in df.assign(_x=emb[:, 0], _y=emb[:, 1]).groupby("dominant_parent"):
            ax.scatter(g["_x"], g["_y"], s=28, alpha=0.8, label=f"{comp} (n={len(g)})",
                       color=palette.get(comp, "#444"))
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.set_title("Self-supervised 3D-CNN deep features by stone composition")
        ax.legend(fontsize=8, loc="best"); ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fp = os.path.join(C.OUTPUT_DIR, "drstone_deep_tsne.png")
        fig.savefig(fp, dpi=200); print(f"t-SNE figure -> {fp}")
    except Exception as e:
        print(f"(t-SNE skipped: {e})")


if __name__ == "__main__":
    main()
