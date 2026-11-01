"""src/generative_models.py — Gap 5: VAE / GAN generative models.

Suggests new nanocomposite formulations via VAE or GMM fallback.
Wrapped in try/except so the module imports without PyTorch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.mixture import GaussianMixture

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False


class _VAE(nn.Module):
    """Simple Variational Autoencoder for formulation vectors."""

    def __init__(self, input_dim: int, latent_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, input_dim), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class FormulationGenerator:
    """Generates novel nanocomposite formulations via VAE or GMM fallback."""

    def __init__(self, feature_cols, cat_maps=None, latent_dim=8):
        self.feature_cols = feature_cols
        self.cat_maps = cat_maps or {}
        self.latent_dim = latent_dim
        self.model = None
        self._use_torch = _HAS_TORCH
        self._input_dim = len(feature_cols)
        self._is_fitted = False

    @staticmethod
    def _normalize(X):
        return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

    def _df_to_matrix(self, df):
        return df[self.feature_cols].values.astype(float)

    def fit(self, df):
        X = self._df_to_matrix(df)
        Xn = self._normalize(X)
        if self._use_torch:
            self._fit_vae(Xn)
        else:
            self._fit_gmm(Xn)
        self._is_fitted = True
        return self

    def _fit_vae(self, X, epochs=100, batch_size=32, lr=1e-3):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        self.model = _VAE(input_dim=self._input_dim, latent_dim=self.latent_dim)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        self.model.train()
        for _ in range(epochs):
            for batch in loader:
                xb = batch[0]
                opt.zero_grad()
                xr, mu, logvar = self.model(xb)
                recon = nn.functional.mse_loss(xr, xb)
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon + 0.01 * kl
                loss.backward()
                opt.step()

    def _fit_gmm(self, X, n_components=5):
        self.model = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        self.model.fit(X)

    def generate(self, n=10, temperature=1.0):
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self._use_torch:
            samples = self._sample_vae(n, temperature)
        else:
            samples = self._sample_gmm(n, temperature)
        import pandas as pd
        return pd.DataFrame(samples, columns=self.feature_cols)

    def _sample_vae(self, n, temperature):
        import torch
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim) * temperature
            samples = self.model.decode(z).numpy()
        return samples

    def _sample_gmm(self, n, temperature):
        samples, _ = self.model.sample(n_samples=n)
        return samples * temperature

    def generate_constrained(self, n=10, constraints=None, temperature=1.0, max_attempts=1000):
        import pandas as pd
        constraints = constraints or {}
        valid_rows = []
        attempts = 0
        while len(valid_rows) < n and attempts < max_attempts:
            batch = self.generate(n - len(valid_rows), temperature)
            attempts += 1
            for _, row in batch.iterrows():
                if self._passes_constraints(row, constraints):
                    valid_rows.append(row.to_dict())
                if len(valid_rows) >= n:
                    break
        return pd.DataFrame(valid_rows)

    def _passes_constraints(self, row, constraints):
        forbidden = constraints.get('forbidden_additives', {})
        for col, val in row.items():
            if col in forbidden and str(val) in forbidden[col]:
                return False
        for col, (lo, hi) in constraints.get('numeric_ranges', {}).items():
            if col in row and not (lo <= row[col] <= hi):
                return False
        return True


def make_dummy_formulations(n=100):
    import pandas as pd
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'recycled_content_pct': rng.uniform(0, 100, n),
        'filler_wt_pct': rng.uniform(0, 40, n),
        'nano_wt_pct': rng.uniform(0, 5, n),
        'extrusion_temp_C': rng.uniform(160, 230, n),
        'screw_speed_rpm': rng.uniform(30, 150, n),
        'sonication_min': rng.uniform(0, 60, n),
    })