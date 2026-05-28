import torch
import torch.nn as nn


class ZCAWhitener(nn.Module):
    """
    ZCA whitening module with one-time initialization.

    z = W (x - mean)
    x = W^{-1} z + mean
    where W ≈ Σ^{-1/2}

    Call initialize_from_data() once before training to compute and fix
    the whitening transform. The transform remains fixed during training.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-4,
        device=None,
        dtype=None,
    ):
        super().__init__()

        factory_kwargs = dict(device=device, dtype=dtype)

        self.dim = dim
        self.eps = eps

        # Running mean μ
        self.register_buffer(
            "running_mean",
            torch.zeros(dim, **factory_kwargs)
        )

        # Running covariance Σ (EMA)
        self.register_buffer(
            "running_cov",
            torch.eye(dim, **factory_kwargs)
        )

        # Whitening matrix W ≈ Σ^{-1/2}
        self.register_buffer(
            "W",
            torch.eye(dim, **factory_kwargs)
        )

    @torch.no_grad()
    def initialize_from_data(self, data: torch.Tensor):
        """
        Compute and fix whitening transform from data samples.

        This computes the mean, covariance, and whitening matrix W = Σ^{-1/2}
        from the provided data. The transform is then fixed (no online updates).

        Parameters
        ----------
        data : torch.Tensor
            Data samples of shape (N, D) to compute whitening from
        """
        data = data.float()  # Ensure float precision, half implementations don't exist
        N = data.shape[0]

        # Compute mean
        mean = data.mean(dim=0)
        self.running_mean.copy_(mean)

        # Compute covariance
        xc = data - mean
        cov = (xc.T @ xc) / N
        self.running_cov.copy_(cov)

        # Compute whitening matrix W = Σ^{-1/2} via eigendecomposition
        I = torch.eye(self.dim, device=data.device, dtype=data.dtype)
        cov_hat = cov + self.eps * I

        # Eigendecomposition: cov_hat = Q @ diag(eigenvalues) @ Q.T
        eigenvalues, Q = torch.linalg.eigh(cov_hat)

        # Clamp eigenvalues for numerical stability
        eigenvalues = eigenvalues.clamp(min=self.eps)

        # W = Q @ diag(1/sqrt(eigenvalues)) @ Q.T
        W_init = Q @ torch.diag(eigenvalues.rsqrt()) @ Q.T
        # ensure dtype matches
        self.W.copy_(W_init.to(self.W))

    def whiten(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ZCA whitening.
        """
        x = x - self.running_mean
        return x @ self.W.T

    def unwhiten(self, z: torch.Tensor) -> torch.Tensor:
        """
        Invert the whitening transform using:
            W^{-1} = Σ W
        """
        I = torch.eye(self.dim, device=z.device, dtype=z.dtype)
        cov_hat = self.running_cov + self.eps * I

        W_inv = cov_hat @ self.W
        x = z @ W_inv.T
        return x + self.running_mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the fixed whitening transform.
        """
        return self.whiten(x)
