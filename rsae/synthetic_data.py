"""Synthetic data generation for evaluating sparse autoencoders."""

import torch
import numpy as np
import hashlib
import pickle
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass, asdict
from .utils.model_utils import extract_decoder_weights
from .utils.topk import apply_topk_and_scatter


@dataclass
class SyntheticDataConfig:
    """
    Configuration for synthetic data generation.

    Args:
        n_samples: Number of samples to generate. If None, enables online mode
                  where data is generated on-the-fly during training.
        concept_dim: Dimension of true concept factors
        observed_dim: Dimension of observed data
        k: Number of active concept factors per sample (sparsity level)
        noise_std: Standard deviation of Gaussian observation noise
        seed: Random seed for reproducibility
        distribution: Distribution for mixing matrix entries (gaussian, beta_half, gamma, cauchy, gaussian_mixture)
        mixture_scale: Variance fraction for gaussian_mixture (component variance fraction, marginal var = 1)
        num_mixtures: Number of mixture components for gaussian_mixture. If None, defaults to max(2, concept_dim // 100)
    """
    n_samples: Optional[int] = None
    concept_dim: int = 100
    observed_dim: int = 50
    k: int = 10
    noise_std: float = 0.
    seed: Optional[int] = None
    distribution: str = "gaussian"
    mixture_scale: float = 0.5
    num_mixtures: Optional[int] = None

    @property
    def regime(self) -> str:
        """Determine if this is undercomplete, exact, or overcomplete regime."""
        if self.concept_dim < self.observed_dim:
            return "undercomplete"
        elif self.concept_dim == self.observed_dim:
            return "exact"
        else:
            return "overcomplete"

    @property
    def is_online_mode(self) -> bool:
        """Check if this config is for online generation mode."""
        return self.n_samples is None


# Increment when generation logic changes to invalidate stale cache
_CACHE_VERSION = 2

def _compute_cache_key(config: SyntheticDataConfig) -> str:
    """Compute a unique hash key for caching based on data generation parameters."""
    cache_dict = asdict(config)
    cache_dict['_cache_version'] = _CACHE_VERSION
    # Create deterministic string representation
    cache_str = str(sorted(cache_dict.items()))
    # Hash it
    return hashlib.sha256(cache_str.encode()).hexdigest()[:16]


def _get_cache_path(cache_key: str, cache_dir: Optional[Path] = None) -> Path:
    """Get the cache file path for a given cache key."""
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "rsae" / "synthetic_data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{cache_key}.pkl"


def _save_to_cache(
    cache_path: Path,
    observations: torch.Tensor,
    concepts: torch.Tensor,
    mixing_matrix: torch.Tensor,
):
    """Save generated data to cache."""
    data = {
        'observations': observations.cpu(),
        'concepts': concepts.cpu(),
        'mixing_matrix': mixing_matrix.cpu(),
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)


def _load_from_cache(cache_path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load generated data from cache."""
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    return data['observations'], data['concepts'], data['mixing_matrix']


def generate_synthetic_data(
    config: SyntheticDataConfig,
    verbose: bool = False,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic data from a sparse concept factor model.

    The generative model is:
        z ~ Sparse (k-sparse per sample)
        x = z @ W + noise

    where W is a random mixing matrix.

    Args:
        config: Configuration object specifying data generation parameters
        verbose: If True, print progress
        use_cache: If True, use cached data if available and cache newly generated data
        cache_dir: Directory to store cached data (default: ~/.cache/rsae/synthetic_data)

    Returns:
        observations: Generated observations [n_samples, observed_dim]
        concepts: True sparse concept factors [n_samples, concept_dim]
        mixing_matrix: True mixing matrix [concept_dim, observed_dim]
    """
    # Check cache if enabled
    if use_cache:
        cache_key = _compute_cache_key(config)
        cache_path = _get_cache_path(cache_key, cache_dir)

        if cache_path.exists():
            if verbose:
                print(f"  Loading cached synthetic data from {cache_path}")
            observations, concepts, mixing_matrix = _load_from_cache(cache_path)
            return observations, concepts, mixing_matrix
        elif verbose:
            print(f"  No cached data found. Generating new data...")
            print(f"  Cache key: {cache_key}")

    # Derive independent seeds for each component to ensure deterministic generation
    # regardless of RNG state or order of operations
    if config.seed is not None:
        mixing_matrix_seed = config.seed
        concepts_seed = config.seed + 1
        noise_seed = config.seed + 2
    else:
        mixing_matrix_seed = None
        concepts_seed = None
        noise_seed = None

    # Generate random mixing matrix with its own seed
    if mixing_matrix_seed is not None:
        torch.manual_seed(mixing_matrix_seed)
        np.random.seed(mixing_matrix_seed)
    mixing_matrix = generate_mixing_matrix(
        config.observed_dim,
        config.concept_dim,
        distribution=config.distribution,
        mixture_scale=config.mixture_scale,
        num_mixtures=config.num_mixtures,
    )

    # Generate sparse concepts with its own seed
    if concepts_seed is not None:
        torch.manual_seed(concepts_seed)
        np.random.seed(concepts_seed)
    concepts = generate_sparse_concepts(
        config.n_samples,
        config.concept_dim,
        config.k,
        normalize=True,
    )

    # Generate observations
    observations = concepts @ mixing_matrix

    # Add Gaussian noise with its own seed
    if config.noise_std > 0:
        if noise_seed is not None:
            torch.manual_seed(noise_seed)
            np.random.seed(noise_seed)
        noise = torch.randn_like(observations) * config.noise_std
        observations = observations + noise

    # Cache the generated data if caching is enabled
    if use_cache:
        if verbose:
            print(f"  Saving generated data to cache: {cache_path}")
        _save_to_cache(cache_path, observations, concepts, mixing_matrix)

    return observations, concepts, mixing_matrix


def generate_mixing_matrix(
    observed_dim: int,
    concept_dim: int,
    distribution: str = "gaussian",
    mixture_scale: float = 0.5,
    num_mixtures: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate a random mixing matrix with entries from specified distribution.

    Each row (a concept's mixing vector) is normalized to unit L2 norm, so the
    per-entry variance is ~ 1/observed_dim regardless of the source distribution.

    Args:
        observed_dim: Dimension of observations
        concept_dim: Dimension of concept factors
        distribution: Distribution type. Options:
            - "gaussian": N(0, 1) entries
            - "beta_half": Beta(1/2, 1/2) entries on [0, 1] (U-shaped)
            - "gamma": Gamma(shape=1, rate=2) entries
            - "cauchy": Cauchy(loc=0, scale=1) - heavy-tailed distribution to violate RIP
            - "gaussian_mixture": Mixture of K Gaussians with marginal variance 1
            All rows are L2-normalized after sampling.
        mixture_scale: Variance fraction for gaussian_mixture. Component means ~ N(0, mixture_scale·I), within ~ N(μ, (1-mixture_scale)·I)
        num_mixtures: Number of mixture components for gaussian_mixture. If None, defaults to max(2, concept_dim // 100)

    Returns:
        Mixing matrix [concept_dim, observed_dim] with unit-norm rows (per-entry variance ~ 1/observed_dim)
    """
    # Generate raw matrix from specified distribution
    if distribution == "gaussian":
        W = torch.randn(concept_dim, observed_dim)
    elif distribution == "beta_half":
        # Beta(1/2, 1/2) - U-shaped distribution on [0, 1]
        W = torch.distributions.Beta(0.5, 0.5).sample((concept_dim, observed_dim))
    elif distribution == "gamma":
        # Gamma(1, 2) - shape=1 (exponential-like), rate=2
        W = torch.distributions.Gamma(1.0, 2.0).sample((concept_dim, observed_dim))
    elif distribution == "cauchy":
        # Cauchy(0, 2) - heavy-tailed distribution with location=0, scale=2
        W = torch.distributions.Cauchy(0.0, 1.0).sample((concept_dim, observed_dim))
    elif distribution == "gaussian_mixture":
        # Mixture of K Gaussians
        # Default: K = max(2, concept_dim // 100) if not specified
        K = num_mixtures if num_mixtures is not None else max(2, concept_dim // 100)

        # Sample K component means from N(0, mixture_scale * I)
        # Using sqrt(mixture_scale) as std dev so variance = mixture_scale
        component_means = torch.sqrt(torch.tensor(mixture_scale)) * torch.randn(K, observed_dim)

        # Balanced assignment: each component gets ~equal concepts
        assignments = torch.arange(concept_dim) % K
        assignments = assignments[torch.randperm(concept_dim)]  # shuffle

        # Sample each concept from its assigned component
        # Using sqrt(1 - mixture_scale) as std dev so within-component variance = 1 - mixture_scale
        # Marginal variance: mixture_scale + (1 - mixture_scale) = 1
        W = torch.zeros(concept_dim, observed_dim)
        within_std = torch.sqrt(torch.tensor(1.0 - mixture_scale))
        for i in range(concept_dim):
            k = assignments[i]
            W[i] = component_means[k] + within_std * torch.randn(observed_dim)
    else:
        raise ValueError(f"Unknown distribution: {distribution}. Choose from: gaussian, beta_half, gamma, cauchy, gaussian_mixture")

    # Normalize to unit norm
    W /= W.norm(dim=1, keepdim=True).clamp_min(1e-8)

    return W


def generate_sparse_concepts(
    n_samples: int,
    concept_dim: int,
    k: int,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Generate sparse concept codes with exactly k active factors per sample.

    For each sample:
        - Draw a dense Gaussian vector and keep the k entries with largest
          magnitude (signs preserved), zeroing the rest
        - Optionally normalize the resulting sparse vector to unit length

    Args:
        n_samples: Number of samples
        concept_dim: Dimension of concept space
        k: Number of active factors per sample
        normalize: If True, normalize each sample to unit length (default: True)

    Returns:
        Sparse concept codes [n_samples, concept_dim] with exactly k nonzero per sample
    """
    concepts = torch.randn(n_samples, concept_dim)
    concepts = apply_topk_and_scatter(concepts, k, use_abstopk=True)
    if normalize:
        concepts = concepts / concepts.norm(dim=1, keepdim=True).clamp_min(1e-8)

    return concepts


def generate_synthetic_batch(
    config: SyntheticDataConfig,
    batch_size: int,
    mixing_matrix: Optional[torch.Tensor] = None,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a single batch of synthetic data on-the-fly.

    This is designed for online generation mode where we generate batches
    as needed rather than pre-generating a full dataset.

    Args:
        config: Configuration for data generation (n_samples is ignored)
        batch_size: Number of samples in this batch
        mixing_matrix: Pre-generated mixing matrix to use. If None, generates a new one.
                      For consistent online generation, pass a fixed mixing matrix.
        normalize: If True, normalize observations (same as _normalize_data in BaseTrainer)

    Returns:
        observations: Generated observations [batch_size, observed_dim]
        concepts: True sparse concept factors [batch_size, concept_dim]
        mixing_matrix: Mixing matrix used [concept_dim, observed_dim]
    """
    # Generate sparse concepts for this batch
    concepts = generate_sparse_concepts(
        n_samples=batch_size,
        concept_dim=config.concept_dim,
        k=config.k,
        normalize=True,
    )

    # Generate or use provided mixing matrix
    if mixing_matrix is None:
        mixing_matrix = generate_mixing_matrix(
            config.observed_dim,
            config.concept_dim,
            distribution=config.distribution,
            mixture_scale=config.mixture_scale,
            num_mixtures=config.num_mixtures,
        )

    # Generate observations
    observations = concepts @ mixing_matrix

    # Add Gaussian noise
    if config.noise_std > 0:
        noise = torch.randn_like(observations) * config.noise_std
        observations = observations + noise

    # Normalize if requested (per-sample centering + L2 normalization)
    if normalize:
        # Center each sample
        centered = observations - observations.mean(dim=1, keepdim=True)
        # Compute L2 norm per sample
        norms = torch.norm(centered, dim=1, keepdim=True)
        # Handle zero norms
        safe_norms = torch.where(norms < 1e-8, torch.ones_like(norms), norms)
        observations = centered / safe_norms

    return observations, concepts, mixing_matrix


class SyntheticDataGenerator:
    """
    Generator for online synthetic data.

    Handles batch generation with a fixed mixing matrix for consistent ground truth.
    Optimized for speed - generates directly on device, no seed setting per batch.
    """

    def __init__(self, config: SyntheticDataConfig, mixing_matrix: torch.Tensor, device: Optional[torch.device] = None):
        """
        Initialize generator with config and mixing matrix.

        Args:
            config: Data generation configuration
            mixing_matrix: Fixed mixing matrix to use across all batches
            device: Device to generate batches on (CPU or GPU)
        """
        self.config = config
        self.device = device if device is not None else torch.device('cpu')
        self.mixing_matrix = mixing_matrix.to(self.device)
        self.k = config.k
        self.noise_std = config.noise_std

    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of observations and concepts.

        Fast generation without seed setting, directly on device.

        Args:
            batch_size: Number of samples to generate

        Returns:
            observations: Generated observations [batch_size, observed_dim]
            concepts: True sparse concept factors [batch_size, concept_dim]
        """
        # Generate sparse concepts (fast, on device)
        concepts = torch.randn(batch_size, self.config.concept_dim, device=self.device)
        concepts = apply_topk_and_scatter(concepts, self.k, use_abstopk=True)
        concepts = concepts / concepts.norm(dim=1, keepdim=True).clamp_min(1e-8)

        # Generate observations
        observations = concepts @ self.mixing_matrix

        # Add noise if needed
        if self.noise_std > 0:
            observations = observations + torch.randn_like(observations) * self.noise_std

        # Normalize (per-sample centering + L2 norm)
        centered = observations - observations.mean(dim=1, keepdim=True)
        norms = torch.norm(centered, dim=1, keepdim=True)
        safe_norms = torch.where(norms < 1e-8, torch.ones_like(norms), norms)
        observations = centered / safe_norms

        return observations, concepts
