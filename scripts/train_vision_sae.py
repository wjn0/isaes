"""Train RIPTopK SAE on DINOv2 vision model activations.

This script:
1. Collects patch activations from DINOv2 using VisionActivationBuffer
2. Trains a RIPTopK SAE with RIP + AuxK + reconstruction loss
3. Logs the trained model + Hydra config to MLflow (periodic checkpoints are
   written under outputs/ by the trainer)

Note: ImageNet requires HuggingFace authentication. Run `huggingface-cli login`
and accept terms at https://huggingface.co/datasets/imagenet-1k
"""

from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import mlflow
import mlflow.pytorch

from rsae import RIPTopK
from rsae.buffers import VisionActivationBuffer
from rsae.utils import flatten_dict, create_rip_topk_model_from_config


def imagenet_to_generator(dataset_name: str = "imagenet-1k", split: str = "train"):
    """
    Create a generator that yields PIL Images from ImageNet.

    Note: ImageNet requires authentication. Users must:
    1. Accept terms at https://huggingface.co/datasets/imagenet-1k
    2. Run `huggingface-cli login`

    Parameters
    ----------
    dataset_name : str
        HuggingFace dataset name
    split : str
        Dataset split ("train" or "validation")

    Yields
    ------
    PIL.Image
        RGB images from the dataset
    """
    from datasets import load_dataset

    print(f"Loading dataset: {dataset_name} (split: {split})")
    dataset = load_dataset(
        dataset_name, split=split, streaming=True, trust_remote_code=True
    )

    for x in iter(dataset):
        img = x["image"]
        # Ensure RGB (some ImageNet images may be grayscale)
        if img.mode != "RGB":
            img = img.convert("RGB")
        yield img


def collect_activations(cfg: DictConfig, device: str) -> VisionActivationBuffer:
    """
    Collect activations from DINOv2 using VisionActivationBuffer.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration
    device : str
        Device for computation

    Returns
    -------
    VisionActivationBuffer
        Buffer filled with patch activations
    """
    model_name = cfg.data.model_name
    hook_layer = cfg.data.hook_layer
    n_images = cfg.data.n_images
    dataset_name = cfg.data.dataset_name
    d_model = cfg.data.d_model
    patches_per_image = cfg.data.get("patches_per_image", 196)
    refresh_batch_size = cfg.data.get("refresh_batch_size", 64)

    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Hook layer: {hook_layer}")
    print(f"  d_model: {d_model}")
    print(f"  n_images: {n_images:,}")
    print(f"  patches_per_image: {patches_per_image}")

    print("\nCreating image generator...")
    data = imagenet_to_generator(dataset_name)

    print("\nCreating VisionActivationBuffer...")

    activation_buffer = VisionActivationBuffer(
        data=data,
        model_name=model_name,
        hook_layer=hook_layer,
        d_submodule=d_model,
        n_images=n_images,
        patches_per_image=patches_per_image,
        refresh_batch_size=refresh_batch_size,
        out_batch_size=cfg.training.batch_size,
        device=device,
        include_cls=False,
    )

    return activation_buffer


def train_riptopk(
    cfg: DictConfig,
    train_buffer: VisionActivationBuffer,
    d_in: int,
    output_dir: Path,
    experiment_name: str,
    device: str,
    val_buffer: Optional[VisionActivationBuffer] = None,
) -> RIPTopK:
    """
    Train a RIPTopK SAE using TransformerTrainer.

    The TransformerTrainer is buffer-agnostic and works with VisionActivationBuffer
    since both provide the same iterator interface.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration
    train_buffer : VisionActivationBuffer
        Training buffer
    d_in : int
        Input dimension
    output_dir : Path
        Directory to save checkpoints
    experiment_name : str
        Name for this experiment
    device : str
        Device for training
    val_buffer : VisionActivationBuffer, optional
        Validation buffer

    Returns
    -------
    RIPTopK
        Trained model
    """
    from rsae.trainers import TransformerTrainer

    # Extract parameters from config for logging
    nb_concepts = cfg.data.nb_concepts
    top_k = cfg.data.k
    rip_weight = cfg.model.rip_weight
    auxk_weight = cfg.model.get('auxk_weight', 1.0 / 32)

    # Create RIPTopK model using shared utility
    print(f"\nInitializing RIPTopK SAE:")
    print(f"  d_in: {d_in}")
    print(f"  nb_concepts: {nb_concepts}")
    print(f"  top_k: {top_k}")
    print(f"  rip_weight: {rip_weight}")
    print(f"  auxk_weight: {auxk_weight}")

    model = create_rip_topk_model_from_config(
        cfg=cfg,
        input_dim=d_in,
        nb_concepts=nb_concepts,
        device=device,
    ).to(device)

    # Create TransformerTrainer (works with any buffer implementing __iter__/__next__)
    trainer = TransformerTrainer(
        cfg=cfg,
        model=model,
        train_buffer=train_buffer,
        device=device,
        output_dir=output_dir,
        experiment_name=experiment_name,
        val_buffer=val_buffer,
    )

    # Train the model
    model = trainer.train()

    return model


@hydra.main(version_base=None, config_path="../configs", config_name=None)
def main(cfg: DictConfig):
    """Main training function."""
    print("=" * 80)
    print("Vision SAE Training with Hydra + MLflow")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Setup device
    if cfg.device.use_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{cfg.device.gpu_id}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Initialize MLflow
    if cfg.mlflow.enabled:
        from rsae.utils import get_or_create_experiment

        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        experiment_id = get_or_create_experiment(cfg.mlflow.experiment_name)
        mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"{cfg.model.name}_{cfg.experiment.name}",
            tags={
                "job_type": cfg.experiment.job_type,
                "group": cfg.experiment.get("group", ""),
            },
        )
        mlflow.log_params(
            {
                f"{k}": v
                for k, v in flatten_dict(
                    OmegaConf.to_container(cfg, resolve=True)
                ).items()
            }
        )

    # Collect activations
    print("\nCollecting activations from vision model...")
    train_buffer = collect_activations(cfg, str(device))
    val_buffer = None  # Validation not supported with dynamic collection
    print("Activation buffer created successfully!")

    # Get input dimension
    d_in = train_buffer.d_submodule
    print(f"  Input dimension (d_model): {d_in}")

    # Train model
    print("\nTraining RIPTopK SAE...")
    model = train_riptopk(
        cfg=cfg,
        train_buffer=train_buffer,
        d_in=d_in,
        output_dir=Path("outputs"),
        experiment_name=cfg.experiment.name,
        device=str(device),
        val_buffer=val_buffer,
    )

    # Save final model
    if cfg.mlflow.enabled and cfg.mlflow.log_artifacts:
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        mlflow.log_dict(config_dict, "model/extra_files/config.yaml")
        mlflow.pytorch.log_model(model, name="model")
        print("Model saved to MLflow!")

    # End MLflow run
    if cfg.mlflow.enabled:
        mlflow.end_run()

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
