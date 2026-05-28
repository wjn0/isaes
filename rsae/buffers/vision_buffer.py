"""Activation buffer for vision models (DINOv2).

This module provides VisionActivationBuffer for collecting patch embeddings
from vision transformers like DINOv2.
"""

import gc
from typing import Generator, Optional

import torch
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model


class VisionActivationBuffer:
    """
    Activation buffer for DINOv2 vision models.

    Extracts patch embeddings from a specified layer, treating each patch
    as an independent sample for SAE training (similar to how LLM buffers
    treat tokens).

    The buffer stores activations from a model, yields them in batches,
    and refreshes them when the buffer is less than half full.

    Parameters
    ----------
    data : Generator
        Generator which yields PIL Images
    model_name : str
        HuggingFace model name (e.g., "facebook/dinov2-base")
    hook_layer : int
        Layer to extract activations from (0-indexed, DINOv2-base has 12 layers)
    d_submodule : int
        Model dimension (768 for dinov2-base)
    n_images : int
        Approximate number of images worth of patches to store in the buffer
    patches_per_image : int
        Number of patches per image (196 for 224x224 with 16x16 patches)
    refresh_batch_size : int
        Number of images to process per refresh batch
    out_batch_size : int
        Number of patches to yield per __next__ call
    device : str
        Device on which to store the activations
    include_cls : bool
        Whether to include CLS token (default False)
    """

    def __init__(
        self,
        data: Generator,
        model_name: str = "facebook/dinov2-base",
        hook_layer: int = 11,
        d_submodule: int = 768,
        n_images: int = 50000,
        patches_per_image: int = 196,
        refresh_batch_size: int = 64,
        out_batch_size: int = 2048,
        device: str = "cuda",
        include_cls: bool = False,
    ):
        self.data = data
        self.model_name = model_name
        self.hook_layer = hook_layer
        self.d_submodule = d_submodule
        self.n_images = n_images
        self.patches_per_image = patches_per_image
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.device = device
        self.include_cls = include_cls

        # Calculate buffer size
        self.activation_buffer_size = int(n_images * patches_per_image)

        # Load model and processor
        print(f"Loading vision model: {model_name}...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Dinov2Model.from_pretrained(model_name).to(device)
        self.model.eval()

        # Get the target layer for hook
        self.target_layer = self.model.encoder.layer[hook_layer]

        # Initialize empty buffer
        self.activations = torch.empty(
            0, d_submodule, device=device, dtype=self.model.dtype
        )
        self.read = torch.zeros(0, dtype=torch.bool, device=device)

        # Storage for hook output
        self._hook_output: Optional[torch.Tensor] = None

    def _hook_fn(self, module, inputs, outputs):
        """Forward hook to capture layer activations."""
        # DINOv2 layer output is a tuple, first element is hidden states
        if isinstance(outputs, tuple):
            self._hook_output = outputs[0]
        else:
            self._hook_output = outputs

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        """Return a batch of patch activations."""
        with torch.no_grad():
            # Refresh if buffer is less than half full
            if (~self.read).sum() < self.activation_buffer_size // 2:
                self.refresh()

            # Sample batch of unread activations
            unreads = (~self.read).nonzero().squeeze()
            if unreads.dim() == 0:
                unreads = unreads.unsqueeze(0)
            idxs = unreads[
                torch.randperm(len(unreads), device=self.device)[: self.out_batch_size]
            ]
            self.read[idxs] = True
            return self.activations[idxs]

    def _image_batch(self, batch_size: Optional[int] = None) -> list:
        """Get a batch of PIL images from the generator."""
        if batch_size is None:
            batch_size = self.refresh_batch_size
        images = []
        for _ in range(batch_size):
            try:
                img = next(self.data)
                # Ensure RGB (some images may be grayscale)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                images.append(img)
            except StopIteration:
                break
        if len(images) == 0:
            raise StopIteration("End of image data stream reached")
        return images

    def refresh(self):
        """Refresh the activation buffer with new patches."""
        gc.collect()
        torch.cuda.empty_cache()

        # Keep unread activations
        self.activations = self.activations[~self.read]

        current_idx = len(self.activations)
        new_activations = torch.empty(
            self.activation_buffer_size,
            self.d_submodule,
            device=self.device,
            dtype=self.model.dtype,
        )

        # Copy existing unread activations
        if current_idx > 0:
            new_activations[:current_idx] = self.activations
        self.activations = new_activations

        # Register forward hook
        handle = self.target_layer.register_forward_hook(self._hook_fn)

        try:
            while current_idx < self.activation_buffer_size:
                with torch.no_grad():
                    # Get batch of images
                    try:
                        images = self._image_batch()
                    except StopIteration:
                        print(
                            f"Warning: Image stream exhausted after {current_idx} patches"
                        )
                        break

                    # Preprocess images
                    inputs = self.processor(images, return_tensors="pt").to(self.device)

                    # Forward pass (hook captures activations)
                    _ = self.model(**inputs)

                    # Get captured activations: (batch, num_patches+1, d_model)
                    hidden_states = self._hook_output

                    if hidden_states is None:
                        raise RuntimeError("Hook did not capture activations")

                    # Remove CLS token if not including it (first position)
                    if not self.include_cls:
                        hidden_states = hidden_states[:, 1:, :]  # (batch, 196, d_model)

                    # Flatten batch and patches: (batch * patches, d_model)
                    batch_size, num_patches, d_model = hidden_states.shape
                    hidden_states = hidden_states.reshape(-1, d_model)

                    # Add to buffer (respecting remaining space)
                    remaining_space = self.activation_buffer_size - current_idx
                    hidden_states = hidden_states[:remaining_space]

                    self.activations[
                        current_idx : current_idx + len(hidden_states)
                    ] = hidden_states
                    current_idx += len(hidden_states)

        finally:
            handle.remove()

        # Trim buffer if we collected fewer than expected
        if current_idx < self.activation_buffer_size:
            self.activations = self.activations[:current_idx]

        self.read = torch.zeros(
            len(self.activations), dtype=torch.bool, device=self.device
        )

    def __len__(self) -> int:
        """Return the number of unread activations in the buffer."""
        return (~self.read).sum().item()

    @property
    def config(self):
        """Return buffer configuration for logging."""
        return {
            "model_name": self.model_name,
            "hook_layer": self.hook_layer,
            "d_submodule": self.d_submodule,
            "n_images": self.n_images,
            "patches_per_image": self.patches_per_image,
            "refresh_batch_size": self.refresh_batch_size,
            "out_batch_size": self.out_batch_size,
            "device": str(self.device),
            "include_cls": self.include_cls,
        }
