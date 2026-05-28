"""
Collect and save transformer activations to disk for static SAE training.

This script pre-computes activations from transformer models and saves them
as memory-mapped NumPy files, enabling efficient reuse across multiple
SAE training runs.

Usage:
    python scripts/collect_embeddings.py \\
        --model-name EleutherAI/pythia-160m \\
        --layers 3 8 12 \\
        --dataset-name HuggingFaceFW/fineweb \\
        --n-contexts 450000 \\
        --ctx-len 128 \\
        --output-dir /path/to/embeddings \\
        --device cuda:0
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import List

import numpy as np
import torch as t
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add dictionary_learning to path
dict_learning_path = Path(__file__).parent.parent / "dictionary_learning"
sys.path.insert(0, str(dict_learning_path))

from dictionary_learning.utils import hf_dataset_to_generator


def get_model_slug(model_name: str) -> str:
    """Extract short slug from model name."""
    return model_name.split('/')[-1]


def get_dataset_slug(dataset_name: str) -> str:
    """Extract short slug from dataset name."""
    return dataset_name.split('/')[-1]


def get_hook_name(layer: int) -> str:
    """Generate hook name for layer."""
    return f"blocks.{layer}.hook_resid_post"


def get_submodule(model: AutoModelForCausalLM, hook_name: str, model_name: str):
    """
    Get submodule from model by hook name.

    Args:
        model: HuggingFace model
        hook_name: TransformerLens-style hook name (e.g., "blocks.3.hook_resid_post")
        model_name: Model name to determine architecture

    Returns:
        Submodule to hook for activation extraction
    """
    # Parse hook name like "blocks.3.hook_resid_post" to extract layer number
    if "blocks" in hook_name and "hook_resid_post" in hook_name:
        # Extract layer number from hook name
        parts = hook_name.split('.')
        layer_idx = int(parts[1])  # "blocks.3.hook_resid_post" -> 3

        # Map to actual model architecture
        if "pythia" in model_name.lower() or "gpt-neox" in model_name.lower():
            # Pythia/GPT-NeoX: model.gpt_neox.layers[layer_idx]
            return model.gpt_neox.layers[layer_idx]
        elif "gemma" in model_name.lower():
            # Gemma: model.model.layers[layer_idx]
            return model.model.layers[layer_idx]
        elif "llama" in model_name.lower():
            # Llama: model.model.layers[layer_idx]
            return model.model.layers[layer_idx]
        elif "gpt2" in model_name.lower():
            # GPT-2: model.transformer.h[layer_idx]
            return model.transformer.h[layer_idx]
        else:
            raise ValueError(
                f"Unsupported model type: {model_name}\n"
                f"Supported models: Pythia, Gemma, Llama, GPT-2"
            )
    else:
        raise ValueError(
            f"Unsupported hook_name format: {hook_name}\n"
            f"Expected format: blocks.LAYER.hook_resid_post"
        )


def infer_d_model(model: AutoModelForCausalLM, layer: int) -> int:
    """Infer hidden dimension from model."""
    # Try to get from config
    if hasattr(model.config, 'hidden_size'):
        return model.config.hidden_size
    elif hasattr(model.config, 'd_model'):
        return model.config.d_model
    elif hasattr(model.config, 'n_embd'):
        return model.config.n_embd
    else:
        raise ValueError(f"Cannot infer d_model from model config: {model.config}")


class EarlyStopException(Exception):
    """Exception for stopping model forward pass early."""
    pass


def collect_activations_batch(
    model: AutoModelForCausalLM,
    submodule: t.nn.Module,
    inputs: dict,
) -> t.Tensor:
    """
    Collect activations from a single batch.

    Registers a forward hook on the submodule to capture activations,
    then raises EarlyStopException to skip unnecessary computation.

    Args:
        model: Transformer model
        submodule: Submodule to hook
        inputs: Tokenized batch (input_ids, attention_mask)

    Returns:
        Activations tensor of shape [batch_size, seq_len, d_model]
    """
    activations = None

    def hook(module, inputs, outputs):
        nonlocal activations
        if isinstance(outputs, tuple):
            activations = outputs[0]
        else:
            activations = outputs
        raise EarlyStopException()

    handle = submodule.register_forward_hook(hook)

    try:
        with t.no_grad():
            _ = model(**inputs)
    except EarlyStopException:
        pass
    finally:
        handle.remove()

    if activations is None:
        raise RuntimeError("Failed to collect activations")

    return activations


def collect_and_save_embeddings(
    model_name: str,
    layer: int,
    dataset_name: str,
    n_contexts: int,
    ctx_len: int,
    output_dir: Path,
    chunk_size: int,
    device: str,
    remove_bos: bool,
    dtype: str,
    refresh_batch_size: int,
):
    """
    Collect activations and save to chunked NumPy files.

    Args:
        model_name: HuggingFace model name
        layer: Layer number to extract activations from
        dataset_name: HuggingFace dataset name
        n_contexts: Number of contexts to collect
        ctx_len: Context length (sequence length)
        output_dir: Base output directory
        chunk_size: Number of activations per chunk file
        device: Device for computation
        remove_bos: Whether to remove BOS tokens
        dtype: Data type for saved arrays ('float16', 'float32', 'bfloat16')
        refresh_batch_size: Batch size for collection
    """
    print(f"\n{'='*80}")
    print(f"Collecting embeddings for {model_name} layer {layer}")
    print(f"{'='*80}\n")

    # Create output directory
    model_slug = get_model_slug(model_name)
    dataset_slug = get_dataset_slug(dataset_name)
    hook_name = get_hook_name(layer)

    embedding_dir = output_dir / model_slug / str(layer) / dataset_slug
    embedding_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {embedding_dir}")

    # Load model and tokenizer
    print(f"\nLoading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=getattr(t, dtype))
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Get submodule
    submodule = get_submodule(model, hook_name, model_name)
    d_model = infer_d_model(model, layer)

    print(f"Model dimension: {d_model}")
    print(f"Hook name: {hook_name}")

    # Create data generator
    print(f"\nLoading dataset: {dataset_name}")
    data_generator = hf_dataset_to_generator(dataset_name)

    # Determine target number of activations
    target_activations = n_contexts * ctx_len
    print(f"\nTarget activations: {target_activations:,} ({n_contexts:,} contexts × {ctx_len} tokens)")
    print(f"Chunk size: {chunk_size:,}")
    print(f"Expected chunks: {(target_activations // chunk_size) + 1}")

    # Collection state
    collected_activations = 0
    chunk_idx = 0
    current_chunk = []
    chunks_metadata = []

    # Convert dtype string to NumPy dtype
    np_dtype = getattr(np, dtype)

    # Progress bar
    pbar = tqdm(total=target_activations, desc="Collecting activations", unit="acts")

    def save_chunk(chunk_data, chunk_idx):
        """Save a chunk to disk."""
        chunk_array = np.concatenate(chunk_data, axis=0).astype(np_dtype)
        chunk_filename = f"embeddings_chunk_{chunk_idx:06d}.npy"
        chunk_path = embedding_dir / chunk_filename

        np.save(chunk_path, chunk_array)

        return {
            'index': chunk_idx,
            'filename': chunk_filename,
            'shape': list(chunk_array.shape),
            'dtype': str(chunk_array.dtype),
        }

    try:
        while collected_activations < target_activations:
            # Get batch of text
            try:
                texts = [next(data_generator) for _ in range(refresh_batch_size)]
            except StopIteration:
                print("\nWarning: Dataset exhausted before reaching target")
                break

            # Tokenize
            inputs = tokenizer(
                texts,
                return_tensors='pt',
                max_length=ctx_len,
                padding=True,
                truncation=True,
                add_special_tokens=True,
            ).to(device)

            # Collect activations
            hidden_states = collect_activations_batch(model, submodule, inputs)

            # Apply mask to filter tokens
            mask = inputs['attention_mask'] != 0

            if remove_bos:
                if tokenizer.bos_token_id is not None:
                    bos_mask = inputs['input_ids'] == tokenizer.bos_token_id
                    mask = mask & ~bos_mask
                else:
                    # Remove first non-pad token
                    first_one = (mask.to(t.int64).cumsum(dim=1) == 1) & mask
                    mask = mask & ~first_one

            # Extract valid activations
            hidden_states = hidden_states[mask].cpu().numpy()

            # Add to current chunk
            remaining_in_chunk = chunk_size - sum(len(a) for a in current_chunk)
            if len(hidden_states) <= remaining_in_chunk:
                current_chunk.append(hidden_states)
            else:
                # Split: fill current chunk, save, start new chunk
                current_chunk.append(hidden_states[:remaining_in_chunk])

                # Save current chunk
                chunk_metadata = save_chunk(current_chunk, chunk_idx)
                chunks_metadata.append(chunk_metadata)
                chunk_idx += 1

                # Start new chunk with remainder
                current_chunk = [hidden_states[remaining_in_chunk:]]

            # Update counters
            collected_activations += len(hidden_states)
            pbar.update(len(hidden_states))

            # Stop if we've reached target
            if collected_activations >= target_activations:
                break

    finally:
        pbar.close()

    # Save any remaining data in current chunk
    if current_chunk:
        chunk_metadata = save_chunk(current_chunk, chunk_idx)
        chunks_metadata.append(chunk_metadata)

    print(f"\nCollected {collected_activations:,} activations")
    print(f"Saved {len(chunks_metadata)} chunks")

    # Save metadata
    metadata = {
        'model_name': model_name,
        'model_revision': 'main',
        'dataset_name': dataset_name,
        'dataset_split': 'train',
        'layer': layer,
        'hook_name': hook_name,
        'd_model': d_model,
        'total_tokens': collected_activations,
        'n_contexts': n_contexts,
        'ctx_len': ctx_len,
        'remove_bos': remove_bos,
        'dtype': dtype,
        'collection_date': datetime.now().isoformat(),
        'collection_script_version': '1.0.0',
    }

    metadata_path = embedding_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved metadata: {metadata_path}")

    # Save manifest
    manifest = {
        'total_chunks': len(chunks_metadata),
        'chunk_size': chunk_size,
        'chunks': chunks_metadata,
    }

    manifest_path = embedding_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved manifest: {manifest_path}")
    print(f"\n{'='*80}")
    print(f"✓ Successfully collected embeddings for layer {layer}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Collect and save transformer activations to disk"
    )

    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='HuggingFace model name (e.g., EleutherAI/pythia-160m)',
    )
    parser.add_argument(
        '--layers',
        type=int,
        nargs='+',
        required=True,
        help='Layer numbers to collect (e.g., 3 8 12)',
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        required=True,
        help='HuggingFace dataset name (e.g., HuggingFaceFW/fineweb)',
    )
    parser.add_argument(
        '--n-contexts',
        type=int,
        default=450000,
        help='Number of contexts to collect (default: 450000)',
    )
    parser.add_argument(
        '--ctx-len',
        type=int,
        default=128,
        help='Context length (default: 128)',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./embeddings'),
        help='Output directory (default: ./embeddings)',
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1024000,
        help='Number of activations per chunk (default: 1024000)',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device for computation (default: cuda:0)',
    )
    parser.add_argument(
        '--remove-bos',
        action='store_true',
        help='Remove BOS tokens from activations',
    )
    parser.add_argument(
        '--dtype',
        type=str,
        choices=['float16', 'float32', 'bfloat16'],
        default='float16',
        help='Data type for saved arrays (default: float16)',
    )
    parser.add_argument(
        '--refresh-batch-size',
        type=int,
        default=512,
        help='Batch size for collection (default: 512)',
    )

    args = parser.parse_args()

    # Collect embeddings for each layer
    for layer in args.layers:
        collect_and_save_embeddings(
            model_name=args.model_name,
            layer=layer,
            dataset_name=args.dataset_name,
            n_contexts=args.n_contexts,
            ctx_len=args.ctx_len,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            device=args.device,
            remove_bos=args.remove_bos,
            dtype=args.dtype,
            refresh_batch_size=args.refresh_batch_size,
        )


if __name__ == '__main__':
    main()
