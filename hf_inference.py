"""
HuggingFace Inference Module

Handles inference for any HuggingFace model (Mistral, Llama, etc.)
- Auto GPU/CPU detection (graceful fallback on MacBook)
- Model caching for reuse across multiple calls
- Response format matches litellm.completion() for compatibility
- Supports temperature, top_p, and other sampling parameters

Usage:
    response, input_tokens, output_tokens = call_hf_with_retry(
        system_prompt="You are helpful",
        user_prompt="Answer this question",
        model="mistralai/Mistral-7B-v0.3",
        temperature=0.7,
        max_tokens=4096
    )
"""

import torch
import logging
from typing import Tuple, Optional, Dict, Any
import time

logger = logging.getLogger(__name__)

# Global model cache - loaded once, reused for all calls
_MODEL_CACHE: Dict[str, Any] = {}
_TOKENIZER_CACHE: Dict[str, Any] = {}
_DEVICE: Optional[str] = None

# ==========================================================
# Device Detection (GPU/CPU)
# ==========================================================

def _detect_device() -> str:
    """
    Detect available device for inference.
    Priority: CUDA > MPS > CPU

    Returns:
        Device string: "cuda" (NVIDIA GPU), "mps" (Apple GPU), or "cpu"
    """
    global _DEVICE

    if _DEVICE is not None:
        return _DEVICE

    # Check for NVIDIA GPU FIRST (cluster priority)
    if torch.cuda.is_available():
        _DEVICE = "cuda"
        logger.info(f"✓ Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        return _DEVICE

    # Check for Apple GPU (MacBook fallback)
    if torch.backends.mps.is_available():
        try:
            _DEVICE = "mps"
            logger.info("✓ Using Apple GPU (MPS)")
            return _DEVICE
        except Exception as e:
            logger.warning(f"MPS failed: {e}")

    # Fallback to CPU
    _DEVICE = "cpu"
    logger.info("ℹ Using CPU for inference")
    return _DEVICE


# ==========================================================
# Model Loading
# ==========================================================

def load_hf_model(model_id: str, device: Optional[str] = None):
    """
    Load HuggingFace model and tokenizer.
    Models are cached globally and reused.
    
    Args:
        model_id: HuggingFace model ID (e.g., "mistralai/Mistral-7B-v0.3")
        device: Device to load model on ("cuda", "mps", "cpu", or None for auto-detect)
    
    Returns:
        Tuple of (model, tokenizer, device_used)
    
    Raises:
        Exception: If model loading fails
    """
    # Return cached model if already loaded
    if model_id in _MODEL_CACHE:
        logger.debug(f"Using cached model: {model_id}")
        model = _MODEL_CACHE[model_id]
        tokenizer = _TOKENIZER_CACHE[model_id]
        device_used = model.device.type if hasattr(model, 'device') else _DEVICE
        return model, tokenizer, device_used
    
    logger.info(f"Loading model: {model_id}")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers is required. Install with: pip install transformers"
        )
    
    # Detect device if not provided
    if device is None:
        device = _detect_device()
    
    try:
        # Load tokenizer
        logger.debug(f"Loading tokenizer from {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # Set pad token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.debug("Set pad_token to eos_token")

        # Load model
        logger.debug(f"Loading model from {model_id}...")

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        }

        # Add device_map for efficient multi-GPU loading on CUDA
        if device == "cuda":
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        
        # Move to device if not already there
        if device != "cuda":  # device_map handles cuda automatically
            model = model.to(device)
        
        logger.info(f"✓ Model loaded successfully on {device}")
        
        # Cache model and tokenizer
        _MODEL_CACHE[model_id] = model
        _TOKENIZER_CACHE[model_id] = tokenizer
        
        return model, tokenizer, device
        
    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {e}")
        raise


# ==========================================================
# Inference
# ==========================================================

def call_hf_with_retry(
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_retries: int = 3,
        top_p: float = 0.9
) -> Tuple[str, int, int]:
    """
    Call HuggingFace model for inference.
    
    Matches litellm.completion() response format for compatibility with existing code.
    
    Args:
        system_prompt: System prompt (instructions to model)
        user_prompt: User prompt (the actual query)
        model: HuggingFace model ID (e.g., "mistralai/Mistral-7B-v0.3")
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        max_retries: Number of retry attempts
        top_p: Nucleus sampling parameter (0.0-1.0)
    
    Returns:
        Tuple of (response_text, input_tokens, output_tokens)
    
    Raises:
        Exception: If inference fails after all retries
    """
    logger.debug(f"Calling HF model: {model}")
    
    # Load model once (cached for subsequent calls)
    model_obj, tokenizer, device = load_hf_model(model)
    
    for attempt in range(max_retries):
        try:
            # Format messages in chat format
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Some models support chat template, apply if available
            if hasattr(tokenizer, 'apply_chat_template'):
                try:
                    prompt_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception as e:
                    logger.debug(f"Chat template failed, using manual format: {e}")
                    # Fallback: manual format
                    prompt_text = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
            else:
                # No chat template, manual format
                prompt_text = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

            # Tokenize
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(device)

            input_token_count = inputs["input_ids"].shape[1]

            # Generate
            with torch.no_grad():
                outputs = model_obj.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode
            response_text = tokenizer.decode(
                outputs[0][input_token_count:],
                skip_special_tokens=True
            ).strip()
            
            output_token_count = outputs[0].shape[0] - input_token_count
            
            logger.debug(
                f"Inference successful - "
                f"Input: {input_token_count} tokens, "
                f"Output: {output_token_count} tokens"
            )
            
            return response_text, input_token_count, output_token_count
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(
                    f"Out of memory on attempt {attempt + 1}/{max_retries}. "
                    f"Try reducing max_tokens or using a smaller model."
                )
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
            else:
                raise
        
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Inference error on attempt {attempt + 1}/{max_retries}: {e}. "
                    f"Retrying..."
                )
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Inference failed after {max_retries} attempts: {e}")
                raise
    
    raise RuntimeError("Should not reach here")


def clear_model_cache():
    """
    Clear cached models and tokenizers from memory.
    Useful for freeing GPU memory between experiments.
    """
    global _MODEL_CACHE, _TOKENIZER_CACHE
    
    logger.info(f"Clearing model cache ({len(_MODEL_CACHE)} models)")
    
    # Move models to CPU to free GPU memory
    for model in _MODEL_CACHE.values():
        if hasattr(model, 'to'):
            model.to('cpu')
    
    _MODEL_CACHE.clear()
    _TOKENIZER_CACHE.clear()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("✓ Model cache cleared")


# ==========================================================
# For Testing/Direct Execution
# ==========================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) < 2:
        print("Usage: python hf_inference.py <model_id> [temperature]")
        print("Example: python hf_inference.py mistralai/Mistral-7B-v0.3 0.7")
        sys.exit(1)
    
    model_id = sys.argv[1]
    temperature = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
    
    # Test inference
    print(f"\n{'='*60}")
    print(f"Testing HF Inference")
    print(f"{'='*60}")
    print(f"Model: {model_id}")
    print(f"Temperature: {temperature}")
    print(f"Device: {_detect_device()}")
    print(f"{'='*60}\n")
    
    try:
        response, input_tokens, output_tokens = call_hf_with_retry(
            system_prompt="You are a helpful assistant.",
            user_prompt="What is 2+2?",
            model=model_id,
            temperature=temperature,
            max_tokens=100
        )
        
        print(f"Response: {response}")
        print(f"\nTokens - Input: {input_tokens}, Output: {output_tokens}")
        print(f"✓ Test successful!")
    
    except Exception as e:
        print(f"✗ Test failed: {e}")
        sys.exit(1)
