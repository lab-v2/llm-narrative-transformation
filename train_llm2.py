# train_mistral_collectivistic.py (Force CPU version)
import torch
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer
from datasets import Dataset
from peft import LoraConfig
import os
import platform
from huggingface_hub import login

# Force CPU usage
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False
os.environ['CUDA_VISIBLE_VICES'] = ''


def authenticate_huggingface():
    """Handle Hugging Face authentication"""
    token = os.getenv('HUGGINGFACE_HUB_TOKEN')

    if token:
        print("🔑 Using HF token from environment variable...")
        login(token=token)
        return token

    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            print("🔑 Using cached HF token...")
            return token
    except:
        pass

    print("🔑 Hugging Face authentication required...")
    token = input("Please enter your HF token (or press Enter to skip): ").strip()

    if token:
        login(token=token)
        print("✅ Authenticated successfully!")
        return token
    else:
        print("⚠️ No token provided. Some models might not be accessible.")
        return None


def load_and_format_dataset(jsonl_path="data/collectivistic_dataset_llm2.jsonl"):
    """Load and format dataset for training"""
    print(f"📊 Loading dataset from {jsonl_path}...")

    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

    # Load JSONL
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"⚠️ Error parsing line {line_num}: {e}")
                continue

    print(f"📚 Loaded {len(data)} stories")

    # Format for training - keep stories shorter
    formatted_data = []
    for example in data:
        prompt = example['prompt']
        # Truncate stories to 500 characters for memory
        response = example['story_text'][:500] + "..." if len(example['story_text']) > 500 else example['story_text']
        text = f"User: {prompt}\nAssistant: {response}"
        formatted_data.append({"text": text})

    return formatted_data


def train_mistral_collectivistic():
    # Configuration
    # MODEL_NAME = "mistralai/Mistral-7B-v0.3"
    # OUTPUT_DIR = "./mistral-collectivistic-finetuned"
    MODEL_NAME = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
    OUTPUT_DIR = "./llam4-finetuned-llm2"
    DATASET_PATH = "data/fine_tuning_data/collectivistic_dataset_llm2.jsonl"

    print("🚀 Starting Mistral-7B-v0.3 fine-tuning (FORCED CPU mode)...")
    print(f"💻 System: {platform.system()} {platform.machine()}")
    print("🔧 MPS disabled, CUDA disabled - Pure CPU training")

    # Authenticate with Hugging Face
    hf_token = authenticate_huggingface()

    # Load tokenizer
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        token=hf_token
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model and explicitly move to CPU
    print("🧠 Loading base model and forcing CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        trust_remote_code=True,
        token=hf_token,
        low_cpu_mem_usage=True,
        device_map="balanced"
    )

    # Explicitly move to CPU
    # model = model.to('cpu')
    # print("✅ Model confirmed on CPU")

    # Minimal LoRA configuration
    print("🔧 Setting up minimal LoRA configuration...")
    peft_config = LoraConfig(
        lora_alpha=4,
        lora_dropout=0.05,
        r=4,  # Extremely small
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj"]  # Just one module
    )

    # Load and format dataset
    formatted_data = load_and_format_dataset(DATASET_PATH)
    train_dataset = Dataset.from_list(formatted_data)

    print(f"📈 Dataset size: {len(train_dataset)} examples")
    print("📖 Sample formatted training example:")
    print(f"{train_dataset[0]['text'][:150]}...")

    # Minimal training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        optim="adamw_torch",
        save_steps=20,
        logging_steps=5,
        learning_rate=1e-4,
        weight_decay=0.01,
        fp16=False,
        bf16=False,
        max_grad_norm=1.0,
        warmup_steps=1,
        lr_scheduler_type="constant",
        report_to="none",
        save_total_limit=2,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        dataloader_pin_memory=False,
        max_steps=-1,  # Just 10 steps to test
        use_cpu=True,
    )

    # Initialize SFTTrainer
    print("🏋️ Initializing CPU-only trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Start training
    print("🚂 Starting FORCED CPU training...")
    print("⏰ This will be slow but should work...")

    try:
        trainer.train()
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training still failed: {e}")
        print("💡 The 7B model might be too large even for CPU training on this system")
        return

    # Save the model
    print("💾 Saving fine-tuned model...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"🎉 Training completed! Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    if not os.path.exists("data/fine_tuning_data/collectivistic_dataset_llm2.jsonl"):
        print("❌ Dataset not found!")
        exit(1)

    print("🚫 Forcing CPU-only mode...")
    train_mistral_collectivistic()


