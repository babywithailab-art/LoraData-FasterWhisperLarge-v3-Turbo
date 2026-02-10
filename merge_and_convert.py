
import os
import argparse
import shutil
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel, PeftConfig
import ctranslate2

def merge_lora_and_convert(
    base_model_name_or_path: str,
    lora_model_path: str,
    output_dir: str,
    quantization: str = "float16"
):
    """
    Merges LoRA adapter into base Whisper model and converts to CTranslate2 format.
    """
    
    print(f"Loading base model: {base_model_name_or_path}")
    print(f"Loading LoRA adapter: {lora_model_path}")

    # 1. Load Base Model & Processor
    # Use CPU for merging to avoid OOM on smaller GPUs, unless CUDA is specifically desired for speed.
    # For merging, CPU is generally safer and sufficient.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        base_model = WhisperForConditionalGeneration.from_pretrained(
            base_model_name_or_path,
            load_in_8bit=False,
            device_map=device
        )
        processor = WhisperProcessor.from_pretrained(base_model_name_or_path)
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    # 2. Load Peft Model (LoRA)
    try:
        model = PeftModel.from_pretrained(base_model, lora_model_path)
    except Exception as e:
        print(f"Error loading LoRA adapter: {e}")
        return

    # 3. Merge LoRA weights
    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    model.eval()

    # 4. Save Merged HF Model (Temporary)
    temp_hf_dir = os.path.join(output_dir, "temp_hf_merged")
    if os.path.exists(temp_hf_dir):
        shutil.rmtree(temp_hf_dir)
    os.makedirs(temp_hf_dir)

    print(f"Saving merged HF model to temporary directory: {temp_hf_dir}")
    model.save_pretrained(temp_hf_dir)
    processor.save_pretrained(temp_hf_dir)

    # 5. Convert to CTranslate2
    print(f"Converting to CTranslate2 format (Quantization: {quantization})...")
    
    # Ensure output directory exists and is empty or handled
    ct2_output_dir = os.path.join(output_dir, "faster_whisper_model")
    if os.path.exists(ct2_output_dir):
        print(f"Warning: Output directory {ct2_output_dir} already exists. It will be overwritten.")
        shutil.rmtree(ct2_output_dir)

    converter = ctranslate2.converters.TransformersConverter(
        model_name_or_path=temp_hf_dir,
        copy_files=["tokenizer.json", "preprocessor_config.json"]
    )
    
    converter.convert(
        output_dir=ct2_output_dir,
        quantization=quantization,
        force=True
    )

    print(f"Conversion complete!")
    print(f"Faster Whisper model saved to: {ct2_output_dir}")
    
    # Cleanup temporary HF model
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_hf_dir)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA and Convert to Faster-Whisper")
    parser.add_argument("--base_model", type=str, default="openai/whisper-large-v3-turbo", help="Base Whisper model (e.g., openai/whisper-large-v3-turbo)")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to your LoRA adapter directory")
    parser.add_argument("--output_dir", type=str, default="./output_model", help="Directory immediately containing the output")
    parser.add_argument("--quantization", type=str, default="float16", help="Quantization type: float16, int8_float16, int8")

    args = parser.parse_args()

    merge_lora_and_convert(
        base_model_name_or_path=args.base_model,
        lora_model_path=args.lora_path,
        output_dir=args.output_dir,
        quantization=args.quantization
    )
