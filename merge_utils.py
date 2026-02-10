
import os
import shutil
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import ctranslate2

def merge_and_convert_model(
    base_model_path: str,
    lora_path: str,
    output_dir: str,
    quantization: str = "float16",
    progress_callback = None
):
    """
    Merges LoRA adapter into base Whisper model and converts to CTranslate2 format.
    """
    
    if progress_callback: progress_callback("Loading base model...")
    print(f"Loading base model: {base_model_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        base_model = WhisperForConditionalGeneration.from_pretrained(
            base_model_path,
            load_in_8bit=False,
            device_map=device
        )
        processor = WhisperProcessor.from_pretrained(base_model_path)
    except Exception as e:
        err_msg = f"Error loading base model: {e}"
        print(err_msg)
        if progress_callback: progress_callback(err_msg)
        return False

    if progress_callback: progress_callback("Loading LoRA adapter...")
    print(f"Loading LoRA adapter: {lora_path}")
    try:
        model = PeftModel.from_pretrained(base_model, lora_path)
    except Exception as e:
        err_msg = f"Error loading LoRA adapter: {e}"
        print(err_msg)
        if progress_callback: progress_callback(err_msg)
        return False

    if progress_callback: progress_callback("Merging weights...")
    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    model.eval()

    # Temporary HF save
    temp_hf_dir = os.path.join(output_dir, "temp_hf_merged")
    if os.path.exists(temp_hf_dir):
        shutil.rmtree(temp_hf_dir)
    os.makedirs(temp_hf_dir)

    if progress_callback: progress_callback("Saving intermediate HF model...")
    print(f"Saving merged HF model to: {temp_hf_dir}")
    model.save_pretrained(temp_hf_dir)
    processor.save_pretrained(temp_hf_dir)

    # Conversion
    if progress_callback: progress_callback(f"Converting to CTranslate2 ({quantization})...")
    print(f"Converting to CTranslate2 format (Quantization: {quantization})...")
    
    ct2_output_dir = os.path.join(output_dir, "ct2_model")
    if os.path.exists(ct2_output_dir):
        shutil.rmtree(ct2_output_dir)

    try:
        converter = ctranslate2.converters.TransformersConverter(
            model_name_or_path=temp_hf_dir,
            copy_files=["tokenizer.json", "preprocessor_config.json"]
        )
        
        converter.convert(
            output_dir=ct2_output_dir,
            quantization=quantization,
            force=True
        )
    except Exception as e:
        err_msg = f"Error during conversion: {e}"
        print(err_msg)
        if progress_callback: progress_callback(err_msg)
        return False
    finally:
        # Cleanup
        if os.path.exists(temp_hf_dir):
            shutil.rmtree(temp_hf_dir)

    success_msg = f"Conversion complete! Model saved to {ct2_output_dir}"
    print(success_msg)
    if progress_callback: progress_callback(success_msg)
    
    return True
