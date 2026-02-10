
import os
import json
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torchaudio
from datasets import Dataset, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def load_manifest(manifest_path: str):
    data = []
    base_dir = os.path.dirname(manifest_path)
    
    # Check if manifest_path is a directory or file
    if os.path.isdir(manifest_path):
        # If directory, look for manifest.jsonl inside
        potential_path = os.path.join(manifest_path, "manifest.jsonl")
        if os.path.exists(potential_path):
            manifest_path = potential_path
            base_dir = os.path.dirname(manifest_path)
        else:
             raise ValueError(f"Directory {manifest_path} does not contain manifest.jsonl")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # Resolve relative audio path
            audio_path = os.path.join(base_dir, item['wav'])
            if not os.path.exists(audio_path):
                 print(f"Warning: Audio file not found: {audio_path}")
                 continue
            
            data.append({
                "audio": audio_path,
                "sentence": item['text']
            })
    
    return Dataset.from_list(data)

def train_lora(
    manifest_path: str,
    output_dir: str = "./lora_output",
    base_model: str = "openai/whisper-large-v3-turbo",
    language: str = "ko",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 1,
    learning_rate: float = 1e-4,
    lora_r: int = 32,
    lora_dropout: float = 0.05,
    progress_callback = None
):
    print(f"Starting training with base model: {base_model}")
    print(f"Language: {language}")
    print(f"Manifest: {manifest_path}")
    print(f"LoRA Config: r={lora_r}, dropout={lora_dropout}, epochs={num_train_epochs}")

    # 1. Load Processor
    processor = WhisperProcessor.from_pretrained(base_model, language=language, task="transcribe")
    
    # 2. Prepare Dataset - Use torchaudio directly instead of datasets Audio (avoids torchcodec requirement)
    dataset = load_manifest(manifest_path)

    def prepare_dataset(batch):
        # Load audio using torchaudio
        waveform, sample_rate = torchaudio.load(batch["audio"])
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Squeeze and convert to numpy for feature extractor
        audio_array = waveform.squeeze().numpy()
        
        batch["input_features"] = processor.feature_extractor(audio_array, sampling_rate=16000).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    print("Processing dataset...")
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names, num_proc=1) # num_proc=1 for safety on windows

    # 3. Load Model
    device_map = "auto"
    # 8-bit loading for memory efficiency if bitsandbytes is available, else float32/16
    try:
        model = WhisperForConditionalGeneration.from_pretrained(
            base_model,
            load_in_8bit=True,
            device_map=device_map,
        )
        model = prepare_model_for_kbit_training(model)
    except Exception as e:
        print(f"Could not load in 8-bit, falling back to full precision: {e}")
        model = WhisperForConditionalGeneration.from_pretrained(base_model, device_map=device_map)

    # 4. LoRA Config
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM # Whisper is Seq2Seq
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # Patch the BASE model's forward to ONLY pass allowed Whisper arguments
    # Whisper's forward signature: input_features, attention_mask, decoder_input_ids, labels, etc.
    ALLOWED_WHISPER_ARGS = {
        'input_features', 'attention_mask', 'decoder_input_ids', 
        'decoder_attention_mask', 'head_mask', 'decoder_head_mask',
        'cross_attn_head_mask', 'encoder_outputs', 'past_key_values',
        'decoder_inputs_embeds', 'labels', 'use_cache', 'output_attentions',
        'output_hidden_states', 'return_dict'
    }
    
    base_model = model.get_base_model()
    original_base_forward = base_model.forward
    def patched_base_forward(*args, **kwargs):
        # Filter kwargs to only include allowed arguments
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in ALLOWED_WHISPER_ARGS}
        return original_base_forward(*args, **filtered_kwargs)
    base_model.forward = patched_base_forward
    
    # Also patch __call__ on the PEFT model itself
    import types
    original_call = model.__class__.__call__
    def patched_call(self, *args, **kwargs):
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in ALLOWED_WHISPER_ARGS}
        return original_call(self, *args, **filtered_kwargs)
    model.__call__ = types.MethodType(patched_call, model)

    # 5. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        warmup_steps=50,
        num_train_epochs=num_train_epochs,
        eval_strategy="no",  # Changed from evaluation_strategy (deprecated)
        fp16=True,
        per_device_eval_batch_size=1,
        generation_max_length=128,
        logging_steps=10,  # More frequent logging
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=["none"]  # Disable wandb/tensorboard for this simple tool
    )

    # Custom callback for progress logging
    class ProgressCallback(TrainerCallback):
        def __init__(self, callback_fn):
            self.callback_fn = callback_fn

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and self.callback_fn:
                step = state.global_step
                total = state.max_steps
                loss = logs.get('loss', 'N/A')
                self.callback_fn(f"Step {step}/{total} | Loss: {loss}")

        def on_epoch_end(self, args, state, control, **kwargs):
            if self.callback_fn:
                self.callback_fn(f"Epoch {int(state.epoch)} completed.")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Custom Trainer for Whisper (handles input_features instead of input_ids)
    class WhisperTrainer(Seq2SeqTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Whisper uses input_features, not input_ids
            labels = inputs.pop("labels")
            outputs = model(input_features=inputs["input_features"], labels=labels)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

    # 6. Trainer with optional progress callback
    callbacks = []
    if progress_callback:
        callbacks.append(ProgressCallback(progress_callback))

    trainer = WhisperTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        callbacks=callbacks,
    )

    print("Starting training...")
    trainer.train()
    
    print("Saving adapter used...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print("Training complete.")
    return output_dir
