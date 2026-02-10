== ThinkSub2_LoRA_Tool ==
🎙 Personal Whisper Model Builder

(LoRA → large-v3-turbo merge → faster-whisper model auto generation)

📢 Please Read

This project helps improve speech recognition accuracy for a specific person (personal speaker) by automatically tuning a Whisper large-v3-turbo model using LoRA training data and generating a model that can be used directly with faster-whisper.

Using accumulated speech data (wav + text), the tool automatically performs:

LoRA-based fine-tuning

Merging with the large-v3-turbo base model

Conversion to faster-whisper (CTranslate2) format

This pipeline makes it easy to build a personalized STT model tailored to a specific speaker.

✨ Main Features

✅ Improves speech recognition accuracy for a specific speaker
✅ Automatic LoRA training → model merging → CT2 conversion
✅ Run by simply selecting a manifest.jsonl file or dataset folder
✅ Outputs models directly usable in faster-whisper
✅ Recognition improves as more personal data is accumulated

🧩 Workflow
Accumulate speech + subtitle data
                ↓
          LoRA training
                ↓
 Merge with large-v3-turbo model
                ↓
        Convert to CT2 model
                ↓
      Use in faster-whisper


As a result, you can build a speech recognition model optimized for:

Your voice

Your pronunciation

Your recording environment

⚙ LoRA Training Configuration Guide

When performing LoRA fine-tuning, you can adjust the following values to control how strongly the model adapts and how stable training remains.

🔧 LoRA Rank (r)

This value determines how strongly LoRA modifies the original model weights.

Value	Description
r = 8	Small model change; stable but adapts slowly
r = 16	Balanced setting commonly used
r = 32	Strong adaptation for a specific speaker, but higher risk of overfitting

✅ For personal STT models, values between 16 and 32 are generally recommended.

🌧 LoRA Dropout (lora_dropout)

Randomly drops connections during training to reduce overfitting.

Value	Description
0.05	Default, stable when enough data is available
0.1	Helps prevent overfitting when data is limited

When starting with a small dataset, 0.1 is often safer.

🔁 Training Epochs

Defines how many times the entire dataset is repeated during training.

Choosing epochs based on dataset size is important.

Dataset Size	Recommended Epochs
≤ 300 samples	6 ~ 10
300 ~ 1000	4 ~ 6
≥ 1000	2 ~ 4

As the dataset grows, reducing epochs helps prevent overfitting.
Too many epochs can cause the model to memorize sentences rather than generalize.

🎯 When This Project Is Useful

Improving subtitle accuracy for personal streaming or video editing

Repeated recognition of the same speaker

Enhancing recognition of specific pronunciations or speech patterns

Personalizing Whisper-based STT for a specific recording environment

🚀 Goal

The goal of this project is to make Whisper-based speech recognition easily personalizable, allowing users to continuously improve recognition performance tailored to their own environment and workflow.