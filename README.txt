== ThinkSub2_LoRA_Tool ==


👇🏻Youtube Link
[![Watch the video](https://img.youtube.com/vi/FoOLdqJKcwU/maxresdefault.jpg)](https://youtu.be/FoOLdqJKcwU)


🎙 Personal Whisper Model Builder
(LoRA → large-v3-turbo 병합 → faster-whisper 모델 자동 생성)
📢 읽어주세요

이 프로젝트는 특정 인물(개인 화자)의 음성을 더 정확하게 인식하기 위해
LoRA 학습 데이터를 기반으로 Whisper large-v3-turbo 모델을 자동 튜닝하고,
faster-whisper에서 바로 사용할 수 있는 모델로 생성해주는 도구입니다.

사용자가 누적한 음성 데이터(wav + text)를 이용해:

LoRA 기반 파인튜닝 수행

large-v3-turbo 베이스 모델과 병합

faster-whisper(CTranslate2) 포맷으로 변환

까지의 과정을 자동으로 처리하여
“개인 맞춤 STT 모델”을 쉽게 만들 수 있도록 돕습니다.

✨ 주요 기능

✅ 특정 화자(개인) 음성 인식률 향상
✅ LoRA 학습 → 모델 병합 → CT2 변환 자동 처리
✅ manifest.jsonl 또는 데이터 폴더 선택만으로 실행
✅ faster-whisper에서 바로 사용 가능한 모델 출력
✅ 데이터가 쌓일수록 개인 맞춤 성능 향상

🧩 사용 흐름
음성 + 자막 데이터 누적
        ↓
LoRA 학습 수행
        ↓
large-v3-turbo 모델 병합
        ↓
CT2 모델 변환
        ↓
faster-whisper에서 사용


결과적으로,
내 목소리 / 내 발음 / 내 환경에 맞춘 음성 인식 모델을 만들 수 있습니다.

⚙ LoRA 학습 설정 가이드

LoRA 파인튜닝 시 아래 값들을 조정하여 모델 변경 강도와 학습 안정성을 조절할 수 있습니다.

🔧 LoRA Rank (r)

LoRA가 원본 모델 가중치를 얼마나 크게 수정할지를 결정하는 값입니다.

값	설명
r = 8	모델 변경이 작아 안정적이지만 적응 속도가 느립니다.
r = 16	일반적으로 많이 사용하는 균형 잡힌 설정입니다.
r = 32	모델을 강하게 변경하여 특정 화자에 빠르게 적응하지만 과적합 위험이 커질 수 있습니다.

✅ 개인 맞춤 STT에는 보통 16~32 범위를 추천합니다.

🌧 LoRA Dropout (lora_dropout)

학습 시 일부 연결을 랜덤하게 제외하여 과적합을 방지하는 역할을 합니다.

값	설명
0.05	기본값, 데이터가 충분할 때 안정적입니다.
0.1	데이터가 적을 때 과적합을 줄이는 데 도움이 됩니다.

데이터가 적은 초기 학습 단계에서는 0.1이 더 안정적일 수 있습니다.

🔁 학습 반복 횟수 (epochs)

모델이 전체 데이터를 몇 번 반복해서 학습할지를 의미합니다.

데이터 양에 따라 적절한 값을 사용하는 것이 중요합니다.

데이터 크기	추천 epochs
≤ 300 샘플	6 ~ 10
300 ~ 1000	4 ~ 6
≥ 1000	2 ~ 4

데이터가 늘어날수록 epochs를 줄이는 방식이 안정적이며,
너무 많이 반복 학습하면 특정 문장을 외워버려 일반화 성능이 떨어질 수 있습니다.

🎯 이 프로젝트가 필요한 경우

개인 방송/영상 편집에서 자막 정확도를 높이고 싶은 경우

반복적으로 같은 화자의 음성을 인식하는 작업

특정 발음/말투를 더 잘 인식시키고 싶은 경우

Whisper 기반 STT 정확도를 개인 환경에 맞게 개선하고 싶은 경우

🚀 목표

이 프로젝트의 목표는
Whisper 기반 음성 인식을 누구나 쉽게 개인화하고,
자신의 작업 환경에 맞게 발전시킬 수 있도록 만드는 것입니다.



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
