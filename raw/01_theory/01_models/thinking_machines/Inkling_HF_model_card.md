---
license: apache-2.0 
license_link: https://www.apache.org/licenses/LICENSE-2.0
pipeline_tag: image-text-to-text
tags:
- conversational
- image-text-to-text
- audio-text-to-text
- moe
---

# Inkling

<img src="https://cdn-uploads.huggingface.co/production/uploads/630e8f0bf6f6d700f50ebd2e/AvmDwmrWRMnKjOWvmLieg.png" style="display: block;margin-left: auto;margin-right: auto;width: 30%;">

<p align="center">
  <a href="https://huggingface.co/thinkingmachines/Inkling">BF16</a> |
  <a href="https://huggingface.co/thinkingmachines/Inkling-NVFP4">NVFP4</a> |
  <a href="https://github.com/thinking-machines-lab/tinker-cookbook">Tinker Cookbook</a> |
  <a href="https://tinker-docs.thinkingmachines.ai/cookbook/inkling/">Documentation</a> |
  <a href="https://thinkingmachines.ai/model-acceptable-use-policy">Acceptable Use</a>
</p>

## 1. General Information

Inkling is a general-purpose multimodal model that accepts text, image and audio inputs and generates text outputs. It is intended for use in English and other languages, and across multiple coding languages. The model is designed to be used by developers building AI-powered applications, including agentic and tool-use systems, coding assistants, chatbots, and retrieval-augmented generation systems, and is suitable for general-purpose conversational use, instruction-following, and other natural language and multimodal tasks. It is released with open weights to support research, fine-tuning and integration into third-party products by downstream developers.

**Languages:** English, with general multilingual capabilities across other languages.

## 2. Getting Started

For accessing Inkling via Tinker: You can get started by referring to the Tinker Cookbook [here](https://github.com/thinking-machines-lab/tinker-cookbook) and associated documentation [here](https://tinker-docs.thinkingmachines.ai/cookbook/inkling/).

Inkling supports local deployment using the following open-source libraries:

* SGLang ([recipe](https://docs.sglang.io/cookbook/autoregressive/ThinkingMachines/Inkling), [PR](https://github.com/sgl-project/sglang/pull/31358))
* vLLM ([recipe](https://recipes.vllm.ai/thinkingmachines/inkling), [PR](https://github.com/vllm-project/vllm/pull/48768))
* TokenSpeed ([recipe](https://lightseek.org/tokenspeed/recipes/models#Inkling), [PR](https://github.com/lightseekorg/tokenspeed/pull/689))
* Unsloth ([recipe](https://unsloth.ai/docs/models/inkling), [PR](https://github.com/ggml-org/llama.cpp/pull/25731))
* Huggingface ([recipe](https://hf.co/blog/thinkingmachines-inkling), [PR](https://github.com/huggingface/transformers/pull/47347))

API access is also available through third party inference providers.

## 3. Model Properties

### Model type

Multimodal autoregressive transformer

### Architecture type

A 66-layer decoder-only transformer with a sparse Mixture-of-Experts (MoE) feed-forward backbone: each token is routed to 6 of 256 experts, plus 2 shared experts active on every token. Attention is a hybrid of local and global layers. The model is natively multimodal — images and video are encoded via a hierarchical patch encoder, and audio via discrete token encoding — with all modalities projected into a shared hidden space and processed jointly by the decoder.

### Parameters

975B total, 41B active

### Numerics support

BF16 and NVFP4

### Input modalities

Inkling accepts text input in UTF-8 encoding, image input in any pixel-based format (with each dimension ideally between 40px and 4096px for optimal performance), and audio input in WAV format sampled at 16kHz (ideally under 20 minutes in length for optimal performance).

### Output modalities

Inkling generates output as UTF-8 encoded text.

## 4. Training

Training data includes a broad variety of content types, including text, images, audio, video. Training data for the model was drawn from publicly available sources, acquired from third-parties, or synthetically generated or augmented. Publicly available data includes content from the public internet and publicly accessible repositories.

The training data curation process includes cleaning, processing, and modifying datasets. These processing steps, which vary by data type, may include deduplication and filtering to remove junk or other low-quality data, or to advance safety or other objectives.

## 5. Evaluations

Inkling results are reported at effort=0.99. Comparison scores are generated Jul 14, 2026. Nemotron 3 Ultra, Kimi K2.5, Kimi K2.6, GLM 5.2, and DeepSeek V4 Pro are open weights models; Gemini 3.1 Pro, Claude Fable 5, and GPT 5.6 Sol are closed weights models.

|     |     | Inkling | Nemotron 3 Ultra | Kimi K2.5 | Kimi K2.6 | GLM 5.2 | DeepSeek V4 Pro | Gemini 3.1 Pro (high) | Claude Fable 5 (max) | GPT 5.6 Sol (xhigh) |
|-----|-----|---------|------------------|-----------|-----------|---------|-----------------|-----------------------|----------------------|---------------------|
| **Reasoning** |     |         |                  |           |           |         |                 |                       |                      |                     |
|     | HLE (text only) | 29.7%   | 26.6%            | 29.4%     | 35.9%     | 40.1%   | 35.9%           | 44.7%                 | 53.3%                | 47.2%               |
|     | HLE (with tools) | 46.0%   | 37.4%            | 50.2%     | 54.0%     | 54.7%   | 48.2%           | 51.4%                 | 64.5%                | 55.0%               |
|     | AIME 2026 | 97.1%   | 94.2%            | 95.8%     | 96.4%     | 99.2%   | 96.7%           | 98.3%                 | –                    | 99.9%               |
|     | GPQA Diamond | 87.2%   | 86.7%            | 87.9%     | 91.1%     | 89.5%   | 88.8%           | 94.1%                 | 92.6%                | 94.1%               |
| **Agentic (coding)** |     |         |                  |           |           |         |                 |                       |                      |                     |
|     | SWEBench Verified | 77.6%   | 70.7%            | 76.8%     | 80.2%     | –       | 80.6%           | 80.6%                 | 95.0%                | –                   |
|     | SWEBench Pro (Public) | 54.3%   | 46.4%            | 50.7%     | 58.6%     | 62.1%   | 55.4%           | 54.2%                 | 80.0%                | 64.6%               |
|     | Terminal Bench 2.1 (Best Harness) | 63.8    | 56.4             | 51.3         | 71.3      | 82.7    | 64              | 73.8                  | 84.6                 | 89.5                |
|     | GDPVal-AA v2 | 1233    | 1164             | 1009         | 1190      | 1514    | 1307            | 962                   | 1760                 | 1748                |
| **Agentic (general)** |     |         |                  |           |           |         |                 |                       |                      |                     |
|     | MCP Atlas | 74.1%   | 44.7%                | 64.0%     | 68.1%     | 77.8%   | 73.2%           | 78.2%                 | 83.3%                | 81.8%               |
|     | Tau 3 Banking | 23.7%   | 13.8%            | 13.2%     | 20.6%     | 26.8%   | 25.8%           | 16.5%                 | 26.8%                | 33.0%               |
| **Factuality** |     |         |                  |           |           |         |                 |                       |                      |                     |
|     | BrowseComp (w/ Ctx) | 77.1%   | –                | 74.9%     | 83.2%     | –       | 83.4%           | 85.9%                 | 88.0%                | 89.4%               |
|     | SimpleQA Verified | 43.9%   | 32.4%            | 36.9%     | 38.7%     | 38.1%   | 57.0%           | 77.3%                 | 68.3%                | 71.6%               |
|     | AA Omniscience | 1.0%    | -1.0%            | -8.0%     | 6.0%      | 4.0%    | -10.0%          | 33.0%                 | 40.0%                | 22.0%               |
| **Chat** |     |         |                  |           |           |         |                 |                       |                      |                     |
|     | IFBench | 79.8%   | 81.4%            | 70.2%     | 76.0%     | 73.3%   | 76.5%           | 77.1%                 | 63.5%                | 72.7%               |
|     | Global-MMLU-Lite | 88.7%   | 85.6%            | 84.0%     | 88.4%     | 89.2%   | 89.3%           | 92.7%                 | 93.3%                | 91.8%               |
| **Vision** |     |         |                  |           |           |         |                 |                       |                      |                     |
|     | MMMU Pro (Standard 10) | 73.3%   | –                | 75.0%     | 79.0%     | –       | –               | 82.0%                 | 84.2%                | 83.0%               |
|     | Charxiv RQ | 78.1%   | –                | 77.5%     | 80.4%     | –       | –               | 80.2%                 | 86.5%                | 84.7%               |
|     | Charxiv RQ (with python) | 82.0%   | –                | 78.7%     | 86.7%     | –       | –               | 89.9%                 | 89.4%                | 87.8%               |
| **Audio** |     |         |                  |           |           |         |                 |                       |                      |                     |
|     | Audio MC | 56.6%   | –                | –         | –         | –       | –               | 66.8%                 | –                    | –                   |
|     | MMAU | 77.2%   | –                | –         | –         | –       | –               | 82.5%                 | –                    | –                   |
|     | VoiceBench | 91.4%   | –                | –         | –         | –       | –               | 94.3%                 | –                    | –                   |
| **Safety** |     |         |                  |           |           |         |                 |                       |                      |                     |
|     | FORTRESS (Adversarial) | 78.0%   | 77.6%            | 54.1%     | 65.6%     | 71.3%   | 36.0%           | 65.2%                 | 96.0%                | 82.4%               |
|     | FORTRESS (Benign) | 95.9%   | 90.5%            | 98.3%     | 97.2%     | 90.0%   | 98.5%           | 98.0%                 | 55.1%                | 98.1%               |
|     | StrongREJECT | 98.6%   | 98.7%            | 99.5%     | 99.8%     | 98.5%   | 98.6%           | 98.0%                 | 98.7%                | 98.5%               |


## 6. Safety

We conducted safety evaluations ahead of release, spanning both everyday human-AI interaction and dangerous-capability testing. Because Inkling is multimodal, we paid attention to whether safety behavior held consistently across text, audio, and image inputs. We applied mitigations to reduce risks before release.

For everyday interaction, we evaluated sycophancy, harmful manipulation, and psychological-harm patterns like parasocial dependency and validation of delusional reasoning, including through multi-turn, open-ended external red-teaming designed to surface issues that only emerge over longer conversations. We also assessed whether the model refuses genuinely harmful requests without over-refusing benign ones. For CBRN and cyber, we assessed knowledge and procedural uplift through internal evaluations, external testing, and refusal-suppressed variants intended to estimate latent capability with safeguards removed. For loss of control, we evaluated agentic capability, strategic deception, and sabotage potential, benchmarked against public frontier models, and found the model materially below frontier capabilities.

Across all areas, we concluded that Inkling did not present risk of material uplift beyond what's already available in the open-weight ecosystem.

The residual risks identified in our evaluations — specifically, Inkling's occasional tendency to comply with role-play and indirectly framed prompts concerning harmful topics — are consistent with what you would see from any open-weight model, and are best addressed with defense-in-depth rather than relying on the model's refusals alone. Common downstream moderation tools, such as Llama Guard, are compatible with Inkling and can be layered around the model to catch jailbreak attempts, filter unsafe outputs, and enforce use-case-specific policies. We would encourage treating this kind of input/output classification as a part of your deployment stack, especially for consumer-facing or high-traffic applications where adversarial prompting is more likely.

## 7. Bias, risks and limitations

Inkling may exhibit general limitations common to foundation models, including hallucination (generating plausible but factually incorrect or unsupported content), occasional failures to follow instructions precisely, and degraded performance in long multi-turn conversations. As with other large-scale models trained on web-derived and synthetic data, Inkling may reflect biases present in its training data, including demographic, cultural, or linguistic biases, and may perform unevenly across languages, dialects, or subject domains that were less represented during training.

Inkling's knowledge is limited to information available as of its training cutoff, and it may not reflect events, developments, or changes that occurred afterward.

We recommend that downstream developers and deployers apply appropriate human oversight and review for outputs used in high-stakes or safety-critical contexts, rather than relying on Inkling's outputs without verification.

* Conduct their own evaluation of Inkling's performance, safety, and fairness for their specific use case, language, and population prior to deployment, particularly for applications involving vulnerable groups.
* Implement additional safeguards — such as content filtering, rate limiting, and monitoring — at the application layer, especially for open deployment contexts where Inkling's built-in mitigations may not be sufficient on their own.
* Avoid deploying Inkling in domains such as medical, legal, or safety-critical decision-making without additional fine-tuning, domain-specific validation, and human oversight.