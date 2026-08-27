---
tags:
- compressed-tensors
license: other
license_name: modified-mit
library_name: transformers
pipeline_tag: image-text-to-text
---
<div align="center">
  <picture>
      <img src="figures/kimi-logo.png" width="30%" alt="Kimi K2.6">
  </picture>
</div>
<hr>
<div align="center" style="line-height:1">
  <a href="https://www.kimi.com" target="_blank"><img alt="Chat" src="https://img.shields.io/badge/🤖%20Chat-Kimi%20K2.6-ff6b6b?color=1783ff&logoColor=white"/></a>
  <a href="https://www.moonshot.ai" target="_blank"><img alt="Homepage" src="https://img.shields.io/badge/Homepage-Moonshot%20AI-white?logo=Kimi&logoColor=white"/></a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/moonshotai" target="_blank"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Moonshot%20AI-ffc107?color=ffc107&logoColor=white"/></a>
  <a href="https://twitter.com/kimi_moonshot" target="_blank"><img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-Kimi.ai-white?logo=x&logoColor=white"/></a>
  <a href="https://discord.gg/TYU2fdJykW" target="_blank"><img alt="Discord" src="https://img.shields.io/badge/Discord-Kimi.ai-white?logo=discord&logoColor=white"/></a>
  <a href="https://modelscope.cn/organization/moonshotai" target="_blank"><img alt="ModelScope" src="https://img.shields.io/badge/ModelScope-Moonshot%20AI-white?labelColor=rgb(99%2C%2074%2C%20255)"/></a>
</div>
<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/moonshotai/Kimi-K2.6/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Modified_MIT-f5de53?&color=f5de53"/></a>
</div>


<p align="center">
🤗&nbsp;&nbsp;<a href="https://huggingface.co/spaces/akhaliq/Kimi-K2.6" target="_blank">huggingchat</a>
&nbsp;|&nbsp;
📰&nbsp;&nbsp;<a href="https://www.kimi.com/blog/kimi-k2-6.html">Tech Blog</a>
</p>


## 1. Model Introduction

Kimi K2.6 is an open-source, native multimodal agentic model that advances practical capabilities in long-horizon coding, coding-driven design, proactive autonomous execution, and swarm-based task orchestration.

### Key Features
- **Long-Horizon Coding**: K2.6 achieves significant improvements on complex, end-to-end coding tasks, generalizing robustly across programming languages (Rust, Go, Python) and domains spanning front-end, DevOps, and performance optimization.
- **Coding-Driven Design**: K2.6 is capable of transforming simple prompts and visual inputs into production-ready interfaces and lightweight full-stack workflows, generating structured layouts, interactive elements, and rich animations with deliberate aesthetic precision.
- **Elevated Agent Swarm**: Scaling horizontally to 300 sub-agents executing 4,000 coordinated steps, K2.6 can dynamically decompose tasks into parallel, domain-specialized subtasks, delivering end-to-end outputs from documents to websites to spreadsheets in a single autonomous run.
- **Proactive & Open Orchestration**: For autonomous tasks, K2.6 demonstrates strong performance in powering persistent, 24/7 background agents that proactively manage schedules, execute code, and orchestrate cross-platform operations without human oversight.

## 2. Model Summary

<div align="center">


| | |
|:---:|:---:|
| **Architecture** | Mixture-of-Experts (MoE) |
| **Total Parameters** | 1T |
| **Activated Parameters** | 32B |
| **Number of Layers** (Dense layer included) | 61 |
| **Number of Dense Layers** | 1 |
| **Attention Hidden Dimension** | 7168 |
| **MoE Hidden Dimension** (per Expert) | 2048 |
| **Number of Attention Heads** | 64 |
| **Number of Experts** | 384 |
| **Selected Experts per Token** | 8 |
| **Number of Shared Experts** | 1 |
| **Vocabulary Size** | 160K |
| **Context Length** | 256K |
| **Attention Mechanism** | MLA |
| **Activation Function** | SwiGLU |
| **Vision Encoder** | MoonViT |
| **Parameters of Vision Encoder** | 400M |
</div>

## 3. Evaluation Results

<div align="center">
<table>
<thead>
<tr>
<th align="center">Benchmark</th>
<th align="center"><sup>Kimi K2.6</sup></th>
<th align="center"><sup>GPT-5.4 <br><sup>(xhigh)</sup></sup></th>
<th align="center"><sup>Claude Opus 4.6 <br><sup>(max effort)</sup></sup></th>
<th align="center"><sup>Gemini 3.1 Pro<br><sup>(thinking high)</sup></sup></th>
<th align="center"><sup>Kimi K2.5</sup></th>
</tr>
</thead>
<tbody>
<tr>
<td align="center" colspan=6><strong>Agentic</strong></td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">HLE-Full<br>(w/ tools)</td>
<td align="center" style="vertical-align: middle">54.0</td>
<td align="center" style="vertical-align: middle">52.1</td>
<td align="center" style="vertical-align: middle">53.0</td>
<td align="center" style="vertical-align: middle">51.4</td>
<td align="center" style="vertical-align: middle">50.2</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">BrowseComp</td>
<td align="center" style="vertical-align: middle">83.2</td>
<td align="center" style="vertical-align: middle" rowspan="2">82.7</td>
<td align="center" style="vertical-align: middle" rowspan="2">83.7</td>
<td align="center" style="vertical-align: middle" rowspan="2">85.9</td>
<td align="center" style="vertical-align: middle">74.9</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">BrowseComp<br>(Agent Swarm)</td>
<td align="center" style="vertical-align: middle">86.3</td>
<td align="center" style="vertical-align: middle">78.4</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">DeepSearchQA<br>(f1-score)</td>
<td align="center" style="vertical-align: middle">92.5</td>
<td align="center" style="vertical-align: middle">78.6</td>
<td align="center" style="vertical-align: middle">91.3</td>
<td align="center" style="vertical-align: middle">81.9</td>
<td align="center" style="vertical-align: middle">89.0</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">DeepSearchQA<br>(accuracy)</td>
<td align="center" style="vertical-align: middle">83.0</td>
<td align="center" style="vertical-align: middle">63.7</td>
<td align="center" style="vertical-align: middle">80.6</td>
<td align="center" style="vertical-align: middle">60.2</td>
<td align="center" style="vertical-align: middle">77.1</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">WideSearch<br> (item-f1)</td>
<td align="center" style="vertical-align: middle">80.8</td>
<td align="center" style="vertical-align: middle">-</td>
<td align="center" style="vertical-align: middle">-</td>
<td align="center" style="vertical-align: middle">-</td>
<td align="center" style="vertical-align: middle">72.7</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">Toolathlon</td>
<td align="center" style="vertical-align: middle">50.0</td>
<td align="center" style="vertical-align: middle">54.6</td>
<td align="center" style="vertical-align: middle">47.2</td>
<td align="center" style="vertical-align: middle">48.8</td>
<td align="center" style="vertical-align: middle">27.8</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">MCPMark</td>
<td align="center" style="vertical-align: middle">55.9</td>
<td align="center" style="vertical-align: middle">62.5*</td>
<td align="center" style="vertical-align: middle">56.7*</td>
<td align="center" style="vertical-align: middle">55.9*</td>
<td align="center" style="vertical-align: middle">29.5</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">Claw Eval (pass^3)</td>
<td align="center" style="vertical-align: middle">62.3</td>
<td align="center" style="vertical-align: middle">60.3</td>
<td align="center" style="vertical-align: middle">70.4</td>
<td align="center" style="vertical-align: middle">57.8</td>
<td align="center" style="vertical-align: middle">52.3</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">Claw Eval (pass@3)</td>
<td align="center" style="vertical-align: middle">80.9</td>
<td align="center" style="vertical-align: middle">78.4</td>
<td align="center" style="vertical-align: middle">82.4</td>
<td align="center" style="vertical-align: middle">82.9</td>
<td align="center" style="vertical-align: middle">75.4</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">APEX-Agents</td>
<td align="center" style="vertical-align: middle">27.9</td>
<td align="center" style="vertical-align: middle">33.3</td>
<td align="center" style="vertical-align: middle">33.0</td>
<td align="center" style="vertical-align: middle">32.0</td>
<td align="center" style="vertical-align: middle">11.5</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">OSWorld-Verified</td>
<td align="center" style="vertical-align: middle">73.1</td>
<td align="center" style="vertical-align: middle">75.0</td>
<td align="center" style="vertical-align: middle">72.7</td>
<td align="center" style="vertical-align: middle">-</td>
<td align="center" style="vertical-align: middle">63.3</td>
</tr>
<tr>
<td align="center" colspan=6><strong>Coding</strong></td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">Terminal-Bench 2.0<br>(Terminus-2)</td>
<td align="center" style="vertical-align: middle">66.7</td>
<td align="center" style="vertical-align: middle">65.4*</td>
<td align="center" style="vertical-align: middle">65.4</td>
<td align="center" style="vertical-align: middle">68.5</td>
<td align="center" style="vertical-align: middle">50.8</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">SWE-Bench Pro</td>
<td align="center" style="vertical-align: middle">58.6</td>
<td align="center" style="vertical-align: middle">57.7</td>
<td align="center" style="vertical-align: middle">53.4</td>
<td align="center" style="vertical-align: middle">54.2</td>
<td align="center" style="vertical-align: middle">50.7</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">SWE-Bench Multilingual</td>
<td align="center" style="vertical-align: middle">76.7</td>
<td align="center" style="vertical-align: middle">-</td>
<td align="center" style="vertical-align: middle">77.8</td>
<td align="center" style="vertical-align: middle">76.9*</td>
<td align="center" style="vertical-align: middle">73.0</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">SWE-Bench Verified</td>
<td align="center" style="vertical-align: middle">80.2</td>
<td align="center" style="vertical-align: middle">-</td>
<td align="center" style="vertical-align: middle">80.8</td>
<td align="center" style="vertical-align: middle">80.6</td>
<td align="center" style="vertical-align: middle">76.8</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">SciCode</td>
<td align="center" style="vertical-align: middle">52.2</td>
<td align="center" style="vertical-align: middle">56.6</td>
<td align="center" style="vertical-align: middle">51.9</td>
<td align="center" style="vertical-align: middle">58.9</td>
<td align="center" style="vertical-align: middle">48.7</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">OJBench (python)</td>
<td align="center" style="vertical-align: middle">60.6</td>
<td align="center" style="vertical-align: middle">-</td>
<td align="center" style="vertical-align: middle">60.3</td>
<td align="center" style="vertical-align: middle">70.7</td>
<td align="center" style="vertical-align: middle">54.7</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">LiveCodeBench (v6)</td>
<td align="center" style="vertical-align: middle">89.6</td>
<td align="center" style="vertical-align: middle">-</td>
<td align="center" style="vertical-align: middle">88.8</td>
<td align="center" style="vertical-align: middle">91.7</td>
<td align="center" style="vertical-align: middle">85.0</td>
</tr>
<tr>
<td align="center" colspan=6><strong>Reasoning &amp; Knowledge</strong></td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">HLE-Full</td>
<td align="center" style="vertical-align: middle">34.7</td>
<td align="center" style="vertical-align: middle">39.8</td>
<td align="center" style="vertical-align: middle">40.0</td>
<td align="center" style="vertical-align: middle">44.4</td>
<td align="center" style="vertical-align: middle">30.1</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">AIME 2026</td>
<td align="center" style="vertical-align: middle">96.4</td>
<td align="center" style="vertical-align: middle">99.2</td>
<td align="center" style="vertical-align: middle">96.7</td>
<td align="center" style="vertical-align: middle">98.3</td>
<td align="center" style="vertical-align: middle">95.8</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">HMMT 2026 (Feb)</td>
<td align="center" style="vertical-align: middle">92.7</td>
<td align="center" style="vertical-align: middle">97.7</td>
<td align="center" style="vertical-align: middle">96.2</td>
<td align="center" style="vertical-align: middle">94.7</td>
<td align="center" style="vertical-align: middle">87.1</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">IMO-AnswerBench</td>
<td align="center" style="vertical-align: middle">86.0</td>
<td align="center" style="vertical-align: middle">91.4</td>
<td align="center" style="vertical-align: middle">75.3</td>
<td align="center" style="vertical-align: middle">91.0*</td>
<td align="center" style="vertical-align: middle">81.8</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">GPQA-Diamond</td>
<td align="center" style="vertical-align: middle">90.5</td>
<td align="center" style="vertical-align: middle">92.8</td>
<td align="center" style="vertical-align: middle">91.3</td>
<td align="center" style="vertical-align: middle">94.3</td>
<td align="center" style="vertical-align: middle">87.6</td>
</tr>
<tr>
<td align="center" colspan=6><strong>Vision</strong></td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">MMMU-Pro</td>
<td align="center" style="vertical-align: middle">79.4</td>
<td align="center" style="vertical-align: middle">81.2</td>
<td align="center" style="vertical-align: middle">73.9</td>
<td align="center" style="vertical-align: middle">83.0*</td>
<td align="center" style="vertical-align: middle">78.5</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">MMMU-Pro (w/ python)</td>
<td align="center" style="vertical-align: middle">80.1</td>
<td align="center" style="vertical-align: middle">82.1</td>
<td align="center" style="vertical-align: middle">77.3</td>
<td align="center" style="vertical-align: middle">85.3*</td>
<td align="center" style="vertical-align: middle">77.7</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">CharXiv (RQ)</td>
<td align="center" style="vertical-align: middle">80.4</td>
<td align="center" style="vertical-align: middle">82.8*</td>
<td align="center" style="vertical-align: middle">69.1</td>
<td align="center" style="vertical-align: middle">80.2*</td>
<td align="center" style="vertical-align: middle">77.5</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">CharXiv (RQ) (w/ python)</td>
<td align="center" style="vertical-align: middle">86.7</td>
<td align="center" style="vertical-align: middle">90.0*</td>
<td align="center" style="vertical-align: middle">84.7</td>
<td align="center" style="vertical-align: middle">89.9*</td>
<td align="center" style="vertical-align: middle">78.7</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">MathVision</td>
<td align="center" style="vertical-align: middle">87.4</td>
<td align="center" style="vertical-align: middle">92.0*</td>
<td align="center" style="vertical-align: middle">71.2*</td>
<td align="center" style="vertical-align: middle">89.8*</td>
<td align="center" style="vertical-align: middle">84.2</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">MathVision (w/ python)</td>
<td align="center" style="vertical-align: middle">93.2</td>
<td align="center" style="vertical-align: middle">96.1*</td>
<td align="center" style="vertical-align: middle">84.6*</td>
<td align="center" style="vertical-align: middle">95.7*</td>
<td align="center" style="vertical-align: middle">85.0</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">BabyVision</td>
<td align="center" style="vertical-align: middle">39.8</td>
<td align="center" style="vertical-align: middle">49.7</td>
<td align="center" style="vertical-align: middle">14.8</td>
<td align="center" style="vertical-align: middle">51.6</td>
<td align="center" style="vertical-align: middle">36.5</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">BabyVision (w/ python)</td>
<td align="center" style="vertical-align: middle">68.5</td>
<td align="center" style="vertical-align: middle">80.2*</td>
<td align="center" style="vertical-align: middle">38.4*</td>
<td align="center" style="vertical-align: middle">68.3*</td>
<td align="center" style="vertical-align: middle">40.5</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">V* (w/ python)</td>
<td align="center" style="vertical-align: middle">96.9</td>
<td align="center" style="vertical-align: middle">98.4*</td>
<td align="center" style="vertical-align: middle">86.4*</td>
<td align="center" style="vertical-align: middle">96.9*</td>
<td align="center" style="vertical-align: middle">86.9</td>
</tr>
</tbody>
</table>
</div>

<details>
<summary><b>Footnotes</b></summary>

1. **General Testing Details**
   - We report results for Kimi K2.6 and Kimi K2.5 with thinking mode enabled, Claude Opus 4.6 with max effort, GPT-5.4 with xhigh reasoning effort, and Gemini 3.1 Pro with a high thinking level. 
   - Unless otherwise specified, all Kimi K2.6 experiments were conducted with temperature = 1.0, top-p = 1.0, and a context length of 262,144 tokens. 
   - Benchmarks without publicly available scores were re-evaluated under the same conditions used for Kimi K2.6 and are marked with an asterisk (`*`). Except where noted with an asterisk, all other results are cited from official reports.
2. **Reasoning Benchmarks**
   - IMO-AnswerBench scores for GPT-5.4 and Claude 4.6 were obtained from [z.ai/blog/glm-5.1](https://z.ai/blog/glm-5.1).
   - Humanity's Last Exam (HLE) and other reasoning tasks were evaluated with a maximum generation length of 98,304 tokens. By default, we report results on the HLE full set. For the text-only subset, Kimi K2.6 achieves 36.4% accuracy without tools and 55.5% with tools. 
3. **Tool-Augmented / Agentic Tasks**
   - Kimi K2.6 was equipped with search, code-interpreter, and web-browsing tools for HLE with tools, BrowseComp, DeepSearchQA, and WideSearch.
   - For HLE-Full with tools, the maximum generation length is 262,144 tokens with a per-step limit of 49,152 tokens. We employ a simple context management strategy: once the context window exceeds the threshold, only the most recent round of tool-related messages is retained.
   - For BrowseComp, we report scores obtained with context management using the same discard-all strategy as Kimi K2.5 and DeepSeek-V3.2.
   - For DeepSearchQA, no context management was applied to Kimi K2.6 tests, and tasks exceeding the supported context length were directly counted as failed. Scores for Claude Opus 4.6, GPT-5.4, and Gemini 3.1 Pro on DeepSearchQA are cited from the [Claude Opus 4.7 System Card](https://cdn.sanity.io/files/4zrzovbb/website/037f06850df7fbe871e206dad004c3db5fd50340.pdf).
   - For WideSearch, we report results under the "hide tool result" context management setting. Once the context window exceeds the threshold, only the most recent round of tool-related messages is retained.
   - The test system prompts are identical to those used in the [Kimi K2.5 technical report](https://arxiv.org/pdf/2602.02276).
   - Claw Eval was conducted using version 1.1  with max-tokens-per-step = 16384.
   - For APEX-Agents, we evaluate 452 tasks from the public 480-task release, as done by [Artificial Analysis](https://artificialanalysis.ai/evaluations/apex-agents-aa)(excluding Investment Banking Worlds 244 and 246, which have external runtime dependencies)
4. **Coding Tasks**
   - Terminal-Bench 2.0 scores were obtained with the default agent framework (Terminus-2) and the provided JSON parser, operating in preserve thinking mode. 
   - For the SWE-Bench series of evaluations (including Verified, Multilingual, and Pro), we used an in-house evaluation framework adapted from SWE-agent. This framework includes a minimal set of tools—bash tool, createfile tool, insert tool, view tool, strreplace tool, and submit tool.
   - All reported scores for coding tasks are averaged over 10 independent runs.
5. **Vision Benchmarks**
   - Max-tokens = 98,304, averaged over three runs (avg@3).
   - Settings with Python tool use max-tokens-per-step = 65,536 and max-steps = 50 for multi-step reasoning.
   - MMMU-Pro follows the official protocol, preserving input order and prepending images.

</details>


## 4. Native INT4 Quantization
Kimi-K2.6 adopts the same native int4 quantization method as [Kimi-K2-Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking#4-native-int4-quantization).

## 5. Deployment

> [!Note]
> You can access Kimi-K2.6's API on https://platform.moonshot.ai and we provide OpenAI/Anthropic-compatible API for you. To verify the deployment is correct, we also provide the  [Kimi Vendor Verifier](https://kimi.com/blog/kimi-vendor-verifier.html).
Currently, Kimi-K2.6 is recommended to run on the following inference engines:
* vLLM
* SGLang
* KTransformers

Kimi-K2.6 has the same architecture as Kimi-K2.5, and the deployment method can be directly reused.

The version requirement for `transformers` is `>=4.57.1, <5.0.0`.

Deployment examples can be found in the [Model Deployment Guide](docs/deploy_guidance.md).


---
## 6. Model Usage

The usage demos below demonstrate how to call our official API. 

For third-party APIs deployed with vLLM or SGLang, please note that:
> [!Note]
> - Chat with video content is an experimental feature and is only supported in our official API for now.
> 
> - The recommended `temperature` will be `1.0` for Thinking mode and `0.6` for Instant mode.
> 
> - The recommended `top_p` is `0.95`.
> 
> - To use instant mode, you need to pass `{'chat_template_kwargs': {"thinking": False}}` in `extra_body`.

### Chat Completion

This is a simple chat completion script which shows how to call K2.6 API in Thinking and Instant modes.

```python
import openai
import base64
import requests
def simple_chat(client: openai.OpenAI, model_name: str):
    messages = [
        {'role': 'system', 'content': 'You are Kimi, an AI assistant created by Moonshot AI.'},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'which one is bigger, 9.11 or 9.9? think carefully.'}
            ],
        },
    ]
    response = client.chat.completions.create(
        model=model_name, messages=messages, stream=False, max_tokens=4096
    )
    print('====== Below is reasoning content in Thinking Mode ======')
    print(f'reasoning content: {response.choices[0].message.reasoning}')
    print('====== Below is response in Thinking Mode ======')
    print(f'response: {response.choices[0].message.content}')

    # To use instant mode, pass {"thinking" = {"type":"disabled"}}
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=False,
        max_tokens=4096,
        extra_body={'thinking': {'type': 'disabled'}},  # this is for official API
        # extra_body= {'chat_template_kwargs': {"thinking": False}}  # this is for vLLM/SGLang
    )
    print('====== Below is response in Instant Mode ======')
    print(f'response: {response.choices[0].message.content}')
```


### Chat Completion with visual content

K2.6 supports Image and Video input.

The following example demonstrates how to call K2.6 API with image input:

```python
import openai
import base64
import requests

def chat_with_image(client: openai.OpenAI, model_name: str):
    url = 'https://huggingface.co/moonshotai/Kimi-K2.6/resolve/main/figures/kimi-logo.png'
    image_base64 = base64.b64encode(requests.get(url).content).decode()
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'Describe this image in detail.'},
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/png;base64, {image_base64}'},
                },
            ],
        }
    ]

    response = client.chat.completions.create(
        model=model_name, messages=messages, stream=False, max_tokens=8192
    )
    print('====== Below is reasoning content in Thinking Mode ======')
    print(f'reasoning content: {response.choices[0].message.reasoning}')
    print('====== Below is response in Thinking Mode ======')
    print(f'response: {response.choices[0].message.content}')

    # Also support instant mode if you pass {"thinking" = {"type":"disabled"}}
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=False,
        max_tokens=4096,
        extra_body={'thinking': {'type': 'disabled'}},  # this is for official API
        # extra_body= {'chat_template_kwargs': {"thinking": False}}  # this is for vLLM/SGLang
    )
    print('====== Below is response in Instant Mode ======')
    print(f'response: {response.choices[0].message.content}')

    return response.choices[0].message.content
```

The following example demonstrates how to call K2.6 API with video input:

```python
import openai
import base64
import requests

def chat_with_video(client: openai.OpenAI, model_name:str):
    url = 'https://huggingface.co/moonshotai/Kimi-K2.6/resolve/main/figures/demo_video.mp4'
    video_base64 = base64.b64encode(requests.get(url).content).decode()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text","text": "Describe the video in detail."},
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{video_base64}"},
                },
            ],
        }
    ]

    response = client.chat.completions.create(model=model_name, messages=messages)
    print('====== Below is reasoning content in Thinking Mode ======')
    print(f'reasoning content: {response.choices[0].message.reasoning}')
    print('====== Below is response in Thinking Mode ======')
    print(f'response: {response.choices[0].message.content}')

    # Also support instant mode if pass {"thinking" = {"type":"disabled"}}
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=False,
        max_tokens=4096,
        extra_body={'thinking': {'type': 'disabled'}},  # this is for official API
        # extra_body= {'chat_template_kwargs': {"thinking": False}}  # this is for vLLM/SGLang
    )
    print('====== Below is response in Instant Mode ======')
    print(f'response: {response.choices[0].message.content}')
    return response.choices[0].message.content
```

### Preserve Thinking
Kimi K2.6 supports `preserve_thinking` mode, which retains full reasoning content across multi-turn interactions and enhances performance in coding agent scenarios.

This feature is disabled by default. The following example demonstrates how to call K2.6 API in `preserve_thinking` mode:

```python
def chat_with_preserve_thinking(client: openai.OpenAI, model_name: str):
    messages = [
        {
            "role": "user",
            "content": "Tell me three random numbers."
        },
        {
            "role": "assistant",
            "reasoning_content": "I'll start by listing five numbers: 473, 921, 235, 215, 222, and I'll tell you the first three.",
            "content": "473, 921, 235"
        },
        {
            "role": "user",
            "content": "What are the other two numbers you have in mind?"
        }
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=False,
        max_tokens=4096,
        extra_body={'thinking': {'type': 'enabled', 'keep': 'all'}},  # this is for official API
        # extra_body={"chat_template_kwargs": {"thinking":True, "preserve_thinking": True}},  # this is for vLLM/SGLang
        # We recommend enabling preserve_thinking only in think mode.
    )
    # the assistant should mention 215 and 222 that appear in the prior reasoning content
    print(f"response: {response.choices[0].message.reasoning}")
    return response.choices[0].message.content

```

### Interleaved Thinking and Multi-Step Tool Call

K2.6 shares the same design of Interleaved Thinking and Multi-Step Tool Call as K2 Thinking. For usage example, please refer to the [K2 Thinking documentation](https://platform.moonshot.ai/docs/guide/use-kimi-k2-thinking-model#complete-example).

### Coding Agent Framework

Kimi K2.6 works best with Kimi Code CLI as its agent framework — give it a try at https://www.kimi.com/code.


---

## 7. License

Both the code repository and the model weights are released under the [Modified MIT License](LICENSE).

---

## 8. Third Party Notices

See [THIRD PARTY NOTICES](THIRD_PARTY_NOTICES.md)

---

## 9. Contact Us

If you have any questions, please reach out at [support@moonshot.ai](mailto:support@moonshot.ai).
