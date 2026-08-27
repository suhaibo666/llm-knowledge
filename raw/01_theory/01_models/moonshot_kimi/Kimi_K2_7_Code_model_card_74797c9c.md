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
      <img src="figures/kimi-logo.png" width="30%" alt="Kimi K2.7 Code">
  </picture>
</div>
<hr>
<div align="center" style="line-height:1">
  <a href="https://www.kimi.com/code" target="_blank"><img alt="Chat" src="https://img.shields.io/badge/🤖-Kimi--Code-ff6b6b?color=1783ff&logoColor=white"/></a>
  <a href="https://www.moonshot.ai" target="_blank"><img alt="Homepage" src="https://img.shields.io/badge/Homepage-Moonshot%20AI-white?logo=Kimi&logoColor=white"/></a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/moonshotai" target="_blank"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Moonshot%20AI-ffc107?color=ffc107&logoColor=white"/></a>
  <a href="https://twitter.com/kimi_moonshot" target="_blank"><img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-Kimi.ai-white?logo=x&logoColor=white"/></a>
  <a href="https://discord.gg/TYU2fdJykW" target="_blank"><img alt="Discord" src="https://img.shields.io/badge/Discord-Kimi.ai-white?logo=discord&logoColor=white"/></a>
  <a href="https://modelscope.cn/organization/moonshotai" target="_blank"><img alt="ModelScope" src="https://img.shields.io/badge/ModelScope-Moonshot%20AI-white?labelColor=rgb(99%2C%2074%2C%20255)"/></a>
</div>
<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Modified_MIT-f5de53?&color=f5de53"/></a>
</div>




## 1. Model Introduction

Kimi K2.7 Code is a coding-focused agentic model built upon Kimi K2.6. With substantial improvements on real-world long-horizon coding tasks, it strengthens end-to-end task completion across complex software engineering workflows while improving token efficiency, reducing thinking-token usage by approximately 30% compared with Kimi K2.6.

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
<th align="center">Kimi K2.6</th>
<th align="center">Kimi K2.7 Code</th>
<th align="center">GPT-5.5</th>
<th align="center">Claude Opus 4.8</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center" colspan=5><strong>Coding</strong></td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">Kimi Code Bench v2</td>
<td align="center" style="vertical-align: middle">50.9</td>
<td align="center" style="vertical-align: middle">62.0</td>
<td align="center" style="vertical-align: middle">69.0</td>
<td align="center" style="vertical-align: middle">67.4</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">Program Bench</td>
<td align="center" style="vertical-align: middle">48.3</td>
<td align="center" style="vertical-align: middle">53.6</td>
<td align="center" style="vertical-align: middle">69.1</td>
<td align="center" style="vertical-align: middle">63.8</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">MLS Bench Lite</td>
<td align="center" style="vertical-align: middle">26.7</td>
<td align="center" style="vertical-align: middle">35.1</td>
<td align="center" style="vertical-align: middle">35.5</td>
<td align="center" style="vertical-align: middle">42.8</td>
</tr>
<tr>
<td align="center" colspan=5><strong>Agentic</strong></td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">Kimi Claw 24/7 Bench</td>
<td align="center" style="vertical-align: middle">42.9</td>
<td align="center" style="vertical-align: middle">46.9</td>
<td align="center" style="vertical-align: middle">52.8</td>
<td align="center" style="vertical-align: middle">50.4</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">MCP Atlas</td>
<td align="center" style="vertical-align: middle">69.4</td>
<td align="center" style="vertical-align: middle">76.0</td>
<td align="center" style="vertical-align: middle">79.4</td>
<td align="center" style="vertical-align: middle">81.3</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">MCP Mark Verified</td>
<td align="center" style="vertical-align: middle">72.8</td>
<td align="center" style="vertical-align: middle">81.1</td>
<td align="center" style="vertical-align: middle">92.9</td>
<td align="center" style="vertical-align: middle">76.4</td>
</tr>
</tbody>
</table>
</div>

<details>
<summary><b>Footnotes</b></summary>

1. **General Testing Details**
   - Unless stated otherwise, Kimi K2.7 Code and K2.6 were tested with thinking mode enabled via Kimi Code CLI at temperature = 1.0, top-p = 0.95, and a 262,144-token context length; GPT-5.5 ran in Codex with xhigh mode, and Opus 4.8 in Claude Code with xhigh mode. Aside from these differences, all benchmarks were evaluated under the same conditions.
2. **Coding Benchmarks**
   - Kimi Code Bench V2 is our in-house benchmark designed to evaluate coding agents on realistic tasks. It has diversed software engineering tasks across 10+ mainstream programming languages and a full production tech stack covering tasks from internal engineering use cases, production incidents, and real-world open-source projects, with emphasis on backend services, infrastructure, performance engineering, systems programming, security, frontend development, and ML/data engineering.
   - [Program Bench](https://programbench.com/) evaluates code-generation agents by asking them to recreate a program’s behavior from only a compiled binary and its documentation. It spans 200 tasks, from small CLI tools to large systems like FFmpeg and SQLite. Submissions are judged against over 248,000 fuzz-generated behavioral tests. In each task, the agent is given an executable and its documentation, but no source code, decompilation, or internet access. It must choose its own implementation language, build the full program from scratch, and pass a behavioral test suite comparing its output against the original binary.
   - [MLS-Bench](https://mls-bench.com) evaluates whether AI systems can invent generalizable and scalable ML methods. MLS-Bench-Lite is the official 30-task subset of MLS-Bench, covering LLM pretraining and post-training, robotics, world models, computer vision, reinforcement learning, optimization, ML systems, AI for Science, and more. Agents are given 5 hours to explore before submitting their solutions. Opus 4.8 is evaluated with the max effort setting in Claude Code.
3. **Agentic Benchmarks**
   - Kimi Claw 24/7 Bench is our in-house benchmark for evaluating long-horizon agentic performance in persistent, multi-day coworking tasks. It spans 17 professional scenarios across 610 evaluation points, covering domains such as software engineering, ML research, recruiting, trading, marketing. All tasks are executed through the OpenClaw harness. The final score is the average pass rate across all evaluation points, and is averaged over 3 runs.
   - [MCP-Atlas](https://labs.scale.com/leaderboard/mcp_atlas) evaluates LLM performance on realistic tool-use tasks through the scalable MCPs. We followed the official MCP-Atlas evaluation configuration with a 100 tool-call budget, and with 32k max tokens per step. The final result is averaged over 3 runs.
   - MCPMark-Verified is a human-verified edition of [MCPMark](https://mcpmark.ai/), a benchmark for evaluating MCP tool use across five real server environments — Notion, GitHub, Filesystem, Postgres, and Playwright. Each task has been re-checked by our team and the benchmark offical and will be open-sourced soon. We followed the official MCPMark evaluation configuration with a 100-step tool-call budget and 32k max tokens per step. The final result is averaged over 3 runs.

</details>


## 4. Native INT4 Quantization
Kimi-K2.7-Code adopts the same native int4 quantization method as [Kimi-K2-Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking#4-native-int4-quantization).

## 5. Deployment

> [!Note]
> You can access Kimi-K2.7-Code's API on https://platform.moonshot.ai and we provide OpenAI/Anthropic-compatible API for you.
Currently, Kimi-K2.7-Code is recommended to run on the following inference engines:
* vLLM
* SGLang
* KTransformers

Kimi-K2.7-Code has the same architecture as Kimi-K2.5/Kimi-K2.6, and the deployment method can be directly reused.

The version requirement for `transformers` is `>=4.57.1, <5.0.0`.

Deployment examples can be found in the [Model Deployment Guide](docs/deploy_guidance.md).


---
## 6. Model Usage

The usage demos below demonstrate how to call our official API. Note that Kimi-K2.7-Code forces thinking and preserve_thinking as True.

For third-party APIs deployed with vLLM or SGLang, please note that:
> [!Note]
> - Chat with video content is an experimental feature and is only supported in our official API for now.
> 
> - The recommended `temperature` will be `1.0` for Thinking mode.
> 
> - The recommended `top_p` is `0.95`.
> 
> - Instant mode is not supported.

### Chat Completion

This is a simple chat completion script which shows how to call K2.7-Code API in Thinking mode.

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
```


### Chat Completion with visual content

K2.7-Code supports Image and Video input.

The following example demonstrates how to call K2.7-Code API with image input:

```python
import openai
import base64
import requests

def chat_with_image(client: openai.OpenAI, model_name: str):
    url = 'https://huggingface.co/moonshotai/Kimi-K2.7-Code/resolve/main/figures/kimi-logo.png'
    image_base64 = base64.b64encode(requests.get(url).content).decode()
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'Describe this image in detail.'},
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/png;base64,{image_base64}'},
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
```

The following example demonstrates how to call K2.7-Code API with video input:

```python
import openai
import base64
import requests

def chat_with_video(client: openai.OpenAI, model_name:str):
    url = 'https://huggingface.co/moonshotai/Kimi-K2.7-Code/resolve/main/figures/demo_video.mp4'
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
```

### Preserve Thinking
Kimi K2.7 Code forces `preserve_thinking` mode, which retains full reasoning content across multi-turn interactions and enhances performance in coding agent scenarios.

This feature is enabled by default and can't be disabled. The following example demonstrates how to call K2.7-Code API in `preserve_thinking` mode:

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
            # Some API (e.g. vLLM) may not support reasoning_content, you can try reasoning instead
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
    )
    # the assistant should mention 215 and 222 that appear in the prior reasoning content
    print(f"response: {response.choices[0].message.reasoning}")
    return response.choices[0].message.content

```

### Interleaved Thinking and Multi-Step Tool Call

K2.7-Code shares the same design of Interleaved Thinking and Multi-Step Tool Call as K2 Thinking. For usage example, please refer to the [K2 Thinking documentation](https://platform.moonshot.ai/docs/guide/use-kimi-k2-thinking-model#complete-example).

### Coding Agent Framework

Kimi K2.7-Code works best with Kimi Code CLI as its agent framework — give it a try at https://www.kimi.com/code.


---

## 7. License

Both the code repository and the model weights are released under the [Modified MIT License](LICENSE).

---

## 8. Third Party Notices

See [THIRD PARTY NOTICES](THIRD_PARTY_NOTICES.md)

---

## 9. Contact Us

If you have any questions, please reach out at [support@moonshot.ai](mailto:support@moonshot.ai).
