# deepseek-ai / deepseek-v4-flash

# DeepSeek-V4-Flash Overview

## Description:

DeepSeek-V4-Flash is a Mixture-of-Experts (MoE) language model with 284 billion total parameters and 13 billion activated parameters.\
DeepSeek-V4-Flash was developed by DeepSeek as a part of DeepSeek-V4 collection.\
*This model is ready for commercial/non-commercial use.*

## Third-Party Community Consideration:

This model is not owned or developed by NVIDIA. This model has been developed and built to a third-party's requirements for this application and use case; see link to Non-NVIDIA [DeepSeek-V4-Flash Model Card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash).

## License and Terms of Use:

**GOVERNING TERMS:** This trial service is governed by the [NVIDIA API Trial Terms of Service](https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf). Use of this model is governed by the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/). Additional Information: [MIT](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/LICENSE).

**You are responsible for ensuring that your use of NVIDIA provided models complies with all applicable laws.**

### Deployment Geography:

Global

### Use Case:

DeepSeek V4 is well-suited for advanced reasoning, agentic AI applications, tool use scenarios, and complex problem-solving in domains such as mathematics, software engineering, and enterprise AI assistants.

### Release Date:

**build.nvidia.com:** April 23, 2026 via [link](https://build.nvidia.com/deepseek-ai/deepseek-v4-flash)\
**Hugging Face:** April 23, 2026 via [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)

## Reference(s):

**References:**

* [DeepSeek-V4-Pro on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
* [DeepSeek V4 Technical Report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)
* [DeepSeek Chat Interface](https://chat.deepseek.com/)
* [DeepSeek Discord Community](https://discord.gg/deepseek)

## Model Architecture:

**Architecture Type:** Transformer\
**Network Architecture:** Mixture of Experts (MoE) with Hybrid Attention (Compressed Sparse Attention + Heavily Compressed Attention)\
**Number of model parameters:** 284 billion total parameters (13 billion activated)

## Input:

**Input Types:** Text\
**Input Formats:** String\
**Input Parameters:** One Dimensional (1D)\
**Other Input Properties:** Supports multi-turn conversations with system prompts, user messages, and assistant responses. Maximum context length of 1 million tokens. Uses a custom encoding pipeline (encoding\_dsv4) with three reasoning modes: Non-think (fast), Think High (logical analysis), and Think Max (full reasoning extent).

### Output:

**Output Types:** Text\
**Output Format:** String\
**Output Parameters:** One Dimensional (1D)\
**Other Output Properties:** Supports structured JSON output, function/tool calling, and reasoning content when enabled.

**Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.**

## Software Integration:

**Runtime Engines:**

* **Transformers:** Compatible with Hugging Face Transformers library
* **vLLM:** Recommended for efficient inference with sparse-attention support

**Supported Hardware:**

* **NVIDIA Ampere:** A100
* **NVIDIA Blackwell:** B200
* **NVIDIA Hopper:** H100, H200

**Operating Systems:** Linux

**The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.**

## Model Version(s):

DeepSeek-V4-Flash

## Training, Testing, and Evaluation Datasets:

## Training Dataset:

**Data Modality:** Text\
**Text Training Data Size:** More than 10 Trillion Tokens\
**Training Data Collection:** Undisclosed\
**Training Labeling:** Undisclosed\
**Training Properties:** Two-stage post-training pipeline: (1) independent cultivation of domain-specific experts via SFT and RL with GRPO, (2) unified model consolidation via on-policy distillation. Uses Muon optimizer for faster convergence and training stability.

### Testing Dataset:

**Testing Data Collection:** Undisclosed\
**Testing Labeling:** Undisclosed\
**Testing Properties:** Undisclosed

### Evaluation Dataset

**Evaluation Benchmark Score:**

| Benchmark (Metric)          | V4-Flash Non-Think | V4-Flash High | V4-Flash Max | V4-Pro Non-Think | V4-Pro High |  V4-Pro Max |
| :-------------------------- | :----------------: | :-----------: | :----------: | :--------------: | :---------: | :---------: |
| **Knowledge & Reasoning**   |                    |               |              |                  |             |             |
| MMLU-Pro (EM)               |        83.0        |      86.4     |     86.2     |       82.9       |     87.1    | **87.5** \| |
| SimpleQA-Verified (Pass\@1) |        23.1        |      28.9     |     34.1     |       45.0       |     46.2    | **57.9** \| |
| Chinese-SimpleQA (Pass\@1)  |        71.5        |      73.2     |     78.9     |       75.8       |     77.7    | **84.4** \| |
| GPQA Diamond (Pass\@1)      |        71.2        |      87.4     |     88.1     |       72.9       |     89.1    | **90.1** \| |
| HLE (Pass\@1)               |         8.1        |      29.4     |     34.8     |        7.7       |     34.5    | **37.7** \| |
| LiveCodeBench (Pass\@1)     |        55.2        |      88.4     |     91.6     |       56.8       |     89.8    | **93.5** \| |
| Codeforces (Rating)         |          -         |      2816     |     3052     |         -        |     2919    | **3206** \| |
| HMMT 2026 Feb (Pass\@1)     |        40.8        |      91.9     |     94.8     |       31.7       |     94.0    | **95.2** \| |
| IMOAnswerBench (Pass\@1)    |        41.9        |      85.1     |     88.4     |       35.3       |     88.0    | **89.8** \| |
| Apex (Pass\@1)              |         1.0        |      19.1     |     33.0     |        0.4       |     27.4    | **38.3** \| |
| Apex Shortlist (Pass\@1)    |         9.3        |      72.1     |     85.7     |        9.2       |     85.5    | **90.2** \| |
| **Long Context**            |                    |               |              |                  |             |             |
| MRCR 1M (MMR)               |        37.5        |      76.9     |     78.7     |       44.7       |     83.3    | **83.5** \| |
| CorpusQA 1M (ACC)           |        15.5        |      59.3     |     60.5     |       35.6       |     56.5    | **62.0** \| |
| **Agentic**                 |                    |               |              |                  |             |             |
| Terminal Bench 2.0 (Acc)    |        49.1        |      56.6     |     56.9     |       59.1       |     63.3    | **67.9** \| |
| SWE Verified (Resolved)     |        73.7        |      78.6     |     79.0     |       73.6       |     79.4    | **80.6** \| |
| SWE Pro (Resolved)          |        49.1        |      52.3     |     52.6     |       52.1       |     54.4    | **55.4** \| |
| SWE Multilingual (Resolved) |        69.7        |      70.2     |     73.3     |       69.8       |     74.1    | **76.2** \| |
| BrowseComp (Pass\@1)        |          -         |      53.5     |     73.2     |         -        |     80.4    | **83.4** \| |
| HLE w/ tools (Pass\@1)      |          -         |      40.3     |     45.1     |         -        |     44.7    | **48.2** \| |
| MCPAtlas (Pass\@1)          |        64.0        |      67.4     |     69.0     |       69.4       |   **74.2**  |     73.6    |
| GDPval-AA (Elo)             |          -         |       -       |     1395     |         -        |      -      | **1554** \| |
| Toolathlon (Pass\@1)        |        40.7        |      43.5     |     47.8     |       46.3       |     49.0    | **51.8** \| |

**Evaluation Data Collection:** Automated\
**Evaluation Labeling:** Human\
**Evaluation Properties:** Evaluated on competitive programming, mathematical reasoning, and general reasoning benchmarks.

## Inference:

**Acceleration Engine:** Transformers, vLLM with sparse-attention optimization <br>\
**Test Hardware:** <br>\
NVIDIA Hopper (H100) <br>\
NVIDIA Hopper (H200) <br>\
Precision formats: FP4 + FP8 Mixed (MoE experts in FP4, other parameters in FP8)

## Ethical Considerations:

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Users are responsible for model inputs and outputs. Users are responsible for ensuring safe integration of this model, including implementing guardrails as well as other safety mechanisms, prior to deployment.

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).