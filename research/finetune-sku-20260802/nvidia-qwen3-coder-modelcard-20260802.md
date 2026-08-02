# qwen / qwen3-coder-480b-a35b-instruct

# Qwen3-Coder-480B-A35B-Instruct

## Model Overview

### Description:

Qwen3-Coder-480B-A35B-Instruct is a state-of-the-art large language model specifically designed for code generation and agentic coding tasks. It is a mixture-of-experts (MoE) model with 480B total parameters and 35B activated parameters, featuring native support for 262,144 tokens context length and extendable up to 1M tokens using YaRN.

This model demonstrates significant performance among open models on Agentic Coding, Agentic Browser-Use, and other foundational coding tasks, achieving results comparable to Claude Sonnet. It supports function calling and tool choice capabilities, making it ideal for complex coding workflows and agentic applications.

This model is ready for commercial use.

### License/Terms of Use

**GOVERNING TERMS:** This trial service is governed by the [NVIDIA API Trial Terms of Service](https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf). Use of this model is governed by the [NVIDIA Community Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-community-models-license/). Additional Information: [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md).

## Deployment Geography

**Deployment Geography**: Global <br>

## Use Cases

* **Code Generation**: Generate high-quality code from natural language descriptions
* **Agentic Coding**: Execute complex coding workflows with function calling
* **Repository Understanding**: Process large codebases with long-context capabilities
* **Tool Integration**: Interface with development tools and APIs
* **Code Review and Analysis**: Analyze and improve existing code
* **Documentation Generation**: Create code documentation and comments
* **Browser Automation**: Agentic browser-use scenarios
* **Function Calling**: Structured tool execution and API integration

## Release Information

**Release Date**: 08/22/2025 <br>\
**Build.NVIDIA.com**: Available via [link](https://build.nvidia.com/qwen/qwen3-coder-480b-a35b-instruct) <br>

### Third-Party Community Consideration

This model is not owned or developed by NVIDIA. This model has been developed by Qwen (Alibaba Cloud). This model has been developed and built to a third-party's requirements for this application and use case; see link to [Qwen3-Coder-480B-A35B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct).

### References

* [Qwen3-Coder: A Large Language Model for Code Generation](https://qwenlm.github.io/blog/qwen3-coder/) <br>
* [Qwen3-Coder GitHub Repository](https://github.com/QwenLM/Qwen3-Coder) <br>
* [Qwen Documentation](https://qwen.readthedocs.io/en/latest/) <br>
* [Hugging Face Model Page](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct) <br>
* [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/abs/2505.09388) <br>

## Model Architecture

**Architecture Type**: mixture-of-experts (MoE) with Sparse Activation <br>\
**Network Architecture**: Qwen3MoeForCausalLM (Transformer-based decoder-only) <br>\
**Parameter Count**: 480B total parameters with 35B activated parameters <br>\
**Expert Configuration**: 160 experts with 8 activated per forward pass <br>\
**Attention Mechanism**: Grouped Query Attention (GQA) with 96 query heads and 8 KV heads <br>\
**Number of Layers**: 62 <br>\
**Hidden Size**: 6144 <br>\
**Head Dimension**: 128 <br>\
**Intermediate Size**: 8192 <br>\
**MoE Intermediate Size**: 2560 <br>\
**Context Length**: 262,144 tokens (native), extendable to 1M with YaRN <br>\
**Vocabulary Size**: 151,936 <br>

## Input

**Input Type(s)**: Text, Code, Function calls <br>\
**Input Format(s)**: Natural language prompts, code snippets, structured function calls <br>\
**Input Parameters**:

* Max input length: 262,144 tokens (native), up to 1M with YaRN
* Support for function calling format
* Tool choice enabled
* Trust remote code execution
* Custom tool call parser (qwen3\_coder)

## Output

**Output Type(s)**: Text, Code, Function responses <br>\
**Output Format(s)**: Natural language responses, code generation, structured function outputs <br>\
**Output Parameters**: One-Dimensional (1D)

* Max output length: Configurable based on remaining context
* Function call responses in structured format

  **Other Properties Related to Output**:
* Non-thinking mode (no `<think></think>` blocks)
* Auto tool choice responses

## Software Integration

**Runtime Engine**: vLLM, Transformers (4.51.0+) <br>\
**Supported Hardware Platform(s)**: NVIDIA Hopper <br>\
**Supported Operating System(s)**: Linux <br>\
**Data Type**: FP8 <br>\
**Data Modality**: Text <br>\
**Model Version**: v1.0 <br>

## Training, Testing, and Evaluation Datasets

### Training Dataset

* **Data Collection Method by dataset**: The model was trained on a diverse dataset including code repositories, documentation, and natural language text related to programming
* **Labeling Method by dataset**: Supervised fine-tuning with instruction-following data
* **Properties**: Multi-language code support, instruction-following capabilities, function calling training

### Testing Dataset

* **Data Collection Method by dataset**: Standard benchmarks for code generation and agentic tasks
* **Labeling Method by dataset**: Automated evaluation metrics
* **Properties**: HumanEval, MBPP, Agentic coding benchmarks

### Evaluation Dataset

* **Data Collection Method by dataset**: Public benchmarks and custom evaluation sets
* **Labeling Method by dataset**: Automated metrics and human evaluation
* **Properties**: Code generation quality, function calling accuracy, agentic task performance

#### Benchmark Results

The model achieves significant performance among open models on:

* Agentic Coding tasks
* Agentic Browser-Use scenarios
* Foundational coding benchmarks
* Results comparable to Claude Sonnet on various coding tasks

## Inference

**Acceleration Engine**: vLLM <br>\
**Test Hardware**: NVIDIA Hopper <br>

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).