---
category: 视觉语言模型
classification_reason: 该论文提出了GenieBlue，一种专为移动设备设计的轻量级多模态大语言模型（MLLM）架构，旨在解决端侧NPU对MoE架构支持不佳的问题，同时保持纯语言能力。
created: '2026-01-18'
status: unread
tags:
- 视觉语言模型
- NPU适配
- LoRA
- 移动端部署
- 架构设计
title: 'GenieBlue: An Efficient MLLM Structural Design for Mobile Devices'
---

# <span id="page-0-2"></span>GenieBlue: Integrating both Linguistic and Multimodal Capabilities for Large Language Models on Mobile Devices

Xudong Lu\*1,2†, Yinghao Chen\*1, Renshou Wu\*1, Haohao Gao¹, Xi Chen¹, Xue Yang³, Xiangyu Zhao³, Aojun Zhou², Fangyuan Li¹, Yafei Wen¹, Xiaoxin Chen¹, Shuai Ren¹‡⊠, Hongsheng Li²⊠¹vivo AI Lab ²CUHK MMLab ³Shanghai Jiao Tong University

{luxudong@link,hsli@ee}.cuhk.edu.hk shuai.ren@vivo.com

#### **Abstract**

Recent advancements in Multimodal Large Language Models (MLLMs) have enabled their deployment on mobile devices. However, challenges persist in maintaining strong language capabilities and ensuring hardware compatibility, both of which are crucial for user experience and practical deployment efficiency. In our deployment process, we observe that existing MLLMs often face performance degradation on pure language tasks, and the current NPU platforms on smartphones do not support the MoE architecture, which is commonly used to preserve pure language capabilities during multimodal training. To address these issues, we systematically analyze methods to maintain pure language capabilities during the training of MLLMs, focusing on both training data and model architecture aspects. Based on these analyses, we propose GenieBlue, an efficient MLLM structural design that integrates both linguistic and multimodal capabilities for LLMs on mobile devices. GenieBlue freezes the original LLM parameters during MLLM training to maintain pure language capabilities. It acquires multimodal capabilities by duplicating specific transformer blocks for full fine-tuning and integrating lightweight LoRA modules. This approach preserves language capabilities while achieving comparable multimodal performance through extensive training. Deployed on smartphone NPUs, GenieBlue demonstrates efficiency and practicality for applications on mobile devices.

#### <span id="page-0-1"></span>1. Introduction

Recent advancements in Large Language Models (LLMs) have significantly improved people's daily lives [1, 28, 33, 68, 80], particularly through multimodal models (MLLMs) that seamlessly integrate information from different sources such as text, images, and videos [3, 4, 7, 12, 49, 59, 66]. As the scope of LLM and MLLM applications continues to expand, efficient deployment on smartphones is gaining increasing attention [14, 15, 53, 78, 82] due to their ability to enhance user privacy and support offline functionality.

<span id="page-0-0"></span>

|          | Model            | MATH  | AlignBench | MT-Bench |
|----------|------------------|-------|------------|----------|
| Base LLM | Qwen2.5-3B       | 61.74 | 6.00       | 5.81     |
| MLLM     | InternVL2.5-4B   | 55.20 | 5.18       | 4.94     |
| Drop (%) |                  | 10.59 | 13.67      | 14.97    |
| Base LLM | Qwen2.5-3B       | 61.74 | 6.00       | 5.81     |
| MLLM     | Qwen2.5-VL-3B    | 58.92 | 5.38       | 4.72     |
| Drop (%) |                  | 4.57  | 10.33      | 18.76    |
| Base LLM | Qwen1.5-7B       | 22.02 | 5.40       | 5.77     |
| MLLM     | Wings-Qwen1.5-8B | 13.96 | 4.86       | 4.56     |
| Drop (%) |                  | 36.60 | 10.00      | 20.97    |
| Base LLM | BlueLM-3B        | 38.94 | 5.67       | 5.42     |
| MLLM     | GenieBlue-3B     | 38.94 | 5.67       | 5.42     |
| Drop (%) |                  | 0     | 0          | 0        |

Table 1. We assess the pure language capabilities of several representative MLLMs alongside their corresponding LLMs. The evaluation reveals that these MLLMs typically exhibit a performance drop exceeding 10% across all three datasets. In contrast, our proposed GenieBlue does not sacrifice any pure language ability.

In the practical process of deploying LLMs and MLLMs on smartphones, we inevitably face the storage and memory limitations inherent to these devices. Therefore, we aim to deploy a single model that can efficiently handle both pure language tasks and multimodal tasks simultaneously [7, 12]. Currently, various MLLMs suitable for ondevice deployment have emerged, such as Qwen2.5-VL-3B [7], MiniCPM-V-2 [82], and InternVL2.5-4B [12], etc. These small models can achieve performance comparable to larger counterparts while having fewer parameters, making them ideal for on-device deployment. However, during the practical deployment of MLLMs on smartphone NPU (neural processing unit), we encounter the following issues:

**Issue** (1) MLLMs still cannot achieve satisfactory pure language capabilities currently:

Current MLLMs, while excelling in multimodal tasks, still perform moderately on pure language tasks, especially in subjective language tasks, where they still exhibit significant performance gaps compared to corresponding pure language models. We here carry out a pilot study to showcase this phenomenon. We evaluate the pure language capabilities of several representative MLLMs along-side their corresponding LLMs. Both Qwen2.5-VL-3B [7]

<sup>\*</sup>Equal contribution <sup>™</sup>Corresponding author <sup>‡</sup>Project lead <sup>†</sup>Intern at vivo.

<span id="page-1-0"></span>and InternVL2.5-4B [\[12\]](#page-8-4) are based on the Qwen2.5-3B [\[80\]](#page-11-0) language model. Additionally, Wings [\[88\]](#page-11-3) introduces a method to train MLLMs without causing text-only forgetting. Therefore, we also assess the NLP metrics of the provided Wings-Qwen1.5 checkpoint based on Qwen1.5- 7B. We select three datasets for evaluation: MATH [\[30\]](#page-9-2), which consists of challenging mathematical reasoning problems; AlignBench [\[46\]](#page-9-3), a subjective dataset for evaluating LLMs' human alignment in Chinese; and MT-Bench [\[91\]](#page-11-4), a subjective benchmark for assessing multi-turn conversational capabilities. For the evaluation of AlignBench and MT-Bench, we leverage Google Gemini 1.5 Pro [\[66\]](#page-10-3) as the judge LLM. As shown in Tab. [1,](#page-0-0) these MLLMs generally suffer a drop of more than 10% across the three datasets.

Remark: For the deployment of LLMs on mobile devices, we prioritize the performance of subjective language tasks. On-device models in smartphone environments frequently engage in more nuanced, subjective tasks in daily usage, such as text refinement, call summarization, etc.

*Issue (2) Mainstream smartphone NPU platforms currently do not support deploying MoE structures:*

Currently, model structural improvements designed to integrate both multimodal and pure language capabilities typically rely on the Mixture of Experts (MoE) architecture, e.g., CogVLM [\[71\]](#page-10-5), Wings [\[88\]](#page-11-3). While the MoE architecture reduces the number of activated parameters during model inference, it still necessitates loading the entire original model into memory during initialization, which is not ideal for practical smartphone deployment given the limited memory available. As of now, NPU platforms of MediaTek and Qualcomm SoCs, e.g., MediaTek Dimensity 9400 and Qualcomm Snapdragon 8 Elite, do not support the deployment of MoE architectures.

Based on issue 1), despite extensive research into data and training methodologies for MLLMs, maintaining the pure language capability of MLLMs remains challenging. Based on issue 2), for end-side scenarios, the design of model architectures must also account for the constraints imposed by deployment environments. Inspired by these two challenges, this paper systematically analyzes how to maintain pure language capabilities during the training of MLLMs from both training data and model architecture aspects, emphasizing end-side deployment considerations.

From the training data perspective, we train LLMs using representative open-source MLLM datasets [\[69\]](#page-10-6), consisting of 2.5M samples for pre-training and 7M for fine-tuning. Our findings reveal a significant decline in pure language capabilities. We then augment the fine-tuning dataset with an additional 2M samples of pure language data and retrain the MLLM. This modification demonstrates moderate benefits for objective NLP tasks, but it yields only minimal improvements in subjective tasks due to the currently limited volume of high-quality training data available for human preference alignment [\[8,](#page-8-7) [89\]](#page-11-5). From these observations, we conclude that simply increasing training data is insufficient to address the decline in pure language capabilities at the current stage. Therefore, we explore the design of model structures considering hardware limitations of mobile NPUs. In this paper, we introduce GenieBlue, which integrates linguistic and multimodal capabilities for LLMs on mobile devices through efficient structural designs.

Specifically, to preserve the pure language capabilities of the original LLM, we freeze all LLM parameters during the multimodal training process. We then copy the transformer block every nth block for full parameter training and add LoRA [\[31\]](#page-9-4) modules to the remaining blocks. During inference, we adopt a non-shared base deployment approach. In the LLM inference process, we utilize the originally frozen model. For MLLM inference, we replace the original transformer blocks (every nth block) with the fully trained ones and incorporate the trained LoRA parameters.

After extensive data training, GenieBlue achieves multimodal capabilities comparable to those of fully fine-tuned MLLMs without sacrificing any pure language capabilities. We also deploy GenieBlue on the NPU of real smartphones, demonstrating its efficiency and practicality for edge computing applications on mobile devices. The contributions of our work can be summarized as follows:

- 1) We examine the deployment of MLLMs on smartphones, identifying performance degradation in text-only tasks and highlighting the limitations of current NPU platforms that do not support the deployment of MoE models.
- 2) We analyze how to maintain pure language performance during the training of MLLMs from both the training data and model structure perspectives. Then, we introduce GenieBlue, which integrates both linguistic and multimodal capabilities for LLMs on mobile devices through efficient and more hardware-friendly model structural designs.
- 3) We train GenieBlue with large amounts of multimodal datasets, achieving multimodal capabilities comparable to fully fine-tuned MLLMs without compromising any pure language abilities. We also support the deployment of GenieBlue on actual smartphone NPUs, demonstrating efficient performance in real-world mobile environments.

## 2. Related Works

### 2.1. On-device LLMs and MLLMs

In recent years, beyond exploring scaling laws and training models with larger numbers of parameters on extensive datasets [\[7,](#page-8-3) [12,](#page-8-4) [28,](#page-9-0) [44\]](#page-9-5), a promising research direction has emerged: enabling smaller LLMs and MLLMs to achieve performance comparable to larger models [\[14,](#page-8-5) [15,](#page-8-6) [32,](#page-9-6) [53,](#page-10-4) [82\]](#page-11-2). Small models with strong performance are more suitable for edge deployment scenarios, especially given the constraints of memory and computational resources on mo-

<span id="page-2-2"></span><span id="page-2-0"></span>

| Type        | #Samples | Datasets                                                                                                                                                                                                                                         |
|-------------|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| General QA  | 840k     | UltraFeedback [22], UltraChat [23], NoRobots [61], LIMA [93], SlimOrca [42], WizardLM-Evol<br>Instruct-70K [76], Llama-3-Magpie-Pro [77], Magpie-Qwen2-Pro [77], Firefly [81], Dolly [19],<br>OpenAI-Summarize-TLDR [9], Know-Saraswati-CoT [35] |
| Code        | 360k     | Code-Feedback [92], Glaive-Code-Assistant [26], XCoder-80K [73], Evol-Instruct-Code [56]                                                                                                                                                         |
| Mathematics | 830k     | GSM8K-Socratic [17], NuminaMath-TIR [37], NuminaMath-CoT [38], InfinityMATH[87],<br>MathQA [2], MetaMathQA [83]                                                                                                                                  |

Table 2. We expand the Cambrian-7M dataset with 2M pure text data training samples, primarily sourced from the InternVL2.5 paper [\[12\]](#page-8-4).

<span id="page-2-1"></span>

| BlueLM-3B  | #Samples         | AI2D                    | ChartQA                 | DocVQA                  | OCRBench                | RealWorldQA             | ScienceQA            | TextVQA              | AVG                     |
|------------|------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|----------------------|----------------------|-------------------------|
| MLLM Tasks | 7M<br>7M+2M      | 74.81<br>74.03          | 68.32<br>69.36          | 74.60<br>74.63          | 55.30<br>56.70          | 62.35<br>58.04          | 67.91<br>68.24       | 60.06<br>62.34       | 66.19<br>66.19          |
| BlueLM-3B  | #Samples         | DROP                    | GPQA                    | GSM8K                   | MATH                    | MMLU                    | AlignBench           | MT-bench             | AVG                     |
| LLM Tasks  | -<br>7M<br>7M+2M | 81.57<br>62.49<br>64.67 | 29.46<br>23.21<br>28.80 | 86.13<br>66.11<br>69.90 | 38.94<br>19.26<br>30.60 | 74.13<br>57.50<br>57.67 | 5.67<br>3.87<br>3.84 | 5.42<br>3.92<br>3.92 | 60.16<br>43.78<br>47.03 |
|            |                  |                         |                         |                         |                         |                         |                      |                      |                         |
| Qwen2.5-3B | #Samples         | AI2D                    | ChartQA                 | DocVQA                  | OCRBench                | RealWorldQA             | ScienceQA            | TextVQA              | AVG                     |
| MLLM Tasks | 7M<br>7M+2M      | 77.20<br>76.98          | 67.36<br>68.48          | 68.84<br>64.25          | 54.70<br>56.20          | 61.05<br>62.09          | 68.19<br>69.43       | 57.72<br>55.54       | 65.01<br>64.71          |
| Qwen2.5-3B | #Samples         | DROP                    | GPQA                    | GSM8K                   | MATH                    | MMLU                    | AlignBench           | MT-bench             | AVG                     |

Table 3. We fully fine-tune BlueLM-V-3B from scratch (with SigLIP [\[86\]](#page-11-13) and BlueLM-3B [\[53\]](#page-10-4)/Qwen2.5-3B [\[80\]](#page-11-0)) using Cambrian 2.5M pre-training data and 7M fine-tuning data. We also conduct fine-tuning by adding 2M text-only data to the Cambrian-7M fine-tuning dataset. The inclusion of text-only data does not cause obvious degradation in MLLM performance and partially improves the accuracy on objective NLP tasks, but does not help with subjective NLP tasks (#Samples denotes the number of fine-tuning data samples).

bile devices like smartphones. With the advancement of this area of research, various small language models (SLMs) have been created [\[54\]](#page-10-10), including the Qwen series models [\[7,](#page-8-3) [70,](#page-10-11) [79,](#page-11-14) [80\]](#page-11-0), InternLM series models [\[12,](#page-8-4) [55,](#page-10-12) [67\]](#page-10-13), and the MiniCPM series models [\[32,](#page-9-6) [82\]](#page-11-2). In addition to exploring methods for training SLMs with high performance, recent research has also focused on how to more effectively deploy these models on edge devices [\[65,](#page-10-14) [78\]](#page-11-1).

### 2.2. Language Capability Maintenance of MLLMs

The maintenance of original pure language capabilities during the training of MLLMs is a critical issue [\[7,](#page-8-3) [71,](#page-10-5) [88\]](#page-11-3). This is particularly significant in scenarios where memory and storage are limited on edge devices, emphasizing the importance of having a model that can efficiently handle both pure language and multimodal tasks. There are now basically two types of approaches used to maintain pure language capabilities during multimodal training. The first is to increase the amount of language data during the multimodal training process [\[7,](#page-8-3) [12,](#page-8-4) [49\]](#page-10-1). However, as demonstrated by the experiments in Sec. [1,](#page-0-1) the current approach provides limited assistance in restoring language capabilities. The second approach is to carefully design the model structure [\[55,](#page-10-12) [71,](#page-10-5) [88\]](#page-11-3). Most existing methods utilize MoE architectures, which separate the "experts" that process text from those that handle other modal information. However, mainstream NPU platforms currently do not support the deployment of MoE structures. Recently, RL methods (e.g., DPO [\[60\]](#page-10-15)) have been utilized to align models with human preference, such as Qwen2.5-VL [\[7\]](#page-8-3). However, these methods still do not fully restore the language capabilities of the model (see Tab. [1](#page-0-0) and Tab. [8\)](#page-7-0). Additionally, most mainstream MLLMs still rely on pre-training and fine-tuning strategies, such as InternVL 2.5 [\[12\]](#page-8-4), DeepSeek-VL2 [\[75\]](#page-11-15), Ovis [\[52\]](#page-10-16), and LLaVA-OneVision [\[36\]](#page-9-12). Therefore, we discuss the pre-training and fine-tuning approach in this paper.

### 2.3. Benchmarks for Evaluating LLMs

The benchmarks for assessing LLMs can now be broadly categorized into two types: objective benchmarks and subjective benchmarks. Objective benchmarks are mainly designed to directly evaluate the knowledge capabilities of LLMs, encompassing areas such as general knowledge [\[16,](#page-8-14) [29,](#page-9-13) [74\]](#page-11-16), mathematics and science [\[18,](#page-8-15) [30,](#page-9-2) [62\]](#page-10-17), coding proficiency [\[5,](#page-8-16) [11\]](#page-8-17), etc. Subjective benchmarks, on the other hand, are characterized by their reliance on human judgment and interpretation [\[46,](#page-9-3) [90\]](#page-11-17), often requiring creativity and nuanced understanding rather than mere factual accuracy [\[39](#page-9-14)[–41\]](#page-9-15). For on-device deployment (e.g., on smartphones), LLMs do not necessarily need to master complex <span id="page-3-5"></span>knowledge but rather require better instruction following abilities, prioritizing a stronger ability in subjective tasks.

## <span id="page-3-4"></span>3. Text Capability Maintenance for MLLMs

In this section, we explore how to maintain the pure language capabilities during the training of MLLMs from both the training data (Sec. [3.1\)](#page-3-0) and model structure perspectives (Sec. [3.2\)](#page-3-1). Based on our analyses, we propose GenieBlue (Sec. [3.3\)](#page-4-0), an efficient and hardware-friendly model structural design for MLLMs that combines both linguistic and multimodal capabilities, specifically tailored for LLMs/MLLMs on the NPUs of mobile devices.

### <span id="page-3-0"></span>3.1. Training Data Perspective

Approach Analysis: To preserve pure language capabilities during the MLLM training process, the most straightforward and commonly used method is to add text-only data to the MLLM's training dataset. Currently, both InternVL2.5 [\[12\]](#page-8-4) and Qwen2.5-VL [\[7\]](#page-8-3) utilize this approach. However, this method presents some challenges. Firstly, it is difficult to collect a large amount of high-quality text-only instruction-tuning data, especially for subjective NLP tasks. Secondly, adding substantial amounts of text-only data during MLLM training will lead to longer training time.

Quantitative Experiments: To validate the effectiveness of this approach, we fully fine-tune an MLLM from scratch using a ViT and an LLM. Specifically, we utilize the BlueLM-V-3B architecture, which is tailored for end-side smartphone deployment, with SigLIP [\[86\]](#page-11-13) as the ViT and BlueLM-3B [\[53\]](#page-10-4)/Qwen2.5-3B [\[80\]](#page-11-0) as the LLM. We follow the training recipe of Cambrian-1 [\[69\]](#page-10-6), using the provided 2.5M alignment data for pre-training and the 7M data[1](#page-3-2) for fine-tuning. For comparison, we add another 2M pure-text data samples to the fine-tuning dataset, primarily sourced from the InternVL2.5 paper [\[12\]](#page-8-4), as shown in Tab. [2.](#page-2-0) We select 7 LLM benchmarks and 7 MLLM benchmarks for evaluation. For multimodal capabilities, we choose AI2Dtest [\[34\]](#page-9-16), ChartQAtest [\[57\]](#page-10-18), DocVQAval [\[58\]](#page-10-19), OCR-Bench [\[47\]](#page-9-17), RealWorldQA [\[21\]](#page-8-18), ScienceQAval [\[50\]](#page-10-20) and TextVQAval [\[64\]](#page-10-21). For pure language capabilities, we choose DROPval [\[25\]](#page-9-18), GPQA Diamond [\[63\]](#page-10-22), GSM8Ktest [\[17\]](#page-8-12), MATHtest [\[30\]](#page-9-2), MMLUtest [\[29\]](#page-9-13), AlignBench [\[46\]](#page-9-3) and MT-Bench [\[91\]](#page-11-4). The first five LLM benchmarks assess objective language capabilities, while the last two evaluate subjective language abilities. The evaluation results are shown in Tab. [3.](#page-2-1) We come across two observations:

*Finding (1) Adding pure-text datasets has little impact on the MLLM performance:*

After adding a pure language dataset containing 2M training samples, we find that the multimodal capabilities of

<span id="page-3-3"></span>![](_page_3_Picture_9.jpeg)

Figure 1. CogVLM [\[71\]](#page-10-5) replicates an identical visual expert module alongside each transformer block to handle multimodal inputs.

the trained MLLM remain virtually unchanged. This phenomenon indicates that incorporating a certain amount of pure text data during the training of an MLLM does not significantly affect its multimodal performance.

*Finding (2) Adding pure text data leads to a moderate improvement in the performance of objective NLP tasks but does not assist with subjective tasks:*

As can be seen from Tab. [3,](#page-2-1) the incorporation of multimodal data (7M) leads to a significant decline in both the objective and subjective language performance of the original LLM. To address this issue, we refer to InternVL2.5 [\[12\]](#page-8-4) and integrate an additional 2M pure text samples for training. As there is still a lack of sufficient high-quality opensource training data for human alignment [\[8\]](#page-8-7), the newly added pure-text data partially restores the performance for objective NLP tasks and provides almost no help for subjective NLP tasks. This indicates that maintaining the purelanguage capabilities of LLMs by adding additional purelanguage data currently remains a challenging endeavor.

### <span id="page-3-1"></span>3.2. Model Structure Perspective

Approach Analysis: Based on the analyses in Sec. [3.1,](#page-3-0) we conclude that maintaining NLP performance during the training of MLLMs by increasing pure text data is currently challenging. Consequently, another research direction focuses on the design of MLLM architectures, aiming to enhance NLP capabilities through architectural innovations rather than solely relying on additional pure text data. Representative works in this area include CogVLM [\[71\]](#page-10-5) and Wings [\[88\]](#page-11-3), both of which utilize the MoE structure.

However, during our deployment journey, we still observe that Wings [\[88\]](#page-11-3) leads to a significant decline in pure language capabilities. As noted in the experiment presented in Sec. [1,](#page-0-1) there is an average drop of over 20% in NLP performance, which is unacceptable for our deployment purposes. Regarding CogVLM [\[71\]](#page-10-5), it replicates an identical visual expert module alongside each transformer block to

<span id="page-3-2"></span><sup>1</sup>Cambrian-7M dataset contains around 1.5M pure-text data samples.

<span id="page-4-3"></span><span id="page-4-2"></span>

| BlueLM-3B             | #Param              | AI2D           | ChartQA        | DocVQA         | OCRBench       | RealWorldQA    | ScienceQA      | TextVQA        | AVG            | Retention (%)  |
|-----------------------|---------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| Full-Finetune         | 3161.26M            | 74.03          | 69.36          | 74.63          | 56.70          | 58.04          | 68.24          | 62.34          | 66.19          | -              |
| LoRA                  | 458.06M             | 68.23          | 61.24          | 66.17          | 48.70          | 55.56          | 68.57          | 56.97          | 60.78          | 91.82          |
| CogVLM-Post           | 1005.69M            | 67.81          | 60.80          | 66.49          | 51.00          | 57.12          | 67.00          | 58.58          | 61.26          | 92.55          |
| CogVLM-Pre            | 1005.69M            | 69.04          | 64.28          | 70.23          | 51.50          | 52.29          | 67.67          | 60.42          | 62.20          | 93.98          |
| CogVLM-Skip           | 1005.69M 70.01      |                | 66.36          | 71.97          | 54.60          | 56.34          | 68.91          | 59.37          | 63.94          | 96.60          |
|                       |                     |                |                |                |                |                |                |                |                |                |
| Qwen2.5-3B            | #Param              | AI2D           | ChartQA        | DocVQA         | OCRBench       | RealWorldQA    | ScienceQA      | TextVQA        | AVG            | Retention (%)  |
|                       |                     |                |                |                |                |                |                |                |                |                |
| Full-Finetune<br>LoRA | 3527.81M<br>456.84M | 76.98<br>65.35 | 68.48<br>54.32 | 64.25<br>55.84 | 56.20<br>48.10 | 62.09<br>55.56 | 69.43<br>72.72 | 55.54<br>58.40 | 64.71<br>58.61 | -<br>90.58     |
| CogVLM-Post           | 1146.75M            |                |                |                |                |                |                |                |                |                |
| CogVLM-Pre            | 1146.75M            | 68.72<br>68.88 | 60.48<br>62.12 | 65.14<br>67.95 | 51.30<br>52.30 | 48.89<br>53.73 | 64.76<br>72.87 | 59.85<br>57.36 | 59.88<br>62.17 | 92.53<br>96.08 |

Table 4. Evaluation results on MLLM benchmarks. We fine-tune all the models using the 9M dataset, comparing full fine-tuning, LoRA fine-tuning, and CogVLM fine-tuning. Post, Pre, and Skip means adding the visual expert module to the last quarter of the layers, the first quarter of the layers, and at every quarter interval of the layers. Apart from full fine-tuning, other methods can maintain pure language capability consistent with the original LLM during inference through the use of the non-shared base deployment strategy. CogVLM-Skip achieves the best MLLM performance retention. We also provide the trainable parameter numbers (#Param) during MLLM training.

handle multimodal inputs while keeping the original LLM frozen during training, as shown in Fig. [1.](#page-3-3) This design ensures that the performance of the original LLM remains unchanged during inference. However, this design still has two shortcomings. 1), during deployment, both the LLM and all corresponding visual expert modules need to be loaded into memory simultaneously, doubling the model's memory requirements. 2), as analyzed in Sec. [1,](#page-0-1) current smartphone NPU platforms do not yet support the deployment of MoE models. This results in deployment issues for CogVLM on real mobile devices.

Quantitative Experiments: To ensure the completeness of our work, we evaluate the MLLM performance of the models after training using the CogVLM approach with both BlueLM-3B and Qwen2.5-3B LLMs. To address the memory issues that arise during deployment, we integrate a visual expert module into one-quarter of the layers. We experiment with adding visual expert modules to the last quarter of the layers, the first quarter of the layers, and at every quarter interval of the layers [\[6\]](#page-8-19). For other transformer blocks, we add LoRA[2](#page-4-1) weights to the attention modules and feed-forward modules. We compare the three CogVLMbased methods with full fine-tuning and full-LoRA training. To provide more insights, we also list the trainable parameters (including ViT and projector layer) during MLLM training. The results are shown in Tab. [4.](#page-4-2)

*Finding (3) Compared to full fine-tuning, LoRA and CogVLM methods lead to a decrease in the multimodal performance of the trained MLLM:*

Due to limitations in the number of trainable parameters, both LoRA and CogVLM methods fall short of the multimodal performance achieved by full fine-tuning. Nevertheless, they typically reach over 90% of the performance seen with full fine-tuning. Besides, CogVLM outperforms LoRA in MLLM performance. It is important to note that full finetuning has a significant negative impact on the performance of pure-text tasks (Tab. [3\)](#page-2-1), while LoRA and CogVLM do not influence the pure language performance through the use of the non-shared base deployment strategy (Sec. [3.3\)](#page-4-0).

*Finding (4) For CogVLM, the addition of visual expert modules at every quarter interval of the layers results in the best MLLM performance:*

Incorporating visual experts at every quarter interval of the layers results in over 96% accuracy retention for the MLLM compared to full fine-tuning. Since CogVLM's training approach does not affect pure-text performance, we have decided to design GenieBlue based on this method.

## <span id="page-4-0"></span>3.3. GenieBlue

Based on the analyses from both data (Sec. [3.1\)](#page-3-0) and structural (Sec. [3.2\)](#page-3-1) perspectives, we propose to integrate linguistic and multimodal capabilities into the training of MLLMs through structural design. In this subsection, we provide a detailed illustration of the GenieBlue structure.

Approach Analysis: We modify from the CogVLM [\[71\]](#page-10-5) structure, particularly paying attention to the limitations of NPUs on the MoE architecture. The main idea behind CogVLM is to separate the processing of text tokens and multimodal tokens. It employs an MoE architecture where different experts handle text and visual tokens. In contrast, our design principle focuses on bypassing the MoE structure by selecting separate model weights for LLM/MLLM deployment, thereby maintaining the original LLM architecture unchanged during the multimodal inference process.

The framework of GenieBlue is shown in Fig. [2.](#page-5-0) To save model storage on smartphones, we replicate the transformer blocks at every quarter interval throughout the layers of the LLM while integrating LoRA modules into the remaining transformer blocks. During multimodal training, we freeze

<span id="page-4-1"></span><sup>2</sup> In all experiments, we set the LoRA rank to 8.

<span id="page-5-0"></span>![](_page_5_Figure_0.jpeg)

Figure 2. Overview of GenieBlue. We replicate the transformer blocks at every quarter interval of the layers in the LLM and incorporate LoRA modules into the other transformer blocks. During multimodal training, we freeze the original LLM while fully training the replicated transformer blocks and the added LoRA parameters. For pure-text inference, we utilize the original LLM. For multimodal inference, we replace the original blocks with the trained transformer blocks at every quarter interval and add LoRA to the remaining transformer blocks. This non-shared base approach avoids the MoE structure while decoupling the inference processes of the LLM and MLLM.

<span id="page-5-1"></span>

| BlueLM-3B                    | #Param               | AI2D           | ChartQA        | DocVQA         | OCRBench       | RealWorldQA    | ScienceQA      | TextVQA        | AVG            | Retention (%)  |
|------------------------------|----------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| Full-Finetune                | 3161.26M             | 74.03          | 69.36          | 74.63          | 56.70          | 58.04          | 68.24          | 62.34          | 66.19          | -              |
| CogVLM-Skip                  | 1005.69M             | 70.01          | 66.36          | 71.97          | 54.60          | 56.34          | 68.91          | 59.37          | 63.94          | 96.60          |
| GenieBlue-Post               | 1005.73M             | 68.49          | 61.68          | 67.78          | 49.80          | 55.42          | 69.96          | 61.59          | 62.10          | 93.82          |
| GenieBlue-Pre                | 1005.73M             | 72.90          | 66.20          | 71.11          | 46.50          | 58.30          | 73.20          | 60.03          | 64.03          | 96.74          |
| GenieBlue-Skip               | 1005.73M 73.67       |                | 69.32          | 74.26          | 55.30          | 57.39          | 68.34          | 60.37          | 65.52          | 98.99          |
|                              |                      |                |                |                |                |                |                |                |                |                |
| Qwen2.5-3B                   | #Param               | AI2D           | ChartQA        | DocVQA         | OCRBench       | RealWorldQA    | ScienceQA      | TextVQA        | AVG            | Retention (%)  |
|                              |                      |                |                |                |                |                |                |                |                |                |
| Full-Finetune<br>CogVLM-Skip | 3527.81M<br>1146.75M | 76.98<br>69.30 | 68.48<br>65.92 | 64.25<br>71.10 | 56.20<br>54.10 | 62.09<br>50.59 | 69.43<br>69.48 | 55.54<br>59.62 | 64.71<br>62.87 | -<br>97.16     |
| GenieBlue-Post               | 1146.79M             |                |                |                |                |                |                |                |                |                |
| GenieBlue-Pre                | 1146.79M             | 67.29<br>69.01 | 59.80<br>58.44 | 60.70<br>56.65 | 49.30<br>43.90 | 56.47<br>58.04 | 75.35<br>75.01 | 59.88<br>62.19 | 61.26<br>60.46 | 94.66<br>93.44 |

Table 5. Evaluation results on MLLM benchmarks after training with the 9M fine-tuning dataset. Similar to the experiment setting of CogVLM, we replicate transformer blocks at the last, first, and every interval quarter of layers. Results show that GenieBlue-Skip demonstrates the best MLLM performance, yielding over 97% retention in MLLM performance compared to full fine-tuning.

the original LLM, allowing ViT, the replicated transformer blocks, and the added LoRA parameters to be fully trained.

For pure-text inference, we utilize the original, unmodified LLM to perform all calculations. In contrast, for multimodal inference, we replace the original blocks with the trained transformer blocks at every quarter interval and incorporate LoRA into the remaining transformer blocks. This non-shared base strategy effectively avoids the MoE structure and decouples the inference processes of the LLM and MLLM. During actual NPU deployment, we only need to replace the weights and adapt the LoRA module. This makes deployment simple and efficient.

Quantitative Experiments: We compare our proposed GenieBlue against full fine-tuning and the CogVLM methods with both BlueLM-3B and Qwen2.5-3B LLMs, using the 2.5M pre-training data and 9M fine-tuning data. For a fair comparison with CogVLM, we replicate transformer blocks at the last (Post), first (Pre), and every interval (Skip) quarter of layers. The results are shown in Tab. [5.](#page-5-1)

*Finding (5) For GenieBlue structure, GenieBlue-Skip achieves the best multimodal performance, GenieBlue-Skip also outperforms CogVLM-Skip:*

Similar to the results of CogVLM, replicating transformer blocks at every interval quarter of layers achieves better multimodal performance. Besides, we find that GenieBlue-Skip outperforms CogVLM-Skip. This could possibly be attributed to CogVLM's approach of incorporating visual expert modules. In CogVLM's design, text features and image features are rigidly separated and processed separately for QKV and FFN calculations. Although CogVLM considers the fusion of text and image features during multi-head attention, this fusion is not as effective as completely sharing weights, which limits better integration throughout the entire MLLM inference process.

Non-shared Base Deployment Strategy: By splitting the LLM and MLLM inference process, deploying GenieBlue with the non-shared base strategy (as shown in Fig. [2\)](#page-5-0) can maintain the pure language capabilities of the original LLM. To validate the importance of this approach, we evaluate GenieBlue's performance on LLM benchmarks, comparing the shared and non-shared base deployment strategies. The shared base deployment strategy refers to unifying the inference processes of LLM and MLLM into the single deployment mode depicted on the right of Fig. [2.](#page-5-0) Specifically, during the inference of pure language tasks, we also

<span id="page-6-4"></span><span id="page-6-0"></span>

| BlueLM-3B                                    | Shared Base  | DROP                             | GPQA                             | GSM8K                            | MATH                             | MMLU                             | AlignBench                   | MT-bench                     | AVG                              | Retention (%)           |
|----------------------------------------------|--------------|----------------------------------|----------------------------------|----------------------------------|----------------------------------|----------------------------------|------------------------------|------------------------------|----------------------------------|-------------------------|
| BlueLM-3B                                    | -            | 81.57                            | 29.46                            | 86.13                            | 38.94                            | 74.13                            | 5.67                         | 5.42                         | 60.16                            | -                       |
| Full-Finetune                                | -            | 64.67                            | 28.80                            | 69.90                            | 30.60                            | 57.67                            | 3.84                         | 3.92                         | 47.03                            | 78.18                   |
| LoRA                                         | ✓            | 79.71                            | 29.02                            | 84.46                            | 39.08                            | 69.76                            | 4.62                         | 4.61                         | 56.33                            | 93.63                   |
| GenieBlue-Post                               | ✓            | 78.64                            | 28.13                            | 85.37                            | 37.08                            | 70.77                            | 4.51                         | 4.65                         | 55.94                            | 92.98                   |
| GenieBlue-Pre                                | $\checkmark$ | 76.95                            | 29.24                            | 74.98                            | 35.66                            | 65.26                            | 4.61                         | 4.71                         | 53.61                            | 89.12                   |
| GenieBlue-Skip                               | ✓            | 75.36                            | 29.02                            | 76.27                            | 38.16                            | 67.78                            | 4.66                         | 4.76                         | 54.40                            | 90.42                   |
| GenieBlue                                    | Х            | 81.57                            | 29.46                            | 86.13                            | 38.94                            | 74.13                            | 5.67                         | 5.42                         | 60.16                            | 100.00                  |
|                                              |              |                                  |                                  |                                  |                                  |                                  |                              |                              |                                  |                         |
| Qwen2.5-3B                                   | Shared Base  | DROP                             | GPQA                             | GSM8K                            | MATH                             | MMLU                             | AlignBench                   | MT-bench                     | AVG                              | Retention (%)           |
| Qwen2.5-3B<br>Qwen2.5-3B                     | Shared Base  | <b>DROP</b> 70.82                | <b>GPQA</b> 30.30                | <b>GSM8K</b> 74.75               | <b>MATH</b> 61.74                | MMLU<br>66.31                    | AlignBench<br>6.00           | MT-bench<br>5.81             | <b>AVG</b> 60.29                 | Retention (%)           |
|                                              |              |                                  |                                  |                                  |                                  |                                  |                              |                              |                                  | Retention (%) - 85.33   |
| Qwen2.5-3B                                   |              | 70.82                            | 30.30                            | 74.75                            | 61.74                            | 66.31                            | 6.00                         | 5.81                         | 60.29                            | -                       |
| Qwen2.5-3B<br>Full-Finetune                  | -            | 70.82<br>71.45                   | 30.30<br>27.78                   | 74.75<br>69.37                   | 61.74<br>40.18                   | 66.31<br>64.34                   | 6.00                         | 5.81<br>4.34                 | 60.29<br>51.45                   | 85.33                   |
| Qwen2.5-3B Full-Finetune LoRA                | -<br>-<br>-  | 70.82<br>71.45<br>53.94          | 30.30<br>27.78<br>23.74          | 74.75<br>69.37<br>70.96          | 61.74<br>40.18<br>43.94          | 66.31<br>64.34<br>66.00          | 6.00<br>4.36<br>4.17         | 5.81<br>4.34<br>4.36         | 60.29<br>51.45<br>49.13          | 85.33<br>81.48          |
| Qwen2.5-3B Full-Finetune LoRA GenieBlue-Post | -            | 70.82<br>71.45<br>53.94<br>60.97 | 30.30<br>27.78<br>23.74<br>20.20 | 74.75<br>69.37<br>70.96<br>72.48 | 61.74<br>40.18<br>43.94<br>43.10 | 66.31<br>64.34<br>66.00<br>64.84 | 6.00<br>4.36<br>4.17<br>4.31 | 5.81<br>4.34<br>4.36<br>4.90 | 60.29<br>51.45<br>49.13<br>50.53 | 85.33<br>81.48<br>83.81 |

Table 6. Comparison of pure language capabilities using the shared base versus non-shared base deployment strategies, trained with 9M fine-tuning data. The non-shared base approach can maintain the pure text capabilities of the original LLM. In the shared-base strategy, training with BlueLM-3B indicates that the fewer trainable parameters involved in multimodal training, the better the retention of pure text capabilities. However, the LoRA-trained MLLM based on Qwen2.5-3B achieves the worst pure-text performance.

leverage the fully trained transformer blocks and incorporate the LoRA module. Additionally, we provide the NLP performances of BlueLM-3B/Qwen2.5-3B, the fully fine-tuned models, and the models trained entirely with LoRA. The results are shown in Tab. 6.

**Finding (6)** Deploying with the non-shared base strategy results in significantly better pure-text capabilities compared to the shared base strategy:

Undoubtedly, using the shared base deployment strategy leads to a loss of pure language capabilities, demonstrating the importance of the non-shared base deployment method. Another interesting finding is that, intuitively, with the same MLLM training data, having fewer trainable parameters results in less loss of the model's pure language performance. Training with BlueLM-3B aligns with this intuition. However, the LoRA-trained MLLM based on Qwen2.5-3B achieves the worst pure-text performance. A plausible explanation for this phenomenon lies in the inherent mechanism of LoRA, which imposes low-rank matrices onto original weights rather than directly training the base parameters. The limited number of adapter parameters may hinder effective integration with the pre-existing model parameters, resulting in suboptimal parameter fusion and consequently injuring the LLM performance.

### 4. Training and Deployment Recipe

After analyzing from both training data and model structure perspectives in Sec. 3, we determine the model structure (GenieBlue-Skip) and deployment approach (non-shared base deployment strategy). In this section, we introduce the detailed training (Sec. 4.1) and deployment details

(Sec. 4.2) of the final GenieBlue model.

### <span id="page-6-1"></span>4.1. Training Recipe

We employ the GenieBlue-Skip structure and strictly adhere to the training recipe and training data of BlueLM-V-3B [53]. Specifically, our training process consists of two stages. In the first stage, we pre-train the MLP projection layer while keeping the ViT and LLM frozen, using the 2.5M pre-training data. In the second stage, we fine-tune the GenieBlue-Skip model (ViT, projector, replicated transformer blocks, and the added LoRA parameters) with 645M fine-tuning data [53] while keeping the original LLM frozen. We use SigLIP as the ViT and BlueLM-3B as the LLM. During training, we set the LoRA rank to 8.

#### <span id="page-6-2"></span>4.2. Deployment Recipe

We deploy GenieBlue on the NPU of the iQOO 13 smartphone, which is equipped with the Qualcomm Snapdragon 8 Elite (Gen 4) SoC. We leverage the Qualcomm QNN SDK<sup>3</sup> for model deployment. For the ViT and projector layer, we employ W8A16 quantization. For the LLM, we adopt W4A16 quantization. Regarding the added LoRA parameters, we utilize a W8A16 quantization scheme. Currently, we support the single-patch ViT inference. It is important to note that the Snapdragon 8 Elite's NPU platform does not support the deployment of MoE structures.

#### 5. Performance of GenieBlue

Through extensive data training and NPU deployment, in this section, we evaluate the MLLM (Sec. 5.1) and LLM

<span id="page-6-3"></span> $<sup>^3 \\ \</sup>text{https://www.qualcomm.com/developer/software/} \\ \\ \text{neural-processing-sdk-for-ai}$ 

<span id="page-7-6"></span><span id="page-7-4"></span>

| Model                   | #Params     | AVG  | MMBench | MMStar | MMMU | MathVista | HallusionBench | AI2D | OCRBench | MMVet |
|-------------------------|-------------|------|---------|--------|------|-----------|----------------|------|----------|-------|
| BlueLM-V-3B [53]        | 3.2B        | 66.1 | 82.7    | 62.3   | 45.1 | 60.9      | 48.0           | 85.3 | 82.9     | 61.8  |
| Ovis2-2B [52]           | 2.46B       | 65.2 | 76.9    | 56.7   | 45.6 | 64.1      | 50.2           | 82.7 | 87.3     | 58.3  |
| Qwen2.5-VL-3B [7]       | 3.75B       | 64.5 | 76.8    | 56.3   | 51.2 | 61.2      | 46.6           | 81.4 | 82.8     | 60.0  |
| SAIL-VL-2B [24]         | 2.1B        | 61.0 | 73.7    | 56.5   | 44.1 | 62.8      | 45.9           | 77.4 | 83.1     | 44.2  |
| InternVL2.5-2B-MPO [72] | 2B          | 60.9 | 70.7    | 54.9   | 44.6 | 53.4      | 40.7           | 75.1 | 83.8     | 64.2  |
| GenieBlue               | 3.2(+0.55)B | 64.2 | 78.2    | 59.4   | 47.6 | 58.0      | 46.3           | 83.1 | 82.9     | 58.1  |
| InternVL2-8B [13]       | 8B          | 64.1 | 79.4    | 61.5   | 51.2 | 58.3      | 45.0           | 83.6 | 79.4     | 54.3  |

Table 7. Performance on MLLM benchmarks under the same evaluation settings as OpenCompass benchmark (≤ 4B, with InternVL2-8B for reference). GenieBlue retains over 97% accuracy of BlueLM-V-3B while outperforming InternVL2-8B on average. <sup>†</sup>The total number of parameters in the replicated transformer blocks and LoRA modules is 0.55B.

<span id="page-7-0"></span>

|              | #Params     | DROP  | GPQA  | GSM8K | MATH  | MMLU  | AlignBench | MT-bench | AVG   | Retention (%) |
|--------------|-------------|-------|-------|-------|-------|-------|------------|----------|-------|---------------|
| BlueLM-3B    | 2.7B        | 81.57 | 29.46 | 86.13 | 38.94 | 74.13 | 5.67       | 5.42     | 60.16 | -             |
| GenieBlue    | 3.2(+0.55)B | 81.57 | 29.46 | 86.13 | 38.94 | 74.13 | 5.67       | 5.42     | 60.16 | 100.00        |
| Qwen2.5-3B   | 3.1B        | 70.82 | 30.30 | 74.75 | 61.74 | 66.31 | 6.00       | 5.81     | 60.29 | -             |
| Qwen2.5VL-3B | 3.75B       | 72.72 | 24.24 | 70.43 | 58.92 | 65.07 | 5.38       | 4.72     | 56.05 | 92.98         |

Table 8. Evaluation results on representative LLM benchmarks, including both objective and subjective benchmarks. GenieBlue retains 100% performance of the original LLM, whereas Qwen2.5VL-3B exhibits some degradation.

<span id="page-7-5"></span>

| Model       | Context (token) | Load Time (s) | ViT Time (s) | Input Speed (token/s) | Output Speed (token/s) | Storage (GB) | Memory (GB) |
|-------------|-----------------|---------------|--------------|-----------------------|------------------------|--------------|-------------|
| BlueLM-V-3B | 2048            | 0.51          | 0.4          | 1515.15               | 33.00                  | 1.77         | 1.73        |
| GenieBlue   | 2048            | 0.80          | 0.4          | 1666.67               | 31.00                  | 1.92         | 2.10        |

Table 9. Deployment efficiency comparison between GenieBlue and BlueLM-V-3B on Qualcomm 8 Elite SoC in peak performance mode. GenieBlue results in a longer model loading time, slightly higher storage and memory usage, and a marginally slower token output speed.

(Sec. 5.2) capabilities of GenieBlue, as well as its deployment efficiency on smartphone NPUs (Sec. 5.3).

#### <span id="page-7-1"></span>**5.1. MLLM Performance**

After extensive data training, we evaluate our model using representative MLLM benchmarks, including MMbench [48], MMStar [10], MMMU [85], MathVista [51], HallusionBench [27], AI2D [34], OCRBench [47], and MM-Vet [84], which are integrated into the OpenCompass benchmark suite [20]. We compare GenieBlue with other MLLMs that have fewer than 4B parameters, and the results are presented in Tab. 7. GenieBlue achieves MLLM accuracy slightly lower than Qwen2.5-VL-3B while retaining 97% performance of BlueLM-V-3B. Besides, GenieBlue slightly outperforms InternVL2-8B on average.

#### <span id="page-7-2"></span>**5.2. LLM Performance**

The most significant feature of GenieBlue is that it does not lose LLM performance when deployed using the non-shared base deployment strategy. Here, we evaluate its LLM performance on representative benchmarks. For comparison, we select Qwen2.5VL-3B, which claims to maintain LLM performance without degradation from MLLM training by incorporating pure-text data. As demonstrated in Tab. 8, GenieBlue achieves no loss in LLM performance, while Qwen2.5VL-3B exhibits some performance degradation, especially in subjective tasks. This indicates that exploring model structure design is more effective for maintaining pure-text capabilities than simply increasing the amount of pure-text data currently.

### <span id="page-7-3"></span>5.3. Deployment Efficiency

We deploy GenieBlue with the non-shared base strategy on Qualcomm Snapdragon 8 Elite (Gen 4) SoC. Different from [53], we now support the 1-patch ViT inference. We here provide the MLLM deployment statistics in Tab. 9, comparing BlueLM-V-3B and GenieBlue. With the inclusion of additional LoRA parameters, GenieBlue incurs longer model loading times, slightly larger storage and memory requirements, and a marginally slower token output speed. However, a token output speed of 30 token/s is fully sufficient for daily use on mobile devices.

#### 6. Conclusion

In this paper, we approach the challenge of maintaining pure language capabilities from a practical deployment perspective on mobile devices (smartphones), analyzing both training data and model structure to identify effective strategies. Based on the analyses, we propose GenieBlue, an efficient and hardware-friendly MLLM design that integrates linguistic and multimodal capabilities for mobile LLMs. By freezing the original LLM parameters during training and acquiring multimodal capabilities through duplicated transformer blocks and lightweight LoRA modules, GenieBlue maintains language performance while achieving competitive multimodal results. Deployed on smartphone NPUs, GenieBlue demonstrates its practicality and efficiency, making it a promising solution for edge computing applications on mobile devices. We hope that our work will provide valuable insights for future research in this field.

## References

- <span id="page-8-0"></span>[1] Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. *arXiv preprint arXiv:2404.14219*, 2024. [1](#page-0-2)
- <span id="page-8-13"></span>[2] Aida Amini, Saadia Gabriel, Shanchuan Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. MathQA: Towards interpretable math word problem solving with operation-based formalisms. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 2357– 2367, Minneapolis, Minnesota, 2019. Association for Computational Linguistics. [3](#page-2-2)
- <span id="page-8-1"></span>[3] Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: A family of highly capable multimodal models. *arXiv preprint arXiv:2312.11805*, 1, 2023. [1](#page-0-2)
- <span id="page-8-2"></span>[4] Anthropic. Claude 3. <https://www.anthropic.com>, 2023. Large Language Model. [1](#page-0-2)
- <span id="page-8-16"></span>[5] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. *arXiv preprint arXiv:2108.07732*, 2021. [3](#page-2-2)
- <span id="page-8-19"></span>[6] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, et al. Openflamingo: An opensource framework for training large autoregressive visionlanguage models. *arXiv preprint arXiv:2308.01390*, 2023. [5](#page-4-3)
- <span id="page-8-3"></span>[7] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. *arXiv preprint arXiv:2502.13923*, 2025. [1,](#page-0-2) [2,](#page-1-0) [3,](#page-2-2) [4,](#page-3-5) [8](#page-7-6)
- <span id="page-8-7"></span>[8] Maosong Cao, Taolin Zhang, Mo Li, Chuyu Zhang, Yunxin Liu, Haodong Duan, Songyang Zhang, and Kai Chen. Condor: Enhance llm alignment with knowledge-driven data synthesis and refinement. *arXiv preprint arXiv:2501.12273*, 2025. [2,](#page-1-0) [4](#page-3-5)
- <span id="page-8-11"></span>[9] CarperAI. openai summarize tldr dataset. [https://](https://huggingface.co/datasets/CarperAI/openai_summarize_tldr) [huggingface.co/datasets/CarperAI/openai\\_](https://huggingface.co/datasets/CarperAI/openai_summarize_tldr) [summarize\\_tldr](https://huggingface.co/datasets/CarperAI/openai_summarize_tldr), 2023. [3](#page-2-2)
- <span id="page-8-21"></span>[10] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? *arXiv preprint arXiv:2403.20330*, 2024. [8](#page-7-6)
- <span id="page-8-17"></span>[11] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evalu-

- ating large language models trained on code. *arXiv preprint arXiv:2107.03374*, 2021. [3](#page-2-2)
- <span id="page-8-4"></span>[12] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and testtime scaling. *arXiv preprint arXiv:2412.05271*, 2024. [1,](#page-0-2) [2,](#page-1-0) [3,](#page-2-2) [4](#page-3-5)
- <span id="page-8-20"></span>[13] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. *arXiv preprint arXiv:2404.16821*, 2024. [8](#page-7-6)
- <span id="page-8-5"></span>[14] Xiangxiang Chu, Limeng Qiao, Xinyang Lin, Shuang Xu, Yang Yang, Yiming Hu, Fei Wei, Xinyu Zhang, Bo Zhang, Xiaolin Wei, et al. Mobilevlm: A fast, reproducible and strong vision language assistant for mobile devices. *arXiv preprint arXiv:2312.16886*, 2023. [1,](#page-0-2) [2](#page-1-0)
- <span id="page-8-6"></span>[15] Xiangxiang Chu, Limeng Qiao, Xinyu Zhang, Shuang Xu, Fei Wei, Yang Yang, Xiaofei Sun, Yiming Hu, Xinyang Lin, Bo Zhang, et al. Mobilevlm v2: Faster and stronger baseline for vision language model. *arXiv preprint arXiv:2402.03766*, 2024. [1,](#page-0-2) [2](#page-1-0)
- <span id="page-8-14"></span>[16] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. *arXiv:1803.05457v1*, 2018. [3](#page-2-2)
- <span id="page-8-12"></span>[17] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*, 2021. [3,](#page-2-2) [4](#page-3-5)
- <span id="page-8-15"></span>[18] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*, 2021. [3](#page-2-2)
- <span id="page-8-10"></span>[19] Mike Conover, Matt Hayes, Ankit Mathur, Jianwei Xie, Jun Wan, Sam Shah, Ali Ghodsi, Patrick Wendell, Matei Zaharia, and Reynold Xin. Free dolly: Introducing the world's first truly open instruction-tuned llm, 2023. [3](#page-2-2)
- <span id="page-8-22"></span>[20] OpenCompass Contributors. OpenCompass: A universal evaluation platform for foundation models. [https://](https://github.com/open-compass/opencompass) [github.com/open-compass/opencompass](https://github.com/open-compass/opencompass), 2023. [8](#page-7-6)
- <span id="page-8-18"></span>[21] X.AI Corp. Grok-1.5 vision preview: Connecting the digital and physical worlds with our first multimodal model. <https://x.ai/blog/grok-1.5v>, 2024. [4](#page-3-5)
- <span id="page-8-8"></span>[22] Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, and Maosong Sun. Ultrafeedback: Boosting language models with high-quality feedback. *arXiv preprint arXiv:2310.01377*, 2023. [3](#page-2-2)
- <span id="page-8-9"></span>[23] Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations. *arXiv preprint arXiv:2305.14233*, 2023. [3](#page-2-2)

- <span id="page-9-19"></span>[24] Hongyuan Dong, Zijian Kang, Weijie Yin, Xiao Liang, Chao Feng, and Jiao Ran. Scalable vision language model training via high quality data curation. *arXiv preprint arXiv:2501.05952*, 2025. [8](#page-7-6)
- <span id="page-9-18"></span>[25] Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. Drop: A reading comprehension benchmark requiring discrete reasoning over paragraphs. *arXiv preprint arXiv:1903.00161*, 2019. [4](#page-3-5)
- <span id="page-9-9"></span>[26] GlaiveAI. Glaive code assistant v3 dataset. [https://](https://huggingface.co/datasets/glaiveai/glaive-code-assistant-v3) [huggingface.co/datasets/glaiveai/glaive](https://huggingface.co/datasets/glaiveai/glaive-code-assistant-v3)[code-assistant-v3](https://huggingface.co/datasets/glaiveai/glaive-code-assistant-v3), 2024. [3](#page-2-2)
- <span id="page-9-20"></span>[27] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. *arXiv preprint arXiv:2310.14566*, 2023. [8](#page-7-6)
- <span id="page-9-0"></span>[28] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025. [1,](#page-0-2) [2](#page-1-0)
- <span id="page-9-13"></span>[29] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. *arXiv preprint arXiv:2009.03300*, 2020. [3,](#page-2-2) [4](#page-3-5)
- <span id="page-9-2"></span>[30] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. *arXiv preprint arXiv:2103.03874*, 2021. [2,](#page-1-0) [3,](#page-2-2) [4](#page-3-5)
- <span id="page-9-4"></span>[31] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. *arXiv preprint arXiv:2106.09685*, 2021. [2](#page-1-0)
- <span id="page-9-6"></span>[32] Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. MiniCPM: Unveiling the potential of small language models with scalable training strategies. *arXiv preprint arXiv:2404.06395*, 2024. [2,](#page-1-0) [3](#page-2-2)
- <span id="page-9-1"></span>[33] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. *arXiv preprint arXiv:2401.04088*, 2024. [1](#page-0-2)
- <span id="page-9-16"></span>[34] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In *Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14*, pages 235– 251. Springer, 2016. [4,](#page-3-5) [8](#page-7-6)
- <span id="page-9-8"></span>[35] knowrohit07. know saraswati cot dataset. [https :](https://huggingface.co/datasets/knowrohit07/know-saraswati-cot) [/ / huggingface . co / datasets / knowrohit07 /](https://huggingface.co/datasets/knowrohit07/know-saraswati-cot) [know-saraswati-cot](https://huggingface.co/datasets/knowrohit07/know-saraswati-cot), 2023. [3](#page-2-2)
- <span id="page-9-12"></span>[36] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. *arXiv preprint arXiv:2408.03326*, 2024. [3](#page-2-2)

- <span id="page-9-10"></span>[37] Jia LI, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu. Numinamath tir. [\[https:]([https://huggingface.co/AI-MO/NuminaMath-TIR](https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf)) [//huggingface.co/AI- MO/NuminaMath- TIR\]]([https://huggingface.co/AI-MO/NuminaMath-TIR](https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf)) [\(https://github.com/project-numina/aimo]([https://huggingface.co/AI-MO/NuminaMath-TIR](https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf))[progress-prize/blob/main/report/numina\\_]([https://huggingface.co/AI-MO/NuminaMath-TIR](https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf)) [dataset.pdf\)]([https://huggingface.co/AI-MO/NuminaMath-TIR](https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf)), 2024. [3](#page-2-2)
- <span id="page-9-11"></span>[38] Jia LI, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu. Numinamath. [\[https :]([https://huggingface.co/AI-MO/NuminaMath-CoT](https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf)) [//huggingface.co/AI- MO/NuminaMath- CoT\]]([https://huggingface.co/AI-MO/NuminaMath-CoT](https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf)) [\(https://github.com/project-numina/aimo]([https://huggingface.co/AI-MO/NuminaMath-CoT](https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf))[progress-prize/blob/main/report/numina\\_]([https://huggingface.co/AI-MO/NuminaMath-CoT](https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf)) [dataset.pdf\)]([https://huggingface.co/AI-MO/NuminaMath-CoT](https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf)), 2024. [3](#page-2-2)
- <span id="page-9-14"></span>[39] Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E Gonzalez, and Ion Stoica. From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder pipeline. *arXiv preprint arXiv:2406.11939*, 2024. [3](#page-2-2)
- [40] Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Banghua Zhu, Joseph E. Gonzalez, and Ion Stoica. From live data to high-quality benchmarks: The arena-hard pipeline, 2024.
- <span id="page-9-15"></span>[41] Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Alpacaeval: An automatic evaluator of instruction-following models. [https://github.com/](https://github.com/tatsu-lab/alpaca_eval) [tatsu-lab/alpaca\\_eval](https://github.com/tatsu-lab/alpaca_eval), 2023. [3](#page-2-2)
- <span id="page-9-7"></span>[42] Wing Lian, Guan Wang, Bleys Goodson, Eugene Pentland, Austin Cook, Chanvichet Vong, and "Teknium". Slimorca: An open dataset of gpt-4 augmented flan reasoning traces, with verification, 2023. [3](#page-2-2)
- <span id="page-9-22"></span>[43] Ji Lin, Hongxu Yin, Wei Ping, Yao Lu, Pavlo Molchanov, Andrew Tao, Huizi Mao, Jan Kautz, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models, 2023. [13](#page-12-0)
- <span id="page-9-5"></span>[44] Dongyang Liu, Renrui Zhang, Longtian Qiu, Siyuan Huang, Weifeng Lin, Shitian Zhao, Shijie Geng, Ziyi Lin, Peng Jin, Kaipeng Zhang, et al. Sphinx-x: Scaling data and parameters for a family of multi-modal large language models. *arXiv preprint arXiv:2402.05935*, 2024. [2](#page-1-0)
- <span id="page-9-21"></span>[45] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. *NeurIPS*, 36, 2024. [13](#page-12-0)
- <span id="page-9-3"></span>[46] Xiao Liu, Xuanyu Lei, Shengyuan Wang, Yue Huang, Zhuoer Feng, Bosi Wen, Jiale Cheng, Pei Ke, Yifan Xu, Weng Lam Tam, et al. Alignbench: Benchmarking chinese alignment of large language models. *arXiv preprint arXiv:2311.18743*, 2023. [2,](#page-1-0) [3,](#page-2-2) [4](#page-3-5)
- <span id="page-9-17"></span>[47] Yuliang Liu, Zhang Li, Hongliang Li, Wenwen Yu, Mingxin Huang, Dezhi Peng, Mingyu Liu, Mingrui Chen, Chunyuan Li, Lianwen Jin, et al. On the hidden mystery of OCR in large multimodal models. *arXiv preprint arXiv:2305.07895*, 2023. [4,](#page-3-5) [8](#page-7-6)

- <span id="page-10-24"></span>[48] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In *European Conference on Computer Vision*, pages 216–233. Springer, 2025. [8](#page-7-6)
- <span id="page-10-1"></span>[49] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Yaofeng Sun, et al. DeepSeek-VL: Towards real-world vision-language understanding. *arXiv preprint arXiv:2403.05525*, 2024. [1,](#page-0-2) [3](#page-2-2)
- <span id="page-10-20"></span>[50] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. *Advances in Neural Information Processing Systems*, 35:2507–2521, 2022. [4](#page-3-5)
- <span id="page-10-25"></span>[51] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. MathVista: Evaluating mathematical reasoning of foundation models in visual contexts. *arXiv preprint arXiv:2310.02255*, 2023. [8](#page-7-6)
- <span id="page-10-16"></span>[52] Shiyin Lu, Yang Li, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, and Han-Jia Ye. Ovis: Structural embedding alignment for multimodal large language model. *arXiv:2405.20797*, 2024. [3,](#page-2-2) [8](#page-7-6)
- <span id="page-10-4"></span>[53] Xudong Lu, Yinghao Chen, Cheng Chen, Hui Tan, Boheng Chen, Yina Xie, Rui Hu, Guanxin Tan, Renshou Wu, Yan Hu, et al. Bluelm-v-3b: Algorithm and system co-design for multimodal large language models on mobile devices. *arXiv preprint arXiv:2411.10640*, 2024. [1,](#page-0-2) [2,](#page-1-0) [3,](#page-2-2) [4,](#page-3-5) [7,](#page-6-4) [8,](#page-7-6) [13,](#page-12-0) [14](#page-13-0)
- <span id="page-10-10"></span>[54] Zhenyan Lu, Xiang Li, Dongqi Cai, Rongjie Yi, Fangming Liu, Xiwen Zhang, Nicholas D Lane, and Mengwei Xu. Small language models: Survey, measurements, and insights. *arXiv preprint arXiv:2409.15790*, 2024. [3](#page-2-2)
- <span id="page-10-12"></span>[55] Gen Luo, Xue Yang, Wenhan Dou, Zhaokai Wang, Jifeng Dai, Yu Qiao, and Xizhou Zhu. Mono-internvl: Pushing the boundaries of monolithic multimodal large language models with endogenous visual pre-training. *arXiv preprint arXiv:2410.08202*, 2024. [3](#page-2-2)
- <span id="page-10-9"></span>[56] Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. Wizardcoder: Empowering code large language models with evol-instruct. *arXiv preprint arXiv:2306.08568*, 2023. [3](#page-2-2)
- <span id="page-10-18"></span>[57] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. *arXiv preprint arXiv:2203.10244*, 2022. [4](#page-3-5)
- <span id="page-10-19"></span>[58] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. DocVQA: A dataset for VQA on document images. In *WACV*, pages 2200–2209, 2021. [4](#page-3-5)
- <span id="page-10-2"></span>[59] OpenAI. Hello GPT-4o, 2024. [1](#page-0-2)
- <span id="page-10-15"></span>[60] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. *NeurIPS*, 36, 2024. [3](#page-2-2)
- <span id="page-10-7"></span>[61] Nazneen Rajani, Lewis Tunstall, Edward Beeching, Nathan Lambert, Alexander M. Rush, and Thomas Wolf. No

- robots. [https://huggingface.co/datasets/](https://huggingface.co/datasets/HuggingFaceH4/no_robots) [HuggingFaceH4/no\\_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots), 2023. [3](#page-2-2)
- <span id="page-10-17"></span>[62] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level googleproof q&a benchmark. *arXiv preprint arXiv:2311.12022*, 2023. [3](#page-2-2)
- <span id="page-10-22"></span>[63] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level googleproof q&a benchmark. In *First Conference on Language Modeling*, 2024. [4](#page-3-5)
- <span id="page-10-21"></span>[64] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 8317–8326, 2019. [4](#page-3-5)
- <span id="page-10-14"></span>[65] Yixin Song, Zeyu Mi, Haotong Xie, and Haibo Chen. Powerinfer: Fast large language model serving with a consumergrade gpu. *arXiv preprint arXiv:2312.12456*, 2023. [3](#page-2-2)
- <span id="page-10-3"></span>[66] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. *arXiv preprint arXiv:2403.05530*, 2024. [1,](#page-0-2) [2](#page-1-0)
- <span id="page-10-13"></span>[67] InternLM Team. Internlm: A multilingual language model with progressively enhanced capabilities. [https : / /](https://github.com/InternLM/InternLM-techreport) [github.com/InternLM/InternLM-techreport](https://github.com/InternLM/InternLM-techreport), 2023. [3](#page-2-2)
- <span id="page-10-0"></span>[68] InternLM Team. Internlm: A multilingual language model with progressively enhanced capabilities, 2023. [1](#page-0-2)
- <span id="page-10-6"></span>[69] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. *arXiv preprint arXiv:2406.16860*, 2024. [2,](#page-1-0) [4,](#page-3-5) [13](#page-12-0)
- <span id="page-10-11"></span>[70] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution. *arXiv preprint arXiv:2409.12191*, 2024. [3](#page-2-2)
- <span id="page-10-5"></span>[71] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. CogVLM: Visual expert for pretrained language models. *arXiv preprint arXiv:2311.03079*, 2023. [2,](#page-1-0) [3,](#page-2-2) [4,](#page-3-5) [5](#page-4-3)
- <span id="page-10-23"></span>[72] Weiyun Wang, Zhe Chen, Wenhai Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Jinguo Zhu, Xizhou Zhu, Lewei Lu, Yu Qiao, and Jifeng Dai. Enhancing the reasoning ability of multimodal large language models via mixed preference optimization. *arXiv preprint arXiv:2411.10442*, 2024. [8](#page-7-6)
- <span id="page-10-8"></span>[73] Yejie Wang, Keqing He, Dayuan Fu, Zhuoma Gongque, Heyang Xu, Yanxu Chen, Zhexu Wang, Yujia Fu, Guanting Dong, Muxi Diao, et al. How do your code llms perform? empowering code instruction tuning with high-quality data. *arXiv preprint arXiv:2409.03810*, 2024. [3](#page-2-2)

- <span id="page-11-16"></span>[74] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. *arXiv preprint arXiv:2406.01574*, 2024. [3](#page-2-2)
- <span id="page-11-15"></span>[75] Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao, Yiyang Ma, Chengyue Wu, Bingxuan Wang, et al. Deepseek-vl2: Mixture-ofexperts vision-language models for advanced multimodal understanding. *arXiv preprint arXiv:2412.10302*, 2024. [3](#page-2-2)
- <span id="page-11-7"></span>[76] Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, Qingwei Lin, and Daxin Jiang. Wizardlm: Empowering large pre-trained language models to follow complex instructions. In *The Twelfth International Conference on Learning Representations*, 2024. [3](#page-2-2)
- <span id="page-11-8"></span>[77] Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, and Bill Yuchen Lin. Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing. *arXiv preprint arXiv:2406.08464*, 2024. [3](#page-2-2)
- <span id="page-11-1"></span>[78] Zhenliang Xue, Yixin Song, Zeyu Mi, Le Chen, Yubin Xia, and Haibo Chen. Powerinfer-2: Fast large language model inference on a smartphone. *arXiv preprint arXiv:2406.06282*, 2024. [1,](#page-0-2) [3](#page-2-2)
- <span id="page-11-14"></span>[79] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report, 2024. [3](#page-2-2)
- <span id="page-11-0"></span>[80] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. *arXiv preprint arXiv:2412.15115*, 2024. [1,](#page-0-2) [2,](#page-1-0) [3,](#page-2-2) [4](#page-3-5)
- <span id="page-11-9"></span>[81] Jianxin Yang. Firefly: A chinese conversational large language model. [https : / / github . com /](https://github.com/yangjianxin1/Firefly) [yangjianxin1/Firefly](https://github.com/yangjianxin1/Firefly), 2023. [3](#page-2-2)
- <span id="page-11-2"></span>[82] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. *arXiv preprint arXiv:2408.01800*, 2024. [1,](#page-0-2) [2,](#page-1-0) [3](#page-2-2)
- <span id="page-11-12"></span>[83] Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. *arXiv preprint arXiv:2309.12284*, 2023. [3](#page-2-2)
- <span id="page-11-19"></span>[84] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang.

- Mm-vet: Evaluating large multimodal models for integrated capabilities. *arXiv preprint arXiv:2308.02490*, 2023. [8](#page-7-6)
- <span id="page-11-18"></span>[85] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 9556– 9567, 2024. [8](#page-7-6)
- <span id="page-11-13"></span>[86] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In *Proceedings of ICCV*, pages 11975–11986, 2023. [3,](#page-2-2) [4](#page-3-5)
- <span id="page-11-11"></span>[87] Bo-Wen Zhang, Yan Yan, Lin Li, and Guang Liu. Infinitymath: A scalable instruction tuning dataset in programmatic mathematical reasoning, 2024. [3](#page-2-2)
- <span id="page-11-3"></span>[88] Yi-Kai Zhang, Shiyin Lu, Yang Li, Yanqing Ma, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, De-Chuan Zhan, and Han-Jia Ye. Wings: Learning multimodal llms without text-only forgetting. *arXiv preprint arXiv:2406.03496*, 2024. [2,](#page-1-0) [3,](#page-2-2) [4](#page-3-5)
- <span id="page-11-5"></span>[89] Xiangyu Zhao, Shengyuan Ding, Zicheng Zhang, Haian Huang, Maosong Cao, Weiyun Wang, Jiaqi Wang, Xinyu Fang, Wenhai Wang, Guangtao Zhai, et al. Omnialign-v: Towards enhanced alignment of mllms with human preference. *arXiv preprint arXiv:2502.18411*, 2025. [2](#page-1-0)
- <span id="page-11-17"></span>[90] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. *Advances in Neural Information Processing Systems*, 36:46595–46623, 2023. [3](#page-2-2)
- <span id="page-11-4"></span>[91] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. *Advances in Neural Information Processing Systems*, 36, 2024. [2,](#page-1-0) [4](#page-3-5)
- <span id="page-11-10"></span>[92] Tianyu Zheng, Ge Zhang, Tianhao Shen, Xueling Liu, Bill Yuchen Lin, Jie Fu, Wenhu Chen, and Xiang Yue. Opencodeinterpreter: Integrating code generation with execution and refinement. *arXiv preprint arXiv:2402.14658*, 2024. [3](#page-2-2)
- <span id="page-11-6"></span>[93] Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: Less is more for alignment. *Advances in Neural Information Processing Systems*, 36:55006–55021, 2023. [3](#page-2-2)

<span id="page-12-1"></span><span id="page-12-0"></span>![](_page_12_Figure_0.jpeg)

Figure 3. Structure detail of GenieBlue during the MLLM inference process.

## A. Structure Details of GenieBlue

We here provide the detailed structure of GenieBlue during the MLLM inference process based on the BlueLM-V-3B [\[53\]](#page-10-4) architecture (Fig. [3\)](#page-12-1). BlueLM-V-3B is modified from the classical LLaVA approach [\[45\]](#page-9-21), incorporating a redesigned dynamic resolution processor and a token downsampler [\[43\]](#page-9-22) to optimize for better on-device deployment. GenieBlue further focuses on the structural design of the transformer blocks within the language model.

## B. Training Data Composition

We here provide the data composition of Cambrian-7M [\[69\]](#page-10-6). It has already included approximately 1.5M pure text training samples. The data composition of the 645M fine-tuning data for GenieBlue can be found in [\[53\]](#page-10-4).

| Type  | OCR       | General | Language | Counting | Code | Math | Science |
|-------|-----------|---------|----------|----------|------|------|---------|
| Ratio | (%) 27.22 | 34.52   | 21.00    | 8.71     | 0.87 | 7.20 | 0.88    |

Table 10. Data composition of the Cambrian-7M [\[69\]](#page-10-6) fine-tuning dataset (with approximately 1.5M pure-text data).

# C. More Discussions

GenieBlue is a plug-and-play training approach that efficiently decouples multimodal training parameters from the original language model. This design allows GenieBlue to achieve good multimodal performance without compromising the language model's performance. In addition, this structural design requires minimal hardware-side adaptation and reduces the engineering difficulty during practical end-side deployment, making it a relatively reasonable approach at the current stage. In the future, we will validate the feasibility of GenieBlue on a wider range of SoC platforms.

## <span id="page-13-0"></span>D. Hyper Parameters

Here, we provide the hyper-parameters used in the pre-training and fine-tuning stage of the final GenieBlue model. We use the same 2.5M pre-training data and 645M fine-tuning data as in BlueLM-V-3B [\[53\]](#page-10-4).

### D.1. Pre-training Stage

| Configuration         | Stage 1                             |
|-----------------------|-------------------------------------|
| LLM Sequence Length   | 4096                                |
| Dynamic Resolution    | None (384×384)                      |
| Optimizer             | AdamW                               |
| Optimizer Hyperparams | = 0.98, ϵ = 10−6<br>β1<br>= 0.9, β2 |
| Peak LR               | 10−3                                |
| LR Schedule           | Cosine Decay                        |
| Weight Decay          | 0.05                                |
| Training Steps        | 3.434k                              |
| Warm-up Steps         | 34                                  |
| Global Batch Size     | 720                                 |
| Gradient Accumulation | 1                                   |
| Numerical Precision   | bfloat16                            |

Table 11. Hyper-parameters for the pre-training stage (stage 1) of GenieBlue with 2.5M training samples.

### D.2. Fine-tuning Stage

In the process of fine-tuning, to enhance the speed of training, we concatenate training samples to achieve a sequence length of 4096.

| Configuration           | Stage 2                             |
|-------------------------|-------------------------------------|
| LLM Sequence Length     | 4096                                |
| Dynamic Resolution      | Up to 16 patches (1536×1536)        |
| Optimizer               | AdamW                               |
| Optimizer Hyperparams   | = 0.98, ϵ = 10−6<br>β1<br>= 0.9, β2 |
| Peak LR                 | 10−4                                |
| LR Schedule             | Cosine Decay                        |
| Weight Decay            | 0.05                                |
| ViT Layer-wise LR Decay | 0.9                                 |
| Training Steps          | 53k                                 |
| Warm-up Steps           | 530                                 |
| Global Batch Size       | 6800                                |
| Gradient Accumulation   | 10                                  |
| Numerical Precision     | bfloat16                            |

Table 12. Hyper-parameters for the fine-tuning stage (stage 2) of GenieBlue with 645M training samples.