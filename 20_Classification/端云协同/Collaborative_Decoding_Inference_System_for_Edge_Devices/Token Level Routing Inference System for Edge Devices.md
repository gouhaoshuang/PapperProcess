---
category: 端云协同
classification_reason: 论文明确提出了一种'协同解码'（collaborative decoding）推理系统，该系统让小模型在端侧运行，同时选择性地咨询云端的大模型来生成关键Token。这种架构通过联合利用端侧资源和云端算力来平衡推理速度与质量，完全符合'端云协同'的技术定义。
created: '2026-01-18'
status: unread
tags:
- 端云协同
- 协同解码
- 混合推理
- 关键Token识别
- 计算卸载
title: Collaborative Decoding Inference System for Edge Devices
---

# Token Level Routing Inference System for Edge Devices\*

Jianshu She<sup>1</sup>† , Wenhao Zheng<sup>2</sup> , Zhengzhong Liu<sup>1</sup> , Hongyi Wang<sup>3</sup> , Eric Xing<sup>1</sup> , Huaxiu Yao<sup>2</sup> , Qirong Ho<sup>1</sup>

<sup>1</sup>Mohamed bin Zayed University of Artificial Intelligence (MBZUAI) <sup>2</sup>University of North Carolina at Chapel Hill <sup>3</sup>Computer Science Department at Rutgers University

# Abstract

The computational complexity of large language model (LLM) inference significantly constrains their deployment efficiency on edge devices. In contrast, small language models offer faster decoding and lower resource consumption but often suffer from degraded response quality and heightened susceptibility to hallucinations. To address this trade-off, collaborative decoding, in which a large model assists in generating critical tokens, has emerged as a promising solution. This paradigm leverages the strengths of both model types by enabling high-quality inference through selective intervention of the large model, while maintaining the speed and efficiency of the smaller model. In this work, we present a novel collaborative decoding inference system that allows small models to perform on-device inference while selectively consulting a cloud-based large model for critical token generation. Remarkably, the system achieves a 60% performance gain on CommonsenseQA using only a 0.5B model on an M1 MacBook, with under 7% of tokens generation uploaded to the large model in the cloud.

# 1 Introduction

Large language models (LLMs) have transformed natural language processing, achieving state-of-theart performance in tasks such as document summarization, question answering, and text generation. Models like Meta's Llama series [\(Touvron et al.,](#page-7-0) [2023\)](#page-7-0), Google's Gemma [\(Team et al.,](#page-7-1) [2024\)](#page-7-1), and DeepSeek series [\(DeepSeek-AI et al.,](#page-6-0) [2025\)](#page-6-0) have demonstrated remarkable capabilities, driving advancements in various applications. However, their deployment in edge devices, such as smartphones, embedded systems, and Internet of Things (IoT) devices, faces significant hurdles due to their high

computational complexity [\(Zhang et al.,](#page-7-2) [2024a;](#page-7-2) [Lin](#page-6-1) [et al.,](#page-6-1) [2024\)](#page-6-1). The role of small language models (SLMs), and the emerging paradigm of collaborative decoding, culminating in a novel framework that balances efficiency and performance.

The computational demands of LLMs, such as the Llama-2 7B parameter model requiring over 8GB of memory in FP16 precision [\(Zhang et al.,](#page-7-2) [2024a\)](#page-7-2) , exceed the capabilities of many edge devices, like the NVIDIA Jetson Orin Nano with 8GB DRAM [\(Shen et al.,](#page-6-2) [2024a;](#page-6-2) [Li et al.,](#page-6-3) [2025\)](#page-6-3). This limitation is compounded by hardware heterogeneity, including ARM processors in smartphones and low-power IoT chips, which further complicates deployment [\(Dao et al.,](#page-6-4) [2022\)](#page-6-4). Recent works, such as [Zheng et al.](#page-7-3) [\(2025b\)](#page-7-3), highlight the need for solutions that can operate within the constraints of memory, processing power, and energy consumption [\(Miao et al.,](#page-6-5) [2024\)](#page-6-5).

One promising approach to leveraging small language models (SLMs) lies in their potential for edge deployment, thanks to their reduced size and faster inference times[\(Xue et al.,](#page-7-4) [2024;](#page-7-4) [Jiang](#page-6-6) [et al.,](#page-6-6) [2023;](#page-6-6) [Zhou et al.,](#page-7-5) [2024\)](#page-7-5). These models consume fewer resources, making them suitable for devices with limited capabilities. However, studies, such as Wang et al.'s work on large and small model trade-offs [\(Zheng et al.,](#page-7-3) [2025b\)](#page-7-3), indicate that SLMs often suffer from degraded response quality and increased susceptibility to hallucinations—generating factually incorrect content [\(Xu](#page-7-6) [et al.,](#page-7-6) [2023\)](#page-7-6). This trade-off between efficiency and performance presents a critical barrier, particularly for applications requiring high accuracy, such as medical data analysis or financial processing [\(Wang](#page-7-7) [et al.,](#page-7-7) [2024\)](#page-7-7).

To mitigate this trade-off, numerous studies have introduced approaches that dynamically route input queries to models of varying sizes, aiming to lower inference costs without compromising output quality [\(Kou et al.,](#page-6-7) [2024;](#page-6-7) [Anagnostidis et al.,](#page-6-8) [2024\)](#page-6-8).

<sup>\*</sup>Demo package available at [https://github.com/](https://github.com/Jianshu1only/Token-Routing) [Jianshu1only/Token-Routing](https://github.com/Jianshu1only/Token-Routing)

<sup>†</sup>Email: jianshu.she@mbzuai.ac.ae

<span id="page-1-0"></span>![](_page_1_Figure_0.jpeg)

Figure 1: System overview: First transfer Huggingface model to ONNX model, then add hidden states of last layer as a output node in ONNX computation graph, deploy ONNX model on Laptop and ONNX-mobile on Mobile phone. Then connect edge divice with router to the SG-Lang backend from server side. The router automatically route token with low confidence to server, and send response back to edge device

Collaborative decoding has emerged as a promising approach [\(Shen et al.,](#page-6-9) [2024b;](#page-6-9) [Shi et al.,](#page-7-8) [2024\)](#page-7-8). This paradigm involves SLMs handling the bulk of the inference process while LLMs assist in generating critical tokens, such as those with high uncertainty or decisive impact on the output. Research suggests that this method leverages the strengths of both model types, maintaining efficiency while enhancing quality. For instance, Wang et al.'s study on Fast and Slow Generating (FS-GEN) [\(Zhang](#page-7-9) [et al.,](#page-7-9) [2024b\)](#page-7-9) categorizes LLMs as System 2 (slow and deliberate) and SLMs as System 1 (fast and intuitive), finding that collaborative interactions require less than 20% of the computations, following scaling laws based on parameter ratios.

Building on these insights, we introduce a novel token-level routing inference system for edge devices, addressing the challenge of balancing efficiency and performance in resource-constrained settings. The system enables on-device SLMs to perform primary decoding while selectively routing critical tokens to a cloud-based LLM using a lightweight, confidence-based MLP router (See Figure [1](#page-1-0) for details). Empirical results on CommonsenseQA demonstrate that routing only 7% of tokens to the LLM yields over 60% accuracy improvement, with more than 80% cost reduction compared to full LLM inference. This system paves the way for practical, low-latency, highquality language model applications on edge hardware, as it mitigates the traditional trade-off between model size and performance, opening new possibilities for deploying high-quality language models in resource-constrained environments. For

example, in privacy-sensitive scenarios like medical data analysis, on-device inference reduces data transmission, protecting user data, while cloudbased LLM assistance ensures accuracy.

Unlike prior works which focus solely on routing algorithms, our contribution lies in building a fully operational client-server token routing system compatible with edge deployment. This includes integration with ONNX inference on laptops and phones, low-latency LLM serving, and practical routing logic—bringing theoretical ideas into realworld applications.

# 2 Token Level Routing

In this section, we introduce serveral token level routing algorithm that can be used on our system.

# 2.1 CITER – Collaborative Inference with Token-level Routing

CITER [\(Zheng et al.,](#page-7-10) [2025a\)](#page-7-10) is a framework that accelerates language model inference through token-level routing between a small, fast but less accurate language model (SLM) and a large, accurate but expensive model (LLM). A trainable router determines, for each token, whether to use the SLM or the LLM, based on routing scores and a predefined threshold τ .

To capture the long-term tradeoff between cost and quality, CITER formulates router training as a preference-based reinforcement learning problem over a Markov Decision Process (MDP). Each state consists of the input prompt and the current generated tokens, and the actions correspond to choosing either the SLM or LLM to generate the next token.

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Figure 2: Computation procedure: Unlike conventional inference, the token routing system involves multiple rounds of prefill and decode within a single request, which prevents full utilization of inference acceleration engines such as SGLang and vLLM, as they only optimize kernel and KV cache on single stage prefill and decode.

Rewards reflect both inference efficiency and the quality of the final generated response.

Rather than specifying explicit reward functions, CITER leverages pairwise routing preferences: whether generating a token with the SLM is preferred over the LLM. These preferences are modeled using the Bradley-Terry model and optimized via a cross-entropy loss on the routing policy. To assign token-level preferences efficiently, a shortcut mechanism is introduced. If the SLM correctly predicts the next ground-truth token, it is preferred; otherwise, if the LLM predicts it correctly, the LLM is preferred. Only when both fail is a full generation trajectory used to assess quality—drastically reducing the need for expensive full-sequence rollouts.

The router is trained iteratively. In each round, the current policy generates routing decisions to collect updated preferences, which are then used to refine the routing policy. During inference, the router deterministically selects the model based on the posterior policy  $\pi(a|\mathbf{s})$ , adjusted by a prior  $(\rho(a_S), \rho(a_L))$ , allowing flexible control of the accuracy-efficiency tradeoff via a tunable threshold  $\tau = \rho(a_L)$ . This enables efficient collaborative inference that maintains high response quality while substantially reducing inference cost.

# 2.2 Co-LLM – Learning to Defer and Collaborate Efficiently

Co-LLM (Shen et al., 2024b) is another token level routing framework that jointly updates the base model and the deferral policy by minimizing the negative log marginal likelihood of the training data. To facilitate training, an initialization

scheme is introduced based on weak supervision: token-level pseudo-labels  $\hat{Z}_t$  indicate whether the assistant model predicts the ground-truth token better than the base model. This initialization helps the base model quickly identify difficult tokens suitable for deferral, which are then refined via unsupervised learning.

At inference time, a threshold  $\eta$  governs the deferral frequency: if  $P_{\theta}(Z_t=1\mid X_{< t})>\eta$ , the base model defers to the assistant. This decoding strategy supports fine-grained, token-level control of collaboration, yielding improved performance on tasks requiring domain expertise or complex reasoning. Empirical results show that CO-LLM not only surpasses single-model baselines but also outperforms other multi-model strategies, while requiring significantly fewer calls to large models during inference.

### 3 System Overview

In the token routing system, we decompose the architecture into three primary modules: (1) a serverside large language model (LLM) serving module, (2) an on-device small model inference module, and (3) a token routing selection module. This system introduces a novel serving paradigm wherein a single request may involve multiple rounds of prefilling, as illustrated in Figure 2. Crucially, interference can arise between the prefilling and decoding phases. While mainstream serving engines offer flexible separation strategies via dynamic partitioning (DP), they are not optimized for scenarios involving multiple alternating prefilling and decoding stages. Consequently, our system requires new strategies for ky-cache management and resource

allocation to support efficient inference under this setting. Therefore, our goal in developing this system is to build a prototype of the token routing framework and optimize it based on its unique computational characteristics.

On the server side, we adopt SGLang [\(Zheng](#page-7-11) [et al.,](#page-7-11) [2024\)](#page-7-11) as our LLM serving engine due to its flexible operator definitions and extensible kvcache management capabilities, which make it well-suited for the optimization techniques we propose. For on-device inference, existing solutions already enable the efficient deployment of small models. However, token routers—such as the routing module in CITER or the deferral mechanism in CO-LLM—often involve substantial computation. Since routing decisions must also be executed on mobile devices, we employ the ONNX [\(ONNX](#page-6-10) [Contributors,](#page-6-10) [2023\)](#page-6-10) framework, which supports both model inference and router execution in a unified and lightweight environment. In the following demonstration and evaluation, we exclusively adopt CITER, as its MLP-based router is more amenable to deployment on edge devices.

### 3.1 Front End

![](_page_3_Picture_4.jpeg)

Figure 3: User interface of the token-level routing system. Users can set prompts, thresholds, and decoding modes. Tokens from the large model are highlighted in red for interpretability.

We design a user-facing interface to support dynamic inference under a token-level routing framework. The interface includes a *prompt input field* for specifying the initial query, and a *threshold slider* that governs the routing decision between the small and large models. The threshold corresponds to the confidence score predicted by an MLP classifier, which operates on the last-layer hidden state of the small model. A token is routed to the large model if its score falls below the specified threshold, reflecting insufficient confidence in the small model's prediction.

The interface supports two inference modes:

joint, which enables collaborative decoding between the small and large models via token-level routing; and small\_only, which disables routing and uses only the small model for decoding. For interpretability, tokens generated by the large model are highlighted in red during generation, allowing users to visualize routing behavior in real time.

### 3.2 API CALL

Since CITER requires the last-layer hidden states of the model as input to the MLP router, we design a custom API schema (See Figure [5\)](#page-4-0) to ensure that each invocation of the large language model includes the necessary internal state information. This allows token-level routing decisions to be made based on contextual representations while maintaining stateless communication across modules.

### 3.3 Backend

On the server side, we adopt SGLang as the inference engine to serve large language models. For ondevice execution, we deploy models in the ONNX format to enable lightweight and efficient inference. However, since the router requires access to the last-layer hidden states of the model to determine whether a token should be routed, we modify the ONNX model accordingly (See Figure [4\)](#page-4-1). Specifically, after loading the model, the backend parses the computational graph to automatically identify the computation node corresponding to the lastlayer hidden states, and programmatically registers it as an additional output.

In cases where automatic matching fails, the node name can be manually identified using tools such as Netron, and the model modification script can be invoked to transform the original ONNX model into a format compatible with the routing system.

# 4 System Evaluation

As a routing system between a small and a large model, the overall system throughput is jointly influenced by the small model's inference speed, the number of routed tokens, the communication latency between the mobile device and the server, and the backend serving system's workload. Meanwhile, the quality of the user response is ensured by the router. Therefore, we evaluate our token routing system from both a system-level perspective and a response quality perspective. We use a Mac-Book Pro with an M1 chip as the edge device and

<span id="page-4-1"></span>![](_page_4_Figure_0.jpeg)

Figure 4: Left: ONNX computation graph of the original Qwen-0.5B model. Right: Modified graph with last-layer hidden states exposed as an output.

run the Qwen/Qwen2.5-32B-Instruct model on two A100 GPUs with the SGLang inference backend, configured with tensor parallelism (–tp=2). Even though onnx provide internal acceleration kernel for M1 chip, we only use CPU for small model and Router inference to simulate other edging device that do not support onnx acceleration kernel.

### 4.1 System Throughput

In our evaluation, we randomly selected 100 multiple-choice questions from the CommonsenseQA dataset. For each inference, the maximum generation length was set to 100 tokens. We varied the threshold of the MLP-based router from 0.4 to 0.9, where the threshold determines the routing score required for a token to be forwarded to the large language model (LLM).

Table [1](#page-5-0) shows the streaming and non-streaming inference speed of our system. The time to first token (TTFT) reflects the prefill time of the SLM. When the threshold is low, all tokens are generated locally by the SLM, which achieves an average generation speed of approximately 4 tokens per second on an M1 chip. When the threshold reaches 0.3, the router begins forwarding some tokens to the LLM for inference.

To simulate a worst-case deployment scenario, we assume a network communication delay of approximately 170 milliseconds between the client and server. Each LLM request incurs a latency of around 0.9 seconds. Furthermore, transferring the generation context from the LLM back to the SLM introduces an additional prefill delay of approximately 4 milliseconds, which accumulates as the

```
1 {
2 "context": "The mitochondria is the
        powerhouse of the",
3 "current_token": "cell",
4 "token_index": 15,
5 "routing_threshold": 0.7,
6 "slm_state": {
7 "hidden_states": [...],
8 "attention_states": [...]
9 },
10 "llm_state": null,
11 "history": {
12 "previous_decisions": [
13 {"token": "mitochondria", "route": "
           SLM"},
14 {"token": "powerhouse", "route": "LLM"
           }
15 ]
16 },
17 "meta_data": {
18 "session_id": "session123",
19 "request_id": "req456"
20 }
21 }
```

Figure 5: An example of the custom API format used to pass internal model state and routing metadata between modules.

number of LLM calls increases.

As the number of routing events increases, the time between tokens (TBT) begins to rise accordingly. This is primarily due to the lack of a keyvalue cache (kv-cache) management mechanism in the current ONNX-based inference system, which necessitates re-prefilling the entire sequence during each routing operation. Consequently, this leads to increased latency. Under more favorable network conditions—such as scenarios where edge devices maintain direct connections to the server—the system is expected to exhibit significantly improved performance.

### 4.2 Response Eval

Since the number of times the large language model (LLM) is involved in the inference process directly affects the quality of the final response, this section evaluates the performance of the token routing system on the CommonsenseQA dataset under various threshold settings. It is worth noting that the LLM and SLM used in the CITER [\(Zheng et al.,](#page-7-10) [2025a\)](#page-7-10) were Qwen2-72B and Qwen2-1.5B, respectively. However, due to the relatively slow inference speed of the 1.5B model on edge devices, we adopt a different configuration in our routing system to ensure a better user experience. Specifically, we use the Qwen2.5-32B model as the serving LLM and

Table 1: Performance Metrics (in seconds) under Different Thresholds – Non-Stream Inference

<span id="page-5-0"></span>

| Threshold                | 0.40  | 0.50  | 0.60  | 0.70  | 0.72  | 0.76  | 0.80  | 0.90   |
|--------------------------|-------|-------|-------|-------|-------|-------|-------|--------|
| Routing Number           | 0     | 0     | 1     | 14    | 17    | 38    | 65    | 76     |
| SLM Inference Time (s)   | 28.19 | 28.10 | 28.40 | 28.04 | 27.58 | 27.59 | 28.02 | 28.20  |
| TTFT (s)                 | 0.67  | 0.50  | 0.45  | 0.34  | 0.46  | 0.41  | 0.47  | 0.49   |
| TBT for SLM (s)          | 0.28  | 0.28  | 0.28  | 0.33  | 0.33  | 0.45  | 0.80  | 1.18   |
| Comm + LLM Inference (s) | 0.00  | 0.00  | 0.94  | 11.97 | 13.43 | 34.00 | 58.23 | 72.76  |
| Overall (s)              | 28.14 | 28.15 | 28.40 | 40.06 | 41.30 | 61.65 | 86.32 | 101.05 |

![](_page_5_Figure_2.jpeg)

![](_page_5_Figure_3.jpeg)

![](_page_5_Figure_4.jpeg)

- (a) Communication + LLM Inference Time
- (b) Complete Request Time (c) Time Between Tokens for SLM

Figure 6: Latency comparisons under different thresholds.

the Qwen2.5-0.5B model for on-device inference, thereby achieving higher overall system throughput.

<span id="page-5-1"></span>![](_page_5_Figure_10.jpeg)

Figure 7: Accuracy vs Threshold on CommonSense QA

<span id="page-5-2"></span>![](_page_5_Figure_12.jpeg)

Figure 8: The ratio of tokens routed to LLM vs Threshold on CommonSense QA

We evaluated the system performance on the CommonsenseQA dataset under various threshold settings. As shown in Figure [7](#page-5-1) and Figure [8,](#page-5-2) when the threshold falls below 0.3, the responses are predominantly generated by the small model, resulting in an accuracy of approximately 50%, which is significantly higher than the random guess baseline of 20%. As the threshold increases beyond 0.4, a portion of the tokens begins to be routed to the large model for decoding, leading to improved answer quality. To strike a balance between response quality and system efficiency—avoiding excessive latency introduced by frequent large model invocations—we typically set the threshold between 0.7 and 0.8 for commonsense reasoning tasks.

# 5 Conclusion

Building upon the token routing algorithm, we design a cloud-assisted token routing system that operates on devices running lightweight models at the edge. By routing a small subset of critical tokens to a large-scale model in the cloud for inference, the system significantly enhances the performance of the edge model while maintaining low inference latency. This architecture is well suited for scenarios where on-device deployment is required but model performance cannot be heavily compromised. Our experiments demonstrate that, on the CommonsenseQA dataset, routing merely 7% of

the tokens to the large model yields over a 60% improvement in the small model's accuracy.

# References

- <span id="page-6-8"></span>Sotiris Anagnostidis, Dario Pavllo, Luca Biggio, Lorenzo Noci, Aurelien Lucchi, and Thomas Hofmann. 2024. [Dynamic context pruning for effi](https://arxiv.org/abs/2305.15805)[cient and interpretable autoregressive transformers.](https://arxiv.org/abs/2305.15805) *Preprint*, arXiv:2305.15805.
- <span id="page-6-4"></span>Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. [Flashattention: Fast and](https://arxiv.org/abs/2205.14135) [memory-efficient exact attention with io-awareness.](https://arxiv.org/abs/2205.14135) *Preprint*, arXiv:2205.14135.
- <span id="page-6-0"></span>DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, and 181 others. 2025. [Deepseek-r1: Incentivizing reasoning capa](https://arxiv.org/abs/2501.12948)[bility in llms via reinforcement learning.](https://arxiv.org/abs/2501.12948) *Preprint*, arXiv:2501.12948.
- <span id="page-6-6"></span>Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023. [Llmlingua: Compressing](https://arxiv.org/abs/2310.05736) [prompts for accelerated inference of large language](https://arxiv.org/abs/2310.05736) [models.](https://arxiv.org/abs/2310.05736) *Preprint*, arXiv:2310.05736.
- <span id="page-6-7"></span>Siqi Kou, Lanxiang Hu, Zhezhi He, Zhijie Deng, and Hao Zhang. 2024. [Cllms: Consistency large lan](https://arxiv.org/abs/2403.00835)[guage models.](https://arxiv.org/abs/2403.00835) *Preprint*, arXiv:2403.00835.
- <span id="page-6-3"></span>Jinhao Li, Jiaming Xu, Shan Huang, Yonghua Chen, Wen Li, Jun Liu, Yaoxiu Lian, Jiayi Pan, Li Ding, Hao Zhou, Yu Wang, and Guohao Dai. 2025. [Large language model inference acceleration: A](https://arxiv.org/abs/2410.04466) [comprehensive hardware perspective.](https://arxiv.org/abs/2410.04466) *Preprint*, arXiv:2410.04466.
- <span id="page-6-1"></span>Xiangning Lin and 1 others. 2024. Tinyllm: Democratizing large language models for edge and mobile devices. *arXiv preprint arXiv:2402.17764*.
- <span id="page-6-5"></span>Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, Chunan Shi, Zhuoming Chen, Daiyaan Arfeen, Reyna Abhyankar, and Zhihao Jia. 2024. [Specinfer: Accel](https://doi.org/10.1145/3620666.3651335)[erating large language model serving with tree-based](https://doi.org/10.1145/3620666.3651335) [speculative inference and verification.](https://doi.org/10.1145/3620666.3651335) In *Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3*, ASPLOS '24, page 932–949. ACM.
- <span id="page-6-10"></span>ONNX Contributors. 2023. Onnx: Open neural network exchange. <https://github.com/onnx/onnx>. Accessed: 2025-03-27.
- <span id="page-6-2"></span>Shannon Zejiang Shen, Hunter Lang, Bailin Wang, Yoon Kim, and David Sontag. 2024a. [Learning to de](https://arxiv.org/abs/2403.03870)[code collaboratively with multiple language models.](https://arxiv.org/abs/2403.03870) *Preprint*, arXiv:2403.03870.
- <span id="page-6-9"></span>Shannon Zejiang Shen and 1 others. 2024b. Learning to decode collaboratively with multiple language models. *arXiv preprint arXiv:2403.03870*.

- <span id="page-7-8"></span>Shuming Shi, Enbo Zhao, Deng Cai, Leyang Cui, Xinting Huang, and Huayang Li. 2024. [Inferflow: an](https://arxiv.org/abs/2401.08294) [efficient and highly configurable inference engine for](https://arxiv.org/abs/2401.08294) [large language models.](https://arxiv.org/abs/2401.08294) *Preprint*, arXiv:2401.08294.
- <span id="page-7-1"></span>Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, Pouya Tafti, Léonard Hussenot, Pier Giuseppe Sessa, Aakanksha Chowdhery, Adam Roberts, Aditya Barua, Alex Botev, Alex Castro-Ros, Ambrose Slone, and 89 others. 2024. [Gemma: Open](https://arxiv.org/abs/2403.08295) [models based on gemini research and technology.](https://arxiv.org/abs/2403.08295) *Preprint*, arXiv:2403.08295.
- <span id="page-7-0"></span>Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. [Llama: Open](https://arxiv.org/abs/2302.13971) [and efficient foundation language models.](https://arxiv.org/abs/2302.13971) *Preprint*, arXiv:2302.13971.
- <span id="page-7-7"></span>Wenxiao Wang, Wei Chen, Yicong Luo, Yongliu Long, Zhengkai Lin, Liye Zhang, Binbin Lin, Deng Cai, and Xiaofei He. 2024. [Model compression and effi](https://arxiv.org/abs/2402.09748)[cient inference for large language models: A survey.](https://arxiv.org/abs/2402.09748) *Preprint*, arXiv:2402.09748.
- <span id="page-7-6"></span>Daliang Xu, Wangsong Yin, Xin Jin, Ying Zhang, Shiyun Wei, Mengwei Xu, and Xuanzhe Liu. 2023. [Llmcad: Fast and scalable on-device large language](https://arxiv.org/abs/2309.04255) [model inference.](https://arxiv.org/abs/2309.04255) *Preprint*, arXiv:2309.04255.
- <span id="page-7-4"></span>Zhenliang Xue, Yixin Song, Zeyu Mi, Xinrui Zheng, Yubin Xia, and Haibo Chen. 2024. [Powerinfer-2:](https://arxiv.org/abs/2406.06282) [Fast large language model inference on a smartphone.](https://arxiv.org/abs/2406.06282) *Preprint*, arXiv:2406.06282.
- <span id="page-7-2"></span>Kaiyan Zhang, Jianyu Wang, Ning Ding, Biqing Qi, Ermo Hua, Xingtai Lv, and Bowen Zhou. 2024a. [Fast](https://arxiv.org/abs/2406.12295) [and slow generating: An empirical study on large](https://arxiv.org/abs/2406.12295) [and small language models collaborative decoding.](https://arxiv.org/abs/2406.12295) *Preprint*, arXiv:2406.12295.
- <span id="page-7-9"></span>Kaiyan Zhang and 1 others. 2024b. Fast and slow generating: An empirical study on large and small language models collaborative decoding. *arXiv preprint arXiv:2406.12295*.
- <span id="page-7-11"></span>Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. 2024. [Sglang: Efficient](https://arxiv.org/abs/2312.07104) [execution of structured language model programs.](https://arxiv.org/abs/2312.07104) *Preprint*, arXiv:2312.07104.
- <span id="page-7-10"></span>Wenhao Zheng and 1 others. 2025a. Citer: Confidencebased token routing for collaborative inference with large language models. *arXiv preprint arXiv:2503.01013*.
- <span id="page-7-3"></span>Yue Zheng, Yuhao Chen, Bin Qian, Xiufang Shi, Yuanchao Shu, and Jiming Chen. 2025b. [A review on](https://arxiv.org/abs/2410.11845) [edge large language models: Design, execution, and](https://arxiv.org/abs/2410.11845) [applications.](https://arxiv.org/abs/2410.11845) *Preprint*, arXiv:2410.11845.

<span id="page-7-5"></span>Zixuan Zhou, Xuefei Ning, Ke Hong, Tianyu Fu, Jiaming Xu, Shiyao Li, Yuming Lou, Luning Wang, Zhihang Yuan, Xiuhong Li, Shengen Yan, Guohao Dai, Xiao-Ping Zhang, Yuhan Dong, and Yu Wang. 2024. [A survey on efficient inference for large lan](https://arxiv.org/abs/2404.14294)[guage models.](https://arxiv.org/abs/2404.14294) *Preprint*, arXiv:2404.14294.