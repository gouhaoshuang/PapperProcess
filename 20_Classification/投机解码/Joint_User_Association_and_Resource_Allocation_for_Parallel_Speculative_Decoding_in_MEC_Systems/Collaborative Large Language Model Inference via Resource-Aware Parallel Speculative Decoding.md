---
category: 投机解码
classification_reason: 论文的核心贡献是提出一种在移动边缘计算（MEC）环境下的并行投机解码框架。虽然目标是加速推理，但其技术路径（端云协同、投机采样）与纯粹的'端侧推理加速'（通常指模型压缩或端侧算子优化）有显著区别，且'投机解码'是移动端LLM优化的一个重要且独立的研究子领域。
created: '2026-01-18'
status: unread
tags:
- 投机解码
- 移动边缘计算
- 协同推理
- 多智能体深度强化学习
- 资源分配
title: Joint User Association and Resource Allocation for Parallel Speculative Decoding
  in MEC Systems
---

# Collaborative Large Language Model Inference via Resource-Aware Parallel Speculative Decoding

Jungyeon Koh, *Member, IEEE*, and Hyun Jong Yang, *Senior Member, IEEE*

*Abstract*— The growing demand for on-device large language model (LLM) inference highlights the need for efficient mobile edge computing (MEC) solutions, especially in resourceconstrained settings. Speculative decoding offers a promising solution by partitioning token generation between a lightweight draft model on mobile devices and a powerful target model on edge servers, but suffers from communication overhead and asynchronous delays. This paper propose a unified framework that jointly optimizes user association and resource allocation (UARA) to support efficient parallel speculative decoding. We solve the UARA problem using a multi-agent deep reinforcement learning algorithm. Simulation results show that our method achieves up to 28.0% and an average of 23.7% reduction in end-to-end latency without compromising inference accuracy, enabling scalable and low-latency LLM services in MEC systems.

*Index Terms*—Speculative Decoding, Collaborative Inference, Multi-Agent Deep Reinforcement Learning

## I. INTRODUCTION

The rising demand for on-device large language models (LLMs) has imposed a significant computational burden on mobile devices, especially in developing countries with limited hardware and infrastructure. This challenge has spurred research interest in the efficient deployment of mobile edge computing (MEC) to assist resource-constrained environments. Although conventional approaches–on-device and cloud-based inference–offer baseline support, both face critical limitations. On-device LLM inference is compute- and memory-intensive; for example, running a Llama2-7B model requires approximately 28 GB of memory, far beyond the capacity of most mobile devices [1]. Even high-end smartphones often suffer from battery drain and degraded quality of experience (QoE). Meanwhile, cloud-based solutions like ChatGPT and Gemini introduce latency, bandwidth costs, and privacy concerns.

To address these challenges, mobile-edge collaborative inference has emerged as a promising solution. By distributing computation between mobile devices and nearby edge servers, this approach enables real-time LLM inference by integrating edge computing, model optimization, and mobile hardware. Existing collaborative methods [2], [3] focus primarily on partitioning deep neural networks (DNNs) and offloading subsequent computation stages to edge servers. For example, [2] proposes spatially tiling the computation graph to enable independent execution among tiles, which works well for

Jungyeon Koh is with the Department of Electrical Engineering, Pohang University of Science and Technology (POSTECH), Korea (e-mail: jungyeon.koh@postech.ac.kr). Hyun Jong Yang is with the Department of Electrical and Computer Engineering, Seoul National University, Korea (email: hjyang@snu.ac.kr).

convolutional neural networks. However, such methods are illsuited for autoregressive language models, where sequential token dependencies make partitioning inherently challenging.

1

To mitigate the inefficiencies of autoregressive generation, speculative decoding [4], [5] employs a lightweight *draft* model on the mobile device and a more powerful *target* model on the edge server. The draft model generates draft tokens with minimal computational cost, while the target model verifies and refines them in batches. This significantly reduces computational overhead while preserving inference accuracy. However, speculative decoding still faces latency issues, mainly due to asynchronous executions. As sequence lengths grow, a *mutual waiting problem* arises– the target model remains idle while awaiting tokens from the draft model, and the draft model similarly stops during target-side verification. This inefficiency hinders the overall performance gain and suggests opportunities for further optimization.

To address this, [6] proposes parallel speculative decoding (PSD), which overlaps the drafting and verifying phases to reduce overall processing time. However, PSD introduces additional communication overhead, limiting its feasibility in mobile-edge collaborative inference. Motivated by these challenges, this paper highlights the need for intelligent user association and resource allocation (UARA) strategies to fully exploit the benefits of speculative decoding in MEC systems. Furthermore, we incorporate device-level constraints–such as remaining battery life–into UARA decisions for sustainable and efficient operations, given the heavy reliance of speculative decoding on mobile-side computation.

Building on these insights, we propose a novel UARA optimization framework for efficient parallel speculative decoding. Our approach addresses user association via Two-Phase Matching-based Association (TMA) [7] and applies a Multi-Agent Soft Actor-Critic (MASAC) [8] network to optimize resource allocation. To the best of our knowledge, this is the first work to jointly optimize UARA problem for speculative decoding in MEC environments.

Our key contributions are summarized as follows:

- Unified UARA strategy for speculative decoding: We propose the first unified framework that jointly optimizes UARA problem to enable low-latency LLM inference via parallel speculative decoding.
- Joint optimization problem formulation: We identify synchronization between mobile-side computation and uplink communication as critical to fully utilizing parallel speculative decoding. To this end, we formulate a mixedinteger, non-convex optimization problem, which is addressed using our proposed MADRL-based solution.

![](_page_1_Figure_1.jpeg)

Fig. 1: Comparison of (a) conventional and (b) parallel speculative decoding. (**Top**) Example of token generation. In (a), the draft model waits for server-side verification and the target model waits for draft generation. In contrast, (b) allows continuous generation for both models, resulting in faster task completion. (**Bottom**) End-to-end timeline illustrating the interplay between mobile computation, network transmission and server computation. In (a), the sequential draft-then-verify process introduces prolonged idle time. In contrast, (b) reduces idle periods by enabling concurrent execution. The shaded regions highlight the mutual waiting problem unique to the conventional approach and the communication overhead present in both methods.

- Realistic simulation using Sionna: We use the Sionna [9] simulator to conduct realistic experiments based on advanced ray tracing and geo-located wireless channels. This allows us to closely mimic real-world MEC systems with dynamic user mobility and varying channel conditions.
- Performance evaluation and analysis: We implement our UARA scheme on Sionna-based setups with a diverse set of mobile devices and edge servers. Numerical results show up to 28.0% and an average of 23.7% reduction in end-to-end latency without compromising inference accuracy, thus validating the practicality of our approach in realistic MEC settings.

## II. PRELIMINARIES

#### A. Iterative LLM Inference

The LLM inference process is inherently iterative and typically consists of two distinct phases: *prompt processing* (*prefill*) and *autoregressive generation*. These phases differ significantly in their computational and memory characteristics.

- **Prompt Processing:** In this phase, the model processes the initial sequence  $\mathbf{x}=(x_1,...,x_n)$  to generate the first new token  $x_{n+1}$  by computing the conditional probability  $P(x_{n+1}|x_1,...,x_n)$ . This step involves generating keyvalue (KV) pairs for all input tokens, which are cached for use in subsequent decoding steps. Since the entire input sequence is available upfront, the phase can be easily accelerated on modern hardware. Thus, prefill is typically compute-bound and exhibits high throughput.
- Autoregressive Generation: In the generation phase, the model produces one token at a time, with each token conditioned on all previously generated tokens. Generation continues until a stopping criterion is met, such as generating an end-of-sequence (EOS) token or reaching a specified

maximum token limit. Unlike the prefill phase, this phase is primarily memory-bound since frequent memory access dominates latency. Consequently, token throughput in this phase is significantly lower, posing a major bottleneck in real-time LLM inference.

#### B. Speculative Decoding

Parallelizing autoregressive token generation is challenging due to its inherent sequential dependency, where each token depends on all previous ones. Speculative decoding mitigates this latency by adopting a *draft-then-verify* strategy using a lightweight *draft* model for fast token generation and a more accurate *target* model for verification.

In speculative decoding, the draft model  $M_q$  and the target model  $M_p$  operate with a fixed hyperparameter  $\gamma$ , which specifies the draft length—the number of tokens generated by the draft model per iteration. The process begins with  $M_q$  generating  $\gamma$  draft tokens  $(x_1, x_2, ..., x_\gamma)$  along with their corresponding logits  $(q_1, q_2, ..., q_\gamma)$ . The target model  $M_p$  then processes the concatenated sequence  $(x, x_1, ..., x_\gamma)$  and produces logits  $(p_1, p_2, ..., p_{\gamma+1})$ . Each draft token  $x_i$  is verified with the acceptance rate  $\alpha_i$  defined as

$$\alpha_i = \begin{cases} 1 & p_i[x_i] \ge q_i[x_i] \\ \frac{p_i[x_i]}{q_i[x_i]} & p_i[x_i] < q_i[x_i] \end{cases} . \tag{1}$$

If a draft token  $x_i$  is rejected, it is resampled from  $p_i-q_i$ ; otherwise, the process proceeds to generate  $x_{\gamma+1}$ via  $p_{\gamma+1}$ . This method ensures that each step generates at most  $\gamma+1$  tokens, thereby speeding up the generation process. However, we eliminate resampling and directly regenerate rejected tokens using the target model, reducing communication overhead from transmitting full logits  $q_i$ .

While most prior works focus mainly on reducing computation latency— via compact or draft-free models [10]–[12] or improved alignment between draft and target distributions [13], [14]— these are often limited to single-device setups. Only few recent efforts have explored speculative decoding in MEC contexts [15], [16]. However, they often overlook the interplay between computation and communication, lacking practical and feasible strategies for real-world deployments.

## C. Parallel Speculative Decoding

We introduce a parallel speculative decoding framework, inspired by recent work in [6], to enable synchronized execution of draft and target models. Unlike conventional speculative decoding, which alternates between draft and verify stages, PSD proposes dynamic scheduling—pre-verify and post-verify—for concurrent execution. These strategies adaptively switch based on token verification outcomes, effectively overlapping computation and reducing idle time. Figure 1 shows the differences between conventional and parallel speculative decoding in terms of token generation behavior and execution timeline.

In the pre-verify phase, the target model  $M_p$  computes logits  $M_p(\mathbf{x})$  while the draft model generates a batch of tokens. If the first token is rejected, the batch is discarded and a new drafting round begins immediately with the updated prefix  $\mathbf{x} + (y_1)$ . This enables a fast recovery from early rejections. If accepted, the process enters the post-verify phase, where the draft model continues generating tokens without interruption and the target model concurrently verifies them. A rejection during this phase triggers a return to pre-verify phase.

By dynamically switching between these two phases, the PSD framework maintains uninterrupted utilization of both models and reduces idle time. However, we observe that this synchronization is fragile and can easily degrade when mobile-side computation and network latency are not properly aligned.

#### D. Motivating Observations

Figure 2 presents the total execution time for conventional and parallel speculative decoding under varying conditions. The key limitations of each method are summarized as follows:

- Conventional speculative decoding lacks adaptability across diverse prompts due to its fixed draft length  $\gamma$ . The optimal  $\gamma$  is 25 for HumanEval and 15 for DailyMail dataset. This motivates the adoption of parallel speculative decoding for adaptive and task-aware generation.
- Parallel speculative decoding benefits from higher bandwidth, but saturates beyond 50 Mbps, where mobile-side computation becomes the bottleneck. This underscores the need for tight synchronization between mobile computation and uplink communication.

### III. SYSTEM MODEL AND PROBLEM FORMULATION

## A. System Overview

Figure 3 illustrates our collaborative MEC system, which consists of M mobile devices and E edge servers. The set of mobile devices is denoted by  $\mathcal{M}$ , where each device may exhibit varying mobility patterns and remaining battery levels.

![](_page_2_Figure_13.jpeg)

Fig. 2: Latency of (a) conventional and (b) parallel speculative decoding using HumanEval dataset. (**Left**) Latency is broken down into mobile computation, network transmission and server computation for varying draft lengths. (**Right**) Total latency is measured across different data rates.

![](_page_2_Figure_15.jpeg)

Fig. 3: Overview of the proposed MEC system. Key UARA decision variables are indicated in red.

The set of edge servers is denoted by  $\mathcal{E}$ , equipped with varying computing resources.

To capture the effect of user mobility and time-varying channel conditions, we adopt a time-slotted model that divides the offloading period into T slots with an equal duration  $\tau$ . In this quasi-static setting, channel states and device parameters are assumed constant within a slot. Each mobile device  $m_i$  generates a task per time slot, which is represented by a tuple  $\{d_i, f_i^{MD}, f_i^{ES}\}$ , where  $d_i$  is the communication load in bits, and  $f_i^{MD}$  and  $f_i^{ES}$  are the required FLOPs for mobile device and edge server, respectively.

We then define the association of mobile device  $m_i$  to edge server  $e_j$  using a binary decision variable.

$$x_{ij} = \begin{cases} 1, & \text{if edge server } j \text{ serves mobile device } i, \\ 0, & \text{otherwise.} \end{cases}$$
 (2)

Each mobile device must offload its task to exactly one edge server, with no partial offloading allowed. Accordingly, the user association variable  $x_{ij}$  is constrained by

$$\sum_{j \in \mathcal{E}} x_{ij} = 1, x_{ij} \in \{0, 1\}, \forall i \in \mathcal{M}, j \in \mathcal{E}$$
 (3)

#### B. Communication Model

We assume that the total network bandwidth W is equally divided among the E edge servers, such that each server  $e_i$  is

allocated with  $W_j = W/E$ . Each server further allocates its bandwidth  $W_j$  among the mobile devices it serves. Let  $y_{ij} \in [0,1]$  denote the fraction of  $W_j$  allocated to mobile device  $m_i$ . To prevent overlapping allocations,  $y_{ij}$  is constrained by

$$\sum_{j \in \mathcal{E}} y_{ij} \le 1, y_{ij} \in [0, 1], \forall i \in \mathcal{M}, j \in \mathcal{E}.$$
(4)

Let  $h_{ij}$  denote the channel gain between device  $m_i$  and edge server  $e_j$ . The corresponding signal-to-noise ratio (SNR) is given by  $SNR_{ij}=\frac{h_{ij}^2P_i}{N_0}$ , where  $P_i$  is the transmission power of device  $m_i$  and  $N_0$  is the noise power spectral density. Then, the data rate  $R_i$  of mobile device  $m_i$  is expressed as

$$R_i = \sum_{j \in \mathcal{E}} x_{ij} y_{ij} W_j \log(1 + SNR_{ij}). \tag{5}$$

### C. Computing Model

Computing latency for a task requiring  $f_i$  FLOPs is calculated by dividing the task size by the available computing capacity. For remote execution, we model the delay in two phases: a waiting phase over k time slots of queuing delay and a computing phase based on the server's processing power. Since edge servers share resources across tasks, we define  $z_{ij}$  as the fraction of computing capacity allocated to the task from mobile device  $m_i$ . The resulting computing delays for local and remote execution are given by

$$\begin{cases} D_i^{\text{MD}} = \frac{f_i}{F_i^{\text{MD}}}, & \text{(local execution)} \\ D_{ij}^{\text{ES}} = k\kappa + \frac{f_i}{z_{ij}F_j^{\text{ES}}}. & \text{(remote execution)} \end{cases}$$
(6)

Similar to Eq. (4),  $z_{ij}$  is constrained by

$$\sum_{i \in \mathcal{M}} z_{ij} \le 1, z_{ij} \in [0, 1], \forall i \in \mathcal{M}, j \in \mathcal{E}.$$
 (7)

#### D. Problem Formulation

Effective parallel speculative decoding requires synchronization between mobile computation and uplink communication overhead. Additionally, minimizing edge-side computing latency is critical for reducing end-to-end delay. To ensure sustainable operation in resource-constrained MEC settings, we further incorporate remaining battery life of mobile devices [17]. Based on these considerations, we formulate a multi-objective cost function that balances latency synchronization, remote computation delay, and energy efficiency.

Here, we define  $\delta^{CP}, \delta^{CM}$  as the computing and transmission energy efficiencies, respectively. The energy consumption of mobile device  $m_i$  can then be expressed as follows.

$$\begin{cases} E_i^{\text{CP}} = \delta^{\text{CP}} f_i^{\text{MD}}, & \text{(mobile computing)} \\ E_i^{\text{CM}} = \delta^{\text{CM}} \sum_{j \in \mathcal{E}} (x_{ij} y_{ij} W_j) P_i. & \text{(uplink transmission)} \end{cases}$$

Algorithm 1 Two-phase Matching-based Association (TMA)

```
1: Initialization: Obtain the channel gain matrix H between
    all mobile devices and edge servers.
2: Phase 1: Channel Gain-based Pre-evaluation
 3: for all m \in \mathcal{M} do
        Select the edge server e with the highest gain \mathbf{H}_{m,e}.
        Set \mathbf{X}_{m,e} \leftarrow 1.
 6: end for
    Phase 2: Iterative Swap-based Optimization
    while swap pair exists do
        for all (m,e),(m',e') with \mathbf{X}_{m,e}=\mathbf{X}_{m',e'}=1 do
           Compute \mathcal{I}_{(m,e),(m',e')} and \mathcal{I}_{(m,e'),(m',e)}.
10:
           if \mathcal{I}_{(m,e'),(m',e)} > \mathcal{I}_{(m,e),(m',e')} then
11:
              Set \mathbf{X}_{m,e'} \leftarrow 1 and \mathbf{X}_{m',e} \leftarrow 1.
12:
              Reset \mathbf{X}_{m,e} \leftarrow 0, \mathbf{X}_{m',e'} \leftarrow 0.
13:
14:
           end if
15:
        end for
16: end while
```

Then, our optimization objective  $\mathcal{I}$  can be formulated as

$$\min_{\mathbf{X}, \mathbf{Y}, \mathbf{Z}} \quad \sum_{i \in \mathcal{M}} \sum_{j \in \mathcal{E}} x_{ij} \left( D_i^{\text{MD}} + D_{ij}^{\text{ES}} + \lambda \left\| D_i^{\text{MD}} - \frac{d_i}{R_i} \right\|_2^2 \right) \\
+ w \sum_{i \in \mathcal{M}} \frac{E_i^{\text{CP}} + E_i^{\text{CM}}}{B_i} \\
\text{s.t.} \quad (3), (4), (7).$$

where  $\lambda$  is a penalty for latency synchronization, and w is a unit conversion factor normalizing energy efficiency. For clarity, we define the decision variable matrices  $\mathbf{X}, \mathbf{Y}$  and  $\mathbf{Z}$  to represent user association, bandwidth allocation, and computing resource allocation, respectively.

The primary challenge in Problem  $\mathcal{P}_1$  lies in its formulation as a mixed-integer, non-convex optimization problem. The binary variable  $\mathbf{X}$  introduces combinatorial complexity and is further coupled with the continuous variables  $\mathbf{Y}$  and  $\mathbf{Z}$ , making the problem particularly difficult to solve.

#### IV. MADRL-BASED UARA SCHEME

## A. Two-Phase Matching-based Association (TMA)

Two-Phase Matching-based Association (TMA) solves the user association problem through a two-phase process. In the first phase, each mobile device pre-evaluates edge servers based on channel gain to establish a favorable initial association for subsequent optimization steps. In the second phase, a matching algorithm iteratively swaps device-server pairs to minimize the objective function  $\mathcal{I}$  formulated in Problem  $\mathcal{P}_1$  until no further improvements are possible. Compared to traditional heuristics such as greedy search, matching-based methods offer faster convergence with lower complexity [18], making them well-suited for real-time applications. The overall procedure is summarized in Alg. 1.

#### B. Proposed TMA-MASAC Framework

We adopt a deep reinforcement learning (DRL) framework to address the resource allocation problem. The system state s(t) includes mobile device and edge server positions, channel gain matrix  $\mathbf{H}$ , task information  $\{d_i, f_i^{MD}, f_i^{ES}\}$ , remaining battery level  $B_i$  for each mobile device  $m_i$ , and queuing delay  $k_j$  at each edge server  $e_j$ . The action a(t) consists of uplink bandwidth allocation  $\mathbf{Y}(t)$  and edge computing resource allocation  $\mathbf{Z}(t)$ . The reward r(t) is defined as the normalized objective value, dividing  $\mathcal{I}$  by a baseline obtained from random user association and uniform resource allocation.

Specifically, we use a multi-agent soft actor-critic (MASAC) network for two reasons. First, the continuous variables **Y** and **Z** require a model-free, policy-gradient method. Second, SAC jointly optimizes expected reward and policy entropy, improving robustness against highly dynamic MEC settings.

The proposed framework consists of a policy network  $\pi_{\theta}(a_t|s_t)$ , a critic network  $Q_{\phi}(s_t, a_t)$ , and a target critic network  $Q_{\bar{\phi}}(s_t, a_t)$ . Introducing  $\alpha$  as the temperature coefficient of policy entropy, MASAC optimizes the following objective

$$J(\pi) = \sum_{t=1}^{T} \mathbb{E}_{(s_t, a_t) \sim \pi_{\theta}} [r(s_t, a_t) - \alpha \log(\pi(\cdot|s_t))] \quad (9)$$

To improve sample efficiency, we adopt offline sampling by reusing historical data. Mini-batches sampled from a replay buffer are used to update the policy and critic networks, while the target critic network is softly updated, following  $\bar{\phi}_i \leftarrow \xi \phi_i + (1 - \xi)\bar{\xi}_i$ , where  $\xi$  denotes the soft update coefficient.

![](_page_4_Figure_7.jpeg)

Fig. 4: Illustration of the proposed TMA-MASAC.

To jointly solve UARA problem, the TMA-based user association is integrated into the MASAC scheme. Fig. 4 shows the overall framework. Specifically, at each time step, the policy network outputs resource allocation decisions, after which Alg. 1 is applied to determine user association  ${\bf X}$  based on the updated  ${\bf Y}$  and  ${\bf Z}$ .

#### V. PERFORMANCE EVALUATION

#### A. Experiment Setup

**Simulation Configuration** We model the MEC system based on the 3GPP small-cell simulation guidelines [19], with parameters detailed in Table I. The system includes a diverse set of mobile device and edge server, including Samsung Galaxy S23, Apple iPhone 14, Huawei Mate 60, and NVIDIA RTX 2080, 3090 and 4090. At each time slot, each device

TABLE I: MEC Simulation Configurations

| Parameter                                      | Value          |
|------------------------------------------------|----------------|
| Target model $M_p$                             | GPT2-XL (1.6B) |
| Draft model $M_q$                              | GPT2 (137M)    |
| Number of mobile devices M                     | 30 - 180       |
| Number of edge servers $E$                     | 4 - 20         |
| Bandwidth W (MHz)                              | 10             |
| Mobile transmission power                      | 16 – 24        |
| P (dBm)                                        | 10 24          |
| Noise power spectral density                   | -174           |
| $N_0$ (dBm/Hz)                                 |                |
| Computing energy                               | $10^{9}$       |
| coefficient $\delta^{CP}$                      | 10             |
| Communication energy coefficient $\delta^{CM}$ | 2.6            |
| Latency weight $\lambda$                       | $10^{-2}$      |
| Energy weight $w$                              | 20 - 100       |
|                                                |                |

randomly executes one of three LLM-based tasks: code generation, text summarization, or chatbot conversation. Diversity in task generation with different expected sequence lengths enables an effective evaluation of the system's adaptability to heterogeneous LLM workloads. Furthermore, to emulate realistic wireless conditions, we adopt the Sionna simulator, which offers a high-fidelity ray-tracing engine and stochastic channel models. Given the locations of mobile devices and edge servers, Sionna computes channel gain H, path loss, multi-path fading, and mobility dynamics.

**Benchmarks** We evaluate the proposed method against three baselines.

- Random: Each mobile device randomly selects an edge server with uniform probability.
- Max SINR [20]: Each mobile device associates with the edge server offering the highest signal-to-interference-plusnoise ratio (SINR).
- Max Compute [21]: Each mobile device associates with the edge server offering the highest computational capacity.

#### B. Performance Evaluation

![](_page_4_Figure_21.jpeg)

Fig. 5: Average latency under different numbers of (**left**) mobile devices and (**right**) edge servers.

Figure 5 compares the latency performance of the proposed framework against multiple baseline strategies under varying numbers of mobile devices and edge servers. As the network scales up, our method consistently outperforms reference schemes, achieving up to 28.0% latency reduction, with an average improvement of 23.7% across all evaluated settings. This notable gain highlights the advantage of our joint UARA

strategy, which enables collaborative LLM inference with efficient resource utilization and minimizes mutual waiting delays even in dense and dynamically changing MEC environments.

![](_page_5_Figure_2.jpeg)

Fig. 6: Impact of energy-efficiency weight w on (left) average latency and (right) energy consumption for mobile devices with high (B<sup>i</sup> ≥ 0.8) and low (B<sup>i</sup> ≤ 0.2) battery levels.

Figure 6 further analyzes how the energy-efficiency weight w influences both average latency and energy consumption across devices with different battery levels. As w increases, a clear tradeoff emerges: energy consumption steadily decreases while latency gradually grows, demonstrating that the system correctly prioritizes energy savings when requested. This impact is especially pronounced for low-battery users, where the framework shifts toward energy-conservative scheduling, preventing premature device shutdown while sustaining inference capability. These results emphasize the importance of carefully tuning w to balance latency and energy efficiency for different deployments—e.g., delay-sensitive applications versus longduration, battery—constrained usage scenarios.

## VI. CONCLUSION

In this paper, we introduce a framework for efficient LLM inference in mobile edge computing (MEC) systems by jointly optimizing the user–association and resource–allocation (UARA) problem to support parallel speculative decoding. To achieve this, we propose the TMA-MASAC framework, which coordinates mobile computation, wireless transmission, and server-side execution through matching algorithm and deep reinforcement learning. Using the high-fidelity Sionna simulator, our evaluation demonstrates that the proposed method achieves up to 28.0% latency reduction across various network conditions, consistently outperforming existing baselines. The results confirm that integrating parallel speculative decoding with intelligent network control can accelerate LLM inference with minimal energy cost, thereby realizing scalable and energy-efficient AI deployment in dense MEC environments.

# REFERENCES

- [1] S. Laskaridis, K. Katevas, L. Minto, and H. Haddadi, "Mobile and edge evaluation of large language models," in *Workshop on Efficient Systems for Foundation Models, ICML*, Jul. 2024.
- [2] K. Bin, J. Park, C. Park, S. Kim, and K. Lee, "CoActo: Coactive neural network inference offloading with fine-grained and concurrent execution," in *Proc. Annu. Int. Conf. Mobile Syst. Appl. Services*, May 2024.
- [3] K. Huang and W. Gao, "Real-time neural network inference on extremely weak devices: agile offloading with explainable ai," in *Proceedings of the 28th Annual International Conference on Mobile Computing And Networking*, 2022, pp. 200–213.

- [4] Y. Leviathan, M. Kalman, and Y. Matias, "Fast inference from transformers via speculative decoding," in *Proc. Int. Conf. Mach. Learn.*, Jul. 2023.
- [5] C. Chen, S. Borgeaud, G. Irving, J.-B. Lespiau, L. Sifre, and J. Jumper, "Accelerating large language model decoding with speculative sampling," 2023, arXiv:2302.01318.
- [6] T. Liu, Y. Li, Q. Lv, K. Liu, J. Zhu, and W. Hu, "Parallel speculative decoding with adaptive draft length," 2024, arXiv:2408.11850.
- [7] B. Wang, H. Kang, J. Li, G. Sun, Z. Sun, J. Wang, and D. Niyato, "UAVassisted joint mobile edge computing and data collection via matchingenabled deep reinforcement learning," 2025, arXiv:2502.07388.
- [8] T. Haarnoja, A. Zhou, K. Hartikainen, G. Tucker, S. Ha, J. Tan, V. Kumar, H. Zhu, A. Gupta, P. Abbeel *et al.*, "Soft actor-critic algorithms and applications," 2018.
- [9] J. Hoydis, S. Cammerer, F. A. Aoudia, A. Vem, N. Binder, G. Marcus, and A. Keller, "Sionna: An open-source library for next-generation physical layer research," 2022, arXiv:2203.11854.
- [10] W. Zhao, Y. Huang, X. Han, C. Xiao, Z. Liu, and M. Sun, "Ouroboros: Speculative decoding with large model enhanced drafting," 2024, arXiv:2402.13720.
- [11] T. Cai, Y. Li, Z. Geng, H. Peng, J. D. Lee, D. Chen, and T. Dao, "Medusa: Simple llm inference acceleration framework with multiple decoding heads," 2024, arXiv:2401.10774.
- [12] Y. Fu, P. Bailis, I. Stoica, and H. Zhang, "Break the sequential dependency of llm inference using lookahead decoding," 2024.
- [13] Y. Zhou, K. Lyu, A. S. Rawat, A. K. Menon, A. Rostamizadeh, S. Kumar, J.-F. Kagy, and R. Agarwal, "DistillSpec: Improving speculative decoding via knowledge distillation," 2023.
- [14] X. Miao, G. Oliaro, Z. Zhang, X. Cheng, Z. Wang, Z. Zhang, R. Y. Y. Wong, A. Zhu, L. Yang, X. Shi *et al.*, "Specinfer: Accelerating large language model serving with tree-based speculative inference and verification," in *Proc. ACM Int. Conf. Archit. Support Program. Lang. Operating Syst.*, vol. 3, 2024, pp. 932–949.
- [15] Z. Hao, H. Jiang, S. Jiang, J. Ren, and T. Cao, "Hybrid SLM and LLM for edge-cloud collaborative inference," in *Workshop on Edge and Mobile Foundation Models, EdgeFM*, Jun. 2024.
- [16] H. Jin and Y. Wu, "CE-CoLLM: Efficient and adaptive large language models through cloud-edge collaboration," 2024, arXiv:2411.02829.
- [17] M. Kim, J. Jang, Y. Choi, and H. J. Yang, "Distributed task offloading and resource allocation for latency minimization in mobile edge computing networks," *IEEE Trans. on Mobile Comput.*, vol. 23, pp. 15 149–15 166, 2024.
- [18] Y. Deng, Z. Chen, X. Chen, and Y. Fang, "Throughput maximization for multiedge multiuser edge computing systems," *IEEE Internet Things J.*, vol. 9, pp. 68–79, 2021.
- [19] 3GPP, "Nr; physical layer procedures for data," Jul. 2020.
- [20] T. Liu, S. Ni, X. Li, Y. Zhu, L. Kong, and Y. Yang, "Deep reinforcement learning based approach for online service placement and computation resource allocation in edge computing," *IEEE Trans. Mobile Comput.*, vol. 22, pp. 3870–3881, 2022.
- [21] X. Ma, A. Zhou, S. Zhang, Q. Li, A. X. Liu, and S. Wang, "Dynamic task scheduling in cloud-assisted mobile edge computing," *IEEE Trans. Mobile Comput.*, vol. 22, pp. 2116–2130, 2021.