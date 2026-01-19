---
category: 能效优化
classification_reason: 该论文的核心目标是通过移动端与边缘服务器的协同训练框架及资源分配，最小化总能耗和延迟。虽然涉及训练任务，但其主要贡献在于解决计算受限下的能效和时延问题，完全符合'能效优化'类别的定义。
created: '2026-01-18'
status: unread
tags:
- 协同训练
- 边缘计算
- 参数高效微调
- 资源分配
- 移动端LLM
title: 5.1 The performance of the proposed collaborative training method
---

![](_page_0_Picture_0.jpeg)

## Resource Allocation for Stable LLM Training in Mobile Edge Computing

Chang Liu Graduate College Nanyang Technological University Singapore liuc0063@e.ntu.edu.sg

Jun Zhao College of Computing and Data Science Nanyang Technological University Singapore junzhao@ntu.edu.sg

### ABSTRACT

As mobile devices increasingly become focal points for advanced applications, edge computing presents a viable solution to their inherent computational limitations, particularly in deploying large language models (LLMs). However, despite the advancements in edge computing, significant challenges remain in efficient training and deploying LLMs due to the computational demands and data privacy concerns associated with these models. This paper explores a collaborative training framework that integrates mobile users with edge servers to optimize resource allocation, thereby enhancing both performance and efficiency. Our approach leverages parameter-efficient fine-tuning (PEFT) methods, allowing mobile users to adjust the initial layers of the LLM while edge servers handle the more demanding latter layers. Specifically, we formulate a multi-objective optimization problem to minimize the total energy consumption and delay during training. We also address the common issue of instability in model performance by incorporating stability enhancements into our objective function. Through novel fractional programming technique, we achieve a stationary point for the formulated problem. Simulations demonstrate that our method reduces the energy consumption as well as the latency, and increases the reliability of LLMs across various mobile settings.

## CCS CONCEPTS

• Networks → Network resources allocation.

## KEYWORDS

Mobile edge computing, large language model, wireless networks.

#### ACM Reference Format:

Chang Liu and Jun Zhao. 2024. Resource Allocation for Stable LLM Training in Mobile Edge Computing. In The Twenty-fifth International Symposium on Theory, Algorithmic Foundations, and Protocol Design for Mobile Networks and Mobile Computing (MOBIHOC '24), October 14–17, 2024, Athens, Greece. ACM, New York, NY, USA, [10](#page-9-0) pages.<https://doi.org/10.1145/3641512.3686358>

Corresponding author: Jun Zhao Chang Liu is a PhD student supervised by Jun Zhao.

![](_page_0_Picture_13.jpeg)

MOBIHOC '24, October 14–17, 2024, Athens, Greece © 2024 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-0521-2/24/10. <https://doi.org/10.1145/3641512.3686358> [This work is licensed under a Creative Commons Attribution International 4.0 License.](https://creativecommons.org/licenses/by/4.0/)

## 1 INTRODUCTION

The advent of large language models (LLMs) marks a significant milestone in the advancement of artificial intelligence and offers unparalleled capabilities in natural language processing, generation, and understanding. The desire for ubiquitous access to Artificial intelligence (AI) capabilities has driven a significant trend and demand toward the deployment and even training of these computationally intensive models directly on mobile devices [\[2,](#page-9-1) [18,](#page-9-2) [30\]](#page-9-3). Users seek real-time, personalized experiences and decision-making support across a diverse array of applications, from healthcare to customer service, which only such advanced models can provide. Additionally, there is a growing emphasis on decentralization in computing to enhance privacy and data security by processing sensitive information locally on the device, rather than transmitting it to distant data centers. However, this aspiration faces a tough challenge: the substantial computational resources required by LLMs. These models necessitate sophisticated hardware configurations that far exceed the capabilities of standard mobile devices [\[19\]](#page-9-4).

Mobile edge computing (MEC) emerges as a transformative solution to this challenge by bringing computational resources closer to the data source [\[26\]](#page-9-5). MEC enables data processing at the network's edge, utilizing distributed computing resources geographically proximate to where the data originates and is consumed. Augmented with LLMs, mobile edge servers can process and understand complex queries locally, minimizing the need for constant communication with centralized cloud infrastructure. This not only improves response times but also enhances privacy and data security by processing sensitive information closer to its source.

However, we still face challenges in how LLMs can be optimized in real-time mobile applications to achieve the best possible performance. The challenges involve tailoring these models in resourceconstrained environments. One solution is in-context learning. It provides the LLM with a few examples of the desired task within the input prompt, allowing the model to adapt its behavior based on these examples without changing its parameters. But the effectiveness of in-context learning is constrained by the model's context window size, and it doesn't lead to persistent improvements in the model's capabilities. Moreover, recent studies have shown that in-context learning may struggle with reliability and hallucination [\[13\]](#page-9-6). Alternatively, many studies propose parameterefficient fine-tuning (PEFT) methods [\[5,](#page-9-7) [12,](#page-9-8) [17\]](#page-9-9). These methods have been demonstrated to yield state-of-the-art performance in model training and optimization, significantly reducing the computational overhead traditionally associated with such processes. Inspired by this body of work, we propose a novel scenario for collaboratively training LLMs, harnessing the combined capabilities

of mobile users and edge servers. In this proposed model, mobile users are responsible for fine-tuning the first several layers of the LLM, capitalizing on the parameter-efficient techniques that require less computational power and are thus more suitable for mobile environments. At the same time, the edge servers undertake the task of training the remaining layers, leveraging their processing capabilities to manage the more resource-intensive aspects of the training process.

Nevertheless, it has been found by many works that the prevalent fine-tuning methods are afflicted with instability concerns [\[9,](#page-9-10) [21,](#page-9-11) [34\]](#page-9-12). The fine-tuning approach, partitioning the computation process between users and edge servers, inherently introduces potential variances in the model's learning dynamics. When the initial several layers of a large language model are tailored to the idiosyncrasies of local data, these layers may start to generate data representations that are highly specialized or customized to the user's local context. Such misalignment can manifest as model instability, where small variations in local data could result in disproportionately large changes in the model's output, reducing its reliability and robustness in real-world applications. Thus, addressing model stability is essential. A stable model ensures minor changes in the training process don't lead to large performance discrepancies, making the model robust and reliable across various conditions and data distributions. This goal is even more critical in our proposed collaborative training framework, where the training workload is divided between mobile devices and edge servers. To solve this problem, we propose to incorporate model stability as a component of our objective function. This approach aims to reduce performance variance across training instances, ensuring that the fine-tuning process yields consistently high-quality results, regardless of the minor fluctuations inherent in distributed training environments. The contributions of this paper are summarized as follows:

- We introduce a collaborative training framework that combines mobile users and edge servers. This framework leverages PEFT methods, allowing mobile users to adjust the initial layers of the LLM while edge servers handle the more demanding latter layers.
- We formulate a multi-objective optimization problem that aims to concurrently minimize total energy consumption and userexperienced delay. At the same time, we enhance the stability of LLMs by integrating model stability considerations into our optimization objectives.
- To quantify the relationship between the number of the finetuned layers and the model stability, we provide the upper bound of the average-replace-one stability through theoretical analysis.
- To address the multi-objective optimization problem, we divide the problem into two parts. In the first part, we optimize the offloading decisions and resource allocation through the application of a novel fractional programming technique, which could find a stationary point with local or global optimal guarantee. For the second part, the Concave-Convex Procedure (CCCP) is employed to optimize the user-to-edge association problem.

The structure of this paper is laid out as follows: Section [2](#page-1-0) reviews the literature and works related to our study. Section [3](#page-2-0) outlines the architecture of the MEC-based LLM framework. Following that, Section [4](#page-4-0) details the analytical exploration. The outcomes of our simulations are presented in Section [5.](#page-6-0) We conclude the paper with Section [6.](#page-8-0)

### <span id="page-1-0"></span>2 RELATED WORK

In this section, we review the existing literature related to our work. Resource allocation in mobile edge computing. In [\[20\]](#page-9-13), the authors propose a Lyapunov optimization-based dynamic computation offloading algorithm to optimize the execution delay and the task-dropping cost in MEC system. When addressing the offloading decision problem, they apply an exhaustive search strategy, assessing the objective values across three discrete options to determine the optimal solution. Dinh et al. [\[6\]](#page-9-14) propose to minimize the execution latency and the user devices' energy consumption in MEC. They use an exhaustive search approach and a semidefinite relaxation (SDR)-based approach to optimize the CPU frequency. However, the exhaustive search approach is not practical in implementation due to its high complexity, and the SDR-based approach has no global or local optimality guarantee. In [\[3\]](#page-9-15), Chen et al. optimize the computation resource allocated to each task to minimize the computation and communication delay. To handle the multiplication of two decision variables (i.e., the computation resource allocation and the offloading decision), they adopt alternative optimization (AO) techniques. Xu et al. [\[31\]](#page-9-16) formulate a cooperative resource optimization problem to optimize the offloading decision and resource allocation in vehicular edge computing. Yet, they decouple the resource allocation variables from the offloading decision variable, and then use a deep reinforcement learning-based approach to solve it. Zhan et al. [\[32\]](#page-9-17) optimize the computation offloading scheduling and resource allocation in unmanned aerial vehicle (UAV)-enabled MEC system. They propose a two-stage alternating optimization algorithm to optimize the offloading scheduling, resource allocation and time duration alternatively.

In contrast, Wang et al. [\[29\]](#page-9-18) obtain the optimal solution in a semi-closed form for offloading decisions and resource allocation in MEC with wireless power transfer. However, their study exclusively focuses on minimizing the total energy consumption without integrating delay considerations into the objective function. Consequently, while it facilitates the determination of the optimal CPU frequency, it inherently simplifies the selection process to the minimal processing unit frequency that meets the latency requirements. Nonetheless, in this paper, we incorporate delay considerations into the objective function, thereby introducing a higher level of complexity to the solution process for resource allocation. As a result, the optimal solution to our problem cannot be directly ascertained. In Table [1,](#page-2-1) a comparative analysis is presented between this paper and the aforementioned related works.

PEFT vs. In-Context Learning (ICL). Recent studies have demonstrated the superiority of PEFT methods over ICL in various scenarios. Mosbach et al. [\[22\]](#page-9-19) conduct a fair comparison of ICL and fine-tuning approaches across different tasks and model sizes. They find that fine-tuning outperforms in-context learning across different performance metrics. Liu et al. [\[16\]](#page-9-20) also rigorously demonstrate that PEFT surpasses ICL in both accuracy and computational efficiency.

The model stability of fine-tuned large language models. Extensive efforts have focused on developing algorithms aimed at improving the stability of the fine-tuning process. Based on the idea of dropout, Lee et al. [\[14\]](#page-9-21) present "Mixout" regularization technique to selectively combine the parameters of two pre-trained

<span id="page-2-1"></span>

| Reference        | Objective Function |              | Optimization technique used to solve       |
|------------------|--------------------|--------------|--------------------------------------------|
|                  | Energy             | Delay        | the multiplication of variables            |
|                  | incorporated       | incorporated | _                                          |
| Mao et al. [20]  | ×                  | ✓            | Exhaustive search-based strategy           |
| Chen et al. [3]  | ×                  | ✓            | Alternative optimization                   |
| Xu et al. [31]   | ×                  | ✓            | Deep reinforcement learning-based approach |
| Zhan et al. [32] | ✓                  | ×            | Alternating optimization                   |
| Wang et al. [29] | ✓                  | ×            | Lagrange duality method                    |
| This paper       | ✓                  | ✓            | Novel fractional programming technique     |

Table 1: A comparative overview of this paper and prior works on MEC.

<span id="page-2-2"></span>![](_page_2_Picture_2.jpeg)

Figure 1: The proposed system model consists of N mobile users and M edge servers. Our optimization problem aims to minimize energy consumption and delay while improving the LLM stability.

language models. This approach effectively regularizes the learning process, improving the stability of the model. Houlsby et al. [12] propose a transfer method based on a bottleneck adapter architecture. He et al. [11] conduct a comprehensive comparison between two PEFT methods: fine-tuning and adapter-based tuning. Their works demonstrate that selectively tuning a subset of the parameters from pre-trained models contributes to enhancing the stability of the model.

For the stability analysis, Fu et al. [7] harmonize the array of PEFT strategies by framing them within the paradigm of sparse fine-tuning models. They provide a theoretical analysis that highlights sparsity's function as a regulatory mechanism for the original model, effectively imposing a constraint on the upper limit of model stability. However, their reliance on pointwise hypothesis stability to evaluate model stability focuses on the sensitivity of individual predictions to changes in the training data. In contrast, our work employs the average-replace-one stability measure which assesses the model's overall performance variation when a single training instance is replaced. In edge computing, we focus more on maintaining high levels of service reliability and efficiency across the entire network, rather than optimizing the outcome for individual predictions. Average-replace-one stability aligns with this objective by providing a macroscopic view of model stability.

#### <span id="page-2-0"></span>SYSTEM MODEL

In this section, we first present the system model, including local computation model, edge computation model and LLM stability. After that, we formulate the multi-objective optimization problem. We consider an MEC system consisting of N users and M edge servers, as described in Figure 1. Assume all the users in the system

train LLMs with the same architecture. Let  $\Upsilon$  be the total number of transformer layers in the LLM. User n fine-tunes the first  $\alpha_n$ layers locally, after which the intermediate results are sent to a certain edge server to complete the remaining training process. Let  $d_n$  denote the length of input tokens of user n for training. For the energy and delay calculation for training LLMs, we follow the setting in [15]. Let  $\psi(d_n)$  be the FLOPs per token required to train one transformer layer,  $\psi(d_n) = 72Bd_nh^2 + 12Bd_n^2h$  where B is the batch size and h is the dimensionality of the hidden states.

### Local Computation Model

When user n is training one transformer layer locally, the delay for computation can be given by:

where  $f_n$  is the GPU frequency of user n,  $C_n^U$  is the number of cores of the GPU at user n and  $D_n^U$  is the number of FLOPs per cycle per core of the GPU. The relationship between the GPU's power consumption and its clock speed is cubic, i.e., power =  $\kappa_1 f_n^3$ . Here,  $\kappa_1$  is the coefficient reflecting the power usage per cubic cycle per second ([in Watt/(cycle/s)<sup>3</sup>]), dependent on the specific GPU architecture. Hence, when training one transformer layer, the energy expenditure for local computations is established as follows:

$$E_n^{cmp} = \kappa_1 f_n^3 \times T_n^{cmp} = \frac{\kappa_1 f_n^2 \psi(d_n)}{C_n^U D_n^U}.$$
 (2)

Upon completing local computations, users transmit the intermediate results to edge servers for further processing. The association between users and edge servers is represented by  $\chi_{n,m}$  with  $\chi_{n,m} = 1$  signifying that user *n* has selected edge server *m* for further computations, and  $\chi_{n,m} = 0$  indicating no such association. In this context, we adopt Frequency-Division Multiple Access (FDMA) such that communications between users and edge servers are free from mutual interference. The power used for transmission by user n is denoted as  $p_n$ . Following the principles of the Shannon-Hartley theorem [4], the transmission rate between user n and edge server *m* can be formulated as  $r_{n,m} = b_{n,m} \log_2(1 + \frac{g_{n,m}p_n}{\sigma^2 b_{n,m}})$ , where  $\sigma^2$ represents the power of the noise,  $b_{n,m}$  denotes the bandwidth that edge server m assigned to user n,  $p_n$  is the transmission power of user n, and  $g_{n,m}$  is the channel gain between user n and edge server m. Let  $s(d_n)$  be the size of the intermediate results for user n. Therefore, the energy consumption of wireless transmission for user *n* is:

 $E_n^{com} = \sum_{m \in \mathcal{M}} \chi_{n,m} \frac{s(d_n)p_n}{r_{n,m}}.$ 

When user n is training the first  $\alpha_n$  layers locally, the computation of both time and energy expenditure is:  $Cost_n^u = \alpha_n \cdot (\omega_t T_n^{cmp} + \omega_e E_n^{cmp}) + \omega_e E_n^{com}.$ 

$$Cost_n^u = \alpha_n \cdot (\omega_t T_n^{cmp} + \omega_e E_n^{cmp}) + \omega_e E_n^{com}. \tag{4}$$

Here,  $\omega_t$  serves as the weighting and normalization factor, reflecting the priority given to minimizing delay, while  $\omega_e$  represents the weighting and normalization factor that underscores the importance of reducing energy consumption.

#### **Edge Computation Model**

When edge server m trains one transformer layer for user n, the

time taken for the computation can be expressed as follows: 
$$T_{n,m}^{cmp} = \frac{\psi(d_n)}{f_{n,m}C_m^ED_m^E}, \tag{5}$$
 where  $f_{n,m}$  denotes the GPU frequency of edge server  $m$  assigned

to user n, and  $C_m^E$  represents the total core count of the GPU within edge server m, and  $D_m^E$  signifies the computational capability of each core, measured in floating point operations per cycle, for the GPU located at edge server m. Thus, the energy required by edge server *m* to train one transformer layer for user *n* can be quantified as follows:

 $E_{n,m}^{cmp} = \frac{\kappa_2 f_{n,m}^2 \psi(d_n)}{C_m^E D_m^E}, \label{eq:energy_energy}$ (6)

where  $\kappa_2$  is a coefficient that varies based on the architecture of the chip. The energy used for downlink transmission from the edge servers to the users is not considered in this calculation, due to the substantially higher power capacities of the edge servers compared to the users. Furthermore, in comparison to the energy requirements for training the LLM, the energy expended on transmission by the edge servers is considered negligible.

Since there are  $\Upsilon$  transformer layers in total,  $(\Upsilon - \alpha_n)$  layers are processed at the corresponding edge server. As a result, the incurred cost for conducting the training tasks for users at edge server m is calculated by integrating both the time delays and

energy expenditures into a weighted sum:
$$Cost_{m}^{E} = \sum_{n \in \mathcal{N}} \chi_{n,m} (\Upsilon - \alpha_{n}) (\omega_{t} T_{n,m}^{cmp} + \omega_{e} E_{n,m}^{cmp}). \tag{7}$$

#### 3.3 LLM Stability

In this paper, we use the Average-replace-one Stability (AS) proposed by [25] to measure the mode stability. AS is a measure of how much an individual prediction is affected by small changes in the training dataset. It serves as a crucial metric for ensuring that our fine-tuned language model remains consistent and reliable, despite the variability in local data from user to user. Next, we give the definition of the average-replace-one stability.

Definition 1 (Average-Replace-one stability). Given a loss function  $\ell$  and training dataset  $S = \{z_1, \ldots, z_k\}$ , an algorithm  $\mathcal A$ demonstrates the average-replace-one stability (AS) with a bound  $\beta$ if the following condition is met:  $\forall i \in \{1, ..., k\}$ ,

$$\mathbb{E}_{\mathcal{S}}\left[\left|\ell(\mathcal{A}(\mathcal{S}),z_i)-\ell(\mathcal{A}(\mathcal{S}^i),z_i)\right|\right] \leq \beta, \tag{8}$$
 where  $\mathcal{A}(\mathcal{S})$  denotes the model obtained after the algorithm  $\mathcal{A}$  has been trained on the dataset  $\mathcal{S}$ , and  $\ell(\mathcal{A}(\mathcal{S}),z_i)$  is the loss function evaluated at a particular data point  $z_i$  using the model given by  $\mathcal{A}(\mathcal{S})$ .  $\mathcal{S}^i$  represents the training dataset with the  $i$ -th sample replaced with  $z_i'$ , i.e.,  $\mathcal{S}^i = \{z_1,\ldots,z_{i-1},z_i',\ldots,z_k\}$ .

This definition implies that for every individual element  $z_i$  in a dataset of size k, the expected disparity in the loss computed by algorithm  $\mathcal{A}$  when trained with the complete dataset versus the dataset lacking that specific sample is bounded by  $\beta$ .

#### **Problem Formulation**

With the computation and communication model above, we then formulate the joint optimization problem that aims to minimize the system's cost while minimizing the Average-replace-one Stability (AS) of the LLMs, by optimizing the following variables: the number of transformer layers that execute locally:  $\alpha := [\alpha_n|_{n \in \mathcal{N}}],$ 

the user-to-edge server association:  $\chi:=[\chi_{n,m}|_{n\in\mathcal{N},m\in\mathcal{M}}],$  the transmission power of the users:  $p := [p_n|_{n \in \mathcal{N}}]$ , the bandwidth allocation:  $b:=[b_{n,m}|_{n\in\mathcal{N},m\in\mathcal{M}}]$ , the users' GPU frequency:  $f^U:=$  $[f_n|_{n\in\mathbb{N}}]$  and the edge servers' GPU frequency allocation:  $f^E:=$  $[f_{n,m}|_{n\in\mathcal{N},m\in\mathcal{M}}]$ . Similar to delay and energy, we also give a weighting and normalization parameter  $\omega_s$  to the AS. The joint optimiza-

tion problem is formulated as follows:

Problem 
$$\mathbb{P}_1$$
: min  $\sum_{\alpha,\chi,\boldsymbol{p},\boldsymbol{b},\boldsymbol{f}^U\boldsymbol{f}^E} \sum_{n\in\mathcal{N}} Cost_n^u + \sum_{m\in\mathcal{M}} Cost_m^E + \omega_s AS$ , (9)

s.t. 
$$\alpha_n \in \{1, 2, \dots, \Upsilon\}, \forall n \in \mathcal{N},$$
 (9a)

<span id="page-3-6"></span><span id="page-3-3"></span><span id="page-3-0"></span>
$$\chi_{n,m} \in \{0,1\}, \forall n \in \mathcal{N}, m \in \mathcal{M},$$
(9b)

$$\sum_{m \in \mathcal{M}} \chi_{n,m} = 1, \forall n \in \mathcal{N},$$

$$p_n \le p_n^{max}, \forall n \in \mathcal{N},$$
(9c)

<span id="page-3-8"></span><span id="page-3-7"></span><span id="page-3-5"></span>
$$p_n \le p_n^{max}, \forall n \in \mathcal{N},$$
 (9d)

$$\sum_{n \in \mathcal{N}} \chi_{n,m} b_{n,m} = b_m^{max}, \forall m \in \mathcal{M},$$
 (9e)

<span id="page-3-4"></span>
$$f_n \le f_n^{max}, \forall n \in \mathcal{N}, \tag{9f}$$

$$\sum_{n \in \mathcal{N}} \chi_{n,m} f_{n,m} = f_m^{max}, \forall m \in \mathcal{M}.$$
 (9g)

Given the inherent challenges in quantifying the average-replaceone stability, we commence by presenting the following theorem to facilitate addressing the optimization problem. We assume the loss function  $\ell(\cdot)$  is L-Lipschitz and strong convex. These two assumptions are widely employed in the analysis of the behavior of neural networks [23, 24].

<span id="page-3-1"></span>THEOREM 1. If a user fine-tunes a proportion  $\alpha$  of the parameters, the expectation of the loss has an AS bounded by  $\frac{2L^2}{k(1-\alpha)}$ . I.e.,  $\forall i \in$ 

<span id="page-3-2"></span>
$$\mathbb{E}_{\mathcal{S}}\left[\left|\ell(\mathcal{A}(\mathcal{S}), z_i) - \ell(\mathcal{A}(\mathcal{S}^i), z_i)\right|\right] \le \frac{2L^2}{k(1-\alpha)}.\tag{10}$$

 $\{1,\ldots,k\}$ ,  $\mathbb{E}_{\mathcal{S}}\left[\left|\ell(\mathcal{A}(\mathcal{S}),z_i)-\ell(\mathcal{A}(\mathcal{S}^i),z_i)\right|\right] \leq \frac{2L^2}{k(1-\alpha)}$ . (10) PROOF. The proof can be found in Appendix A. Theorem 1 provides a quantifiable measure of model stability and bridges the concept of "model stability" with a measurable quantity. Since the "model stability" term is not quantitative in problem  $\mathbb{P}_1$ , we re-formulate problem  $\mathbb{P}_1$  into the following  $\mathbb{P}_2$  by replacing the

we re-formulate problem 
$$\mathbb{F}_1$$
 into the following  $\mathbb{F}_2$  by replacing the sum of AS with the sum of the AS's upper bound of all the users:  
Problem  $\mathbb{F}_2$ :  $\min_{\boldsymbol{\alpha}, \boldsymbol{\chi}, \boldsymbol{p}, \boldsymbol{b}, \boldsymbol{f}^U f^E} \sum_{n \in \mathcal{N}} Cost_n^u + \sum_{m \in \mathcal{M}} Cost_m^E + \omega_s \sum_{n \in \mathcal{N}} \frac{2L^2}{k_n(1 - \frac{\alpha_n}{1})}$ , (11) s.t. (9a) – (9g).

While the optimal solutions to problem  $\mathbb{P}_1$  and  $\mathbb{P}_2$  may not be strictly equivalent in a mathematical sense,  $\mathbb{P}_2$  serves as a practical approximation of  $\mathbb{P}_1$ . By using the upper bound from Theorem 1, we are optimizing for the worst-case scenario of model instability. Problem  $\mathbb{P}_2$  falls into the category of Mixed Integer Nonlinear Programming (MINLP) problem. This classification arises due to the inclusion of both integer-valued decision variables and nonlinear terms involving products of variables, a combination that inherently induces non-convexity in the problem space. The non-convex nature of this problem makes it especially challenging to solve because it cannot be addressed using standard optimization methods, which typically rely on the problem being convex. In order to tackle the non-convex problem, we optimize  $\alpha$ ,  $\rho$ , b,  $f^U$ ,  $f^E$  and  $\gamma$ iteratively. Specifically, in the first step, we fix  $\chi$  and utilize a novel fractional programming technique motivated by Zhao et al. [33] to optimize  $\alpha$ , p, b,  $f^U$ ,  $f^E$  by transforming the non-convex problem into a series of parametric convex problems. In the second step, given  $\alpha$ , p, b,  $f^{\bar{U}}$ ,  $f^{E}$ , the method of CCCP is adopted to facilitate the solution to  $\gamma$  by solving a sequence of convex problems.

#### <span id="page-4-0"></span>PROPOSED ALGORITHM

In this section, we provide a detailed solution to the optimization

## Optimizing $\alpha$ , p, b, $f^U$ , $f^E$ given $\chi$

The discrete variable  $\alpha_n$  is difficult to handle. Thus, we first relax  $\alpha_n$ to continuous variables, which will be rounded back to the nearest integer later. For problem  $\mathbb{P}_2$ , to optimize  $\pmb{\alpha}, \pmb{p}, \pmb{b}, f^U, f^E$  given  $\pmb{\chi}$ means to solve the following optimization problem:

Problem 
$$\mathbb{P}_{3}(\boldsymbol{\chi}): \min_{\boldsymbol{\alpha},\boldsymbol{p},\boldsymbol{b},\boldsymbol{f}^{U},\boldsymbol{f}^{E}} H(\boldsymbol{\alpha},\boldsymbol{p},\boldsymbol{b},\boldsymbol{f}^{U},\boldsymbol{f}^{E}) = \sum_{n\in\mathcal{N}} Cost_{n}^{u} + \sum_{m\in\mathcal{M}} Cost_{m}^{E} + \omega_{s} \sum_{n\in\mathcal{N}} \frac{2L^{2}}{k_{n}\cdot(1-\frac{\alpha_{n}}{1})},$$
 (12) s.t.  $1 \leq \alpha_{n} \leq \Upsilon, \forall n \in \mathcal{N},$  (9d)-(9g).

Problem  $\mathbb{P}_3$  involves fraction term and multiplication terms, which makes it difficult to solve using standard optimization algorithms. Motivated by the novel fractional programming technique proposed in [33], we next transform problem  $\mathbb{P}_3$  into a series of  $\mathbb{P}_4$ :

Problem 
$$\mathbb{P}_4(\chi, z, \nu, q) : \min_{\alpha, p, b, f^U, f^E} K(\alpha, p, b, f^U, f^E, z, \nu, q) =$$

$$\begin{split} & \sum_{n \in \mathcal{N}} \left( \alpha_{n}^{2} z_{n} + \frac{(\omega_{t} \frac{\psi(d_{n})}{f_{n} C_{n}^{U} D_{n}^{U}} + \omega_{e} \frac{\kappa_{1} f_{n}^{2} \psi(d_{n})}{C_{n}^{U} D_{n}^{U}} \right)^{2}}{4z_{n}} \right) + \\ & \omega_{e} \sum_{n \in \mathcal{N}} \sum_{m \in \mathcal{M}} \chi_{n,m} \left( (p_{n} d_{n})^{2} v_{n,m} + \frac{1}{4r_{n,m}^{2} v_{n,m}} \right) + \\ & \sum_{n \in \mathcal{N}} \sum_{m \in \mathcal{M}} \chi_{n,m} \left( (\Upsilon - \alpha_{n})^{2} q_{n,m} + \frac{\left( \omega_{t} \frac{\psi(d_{n})}{f_{n,m} C_{m}^{E} D_{m}^{E}} + \omega_{e} \frac{\kappa_{2} f_{n,m}^{2} \psi(d_{n})}{C_{m}^{E} D_{m}^{E}} \right)^{2}}{4q_{n,m}} \right) \\ & + \omega_{s} \sum_{n \in \mathcal{N}} \frac{2L^{2}}{k_{n} \cdot (1 - \frac{\alpha_{n}}{\Upsilon})}, \end{split} \tag{13}$$

where the auxiliary variables  $z := [z_1, z_2, ..., z_n]$  with  $z_n > 0$ ,  $\nu :=$  $[v_{1,1}, v_{1,2}, \dots, v_{1,m}, \dots, v_{n,m}]$  with  $v_{n,m} > 0$  and  $q := [q_{1,1}, q_{1,2}, \dots, q_{n,m}]$ 

 $q_{1,m}, \ldots, q_{n,m}$ ] with  $q_{n,m} > 0$ .

Problem  $\mathbb{P}_3$  involves non-convex terms and is difficult to handle. Therefore, we formulate the above problem  $\mathbb{P}_4$ . Next, we introduce an AO algorithm for problem  $\mathbb{P}_4$ . After that, we propose Proposition 1 to explain how we can tackle problem  $\mathbb{P}_3$  through a series of convex problem  $\mathbb{P}_4$  instances.

First, we introduce the AO algorithm for problem  $\mathbb{P}_4$ . Overall, we alternatively optimize  $\alpha$ , p, b,  $f^U$ ,  $f^E$  and z, v, q. Specifically, we begin with an initial feasible  $[\boldsymbol{\alpha}^{(0)}, \boldsymbol{p}^{(0)}, \boldsymbol{b}^{(0)}, f^{U^{(0)}}, f^{E^{(0)}}]$ . Next, we denote  $A(f_n)$  and  $B(f_{n,m})$  as:

$$A(f_n) = \omega_t \frac{\psi(d_n)}{f_n C_n^U D_n^U} + \omega_e \frac{\kappa_1 f_n^2 \psi(d_n)}{C_n^U D_n^U}, \tag{14}$$

$$B(f_{n,m}) = \omega_t \frac{\psi(d_n)}{f_{n,m} C_m^E D_m^E} + \omega_e \frac{\kappa_2 f_{n,m}^2 \psi(d_n)}{C_m^E D_m^E}.$$
 (15)

We assign  $z_n^{(0)}$  to be  $\frac{A(f_n^{(0)})}{2\alpha_n^{(0)}}$ , which is the optimal value of  $z_n$  when optimizing  $\alpha_n^2 z_n + \frac{A^2(f_n)}{4z_n}$  with respect to  $z_n$ , while keeping  $\alpha_n$ ,  $f_n$  fixed at  $\alpha_n^{(0)}$ ,  $f_n^{(0)}$ ; we assign  $v_{n,m}^{(0)}$  to be  $\frac{1}{2p_n^{(0)}d_n^{(0)}r_{n,m}^{(0)}}$ , which is the optimal value of  $v_{n,m}$  when optimizing  $(p_n d_n)^2 v_{n,m} + \frac{1}{4r_{n,m}^2}$ with respect to  $v_{n,m}$ , while keeping  $p_n, d_n$  fixed at  $p_n^{(0)}, d_n^{(0)}$ ; we assign  $q_{n,m}^{(0)}$  to be  $\frac{B(f_{n,m}^{(0)})}{2(\Upsilon-\alpha_n^{(0)})}$ , which is the optimal value of  $q_{n,m}$  when optimizing  $(\Upsilon - \alpha_n)^2 q_{n,m} + \frac{B^2(f_{n,m})}{4q_{n,m}}$  with respect to  $q_{n,m}$ , while keeping  $\alpha_n$ ,  $f_{n,m}$  fixed at  $\alpha_n^{(0)}$ ,  $f_{n,n}^{(0)}$ 

<span id="page-4-1"></span>After that, we solve problem  $\mathbb{P}_4(\chi, z^{(0)}, v^{(0)}, q^{(0)})$ , from which we derive the solution  $[\boldsymbol{\alpha}^{(1)}, \boldsymbol{p}^{(1)}, \boldsymbol{b}^{(1)}, f^{U(1)}, f^{E(1)}]$ , and subsequently update  $[z^{(1)}, v^{(1)}, q^{(1)}]$ . This procedure is repeated in an iterative fashion: during the (t + 1)-th iteration, we set  $z_n^{(t)}$ ,  $v_{n,m}^{(t)}$ and  $q_{n,m}^{(t)}$  as  $\frac{A(f_n^{(t)})}{2a_n^{(t)}}$ ,  $\frac{1}{2p_n^{(t)}d_n^{(t)}r_{n,m}^{(t)}}$  and  $\frac{B(f_{n,m}^{(t)})}{2(\Upsilon-\alpha_n^{(t)})}$ , and then solve  $\mathbb{P}_{4}(\chi, z^{(t)}, \boldsymbol{\nu}^{(t)}, \boldsymbol{q}^{(t)}), \text{ to obtain } [\boldsymbol{\alpha}^{(t+1)}, \boldsymbol{p}^{(t+1)}, \boldsymbol{b}^{(t+1)}, \boldsymbol{f}^{U^{(t+1)}}].$  $f^{E\,(t+1)}$  ]. The above AO process converges when the difference between the objective function of problem  $\mathbb{P}_4(\pmb{\chi},\pmb{z}^{(t-1)},\pmb{v}^{(t-1)},\pmb{q}^{(t-1)})$ and problem  $\mathbb{P}_4(\chi, \mathbf{z}^{(t)}, \mathbf{v}^{(t)}, \mathbf{q}^{(t)})$  falls below a predefined small error tolerance. Then, we propose the following proposition to explain how we solve  $\mathbb{P}_3$  through the AO process for  $\mathbb{P}_4$ .

<span id="page-4-4"></span><span id="page-4-3"></span>PROPOSITION 1. We can derive a stationary point for problem  $\mathbb{P}_3$ by applying the AO process outlined above for problem  $\mathbb{P}_4$  until con-

<span id="page-4-2"></span>PROOF. Denote " $\alpha$ , p, b,  $f^U$ ,  $f^{E}$ " as " $\bigstar$ " and "z,  $\nu$ , q" as " $\spadesuit$ ". In the first step in the AO process, we optimize  $\spadesuit$  while keeping  $\bigstar$  fixed, i.e., letting  $z_n^{\#} = \frac{A(f_n)}{2\alpha_n}$ ,  $v_{n,m}^{\#} = \frac{1}{2p_nd_nr_{n,m}}$ ,  $q_{n,m}^{\#} = \frac{B(f_{n,m})}{2(\Upsilon-\alpha_n)}$ . When we substitute back  $z_n^{\#}$ ,  $v_{n,m}^{\#}$ ,  $q_{n,m}^{\#}$  to  $K(\bigstar, \spadesuit)$ ,  $K(\bigstar, \spadesuit)$  will become  $H(\bigstar)$ , i.e.,

<span id="page-4-5"></span>
$$K(\bigstar, \blacklozenge)|_{z_n = z_n^{\sharp}, \nu_{n,m} = \nu_{n,m}^{\sharp}, q_{n,m} = q_{n,m}^{\sharp}} = H(\bigstar). \tag{16}$$

Next, we investigate the partial derivative of  $K(\bigstar, \spadesuit)$  w.r.t  $\bigstar$ :

ext, we investigate the partial derivative of 
$$K(\bigstar, \blacklozenge)$$
 w.r.t  $\bigstar$ :
$$\frac{\partial K(\bigstar, \blacklozenge)}{\partial \alpha_n} = \frac{2L^2 \omega_s}{k_n \Upsilon \cdot (1 - \frac{\alpha_n}{\Upsilon})^2} + 2z_n \alpha_n - 2\sum_{m \in \mathcal{M}} \chi_{n,m} q_{n,m} \cdot (\Upsilon - \alpha_n), \quad (17)$$

$$\frac{\partial K(\bigstar, \blacklozenge)}{\partial p_n} = \sum_{m \in \mathcal{M}} \chi_{n,m} \omega_e \cdot \left( \frac{2d_n^2 v_{n,m} p_n - \frac{\ln^2(2) g_{n,m}}{2b_{n,m}^3 v_{n,m} \sigma^2} \cdot \left( \frac{g_{n,m} p_n}{b_{n,m} \sigma^2} + 1 \right) \ln^3 \left( \frac{g_{n,m} p_n}{b_{n,m} \sigma^2} + 1 \right) \right)$$
(18)

$$\frac{\partial K(\bigstar, \spadesuit)}{\partial b_{n,m}} = \frac{\chi_{n,m}\omega_e \ln^2(2)}{2\nu_{n,m}\ln^2\left(\frac{g_{n,m}p_n}{\sigma^2 b_{n,m}} + 1\right)b_{n,m}^3}.$$

$$\frac{\partial K(\bigstar, \blacklozenge)}{\partial b_{n,m}} = \frac{\chi_{n,m}\omega_e \ln^2(2)}{2\nu_{n,m} \ln^2\left(\frac{g_{n,m}p_n}{\sigma^2 b_{n,m}} + 1\right) b_{n,m}^3} \cdot \left(\frac{g_{n,m}p_n}{\sigma^2 \ln\left(\frac{g_{n,m}p_n}{\sigma^2 b_{n,m}} + 1\right) \cdot \left(\frac{g_{n,m}p_n}{\sigma^2 b_{n,m}} + 1\right) b_{n,m}} - 1\right), \tag{19}$$

<span id="page-4-6"></span>
$$\frac{\partial K(\bigstar, \blacklozenge)}{\partial f_n} = \frac{\psi^2(d_n) \cdot \left(\kappa_1 \omega_e f_n^3 + \omega_t\right) \cdot \left(2\kappa_1 \omega_e f_n^3 - \omega_t\right)}{2z_n f_n^3 \cdot (C_n^U D_n^U)^2},\tag{20}$$

$$\frac{\partial K(\mathbf{x}, \mathbf{\phi})}{\partial f_n} = \frac{\psi^2(d_n) \cdot (\kappa_1 \omega_e f_n^3 + \omega_t) \cdot (2\kappa_1 \omega_e f_n^3 - \omega_t)}{2z_n f_n^3 \cdot (C_n^U D_n^U)^2}, \qquad (20)$$

$$\frac{\partial K(\mathbf{x}, \mathbf{\phi})}{\partial f_{n,m}} = \frac{\chi_{n,m} \psi^2(d_n) \cdot (\kappa_2 \omega_e f_{n,m}^3 + \omega_t) \cdot (2\kappa_2 \omega_e f_{n,m}^3 - \omega_t)}{2q_{n,m} f_{n,m}^3 \cdot (C_n^E D_n^E)^2}. (21)$$

From (17) to (21), it can be found that

<span id="page-4-7"></span>
$$\left(\frac{\partial K(\bigstar, \blacklozenge)}{\partial \alpha_{n}}\right)\Big|_{z_{n}=\frac{A(f_{n})}{2\alpha_{n}},q_{n,m}=\frac{B(f_{n,m})}{2(\Upsilon-\alpha_{n})}} = \frac{2L^{2}\omega_{s}}{k_{n}\Upsilon\cdot\left(1-\frac{\alpha_{n}}{\Upsilon}\right)^{2}} + A(f_{n}) - \sum_{m\in\mathcal{M}}\chi_{n,m} \cdot B(f_{n,m}), \tag{22}$$

$$\left(\frac{\partial K(\bigstar, \blacklozenge)}{\partial p_{n}}\right)\Big|_{v_{n,m}=\frac{1}{2p_{n}d_{n}r_{n,m}}} = \sum_{m\in\mathcal{M}}\chi_{n,m}\omega_{e} \cdot \left(\frac{d_{n}}{b_{n,m}\log_{2}(1+\frac{g_{n,m}p_{n}}{\sigma^{2}b_{n,m}})} - \frac{\ln(2)g_{n,m}p_{n}d_{n}}{b_{n,m}(g_{n,m}p_{n}+b_{n,m}\sigma^{2})\ln^{2}(\frac{g_{n,m}p_{n}}{b_{n,m}\sigma^{2}}+1)}\right), \tag{23}$$

$$\left(\frac{\partial K(\bigstar, \blacklozenge)}{\partial b_{n,m}}\right)\Big|_{v_{n,m}=\frac{1}{2p_{n}d_{n}r_{n,m}}} = \frac{\chi_{n,m}\omega_{e}\ln(2)p_{n}d_{n}}{b_{n}^{2}m\ln(\frac{g_{n,m}p_{n}}{\sigma^{2}}+1)} \cdot \frac{g_{n,m}p_{n}}{b_{n}^{2}m\ln(\frac{g_{n,m}p_{n}}{\sigma^{2}}+1)}\right)$$

$$\left(\frac{g_{n,m}p_n}{\sigma^2 \ln\left(\frac{g_{n,m}p_n}{\sigma^2 b_{n,m}}+1\right) \cdot \left(\frac{g_{n,m}p_n}{\sigma^2 b_{n,m}}+1\right) b_{n,m}}-1\right),\tag{24}$$

$$\left(\frac{\partial K(\bigstar, \spadesuit)}{\partial f_n}\right)|_{z_n = \frac{A(f_n)}{2\alpha_n}} = \frac{\alpha_n \psi(d_n) \cdot (2\kappa_1 \omega_e f_n^3 - \omega_t)}{f_n^2 C_n^V D_n^V},\tag{25}$$

$$\left(\frac{\partial K(\bigstar, \spadesuit)}{\partial f_{n,m}}\right)\Big|_{q_{n,m} = \frac{B(f_{n,m})}{2(1-\alpha_n)}} = \frac{(\Upsilon - \alpha_n)\chi_{n,m}\psi(d_n) \cdot (2\kappa_2\omega_e f_{n,m}^3 - \omega_t)}{f_{n,m}^2 C_m^R D_m^R}. (26)$$

Besides, the partial derivative of  $H(\bigstar)$  is given by:

$$\frac{\partial H(\bigstar)}{\partial \alpha_n} = \omega_t \frac{\psi(d_n)}{f_n C_n^U D_n^U} + \omega_e \frac{\kappa_1 f_n^2 \psi(d_n)}{C_n^U D_n^U} + \frac{2L^2 \omega_s}{k_n \Upsilon \cdot \left(1 - \frac{\alpha_n}{\Upsilon}\right)^2} - \sum_{m \in \mathcal{M}} \chi_{n,m}(\omega_t \frac{\psi(d_n)}{f_{n,m} C_m^E D_m^E} + \omega_e \frac{\kappa_2 f_{n,m}^2 \psi(d_n)}{C_m^E D_m^E}), \tag{27}$$

$$\frac{\partial H(\bigstar)}{\partial p_n} = \sum_{m \in \mathcal{M}} \ln\left(2\right) \omega_e \chi_{n,m} d_n \cdot$$

$$\frac{(g_{n,m}p_n + b_{n,m}\sigma^2)\ln(\frac{g_{n,m}p_n}{b_{n,m}\sigma^2} + 1) - g_{n,m}p_n}{b_{n,m}\sigma^2 + b_{n,m}\sigma^2 \ln^2(\frac{g_{n,m}p_n}{b_n} + 1)},$$
(28)

$$\frac{\left(g_{n,m}p_{n}+b_{n,m}\sigma^{2}\right)\ln\left(\frac{g_{n,m}p_{n}}{b_{n,m}\sigma^{2}}+1\right)-g_{n,m}p_{n}}{b_{n,m}\cdot\left(g_{n,m}p_{n}+b_{n,m}\sigma^{2}\right)\ln^{2}\left(\frac{g_{n,m}p_{n}}{b_{n,m}\sigma^{2}}+1\right)}, \qquad (28)$$

$$\frac{\partial H(\bigstar)}{\partial b_{n,m}} = \frac{\omega_{e}\ln\left(2\right)\chi_{n,m}d_{n}g_{n,m}p_{n}^{2}}{\sigma^{2}\ln^{2}\left(\frac{g_{n,m}p_{n}}{\sigma^{2}b_{n,m}}+1\right)\left(\frac{g_{n,m}p_{n}}{\sigma^{2}b_{n,m}}+1\right)b_{n,m}^{3}} - \frac{\omega_{e}\ln\left(2\right)\chi_{n,m}d_{n}p_{n}}{\ln\left(\frac{g_{n,m}p_{n}}{\sigma^{2}b_{n,m}}+1\right)b_{n,m}^{2}} + \frac{\omega_{e}\ln\left(2\right)\chi_{n,m}d_{n}p_{n}}{\ln\left(\frac{g_{n,m}p_{n}}{\sigma^{2}b_{n,m}}+1\right)b_{n,m}^{2}} + \frac{\omega_{e}\ln\left(2\right)\chi_{n,m}d_{n}p_{n}}{\ln\left(\frac{g_{n,m}p_{n}}{\sigma^{2}b_{n,m}}+1\right)b_{n,m}^{2}} + \frac{\omega_{e}\ln\left(2\right)\chi_{n,m}d_{n}p_{n}}{\ln\left(\frac{g_{n,m}p_{n}}{\sigma^{2}b_{n,m}}+1\right)b_{n,m}^{2}} + \frac{\omega_{e}\ln\left(2\right)\chi_{n,m}d_{n}p_{n}}{\ln\left(\frac{g_{n,m}p_{n}}{\sigma^{2}b_{n,m}}+1\right)b_{n,m}^{2}} + \frac{\omega_{e}\ln\left(2\right)\chi_{n,m}d_{n}p_{n}}{\ln\left(\frac{g_{n,m}p_{n}}{\sigma^{2}b_{n,m}}+1\right)b_{n,m}^{2}} + \frac{\omega_{e}\ln\left(2\right)\chi_{n,m}d_{n}p_{n}}{\ln\left(\frac{g_{n,m}p_{n}}{\sigma^{2}b_{n,m}}+1\right)b_{n,m}^{2}} + \frac{\omega_{e}\ln\left(2\right)\chi_{n,m}d_{n}p_{n}}{\ln\left(\frac{g_{n,m}p_{n}}{\sigma^{2}b_{n,m}}+1\right)b_{n,m}^{2}} + \frac{\omega_{e}\ln\left(2\right)\chi_{n,m}d_{n}p_{n}}{\ln\left(\frac{g_{n,m}p_{n}}{\sigma^{2}b_{n,m}}+1\right)b_{n,m}^{2}} + \frac{\omega_{e}\ln\left(2\right)\chi_{n,m}d_{n}p_{n}}{\ln\left(\frac{g_{n,m}p_{n}}{\sigma^{2}b_{n,m}}+1\right)b_{n,m}^{2}} + \frac{\omega_{e}\ln\left(2\right)\chi_{n,m}d_{n}p_{n}}{\ln\left(\frac{g_{n,m}p_{n}}{\sigma^{2}b_{n,m}}+1\right)b_{n,m}^{2}} + \frac{\omega_{e}\ln\left(2\right)\chi_{n,m}d_{n}p_{n}}{\ln\left(\frac{g_{n,m}p_{n}}{\sigma^{2}b_{n,m}}+1\right)b_{n,m}^{2}} + \frac{\omega_{e}\ln\left(2\right)\chi_{n,m}d_{n}p_{n}}{\ln\left(\frac{g_{n,m}p_{n}}{\sigma^{2}b_{n,m}}+1\right)b_{n,m}^{2}} + \frac{\omega_{e}\ln\left(2\right)\chi_{n,m}d_{n}p_{n}}{\ln\left(\frac{g_{n,m}p_{n}}{\sigma^{2}b_{n,m}}+1\right)b_{n,m}^{2}} + \frac{\omega_{e}\ln\left(2\right)\chi_{n,m}d_{n}p_{n}}{\ln\left(\frac{g_{n,m}p_{n}}{\sigma^{2}b_{n,m}}+1\right)b_{n,m}^{2}} + \frac{\omega_{e}\ln\left(2\right)\chi_{n,m}d_{n}p_{n}}{\ln\left(\frac{g_{n,m}p_{n}}{\sigma^{2}b_{n,m}}+1\right)b_{n,m}^{2}} + \frac{\omega_{e}\ln\left(2\right)\chi_{n,m}d_{n}p_{n}}{\ln\left(\frac{g_{n,m}p_{n}}{\sigma^{2}b_{n,m}}+1\right)b_{n,m}^{2}} + \frac{\omega_{e}\ln\left(2\right)\chi_{n,m}d_{n}p_{n}}{\ln\left(\frac{g_{n,m}p_{n}}{\sigma^{2}b_{n,m}}+1\right)b_{n,m}^{2}} + \frac{\omega_{e}\ln\left(2\right)\chi_{n,m}d_{n}p_{n}}{\ln\left(\frac{g_{n,m}p_{n}}{\sigma^{2}b_{n}}+1\right)b_{n,m}^{2}} + \frac{\omega_{e}\ln\left(2\right)\chi_{n,m}d_{n}p_{n}}{\ln\left(\frac{g_{n,m}p_{n}}{\sigma^{2}b_{n}}+1\right)b_{n,m}^{2}} + \frac{\omega_{e}\ln\left(2\right)\chi_{n,m}d_{n}p_{n}}{\ln\left(\frac{g_{n,m}p_{n}}{\sigma^$$

$$\frac{\partial H(\star)}{\partial f_{-}} = -\frac{\alpha_n \omega_t \psi(d_n)}{C^V D^V f^2} + \frac{2\alpha_n \omega_e \kappa_1 f_n \psi(d_n)}{C^V D^V},$$
(30)

$$\frac{\partial H(\bigstar)}{\partial f_n} = -\frac{\alpha_n \omega_t \psi(d_n)}{C_n^V D_n^V f_n^2} + \frac{2\alpha_n \omega_e \kappa_1 f_n \psi(d_n)}{C_n^V D_n^V}, \tag{30}$$

$$\frac{\partial H(\bigstar)}{\partial f_{n,m}} = \chi_{n,m} (\Upsilon - \alpha_n) \left( \frac{2\omega_e \kappa_2 f_{n,m} \psi(d_n)}{C_m^R D_m^R} - \frac{\omega_t \psi(d_n)}{C_m^R D_m^R f_{n,m}^2} \right). \tag{31}$$

From (22)–(26) and (27)–(31), it can be observed

$$\frac{\partial H(\bigstar)}{\partial \alpha_n} = \left(\frac{\partial K(\bigstar, \blacklozenge)}{\partial \alpha_n}\right) |_{z_n = \frac{A(f_n)}{2\alpha_n}, q_{n,m} = \frac{B(f_{n,m})}{2(T - \alpha_n)}}, \tag{32a}$$

$$\frac{\partial H(\bigstar)}{\partial p_n} = \left(\frac{\partial K(\bigstar, \blacklozenge)}{\partial p_n}\right)|_{v_{n,m} = \frac{1}{2p_n d_n r_{n,m}}},$$

$$\frac{\partial H(\bigstar)}{\partial b_{n,m}} = \left(\frac{\partial K(\bigstar, \blacklozenge)}{\partial b_{n,m}}\right)|_{v_{n,m} = \frac{1}{2p_n d_n r_{n,m}}},$$
(32b)

$$\frac{\partial H(\bigstar)}{\partial b_{n,m}} = \left(\frac{\partial K(\bigstar, \blacklozenge)}{\partial b_{n,m}}\right)|_{v_{n,m} = \frac{1}{2p_{n}d_{n}r_{n,m}}},$$
(32c)

$$\frac{\partial H(\bigstar)}{\partial f_n} = \left(\frac{\partial K(\bigstar, \spadesuit)}{\partial f_n}\right)|_{z_n = \frac{A(f_n)}{2\sigma c_n}},$$
 (32d)

$$\frac{\partial H(\bigstar)}{\partial f_n} = \left(\frac{\partial K(\bigstar, \blacklozenge)}{\partial f_n}\right)|_{z_n = \frac{A(f_n)}{2\alpha_n}}, \tag{32d}$$

$$\frac{\partial H(\bigstar)}{\partial f_{n,m}} = \left(\frac{\partial K(\bigstar, \blacklozenge)}{\partial f_{n,m}}\right)|_{q_{n,m} = \frac{B(f_{n,m})}{2(\Gamma - \alpha_n)}}. \tag{32e}$$

The process of AO in minimizing  $K(\bigstar, \spadesuit)$  is non-increasing. To elaborate, it's observed that  $K(\bigstar^{(i)}, \blacklozenge^{(i)}) \leq K(\bigstar^{(i-1)}, \blacklozenge^{(i)}) \leq$  $K(\bigstar^{(i-1)}, \blacklozenge^{(i-1)})$ . Thus,  $K(\bigstar^{(i)}, \blacklozenge^{(i)})$  tends towards convergence as the iteration count i increases infinitely. Assume after the AO process,  $\bigstar$  and  $\blacklozenge$  converges to  $(\bigstar^*, \blacklozenge^*)$ . It indicates: (1)  $\blacklozenge^*$  is the optimial value of  $\blacklozenge$  if we fix  $\bigstar$  as  $\bigstar^*$  and minimize  $K(\bigstar^*, \spadesuit)$ , and ② ★\* is the optimial value of ★ if we fix ♦ as  $\blacklozenge$ \* and minimize  $K(\bigstar, \spadesuit^*).$ 

From result ②, we know that ★\* satisfies the Karush-Kuhn-Tucker (KKT) conditions of problem  $\mathbb{P}_4$ . With  $\gamma$ ,  $\mu$  denoting the Lagrange multipliers for the inequality constraints and equality constraints, respectively, the Lagrangian function of problem  $\mathbb{P}_4$  is

$$L_{\mathbb{P}_4}(\bigstar, \blacklozenge, \gamma, \mu) = K(\bigstar, \blacklozenge) + \sum_{n \in \mathcal{N}} \Big( \gamma_{1,n} (1 - \alpha_n) +$$

 $\gamma_{2,n}(\alpha_n - \Upsilon) + \gamma_{3,n}(p_n - p_n^{max}) + \gamma_{4,n}(f_n - f_n^{max}) + \gamma_{4,n}(f_n - f_n^{max})$ 

$$\sum_{m \in \mathcal{M}} \left( \mu_{1,m} \left( \sum_{n \in \mathcal{N}} \chi_{n,m} b_{n,m} - b_m^{max} \right) + \mu_{2,m} \left( \sum_{n \in \mathcal{N}} \chi_{n,m} f_{n,m} - f_m^{max} \right) \right). \tag{33}$$

The KKT conditions of problem  $\mathbb{P}_4$  is given by:

$$\frac{\partial L_{\mathbb{P}_4}}{\partial \boldsymbol{\alpha}^*} = \frac{\partial L_{\mathbb{P}_4}}{\partial \boldsymbol{p}^*} = \frac{\partial L_{\mathbb{P}_4}}{\partial \boldsymbol{b}^*} = \frac{\partial L_{\mathbb{P}_4}}{\partial \boldsymbol{f}^{U^*}} = \frac{\partial L_{\mathbb{P}_4}}{\partial \boldsymbol{f}^{E^*}} = 0, \tag{34}$$

Primal feasibility:

<span id="page-5-0"></span>
$$\star^*$$
 satisfy (12a), (9d) – (9g), (35)

<span id="page-5-3"></span>Dual feasibility:

$$\gamma_{1,n}, \gamma_{2,n}, \gamma_{3,n}, \gamma_{4,n} \ge 0, \forall n \in \mathcal{N},$$
 (36)

Complementary slackness:

$$\begin{cases} \gamma_{1,n}(1-\alpha_n^*) = 0, \gamma_{3,n}(p_n^* - p_n^{max}) = 0, \\ \gamma_{2,n}(\alpha_n^* - \Upsilon) = 0, \gamma_{4,n}(f_n^* - f_n^{max}) = 0, \forall n \in \mathcal{N}. \end{cases}$$
(37)

<span id="page-5-1"></span>We rewrite  $L_{\mathbb{P}_4}$  as  $L_{\mathbb{P}_4} = K(\bigstar, \blacklozenge) + Q(\bigstar, \gamma, \mu)$ . Then, (34) can be rewritten as:

<span id="page-5-6"></span>
$$\frac{\partial L_{\mathbb{P}_4}}{\partial \boldsymbol{\alpha}} = \frac{\partial K(\bigstar^*, \spadesuit^*)}{\partial \boldsymbol{\alpha}} + \frac{\partial Q(\bigstar^*, \gamma, \boldsymbol{\mu})}{\partial \boldsymbol{\alpha}} = 0, \tag{38a}$$

$$\frac{\partial L_{\mathbb{P}_4}}{\partial \boldsymbol{p}} = \frac{\partial K(\boldsymbol{\star}^*, \boldsymbol{\diamond}^*)}{\partial \boldsymbol{p}} + \frac{\partial Q(\boldsymbol{\star}^*, \boldsymbol{\gamma}, \boldsymbol{\mu})}{\partial \boldsymbol{p}} = 0, \tag{38b}$$

$$\frac{\partial L_{\mathbb{P}_4}}{\partial b} = \frac{\partial K(\bigstar^*, \spadesuit^*)}{\partial b} + \frac{\partial Q(\bigstar^*, \gamma, \mu)}{\partial b} = 0, \tag{38c}$$

$$\frac{\partial L_{\mathbb{P}_4}}{\partial \mathbf{f}^U} = \frac{\partial K(\mathbf{x}^*, \mathbf{\phi}^*)}{\partial \mathbf{f}^U} + \frac{\partial Q(\mathbf{x}^*, \mathbf{y}, \boldsymbol{\mu})}{\partial \mathbf{f}^U} = 0, \tag{38d}$$

$$\frac{\partial L_{\mathbb{P}_4}}{\partial f^E} = \frac{\partial K(\bigstar^*, \spadesuit^*)}{\partial f^E} + \frac{\partial Q(\bigstar^*, \gamma, \mu)}{\partial f^E} = 0.$$
(38e)

<span id="page-5-4"></span><span id="page-5-2"></span>Substituting (32a) – (32e) into (38a) – (38e), we can obtain:

<span id="page-5-8"></span><span id="page-5-7"></span>
$$\frac{\partial H(\bigstar^*)}{\partial \alpha} + \frac{\partial Q(\bigstar^*, \gamma, \mu)}{\partial \alpha} = 0, \tag{39a}$$

$$\frac{\partial H(\bigstar^*)}{\partial \alpha} + \frac{\partial Q(\bigstar^*, \gamma, \mu)}{\partial \alpha} = 0,$$

$$\frac{\partial H(\bigstar^*)}{\partial p} + \frac{\partial Q(\bigstar^*, \gamma, \mu)}{\partial p} = 0,$$
(39a)
(39b)

$$\frac{\partial H(\bigstar^*)}{\partial h} + \frac{\partial Q(\bigstar^*, \gamma, \mu)}{\partial h} = 0, \tag{39c}$$

$$\frac{\partial H(\bigstar^*)}{\partial b} + \frac{\partial Q(\bigstar^*, \gamma, \mu)}{\partial b} = 0, \qquad (39c)$$

$$\frac{\partial H(\bigstar^*)}{\partial f^U} + \frac{\partial Q(\bigstar^*, \gamma, \mu)}{\partial f^U} = 0, \qquad (39d)$$

<span id="page-5-9"></span>
$$\frac{\partial H(\bigstar^*)}{\partial f^E} + \frac{\partial Q(\bigstar^*, \gamma, \mu)}{\partial f^E} = 0. \tag{39e}$$

<span id="page-5-5"></span>At the same time, with  $\gamma$ ,  $\mu$  denoting the Lagrange multipliers, the Lagrangian function of problem  $\mathbb{P}_3$  can be given by:

$$L_{\mathbb{P}_3}(\bigstar, \blacklozenge, \gamma, \mu) = H(\bigstar) + Q(\bigstar, \gamma, \mu). \tag{40}$$

Therefore, (39a) - (39e) indicate that

$$\frac{\partial L_{\mathbb{P}_3}}{\partial \boldsymbol{\alpha}} = \frac{\partial L_{\mathbb{P}_3}}{\partial \boldsymbol{p}} = \frac{\partial L_{\mathbb{P}_3}}{\partial \boldsymbol{b}} = \frac{\partial L_{\mathbb{P}_3}}{\partial \boldsymbol{f}^U} = \frac{\partial L_{\mathbb{P}_3}}{\partial \boldsymbol{f}^E} = 0. \tag{41}$$

Then, (34) - (37) are equivalent to:

$$\frac{\partial L_{\mathbb{P}_3}}{\partial \boldsymbol{\alpha}} = \frac{\partial L_{\mathbb{P}_3}}{\partial \boldsymbol{p}} = \frac{\partial L_{\mathbb{P}_3}}{\partial \boldsymbol{b}} = \frac{\partial L_{\mathbb{P}_3}}{\partial \boldsymbol{f}^U} = \frac{\partial L_{\mathbb{P}_3}}{\partial \boldsymbol{f}^E} = 0, \tag{42}$$

$$\star^*$$
 satisfy (12a), (9d) – (9g), (43)

<span id="page-5-10"></span>Dual feasibility:

$$\gamma_{1,n}, \gamma_{2,n}, \gamma_{3,n}, \gamma_{4,n} \ge 0, \forall n \in \mathcal{N},$$
(44)

Complementary slackness:

$$\begin{cases} \gamma_{1,n}(1-\alpha_n^*) = 0, \gamma_{3,n}(p_n^* - p_n^{max}) = 0, \\ \gamma_{2,n}(\alpha_n^* - \Upsilon) = 0, \gamma_{4,n}(f_n^* - f_n^{max}) = 0, \forall n \in \mathcal{N}. \end{cases}$$
(45)

The above (42) – (45) indicate that  $\bigstar^*$  is a stationary point for problem  $\mathbb{P}_3$ . Therefore, the proof is concluded.

It can be easily verified that  $\mathbb{P}_4$  is a convex optimization problem and can be solved by utilizing convex optimization solvers such as CVX [8]. According to Proposition 1, we are able to obtain a stationary point for  $\mathbb{P}_3$  after solving  $\mathbb{P}_4$ .

## 4.2 Optimizing $\chi$ given $\alpha$ , p, b, $f^U$ , $f^E$

Firstly, to reduce the computational complexity, we convert discrete variables into continuous ones. Without loss of equivalence, constraint (9b) can be reformulated as:

$$\begin{cases} \chi_{n,m} \in [0,1], \ \forall n \in \mathcal{N}, \ m \in \mathcal{M}, \\ \sum_{n \in \mathcal{N}} \sum_{m \in \mathcal{M}} \chi_{n,m} \cdot (1 - \chi_{n,m}) \le 0. \end{cases}$$
(46)

<span id="page-6-1"></span>By replacing constraint (9b) with constraints (46) and (47), the discrete variables are transformed into continuous ones, thereby reducing the computation complexity of the problem.

With fixed  $\alpha$ , p, b,  $f^U$ ,  $f^E$ , solving problem  $\mathbb{P}_1$  is equivalent to solving the following problem:

<span id="page-6-2"></span>Problem 
$$\mathbb{P}_{5}(\boldsymbol{\alpha}, \boldsymbol{p}, \boldsymbol{b}, f^{U}, f^{E}):$$

$$\min_{\boldsymbol{\chi}} \sum_{n \in \mathcal{N}} Cost_{n}^{u} + \sum_{m \in \mathcal{M}} Cost_{m}^{E}, \qquad (48)$$
s.t. (9c), (9e), (9g), (46), (47).

However, constraint (47) remains a non-convex constraint. Thus, further measures are required to efficiently tackle this challenge. Next, we convert problem  $\mathbb{P}_5$  into an equivalent problem that has linear constraints, which we then address using the CCCP method. To this end, we introduce the following lemma:

LEMMA 1. Let  $G(\chi_{n,m}) = \sum_{n \in \mathcal{N}} Cost_n^u + \sum_{m \in \mathcal{M}} Cost_m^E$ . With any  $\chi_{n,m}^0$  satisfying (9c), (9e), (9g), and (46), for all  $\varrho > \varrho_0$  where

$$\varrho_{0} = \frac{G(\chi_{n,m}^{0}) - \min\{G(\chi_{n,m}): (9c), (9e), (9g), (46)\}}{\min\{\sum_{n \in \mathcal{N}} \sum_{m \in \mathcal{M}} \chi_{n,m} (1 - \chi_{n,m}): (9c), (9e), (9g), (46)\}},$$
(49)

problem  $\mathbb{P}_5$  has the same optimal solution with problem  $\mathbb{P}_6$ , which is defined as follows:

Problem 
$$\mathbb{P}_{6}(\boldsymbol{\alpha}, \boldsymbol{p}, \boldsymbol{b}, f^{U}, f^{E})$$
:
$$\min_{\boldsymbol{\chi}} \sum_{n \in \mathcal{N}} Cost_{n}^{u} + \sum_{m \in \mathcal{M}} Cost_{m}^{E} + \varrho \sum_{n \in \mathcal{N}} \sum_{m \in \mathcal{M}} \chi_{n,m} (1 - \chi_{n,m}), \quad (50)$$
s.t. (9c), (9e), (9g), (46).

It is worth noting that problem  $\mathbb{P}_6$  is derived from problem  $\mathbb{P}_5$  by integrating the concave constraint (47) into the objective function as a penalization term.

Proof. The proof can be directly derived from Theorem 1 in [1].

Problem  $\mathbb{P}_6$  involves subtracting a quadratic convex function from a linear function, while its constraints are linear in nature. According to [27], problem  $\mathbb{P}_6$  falls under the category of indefinite quadratic problem, which is a subset of the broader class of problems known as the difference of convex problems. With the objective function of problem  $\mathbb{P}_6$  being differentiable, we can effectively address problem  $\mathbb{P}_6$  using the CCCP method, which involves employing a first-order Taylor series approximation [10] to refine

 $\sum_{n\in\mathcal{N}}\sum_{m\in\mathcal{M}}\chi_{n,m}(\chi_{n,m}-1)$ . Specifically, it updates the expression to:

$$\sum_{n \in \mathcal{N}} \sum_{m \in \mathcal{M}} \chi_{n,m}^{(i)}(\chi_{n,m}^{(i)} - 1) + \sum_{n \in \mathcal{N}} \sum_{m \in \mathcal{M}} (2\chi_{n,m}^{(i)} - 1)(\chi_{n,m} - \chi_{n,m}^{(i)}), (51)$$

where  $\chi_{n,m}^{(i)}$  indicates the value of  $\chi_{n,m}$  at the i-th iteration. After that, the CCCP method systematically approaches resolution by iteratively engaging in a sequence of linear problems. The CCCP method not only simplifies complex issues by breaking them down into more manageable linear tasks but also ensures a structured progression towards finding an optimal solution through successive approximations. However, directly solving problem  $\mathbb{P}_6$  to reach a feasible solution for problem  $\mathbb{P}_5$  might not always be viable. To navigate this challenge, our approach involves generating several local optimum solutions for problem  $\mathbb{P}_6$ . This is achieved by applying the CCCP algorithm multiple times, initiating each iteration from a different feasible starting point specific to problem  $\mathbb{P}_6$ . The optimal solution is then determined by selecting the one that presents the smallest average value among these.

#### <span id="page-6-0"></span>5 SIMULATIONS

In this section, we present the performance of the proposed approach through simulations. The simulated MEC network has 50 mobile users and 10 edge servers by default. Assume the users and edge servers collaboratively train Meta's open-source large language model Meta-AI (LLaMA-7B) which consists of 32 transformer layers [28]. The path loss is modeled as  $128.1 + 37.6 \log(\text{distance})$  and Gaussian noise power is  $\sigma^2 = -134 \text{dBm}$ . The maximum transmission power  $p_n^{max}$  for the users is set in the range of 1 to 2 W. The maximum GPU frequency  $f_n^{max}$  for users and  $f_m^{max}$  for edge servers are chosen from [0.5,1] and [1,3] respectively. The total bandwidth  $b_m^{max}$  for each edge server is 20 MHz.

For LLM training, the batch size B is set to 512 and the dimensionality of the hidden states h is set to 1024. The lengths of input tokens for each user are randomly generated from 512 to 1024. We assume the mobile users are equipped with mobile devices with GPU such as Apple A15, whose GPU has 4 to 6 cores. Thus, the number of cores of the GPU at the user side  $C_n^U$  is chosen between 4 to 6. The number of FLOPs per cycle per core  $D_n^U$  is all set to 1. The edge servers are presumed to be equipped with advanced GPUs such as NVIDIA Tesla T4 and NVIDIA Tesla V100, therefore the number of cores of the GPU  $C_m^E$  is randomly assigned values from the interval [2560, 5120]. The number of FLOPs per cycle per core  $D_m^E$  is chosen between 1 and 2.

# 5.1 The performance of the proposed collaborative training method

In Figure 2, we present an analysis of system performance across three computing approaches: the proposed collaborative training method, edge server training method and local training method. Figure 2 (a) illustrates the energy consumption associated with each approach, where the proposed collaborative method demonstrates a balanced reduction in energy usage compared to the edge server training and local training methods. Figure 2 (b) depicts the delay experienced under each approach, showing that the proposed method effectively minimizes delay, achieving a performance closer

<span id="page-7-0"></span>![](_page_7_Figure_0.jpeg)

![](_page_7_Figure_1.jpeg)

- (a) The system performance on energy
- (b) The system performance on average delay.

Figure 2: Comparison of system performance with and without the proposed collaborative training approach.

to the edge server training approach while significantly outperforming the local training method. These results demonstrate the efficiency and effectiveness of the proposed collaborative training method in optimizing both energy consumption and system delay.

# 5.2 The performance of proposed algorithms under weighting factors

Next, we compare the performance of the proposed method when the weighting factors for energy, delay and model stability vary. The default weighting factors after normalization are all set to 1 for energy, delay and model stability. A larger weighting factor denotes enhanced prioritization of system attributes such as energy efficiency, latency, or model stability. The three additional methodologies employed for comparative analysis with our proposed method are listed as follows:

- Alternating optimization: This method is the most commonly employed strategy in the related literature as discussed in Section 2. It systematically alternates between optimizing offloading decisions and the allocation of computational or communication
- **Optimize**  $\alpha$  **only:** This approach solely focuses on optimizing the offloading decision  $\alpha$ , while implementing a random strategy for resource allocation.
- Optimize resource only: This method concentrates exclusively on the optimization of resource allocation, while employing a random approach to the offloading decision α.

In Figure 3, we adjust the weighting factors for energy, delay and model stability from 1 to 10, respectively. For each setting where one attribute's weighting factor varied from 1 to 10, the weighting factors for the other two attributes are held constant at 1. In Figure 3 (a), the proposed methodology consistently attains the lowest energy consumption among the methods evaluated. The alternating optimization approach secures the second-best performance. Conversely, the method that solely optimizes  $\alpha$  exhibits the poorest results. This suboptimal performance can be attributed to the fact that the  $\alpha$ -only optimization method neglects resource allocation considerations, which are crucial in minimizing energy consumption. Furthermore, as the weighting factor for energy is incrementally increased, the reductions in optimal energy consumption diminish progressively, eventually converging. Figure 3 (b) depicts the average delay experienced under various weighting factors for delay. Consistently, the proposed approach yields

the minimal delay, surpassing the performance of the alternating optimization method. Notably, the strategy focusing solely on optimizing  $\alpha$  demonstrates superior results compared to that which exclusively optimizes resource allocation. This advantage is attributed to the enhanced computational capabilities of the edge servers, which significantly reduce computational delays. Figure 3 (c) illustrates the mean stability of the model across a range of weighting factors for model stability. The method we proposed consistently attains the highest level of model stability. The alternating optimization approach outperforms the strategy that solely optimizes  $\alpha$  a little, although both methods converge to nearly identical point in the long term. Conversely, the technique that focuses exclusively on optimizing resource allocation exhibits the poorest performance, primarily due to the arbitrary selection of the offloading decision  $\alpha$ , which significantly impacts model stability.

## 5.3 The impact of the number of users and the number of edge servers

In this part, we assess the influence of both the user population and the number of edge servers on the effectiveness of the proposed approach to addressing the user-to-edge association challenge. We employ a comparative methodology as outlined below:

- Baseline: The baseline approach we choose is a greedy-based strategy. Under this strategy, each user opts for the edge server offering the highest available transmission rate, subject to bandwidth limitations.
- Random: The random user-to-edge server association method distributes users among edge servers in a stochastic manner, also adhering to bandwidth constraints.

In Figure 5 (a), we present the total energy consumption when there are different numbers of users. It can be observed that the proposed method always outperforms the two alternative strategies. The baseline strategy selects the edge server with the highest available transmission rate for each user; however, this approach may inadvertently overload servers that possess lower computational efficiency, thereby causing a marginal increase in energy consumption relative to our method. The random strategy invariably results in the highest energy expenditure. Subsequently, we analyze the average delay contingent on varying user quantities, as depicted in Figure 5 (b). It is evident that the proposed methodology consistently surpasses the two alternative strategies. Specifically, the random strategy yields the least favorable outcomes due to its arbitrary selection of edge servers for user allocation. While the baseline strategy may attain minimal communication delays, it tends to allocate an excessive number of users to a single edge server, thereby exacerbating the computational delays.

In Figure 4, the convergence performance of the algorithm is analyzed for the user-to-edge server association problem across varying quantities of edge servers. For each scenario considered, the user count remains constant at 100. The analysis reveals that the algorithm attains a stationary point as evidenced by the stabilization of the objective value. It is worth noticing that configurations with a smaller number of edge servers exhibit faster convergence rates. This enhanced speed of convergence can be attributed to the diminished complexity of the optimization challenge: a reduced number of servers correlates with fewer constraints and a lower number

<span id="page-8-2"></span>![](_page_8_Figure_0.jpeg)

![](_page_8_Figure_1.jpeg)

![](_page_8_Figure_2.jpeg)

![](_page_8_Figure_3.jpeg)

(a) The total energy consumption under different weighting factors for energy

(b) The average delay under different weighting factors for energy

(c) The average model stability under different weighting factors for model stabil-

Figure 4: The convergence performance with different numbers of edge servers.

Figure 3: The performance of the proposed method under different weighting factors.

<span id="page-8-3"></span>![](_page_8_Figure_9.jpeg)

![](_page_8_Figure_10.jpeg)

(a) The energy consumption under differ- (b) The average delay under different nument numbers of users. bers of users

Figure 5: The performance of the algorithm under different numbers of mobile users.

of parameters requiring adjustment throughout the optimization procedure. In all tested configurations, the algorithm consistently achieved convergence within 13 iterations, thereby demonstrating its robust capability to efficiently resolve the user-to-edge server association problem.

#### <span id="page-8-0"></span>CONCLUSION

In this study, we present a collaborative LLM training framework that merges the efforts of mobile users and edge servers. Here, mobile users are tasked with training the preliminary layers of the LLM, while the computationally intensive later layers are managed by the edge servers. We develop a multi-objective optimization strategy aimed at reducing overall energy usage and latency experienced by users, while also improving the stability of LLMs. Through analytical analysis, we establish an upper bound for the average-replace-one stability. The proposed algorithm leverages fractional programming and the CCCP method to derive solutions. Simulation results indicate that our approach effectively reduces energy usage and delay, and enhances the stability of LLMs in the mobile edge computing environments.

#### ACKNOWLEDGEMENT

This research is supported by the National Research Foundation, Singapore and Infocomm Media Development Authority under its Trust Tech Funding Initiative, Singapore MOE AcRF Tier 1 RT5/23, Tier 1 RG90/22, and NTU-WASP Joint Project. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of National Research Foundation, Singapore and Infocomm Media Development Authority.

#### <span id="page-8-1"></span>A PROOF OF THEOREM 1

Denote **w** as the parameters of a language model, and  $\xi = \dim(\mathbf{w})$ as the number of the parameters. When the users fine-tune a pretrained model, denoting  $M^{\xi \times \xi}$  as a mask matrix which is a diagonal matrix with  $M_{ii} = \{0, 1\}$  and  $\mathcal{L}_{\mathcal{S}}(\mathbf{w}) = \sum_{i=1}^{k} \ell(\mathbf{w}, z_i)$  for a training dataset S, the training process of fine-tuning the pre-trained model is to solve the following problem:

<span id="page-8-4"></span>
$$\min_{\mathbf{w}} \mathcal{L}_{\mathcal{S}}(\mathbf{w}), \tag{A.1}$$
 s.t.  $\|(I-M)(\mathbf{w}-\mathbf{w}^0)\|^2 = 0, \tag{A.1a}$ 

s.t. 
$$||(I-M)(\mathbf{w} - \mathbf{w}^0)||^2 = 0,$$
 (A.1a)

where  $\mathbf{w}^0$  are the parameters of the pre-trained model. According to the Lagrangian duality, problem (A.1) is equivalent to:

$$\min_{\mathbf{w}} \max_{\lambda} \mathcal{L}_{\mathcal{S}}(\mathbf{w}) + \lambda \| (I - M)(\mathbf{w} - \mathbf{w}^0) \|^2, \tag{A.2}$$

where  $\lambda \geq 0$  is the Lagrangian multiplier. Since

$$\min_{\mathbf{w}} \max_{\lambda} \mathcal{L}_{\mathcal{S}}(\mathbf{w}) + \lambda \| (I - M)(\mathbf{w} - \mathbf{w}^{0}) \|^{2} \ge$$

$$\min_{\mathbf{w}} \mathcal{L}_{\mathcal{S}}(\mathbf{w}) + \| (I - M)(\mathbf{w} - \mathbf{w}^{0}) \|^{2},$$
(A.3)

we then focus on optimizing the lower bound of problem (A.1), which is given by:

$$\min \mathcal{L}_{\mathcal{S}}'(\mathbf{w}) = \mathcal{L}_{\mathcal{S}}(\mathbf{w}) + \|(I - M)(\mathbf{w} - \mathbf{w}^0)\|^2. \tag{A.4}$$

It indicates that minimizing initial loss function  $\mathcal{L}_{\mathcal{S}}(\mathbf{w})$ , augmented by the regularization term  $||(I - M)(\mathbf{w} - \mathbf{w}^0)||^2$  provides a lower bound on the optimal value of problem (A.1).

By taking the expectation with respect to M, we can get:

$$\mathbb{E}_{M}\left(\mathcal{L}_{S}'(\mathbf{w})\right) = \mathcal{L}_{S}(\mathbf{w}) + \mathbb{E}\|(I - M)(\mathbf{w} - \mathbf{w}^{0})\|^{2}$$

$$= \mathcal{L}_{S}(\mathbf{w}) + \|(\mathbf{w} - \mathbf{w}^{0})\|^{2}\mathbb{E}\left(\sum_{i=1}^{\xi} (1 - M_{ii})^{2}\right)$$

$$= \mathcal{L}_{S}(\mathbf{w}) + \|(\mathbf{w} - \mathbf{w}^{0})\|^{2}(1 - \alpha), \tag{A.5}$$

where the validity of the last equality is attributed to the fact that the fraction of parameters subjected to fine-tuning is  $\alpha$ . Therefore,

<span id="page-8-5"></span>
$$\mathcal{A}(S) = \underset{\mathbf{w}}{\operatorname{argmin}} \ \mathcal{L}_{S}(\mathbf{w}) + (1 - \alpha) \|\mathbf{w} - \mathbf{w}^{0}\|^{2}. \tag{A.6}$$

Next, we denote  $f_{\mathcal{S}}(\mathbf{w}) = \mathcal{L}_{\mathcal{S}}(\mathbf{w}) + (1-\alpha)\|\mathbf{w} - \mathbf{w}^0\|^2$ . Subsequently,  $\forall \mathbf{u}, \mathbf{v}, \forall i \in \{1, \dots, k\}$ , we can get:

$$f_{\mathcal{S}}(\mathbf{u}) - f_{\mathcal{S}}(\mathbf{v})$$

$$= \mathcal{L}_{\mathcal{S}}(\mathbf{u}) + (1-\alpha)\|\mathbf{u} - \mathbf{u}^{0}\|^{2} - \left(\mathcal{L}_{\mathcal{S}}(\mathbf{v}) + (1-\alpha)\|\mathbf{v} - \mathbf{v}^{0}\|^{2}\right)$$

$$= \mathcal{L}_{\mathcal{S}^{i}}(\mathbf{u}) + (1-\alpha)\|\mathbf{u} - \mathbf{u}^{0}\|^{2} - \left(\mathcal{L}_{\mathcal{S}^{i}}(\mathbf{v}) + (1-\alpha)\|\mathbf{v} - \mathbf{v}^{0}\|^{2}\right)$$

$$- \frac{\ell(\mathbf{u}, z_{i}') - \ell(\mathbf{v}, z_{i}')}{k} + \frac{\ell(\mathbf{u}, z_{i}) - \ell(\mathbf{v}, z_{i})}{k}$$

$$= f_{\mathcal{S}^{i}}(\mathbf{u}) - f_{\mathcal{S}^{i}}(\mathbf{v}) - \frac{\ell(\mathbf{u}, z_{i}') - \ell(\mathbf{v}, z_{i}')}{k} + \frac{\ell(\mathbf{u}, z_{i}) - \ell(\mathbf{v}, z_{i})}{k}. \tag{A.7}$$

<span id="page-9-0"></span>Let's set  $\mathbf{u}=\mathcal{A}(\mathcal{S}^i)$  and  $\mathbf{v}=\mathcal{A}(\mathcal{S})$  in (A.7). Using the fact that  $f_{S^i}(\mathbf{u}) \leq f_{S^i}(\mathbf{v})$  because  $\mathcal{A}(S^i)$  is the minimizer of  $f_{S^i}(\mathbf{w})$ , we can get:

$$f_{\mathcal{S}}(\mathcal{A}(\mathcal{S}^i)) - f_{\mathcal{S}}(\mathcal{A}(\mathcal{S})) \le$$

$$\frac{\ell(\mathcal{A}(S^i), z_i) - \ell(\mathcal{A}(S), z_i)}{k} - \frac{\ell(\mathcal{A}(S^i), z_i') - \ell(\mathcal{A}(S), z_i')}{k}. \quad \text{(A.8)}$$
 Since the loss function is strong convex,  $f_S$  is  $2(1 - \alpha)$ -strong

$$f_{\mathcal{S}}(\mathbf{u}) \ge f_{\mathcal{S}}(\mathbf{v}) + \nabla f_{\mathcal{S}}(\mathbf{v})^{\mathsf{T}}(\mathbf{u} - \mathbf{v}) + (1 - \alpha)\|\mathbf{u} - \mathbf{v}\|^2$$
. (A.9)  
Let  $\mathbf{u} = \mathcal{A}(\mathcal{S}^i)$  and  $\mathbf{v} = \mathcal{A}(\mathcal{S})$ , then  $\nabla f_{\mathcal{S}}(\mathcal{A}(\mathcal{S})) = 0$  since  $\mathcal{A}(\mathcal{S})$  is the minimizer of  $f_{\mathcal{S}}(\mathbf{w})$ . Therefore, (A.9) becomes:

$$f_{\mathcal{S}}(\mathcal{A}(\mathcal{S}^i)) \ge f_{\mathcal{S}}(\mathcal{A}(\mathcal{S})) + (1-\alpha)\|\mathcal{A}(\mathcal{S}^i) - \mathcal{A}(\mathcal{S})\|^2$$
, (A.10) which can be rearranged as follows:

$$f_{\mathcal{S}}(\mathcal{A}(\mathcal{S}^i)) - f_{\mathcal{S}}(\mathcal{A}(\mathcal{S})) \ge (1 - \alpha) \|\mathcal{A}(\mathcal{S}^i) - \mathcal{A}(\mathcal{S})\|^2.$$
 (A.11) Combing (A.8) and (A.11), it yields:

$$(1-\alpha)\|\mathcal{A}(\mathcal{S}^i) - \mathcal{A}(\mathcal{S})\|^2 \le$$

$$\frac{\ell(\mathcal{A}(\mathcal{S}^i),z_i) - \ell(\mathcal{A}(\mathcal{S}),z_i)}{k} - \frac{\ell(\mathcal{A}(\mathcal{S}^i),z_i') - \ell(\mathcal{A}(\mathcal{S}),z_i')}{k}. \quad \text{(A.12)}$$
 Since we assume  $\ell(\cdot,z_i)$  is  $L$ -Lipschitz, which means: 
$$\ell(\mathcal{A}(\mathcal{S}^i),z_i) - \ell(\mathcal{A}(\mathcal{S}),z_i) \leq L\|\mathcal{A}(\mathcal{S}^i) - \mathcal{A}(\mathcal{S})\|, \quad \text{(A.13)}$$

$$\ell(\mathcal{A}(\mathcal{S}^i), z_i) - \ell(\mathcal{A}(\mathcal{S}), z_i) \le L \|\mathcal{A}(\mathcal{S}^i) - \mathcal{A}(\mathcal{S})\|, \tag{A.13}$$

$$\ell(\mathcal{A}(\mathcal{S}), z_i') - \ell(\mathcal{A}(\mathcal{S}^i), z_i') \le L \|\mathcal{A}(\mathcal{S}^i) - \mathcal{A}(\mathcal{S})\|. \tag{A.14}$$

Substituting (A.13) and (A.14) into (A.12) yields the following result:

$$(1-\alpha)\|\mathcal{A}(\mathcal{S}^i) - \mathcal{A}(\mathcal{S})\|^2 \le \frac{2L\|\mathcal{A}(\mathcal{S}^i) - \mathcal{A}(\mathcal{S})\|}{k}, \tag{A.15}$$

which indicates:

$$\|\mathcal{A}(S^i) - \mathcal{A}(S)\| \le \frac{2L}{(1-\alpha)k}.$$
 (A.16) Inserting (A.16) into (A.13) leads us to determine the following:

$$\ell(\mathcal{A}(S^{i}), z_{i}) - \ell(\mathcal{A}(S), z_{i}) \leq \frac{2L^{2}}{(1-\alpha)k}. \tag{A.17}$$
 Given that this is true for any  $S$  and  $z_{i}$ , we can finally deduce that

 $\forall i \in \{1, \ldots, k\}$ :

$$\mathbb{E}_{\mathcal{S}}[\left|\ell(\mathcal{A}(\mathcal{S}), z_i) - \ell(\mathcal{A}(\mathcal{S}^i), z_i)\right|] \le \frac{2L^2}{(1-\alpha)k}.$$
 (A.18)

#### REFERENCES

- <span id="page-9-31"></span>[1] LT Hoai An, HV Ngai, and PD Tao. 2012. Exact penalty and error bounds in DC programming. Journal of Global Optimization 52, 3 (2012), 509-535
- <span id="page-9-1"></span>Dongqi Cai, Yaozong Wu, Shangguang Wang, Felix Xiaozhu Lin, and Mengwei Xu. 2023. Efficient federated learning for modern NLP. In Proceedings of the 29th Annual International Conference on Mobile Computing and Networking. 1-16.
- <span id="page-9-15"></span>[3] Min Chen and Yixue Hao. 2018. Task offloading for mobile edge computing in software defined ultra-dense network. IEEE Journal on Selected Areas in Communications 36, 3 (2018), 587-597.
- <span id="page-9-25"></span>Thomas M Cover. 1999. Elements of Information Theory. John Wiley & Sons.
- <span id="page-9-7"></span>Ning Ding, Yujia Qin, Guang Yang, Fuchao Wei, Zonghan Yang, Yusheng Su, Shengding Hu, Yulin Chen, Chi-Min Chan, Weize Chen, et al. 2022. Delta tuning: A comprehensive study of parameter efficient methods for pre-trained language models. arXiv preprint arXiv:2203.06904 (2022).
- <span id="page-9-14"></span>[6] Thinh Quang Dinh, Jianhua Tang, Quang Duy La, and Tony QS Quek. 2017. Offloading in mobile edge computing: Task allocation and computational frequency scaling. IEEE Transactions on Communications 65, 8 (2017), 3571-3584.
- <span id="page-9-23"></span>Zihao Fu, Haoran Yang, Anthony Man-Cho So, Wai Lam, Lidong Bing, and Nigel Collier. 2023. On the effectiveness of parameter-efficient fine-tuning. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 37. 12799-12807.
- <span id="page-9-30"></span>Michael Grant, Stephen Boyd, and Yinyu Ye. 2008. CVX: Matlab software for disciplined convex programming.
- <span id="page-9-10"></span>[9] Wenjuan Han, Bo Pang, and Ying Nian Wu. 2021. Robust transfer learning with pretrained language models through adapters. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing. 854-861.
- <span id="page-9-33"></span>[10] Muhammad Fainan Hanif, Zhiguo Ding, Tharmalingam Ratnarajah, and George K Karagiannidis. 2015. A minorization-maximization method for optimizing sum rate in the downlink of non-orthogonal multiple access systems. IEEE Transactions on Signal Processing 64, 1 (2015), 76-88.

- <span id="page-9-22"></span>[11] Ruidan He, Linlin Liu, Hai Ye, Qingyu Tan, Bosheng Ding, Liying Cheng, Jia-Wei Low, Lidong Bing, and Luo Si. 2021. On the effectiveness of adapter-based tuning for pretrained language model adaptation. arXiv preprint arXiv:2106.03164 (2021).
- <span id="page-9-8"></span>Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019. Parameter-efficient transfer learning for NLP. In International conference on  $machine\ learning.\ PMLR,\ 2790-2799.$
- <span id="page-9-36"></span><span id="page-9-6"></span>Yunpeng Huang, Yaonan Gu, Jingwei Xu, Zhihong Zhu, Zhaorun Chen, and Xiaoxing Ma. 2024. Securing reliability: A brief overview on enhancing in-context learning for foundation models. arXiv preprint arXiv:2402.17671 (2024).
- <span id="page-9-21"></span>Cheolhyoung Lee, Kyunghyun Cho, and Wanmo Kang. 2019. Mixout: Effective regularization to finetune large-scale pretrained language models. arXiv preprint arXiv:1909.11299 (2019).
- <span id="page-9-35"></span><span id="page-9-24"></span>[15] Chang Liu and Jun Zhao. 2024. Resource allocation in large language model integrated 6G vehicular networks. In IEEE 99th Vehicular Technology Conference (VTC2024-Spring). IEEE.
- <span id="page-9-20"></span>[16] Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin A Raffel. 2022. Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. In  $\hat{A}dvances$  in Neural Information Processing Systems.
- <span id="page-9-37"></span><span id="page-9-9"></span>[17] Xiao Liu, Kaixuan Ji, Yicheng Fu, Weng Lam Tam, Zhengxiao Du, Zhilin Yang, and Jie Tang. 2021. P-tuning v2: Prompt tuning can be comparable to fine-tuning universally across scales and tasks. arXiv preprint arXiv:2110.07602 (2021).
- <span id="page-9-40"></span><span id="page-9-2"></span>Zechun Liu, Changsheng Zhao, Forrest Iandola, Chen Lai, et al. 2024. MobileLLM: Optimizing sub-billion parameter language models for on-device use cases. In Forty-first International Conference on Machine Learning.
- <span id="page-9-38"></span><span id="page-9-4"></span>[19] Ruilong Ma, Jingyu Wang, Qi Qi, Xiang Yang, Haifeng Sun, Zirui Zhuang, and Jianxin Liao, 2023, Poster: PipeLLM: Pipeline LLM inference on heterogeneous devices with sequence slicing. In Proceedings of the ACM SIGCOMM 2023 Conference. 1126-1128
- <span id="page-9-39"></span><span id="page-9-13"></span>[20] Yuyi Mao, Jun Zhang, and Khaled B Letaief. 2016. Dynamic computation offloading for mobile-edge computing with energy harvesting devices. IEEE Journal on Selected Areas in Communications 34, 12 (2016), 3590-3605.
- <span id="page-9-11"></span>[21] Marius Mosbach, Maksym Andriushchenko, and Dietrich Klakow. 2020. On the stability of fine-tuning BERT: Misconceptions, explanations, and strong baselines. arXiv preprint arXiv:2006.04884 (2020).
- <span id="page-9-41"></span><span id="page-9-19"></span>Marius Mosbach, Tiago Pimentel, Shauli Ravfogel, Dietrich Klakow, and Yanai Elazar. 2023. Few-shot fine-tuning vs. in-context learning: A Fair comparison and evaluation. In Findings of the Association for Computational Linguistics: ACL.
- <span id="page-9-27"></span>Matan Schliserman and Tomer Koren. 2022. Stability vs implicit bias of gradient methods on separable data and beyond. In Proceedings of Thirty Fifth Conference on Learning Theory (Proceedings of Machine Learning Research, Vol. 178). PMLR, 3380-3394.
- <span id="page-9-28"></span>Shai Shalev-Shwartz and Shai Ben-David. 2014. Understanding Machine Learning: From Theory to Algorithms. Cambridge university press.
- <span id="page-9-26"></span>Shai Shaley-Shwartz, Ohad Shamir, Nathan Srebro, and Karthik Sridharan, 2010. Learnability, stability and uniform convergence. The Journal of Machine Learning Research 11 (2010), 2635-2670.
- <span id="page-9-5"></span>Weisong Shi, Jie Cao, Quan Zhang, Youhuizi Li, and Lanyu Xu. 2016. Edge computing: Vision and challenges. IEEE Internet of Things Journal 3, 5 (2016), 637-646
- <span id="page-9-32"></span>Le Thi Hoai An and Pham Dinh Tao. 1997. Solving a class of linearly constrained indefinite quadratic problems by DC algorithms. Journal of global optimization 11 (1997), 253-285.
- <span id="page-9-34"></span>[28] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023).
- <span id="page-9-18"></span>[29] Feng Wang, Jie Xu, Xin Wang, and Shuguang Cui. 2017. Joint offloading and computing optimization in wireless powered mobile-edge computing systems. IEEE Transactions on Wireless Communications 17, 3 (2017), 1784-1797
- <span id="page-9-3"></span>[30] Daliang Xu, Wangsong Yin, Xin Jin, Ying Zhang, Shiyun Wei, Mengwei Xu, and Xuanzhe Liu. 2023. LLMCad: Fast and scalable on-device large language model inference. arXiv preprint arXiv:2309.04255 (2023).
- <span id="page-9-16"></span>Xincao Xu, Kai Liu, Penglin Dai, Feiyu Jin, Hualing Ren, Choujun Zhan, and Songtao Guo. 2023. Joint task offloading and resource optimization in NOMAbased vehicular edge computing: A game-theoretic DRL approach. Journal of Systems Architecture 134 (2023), 102780.
- <span id="page-9-17"></span>[32] Cheng Zhan, Han Hu, Xiufeng Sui, Zhi Liu, and Dusit Niyato. 2020. Completion time and energy optimization in the UAV-enabled mobile-edge computing system. IEEE Internet of Things Journal 7, 8 (2020), 7808-7822.
- <span id="page-9-29"></span>[33] Jun Zhao, Liangxin Qian, and Wenhan Yu. 2024. Human-centric resource allocation in the Metaverse over wireless communications. IEEE Journal on Selected Areas in Communications (JSAC) 42, 3 (2024), 514-537. https://arxiv.org/pdf/
- <span id="page-9-12"></span>[34] Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate before use: Improving few-shot performance of language models. In International conference on machine learning. PMLR, 12697-12706.