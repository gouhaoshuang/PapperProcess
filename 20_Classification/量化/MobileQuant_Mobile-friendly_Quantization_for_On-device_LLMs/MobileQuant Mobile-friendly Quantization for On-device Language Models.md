---
category: 量化
classification_reason: 论文的核心贡献是提出MobileQuant，一种针对移动端NPU优化的全整数训练后量化（PTQ）方法，重点解决了激活量化的难题。虽然属于模型压缩的大范畴，但为了建立细粒度索引，'量化'作为独立的技术类别比'模型压缩'更为精准且符合推荐示例。
created: '2026-01-18'
status: unread
tags:
- 训练后量化
- 激活量化
- 全整数量化
- 权重等价变换
- NPU加速
title: 'MobileQuant: Mobile-friendly Quantization for On-device LLMs'
---

## MobileQuant: Mobile-friendly Quantization for On-device Language Models

Fuwen Tan<sup>1</sup> , Royson Lee<sup>1</sup>,<sup>2</sup> , Łukasz Dudziak<sup>1</sup>,<sup>2</sup> , Shell Xu Hu<sup>1</sup> , Sourav Bhattacharya<sup>1</sup> , Timothy Hospedales<sup>1</sup>,<sup>3</sup> , Georgios Tzimiropoulos<sup>1</sup>,<sup>4</sup> , Brais Martinez<sup>1</sup>

> Samsung AI Center, Cambridge<sup>1</sup> University of Cambridge<sup>2</sup> University of Edinburgh<sup>3</sup> Queen Mary University of London<sup>4</sup> {fuwen.tan, royson.lee, l.dudziak, shell.hu, sourav.b1, t.hospedales, georgios.t, brais.mart}@samsung.com

## Abstract

Large language models (LLMs) have revolutionized language processing, delivering outstanding results across multiple applications. However, deploying LLMs on edge devices poses several challenges with respect to memory, energy, and compute costs, limiting their widespread use in devices such as mobile phones. A promising solution is to reduce the number of bits used to represent weights and activations. While existing works have found partial success at quantizing LLMs to lower bitwidths, *e.g.* 4-bit weights, quantizing activations beyond 16 bits often leads to large computational overheads due to poor on-device quantization support, or a considerable accuracy drop. Yet, 8-bit activations are very attractive for on-device deployment as they would enable LLMs to fully exploit mobile-friendly hardware, e.g. Neural Processing Units (NPUs). In this work, we make a first attempt to facilitate the on-device deployment of LLMs using integer-only quantization. We first investigate the limitations of existing quantization methods for on-device deployment, with a special focus on activation quantization. We then address these limitations by introducing a simple post-training quantization method, named MobileQuant, that extends previous weight equivalent transformation works by jointly optimizing the weight transformation and activation range parameters in an end-to-end manner. MobileQuant demonstrates superior capabilities over existing methods by 1) achieving nearlossless quantization on a wide range of LLM benchmarks, 2) reducing latency and energy consumption by 20%-50% compared to current on-device quantization strategies, 3) requiring limited compute budget, 4) being compatible with mobile-friendly compute units, *e.g.* NPU.

## 1 Introduction

Large language models (LLMs) have markedly advanced language processing capabilities, paving

the way for expansive applications in artificial intelligence. However, the deployment of LLMs is costly in terms of memory, computation, and energy, which can be prohibitive on edge devices like mobile phones. A standard approach to facilitate running these models on edge devices is to quantize them, representing weights and activations with fewer bits, thereby mitigating these costs.

Existing LLM quantization works can be grouped into two categories: *weight-only quantization* and *weight-activation quantization*. Weightonly quantization approaches [\(Frantar et al.,](#page-8-0) [2023;](#page-8-0) [Lin et al.,](#page-9-0) [2024\)](#page-9-0) convert model weights into lowbitwidth integers, most commonly 4-bit, and maintain the activations in 16-bit floating-point. Weightonly quantization often preserves accuracy while significantly reducing the model storage footprint. In addition, weight-only quantization can result in minor gains in inference latency due to the reduction in memory access overheads. However, these approaches still suffer from high energy consumption and high latency, as computation is performed in floating point. Costly on-the-fly weight dequantization is also required during inference. Instead, weight-activation quantization approaches forgo the need for on-the-fly dequantization by quantizing both weights and activations, and potentially utilizing efficient fixed-point operators. Despite its efficiency benefits, quantizing activations typically degrades accuracy given the activation outliers [\(Xiao et al.,](#page-9-1) [2023;](#page-9-1) [Wu et al.,](#page-9-2) [2024;](#page-9-2) [Luo](#page-9-3) [et al.,](#page-9-3) [2024\)](#page-9-3), especially in the case where static per-tensor quantization parameters are applied. To counteract this accuracy drop, previous works include quantizing activations for certain expensive operations [\(Xiao et al.,](#page-9-1) [2023\)](#page-9-1), *e.g.* matrix multiplication, or employing dynamic per-token quantization [\(Shao et al.,](#page-9-4) [2024;](#page-9-4) [Liu et al.,](#page-9-5) [2023;](#page-9-5) [Ashkboos](#page-8-1) [et al.,](#page-8-1) [2024;](#page-8-1) [Liu et al.,](#page-9-6) [2024\)](#page-9-6), which is often slow on Graphic Processing Units (GPUs) and, most importantly, lacks hardware support on edge devices. Notably, none of these methods support lossless 8 bit (int8) per-tensor quantization for the activations, or fully leverage low-precision fixed-point engines, such as the Digital Signal Processor (DSP), or dedicated Neural Processing Unit (NPU) [\(Qualcomm,](#page-9-7) [2024;](#page-9-7) [Google,](#page-8-2) [2021\)](#page-8-2), commonly found in mobile devices [\(Mahurin,](#page-9-8) [2023\)](#page-9-8). Towards on-device quantization for LLMs, we introduce MobileQuant, a post-training quantization approach that not only effectively handles the conventional accuracy and efficiency challenges of quantization but is also seamlessly supported by existing mobile hardware. To achieve this, MobileQuant consists of three simple yet effective methodological extensions, motivated by the shortcomings of existing state-of-theart works when deployed on device, and building on top of these works. These extensions include: *1)* applying weight equivalent transformation on *all possible layers*, *2)*, learning the optimal quantization range for activations, *3)* jointly optimizing all weight transformation and range parameters in an end-to-end manner. As such, MobileQuant applies a combination of per-tensor and per-channel weight quantization at 4-bit or 8-bit and per-tensor activation quantization at 8-bit or 16-bit, utilizing fixed-point integer representations for all operations.

The benefits of MobileQuant over previous works are multifold. Firstly, MobileQuant enables the quantization of the weights to either 4-bit or 8-bit and the activations to 8-bit integers, except for non-linearities like softmax and normalization, with minimal impact on performance. Mobile-Quant, hence, maximizes the potential of equivalent transformation-based methods [\(Nagel et al.,](#page-9-9) [2019;](#page-9-9) [Xiao et al.,](#page-9-1) [2023;](#page-9-1) [Lin et al.,](#page-9-0) [2024;](#page-9-0) [Shao](#page-9-4) [et al.,](#page-9-4) [2024\)](#page-9-4) that achieve linear-invariant weight equalization. Deploying LLMs on device using MobileQuant results in a significant reduction in inference speed and energy usage as the latency and energy consumption of multiply-accumulate operations correlate directly with the bit-widths. Besides substantial gains during inference, we also show that MobileQuant's end-to-end optimization benefits from more calibration samples and extended training samples through our ablation study. In contrast, previous works that adopt closed-form solutions [\(Nagel et al.,](#page-9-9) [2019\)](#page-9-9), search-based optimization [\(Lin et al.,](#page-9-0) [2024\)](#page-9-0), and block-wise error minimization [\(Shao et al.,](#page-9-4) [2024;](#page-9-4) [Liu et al.,](#page-9-6) [2024\)](#page-9-6) struggle to scale with the number of samples and training steps. Lastly, in comparison with

other learnable-based quantization methods such as Quantization Aware Training (QAT) [\(Liu et al.,](#page-9-5) [2023;](#page-9-5) [Bondarenko et al.,](#page-8-3) [2023\)](#page-8-3), MobileQuant retains the model generalizability as the model remains mathematically equivalent to its unquantized variant. Our contributions are summarized as follows:

- 1. We introduce a post-training quantization approach for large language models (LLMs) that is supported by current mobile hardware implementations (i.e. DSP, NPU), thus being directly deployable on real edge devices.
- 2. Our method improves upon prior works through simple yet effective methodological extensions that enable us to effectively quantize most activations to a lower bitwidth (*i.e.* 8-bit) with near-lossless performance.
- 3. We conduct a comprehensive on-device evaluation of model accuracy, inference latency, and energy consumption. Our results indicate that our method reduces both inference latency and energy usage by 20%-50% while still maintaining accuracy compared to models using 16-bit activations.

## 2 Related Work

#### 2.1 Post-training Quantization (PTQ)

Previous research in post-training quantization for LLMs can be categorized into three main groups: Weight-only Quantization focuses on compressing the model weights to reduce storage requirements and memory transfer overheads. Representative works [\(Frantar et al.,](#page-8-0) [2023;](#page-8-0) [Lin et al.,](#page-9-0) [2024;](#page-9-0) [Shao et al.,](#page-9-4) [2024;](#page-9-4) [Liu et al.,](#page-9-6) [2024\)](#page-9-6) generally achieve performance comparable to full-precision models and maintain similar inference speeds on GPUs. However, these methods dequantize weights to 16 bit values on the fly, resulting in high-precision floating-point computations and hence leading to high inference latency and energy consumption, particularly on edge devices such as mobile phones. Weight-activation Quantization extends quantization to both model weights and activations, aiming to further reduce computational overhead. However, as indicated in prior works [\(Dettmers et al.,](#page-8-4) [2022;](#page-8-4) [Xiao et al.,](#page-9-1) [2023\)](#page-9-1), activations have dynamic ranges across different data distributions and are hence more challenging to quantize compared to weights. As a result, quantizing activations to a

lower bit-width often results in a significant performance decline. Leading solutions either retain some compute-intensive matrix multiplications in full precision (Dettmers et al., 2022; Xiao et al., 2023) or utilize dynamic per-token activation quantization, which lacks hardware support on mobile platforms. In contrast, our approach quantizes all linear operations and is compatible with current hardware support on edge devices.

Learning to Round. Notable works like (Nagel et al., 2020; Lee et al., 2023) also focus on weight-only quantization but introduce techniques for learning optimal weight rounding. The key argument is that the conventional round-to-nearest method is suboptimal, as it does not account for the interdependencies among adjacent weights. Our work is orthogonal with this research and can hence be integrated with these techniques.

#### 2.2 Quantization Aware Training (QAT)

Quantization aware training (QAT) involves retraining or fine-tuning full-precision models using differentiable quantizers. Recent research (Liu et al., 2023; Bondarenko et al., 2023) has shown that QAT outperforms PTQ methods, particularly with in-domain training data. However, QAT requires extensive training, which is often impractical for LLMs. Additionally, QAT may be vulnerable to domain shifts if the data used for pretraining is unavailable. In contrast, our approach is zero-shot, only requiring a minimal set of calibration samples and a limited compute budget. Once trained, our model remains mathematically equivalent to the original model when unquantized, enhancing its adaptability to various downstream tasks.

#### 3 Preliminaries

#### 3.1 Mobile-friendly Design Choices

Quantization methods are differentiated by several main design choices, with varying levels of hardware support. In this section, we first list these design choices and then highlight the limitations of existing works with respect to these choices.

**Support for mobile-friendly bitwidth:** int8-int8 operations are widely supported and most often optimized for, while int4-int16 and int8-int16 are typically supported although often slower than int8-int8.

**Quantization groups:** Quantizing using per-tensor and per-channel statistics is widely supported while using per-token statistics is not.

**Dynamic vs static:** Static quantization statistics that do not depend on the input data, typically computed on a holdout calibration set, are widely supported. Dynamic quantization, on the other hand, requires online calibration from the input data and is not widely supported.

State-of-the-art quantization methods demonstrate strong performance on a server use case (i.e. high-end GPU). However, they either utilize on-the-fly dequantization and 16-bit floating point operations (Frantar et al., 2023; Lin et al., 2024), which are computationally inefficient, or dynamic pertoken quantization (Xiao et al., 2023; Shao et al., 2024), which, as previously mentioned, has no support on edge devices.

We, instead, consider design choices that are widely supported and optimized on modern edge devices (*e.g.* Mobile NPUs), namely *i*) fixed-point weight and activation quantization with integer arithmetic operations, and *ii*) per-tensor/channel quantization with static pre-computed ranges. Our objective is hence to improve existing state-of-theart approaches such as SmoothQuant (Xiao et al., 2023) and OmniQuant (Shao et al., 2024) while staying within the limits of hardware support on device.

#### <span id="page-2-0"></span>3.2 Weight Equivalent Transformation

Prior efforts on LLM quantization (Dettmers et al., 2022; Xiao et al., 2023) observed that activations are harder to quantize compared to the model weights due to the outlier channel dimensions with diverse min-max ranges. As an example, given a fully connected layer  $\mathbf{Y} = \mathbf{X}\mathbf{W}, \mathbf{W} \in \mathbb{R}^{N \times M}, \mathbf{X} \in \mathbb{R}^{N}, \mathbf{Y} \in \mathbb{R}^{M}$ , specific channel dimensions  $\{i: 0 \leq i < N\}$  in  $\mathbf{X}$  may have a wide min-max range across different data samples, causing substantial quantization errors. To counteract this, previous methods proposed a weight equivalent transform defined by a scaling vector  $\mathbf{S} \in \mathbb{R}^{N}$ :

$$\mathbf{Y} = \mathbf{X}\mathbf{W} = (\mathbf{X}\mathbf{S}^{-1}) \cdot (\mathbf{S}\mathbf{W}) = \hat{\mathbf{X}}\hat{\mathbf{W}}$$
 (1)

The goal is to find the optimal scaling vector  $\mathbf{S}$  such that both  $\hat{\mathbf{X}}$  and  $\hat{\mathbf{W}}$  are easier to quantize compared to the original  $\mathbf{X}$  and  $\mathbf{W}$ . SmoothQuant (Xiao et al., 2023) reparameterized  $\mathbf{S}$  as  $\mathbf{s}_i = \frac{max(|\mathbf{X}_i|)^{\alpha}}{max(|\mathbf{W}_i|)^{(1-\alpha)}}$ ),  $0 \le i < N$ , and searched for the hyper-parameter  $\alpha$ . The obtained  $\mathbf{S}$  is similar to the closed-form solution derived in (Nagel et al., 2019). OmniQuant (Shao et al., 2024) extended SmoothQuant (Xiao et al., 2023) by learning

<span id="page-3-0"></span>
$$X_0 = Norm(X) \qquad X_1 = X_0 A \qquad X_1 = X_1 W_1 \qquad X_2 = X_1 W_1 \qquad Y = X_2 W_2$$

$$X \in \mathbb{R}^N, \ A \in \mathbb{R}^N, \ S \in \mathbb{R}^N, \ W \in \mathbb{R}^{N \times N}$$

$$X_0 = Norm(X) \qquad \bar{X}_1 = X_0 \ (AS_0^{-1}) \qquad \bar{X}_1 = \bar{X}_1 (S_0 W_1 S_1^{-1}) \qquad \bar{X}_2 \qquad Y = \bar{X}_2 \ (S_1 W_2)$$

Figure 1: Weight equalization transformation proposed in (Nagel et al., 2019; Xiao et al., 2023; Shao et al., 2024). In this example, we use three consecutive layers: one normalization layer, e.g. LayerNorm (Ba et al., 2016)/RMSNorm (Zhang and Sennrich, 2019), and two linear layers, and assume the activations of all layers have the same hidden dimension N. Here  $A \in \mathbb{R}^N$  refers to the affinity transformation of the normalization layer. The goal of weight transformation is to learn the scaling vector S such that the resulting weight matrices (i.e.  $S_0W_1S_1^{-1}$  and  $S_1W_2$ ) and activations  $\bar{X}$ , are easier to quantize. S is hence the only learnable parameters. Note that the new model is mathematically equivalent to the original model when unquantized.

S, together with the weight clipping parameters via block-wise error minimization. Here, both S and  $S^{-1}$  can be fused to the adjacent linear layers, making the transformation mathematically equivalent to the original models. Figure 1 provides an illustration of the transformation among consecutive linear layers.

## 4 MobileQuant: Towards Mobile-friendly Quantization

# 4.1 Challenges for Mobile-friendly Quantization

The weight equivalent transformation approaches used in SmoothQuant and OmniQuant, as described in Section 3.2, demonstrate strong performance on GPU-like hardware. However, they do not work out of the box for edge devices. Specifically, two challenges remain: i) the weight transformations cannot propagate beyond non-linear operators, e.g. Softmax, RMSNorm (Zhang and Sennrich, 2019), LayerNorm (Ba et al., 2016), SiLU/GELU (Hendrycks and Gimpel, 2016). To counteract this, we apply weight transformations on all consecutive layers with linear components, e.g. between linear layers or affine transformations in the normalization layers, while keeping the nonlinear activations in 16-bit integers; ii) with the weight transformation, the distribution of the activations shifts accordingly. This causes essential difficulty for learning-based approaches like Omni-Quant (Shao et al., 2024), when the min-max range for the activations changes after each training iteration. OmniQuant (Shao et al., 2024) proposed to bypass the issue with dynamic per-token quantization, which has no hardware support on-device.

## <span id="page-3-1"></span>4.2 Learning the Per-tensor Range of the Activations

Given the distribution of the activations shifts accordingly with the weight transformation, the ideal solution is to re-estimate the activation ranges across the training set after each training iteration. However, doing so is computationally prohibited. Hence, we propose to learn the activation range jointly with the weight transformation. Given an activation tensor  $\mathbf{X}$ , instead of learning the min and max values  $f_{min}(\mathbf{X})$ ,  $f_{max}(\mathbf{X})$  directly, we leverage the correlation between  $f_{min}$ ,  $f_{max}$  and the scale and offset parameters,  $\alpha, \beta \in \mathbb{R}$ , for quantization. With the targeted bit-width bw, quantizing  $\mathbf{X}$  can be formulated as:

$$q_{max} = 2^{bw} - 1,$$
 (2)

$$\alpha = \frac{f_{max} - f_{min}}{q_{max}}, \beta = \frac{f_{min}}{\alpha}$$
 (3)

$$\mathbf{X}_{int} = min(max(ste(\frac{\mathbf{X}}{\alpha}) - \beta, 0), q_{max})$$
 (4)

Here,  $\mathbf{X}_{int}$  refers to the quantized tensor of  $\mathbf{X}$ , ste refers to straight-through estimator. We can therefore learn  $f_{min} = \alpha \beta$  and  $f_{max} = \alpha q_{max} + \alpha \beta$  indirectly by learning  $\alpha$  and  $\beta$ , which are computationally more stable.

<span id="page-4-0"></span>

| WikiText (↓)                       | TinyLLaMA<br>1.1B | StableLM-2<br>1.6B | Gemma<br>2B    |  |
|------------------------------------|-------------------|--------------------|----------------|--|
| FP16                               | 14.9              | 28.4               | 18.0           |  |
|                                    | W8A8              |                    |                |  |
| SmoothQuant-Static                 | 177               | 583                | >1E+03         |  |
| SmoothQuant-Edge                   | 27.1              | 74.5               | 45.3           |  |
| OmniQuant-Static                   | 51.0              | 298.6              | >1E+03         |  |
| OmniQuant-Edge                     | 16.3              | 30.9               | 23.4           |  |
| W4A8                               |                   |                    |                |  |
| OmniQuant-Static<br>OmniQuant-Edge | 416.3<br>18.8     | 258.5<br>36.0      | >1E+03<br>23.9 |  |

Table 1: Adapting quantization SOTA to the ondevice setting. OmniQuant and SmoothQuant are not fully supported for on-device deployment. We introduce mobile-friendly variants. Evaluation: perplexity on WikiText [\(Merity et al.,](#page-9-14) [2016\)](#page-9-14). We adopt the "Edge" variants as strong on-device baselines.

## 4.3 End-to-end Optimization vs Layer-wise Optimization

To learn the equivalent transformation, previous works either resort to closed-form solutions [\(Nagel](#page-9-9) [et al.,](#page-9-9) [2019\)](#page-9-9), search-based methods [\(Xiao et al.,](#page-9-1) [2023;](#page-9-1) [Lin et al.,](#page-9-0) [2024\)](#page-9-0), or layer-wise error minimization [\(Shao et al.,](#page-9-4) [2024\)](#page-9-4). These solutions require limited training budget, but, as shown in Section. [5.4,](#page-5-0) lead to sub-optimal performance. Particularly, given the restricted form of supervision, we show that these methods cannot scale with more training samples or iterations. We, instead, propose to jointly optimize all the training parameters, including the weight equalization parameters S, weight clipping parameters used in Omni-Quant [\(Shao et al.,](#page-9-4) [2024\)](#page-9-4), and the range parameters α, β for all layers in an end-to-end manner. Compared to previous PTQ approaches, which struggle with more training samples and epochs, we demonstrate that our holistic optimization approach consistently improves the performance with larger training settings for different LLM architectures. Compared to QAT, our method preserves model generalizability and does not overfit to specific calibration samples, achieving near-lossless zero-shot performance.

## 5 Experiments

#### 5.1 Setup

We perform experiments by training and simulating the quantization on GPUs and further evaluate the on-device performance on a Samsung Galaxy S24, with the Snapdragon 8 Gen 3 HTP as the compute

unit. All models were trained on two A100 GPUs, with a maximum sequence length of 2048.

Architectures: MobileQuant focuses on lightweight LLMs that are suitable to be deployed on mobile devices. Hence, we experiment with representative pretrained models with different architectures: TinyLlaMA-1.1B-Chat-v1.0 [\(Zhang](#page-9-15) [et al.,](#page-9-15) [2024\)](#page-9-15), StableLM-2-1.6B [\(Bellagente et al.,](#page-8-6) [2024\)](#page-8-6), and Gemma-2B [\(Google,](#page-8-7) [2024\)](#page-8-7).

Quantization details. MobileQuant use a subset of the Pile [\(Gao et al.,](#page-8-8) [2020\)](#page-8-8) dataset as the calibration set. We explore two quantization settings: i) W8A8: 8-bit weight quantization with per-tensor statistics except for the last linear projection in each MLP block (e.g. down\_proj in LLaMA-like [\(Tou](#page-9-16)[vron et al.,](#page-9-16) [2023\)](#page-9-16) models) which uses per-channel statistics, and 8-bit per-tensor quantization for the activations, except those linked to non-linear operators. ii) W4A8: 4-bit per-channel quantization for model weights, and 8-bit per-tensor quantization, likewise excluding non-linear operators.

We consider asymmetric quantization for both settings, which can utilize the full quantized range. We also provide extra experiments on symmetric per-channel W4A8 quantization in the supplemental material, which is better supported by the current on-device toolchain we use.

Evaluation datasets. We evaluate our quantization approach in a zero-shot setting on representative tasks from the Language Model Evaluation Harness benchmark (Harness) [\(Gao et al.,](#page-8-9) [2023\)](#page-8-9) including WikiText [\(Merity et al.,](#page-9-14) [2016\)](#page-9-14), AI2 Reasoning Challenge (arc\_challenge) [\(Clark](#page-8-10) [et al.,](#page-8-10) [2018\)](#page-8-10), Hellaswag [\(Zellers et al.,](#page-9-17) [2019\)](#page-9-17), and MMLU [\(Hendrycks et al.,](#page-9-18) [2021\)](#page-9-18).

#### 5.2 On-device Baselines

In this section, we extend state-of-the-art weight-activation quantization methods, SmoothQuant [\(Xiao et al.,](#page-9-1) [2023\)](#page-9-1) and Omni-Quant [\(Shao et al.,](#page-9-4) [2024\)](#page-9-4) on device and use them as baselines. As these approaches utilize dynamic per-token quantization for the activation, which is not supported on edge devices, we modify these methods to work on device by using static per-tensor activation quantization, referring to these variants as OmniQuant-Static and SmoothQuant-Static respectively. Note that, for SmoothQuant, we only include evaluations on W8A8, which is the default setting used in the original work.

<span id="page-5-2"></span>

|          |         | TinyLlaMA-1.1B |            | StableLM-2-1.6B |            | Gemma-2B   |            |
|----------|---------|----------------|------------|-----------------|------------|------------|------------|
| #Samples | #Epochs | Block-wise     | End-to-end | Block-wise      | End-to-end | Block-wise | End-to-end |
| 128      | 20      | 18.3           | 19.9       | 35.4            | 40.4       | 23.0       | 32.5       |
| 128      | 60      | 18.3           | 17.4       | 37.0            | 36.5       | 23.7       | 26.1       |
| 128      | 120     | 18.1           | 17.1       | 37.1            | 35.1       | 24.0       | 23.1       |
| 256      | 60      | 17.9           | 17.1       | 35.9            | 34.2       | 24.5       | 22.0       |
| 1024     | 60      | 17.7           | 17.1       | 35.4            | 33.6       | 24.9       | 21.4       |

Table 2: End-to-end range optimization: Perplexity on WikiText for OmniQuant-Edge W4A8 setting with block-wise vs end-to-end range optimization. Best overall performance is in bold, best block-wise performance is underlined. Compared to block-wise, end-to-end optimization benefits from larger training settings with more samples/iterations, leading to better performance.

As shown in Table [1,](#page-4-0) both "Static" variants suffer from large performance degradation when evaluated on WikiText [\(Merity et al.,](#page-9-14) [2016\)](#page-9-14). We further observe that the performance drop is mainly caused by quantizing the activations for the last linear layer in each MLP head (*i.e*. down\_proj in LLaMA-like [\(Touvron et al.,](#page-9-16) [2023\)](#page-9-16) models). To further alleviate this issue, we introduce an extra weight equalization transformation between consecutive linear layers in each MLP head (*i.e*. S between the up\_proj and down\_proj layers in TinyLLaMA [\(Zhang et al.,](#page-9-15) [2024\)](#page-9-15)). The new models, which we termed *SmoothQuant-Edge* and *OmniQuant-Edge* respectively, significantly alleviate the performance degradation. For the remainder of this section, we use these adapted models as strong on-device baselines.

<span id="page-5-1"></span>

| WikiText (↓)         | TinyLLaMA<br>1.1B | StableLM-2<br>1.6B | Gemma<br>2B |  |
|----------------------|-------------------|--------------------|-------------|--|
| FP16                 | 14.9              | 28.4               | 18.0        |  |
| W8A8                 |                   |                    |             |  |
| OmniQuant-Edge       | 16.3              | 30.9               | 23.4        |  |
| OmniQuant-Edge w ARL | 15.9              | 30.5               | 22.8        |  |
| W4A8                 |                   |                    |             |  |
| OmniQuant-Edge       | 18.8              | 36.0               | 23.9        |  |
| OmniQuant-Edge w ARL | 18.3              | 35.4               | 23.0        |  |

Table 3: Activation range learning (ARL): Perplexity on WikiText for OmniQuant-Edge with/without ARL for W8A8 and W4A8 settings. The performance gains are larger on models with larger quantization errors.

#### 5.3 Impact of Activation Range Learning

Table [1](#page-4-0) shows that the learning-based approach, OmniQuant [\(Shao et al.,](#page-9-4) [2024\)](#page-9-4), outperforms the search-based method, SmoothQuant [\(Xiao et al.,](#page-9-1) [2023\)](#page-9-1), for all models by a notable margin. However, learning to transform the weights with fixed

activation ranges is suboptimal, as the activation ranges shift after each training iteration. We further evaluate the impact of incorporating activation range learning (ARL), described in Section. [4.2,](#page-3-1) into OmniQuant [\(Shao et al.,](#page-9-4) [2024\)](#page-9-4). In other words, we learn the per-tensor scale and offset parameters, together with the weight transformation via blockwise error minimization.

Table [3](#page-5-1) demonstrates that activation range learning (ARL) consistently improves the performance for all LLM models across all settings. The gains are larger for quantized models exhibiting a larger performance gap compared to the FP16 models. Notably, these models require more training steps to mitigate the quantization errors, leading to larger range shifts for the activation.

### <span id="page-5-0"></span>5.4 Impact of End-to-End Optimization

In the previous section, we show that incorporating ARL into our baselines results in consistent improvements. Nonetheless, there is still a notable performance gap between the quantized models and the FP16 models, especially under the W4A8 setting. In order to reduce this gap, we attempt to improve the performance by scaling up the performance, namely increasing the number of calibration samples and the number of training epochs. However, Table [2](#page-5-2) shows that the performance of all considered models saturate as we scale the training up using the block-wise approach proposed in OmniQuant [\(Shao et al.,](#page-9-4) [2024\)](#page-9-4). We therefore conjecture that the optimization is hindered by the block-wise error minimization objective that provides limited global supervision. To verify this, we use our end-to-end training pipeline and jointly optimize all trainable parameters of the whole model, namely the weight transformation, clipping, and activation range learning parameters.

<span id="page-6-0"></span>

|                 |                  | WikiText ↓  | ARC-Challenge ↑ | HellaSwag ↑ | MMLU ↑      |
|-----------------|------------------|-------------|-----------------|-------------|-------------|
|                 |                  | W8A8        |                 |             |             |
|                 | FP16             | 14.9        | 33              | 60          | 25          |
|                 | SmoothQuant-Edge | 27.1        | 29.6            | 52.8        | 24.9        |
| TinyLlaMA-1.1B  | OmniQuant-Edge   | 16.3        | 31.7            | 58.4        | 24.9        |
|                 | MobileQuant      | 15.5 (-0.8) | 31.9 (+0.2)     | 59.2 (+0.8) | 25.0 (+0.1) |
|                 | FP16             | 28.4        | 39              | 65          | 32          |
|                 | SmoothQuant-Edge | 70.2        | 35.9            | 61.8        | 26.0        |
| StableLM-2-1.6B | OmniQuant-Edge   | 30.9        | 36.3            | 63.4        | 29.3        |
|                 | MobileQuant      | 29.7 (-1.2) | 37.1 (+0.8)     | 63.6 (+0.2) | 30.0 (+0.7) |
|                 | FP16             | 18.0        | 23              | 42          | 28          |
|                 | SmoothQuant-Edge | 45.3        | 23.0            | 39.0        | 25.8        |
| Gemma-2B        | OmniQuant-Edge   | 23.4        | 22.4            | 39.9        | 26.8        |
|                 | MobileQuant      | 20.3 (-3.1) | 21.8 (-1.2)     | 40.9 (+1.0) | 25.8 (-1.0) |
|                 |                  | W4A8        |                 |             |             |
|                 | FP16             | 14.9        | 33              | 60          | 25          |
| TinyLlaMA-1.1B  | OmniQuant-Edge   | 18.8        | 28.8            | 56.4        | 25.5        |
|                 | MobileQuant      | 17.1 (-1.7) | 32.3 (+3.5)     | 57.0 (+0.6) | 25.5 (+0.0) |
|                 | FP16             | 28.4        | 39              | 65          | 32          |
| StableLM-2-1.6B | OmniQuant-Edge   | 36.0        | 34.9            | 60.2        | 25.9        |
|                 | MobileQuant      | 33.6 (-2.4) | 35.6 (+0.7)     | 60.5 (+0.3) | 24.1 (-1.8) |
|                 | FP16             | 18.0        | 23              | 42          | 28          |
| Gemma-2B        | OmniQuant-Edge   | 23.9        | 23.1            | 38.1        | 25.5        |
|                 | MobileQuant      | 21.4 (-2.5) | 23.0 (-0.1)     | 38.9 (+0.8) | 25.6 (+0.1) |

Table 4: Comparisons with existing state-of-the-art methods on Harness: Best performance is bold, second-best underlined. We indicate the gain/drop of our approach vs the next strongest on-device baseline. Our method, MobileQuant, demonstrates consistent improvements across models, quantization configurations, and tasks, achieving best performance in most cases.

As shown in Table [2,](#page-5-2) our end-to-end trained models demonstrate consistent improvements with more training samples and iterations, only underperforming the blockwise optimized models in the smallest settings when the models were undertrained. We currently train the models with up to 1024 samples for 60 epochs but posit that the models could be further improved with more diverse samples and larger training settings.

#### 5.5 Harness Benchmark Results

Following previous approaches [\(Xiao et al.,](#page-9-1) [2023;](#page-9-1) [Shao et al.,](#page-9-4) [2024;](#page-9-4) [Liu et al.,](#page-9-5) [2023\)](#page-9-5), we perform zero-shot evaluations on representative tasks from the Harness benchmark [\(Gao et al.,](#page-8-9) [2023\)](#page-8-9). Table [4](#page-6-0) shows that, in addition to the WikiText perplexity, our method also improves the quantization performance for the common sense reasoning tasks in general, without using any in-domain data. The

<span id="page-6-1"></span>

| TinyLlaMA-1.1B   | WikiText ↓ | Lambada ↑ |
|------------------|------------|-----------|
| FP16             | 14.9       | 82.9      |
| W8A16            | 15.2       | 82.9      |
| MobileQuant W8A8 | 15.6       | 82.4      |
| full W8A8        | 8e5        | 1.3       |

Table 5: On-device accuracy of the quantized TinyLLaMA-1.1B-Chat-v1.0 on WikiText and LAM-BADA. Models run on a Snapdragon 8 Gen 3 HTP processor.

improvements are consistent for most benchmarks and we believe that the performance of our method could be further improved with in-domain data, especially for benchmarks with a large domain shift relative to our calibration set (i.e. Pile [\(Gao et al.,](#page-8-8) [2020\)](#page-8-8)).

<span id="page-7-0"></span>

| Seq. Length | Method                    | Avg. lat. (ms) | Avg. energy (mJ) | Peak mem. (MiB) |  |  |
|-------------|---------------------------|----------------|------------------|-----------------|--|--|
|             | Prompt Encoding           |                |                  |                 |  |  |
|             | W8A16                     | 510            | 1000             | 1019            |  |  |
| 256         | MobileQuant (W8A8)        | 276            | 490              | 1011            |  |  |
|             | full W8A8                 | 89             | 183              | 1006            |  |  |
|             | Autoregressive Generation |                |                  |                 |  |  |
|             | W8A16                     | 54             | 69               | 1007            |  |  |
| 1024        | MobileQuant (W8A8)        | 46             | 61               | 1005            |  |  |
|             | full W8A8                 | 42             | 61               | 1003            |  |  |
|             | W8A16                     | 119            | 165              | 1010            |  |  |
| 2048        | MobileQuant (W8A8)        | 95             | 110              | 1007            |  |  |
|             | full W8A8                 | 94             | 106              | 1006            |  |  |

Table 6: On-device execution cost. Measurements of latency, energy and memory are computed under sustained execution (30 minutes). Values are reported per single forward pass.

#### 5.6 On-device Evaluation

On-device Setup. We further deploy the quantized LLM model on a mobile device and provide evaluations on the accuracy, latency, memory usage, and power consumption. Specifically, we evaluate the W8A8 quantized TinyLLaMA-1.1B-Chatv1.0 [\(Zhang et al.,](#page-9-15) [2024\)](#page-9-15) model on a Samsung Galaxy S24, using the Snapdragon 8 Gen 3 HTP as the compute unit. We evaluate the model under three different quantization settings: 1) *W8A16*, which keeps activations as 16-bit; Note that the matrix multiplication for the self-attention computation is still between 8-bit and 16-bit unsigned integer activations to avoid potential overflowing, 2) *full W8A8*, keeps all activations in 8-bit, and 4) our proposed MobileQuant for W8A8.

On-device Accuracy. We compute the accuracy of the quantized models on two tasks: i) Wiki-Text [\(Merity et al.,](#page-9-14) [2016\)](#page-9-14) from Harness [\(Gao et al.,](#page-8-9) [2023\)](#page-8-9), as we used in our previous evaluations and ii) LAMBADA [\(Paperno et al.,](#page-9-19) [2016\)](#page-9-19), which predicts the last token of a sentence given the previous context. Following SmoothQuant [\(Xiao et al.,](#page-9-1) [2023\)](#page-9-1), we use the first 1000 samples from LAM-BADA for this task. Table [5](#page-6-1) shows that using 16-bit activations (i.e. *W8A16*) achieves lossless performance. However, quantizing all activations into 8-bit leads to near-zero performance, highlighting the difficulty of activation quantization. Our W8A8 MobileQuant model achieves near-lossless performance in both tasks, approaching the performance of the FP16 model.

On-device Latency. We provide the on-device

latency evaluation by running the quantized model in two modes: i) prompt encoding with a context length of 256, ii) auto-regressive generation with a maximum sequence length of 1024 and 2048. Table [6](#page-7-0) shows that, for prompt encoding, using lower-bitwidth activations is critical to reducing the inference latency, as some of the operations, e.g. self-attention (batched matrix multiplication), are compute-intensive. Our model demonstrates significant advantages over the full W8A16 solution, reducing the latency by 40%. However, there is still a large gap between MobileQuant and the *full W8A8* model, indicating the improvement margin. For auto-regressive generation, the latency gaps are smaller. We posit that the auto-regressive generation is not as compute-bound as prompt encoding, especially for lightweight models, but instead is partially memory access-bound. Our solution demonstrates a 20% latency reduction compared to *W8A16*, achieving the same latency as the *full W8A8* model. We include a video demo that showcases the auto-regressive generation of the quantized model on device in the supplemental material. In general, the advantage of low-bitwidth activations correlates strongly with the scale of the computation. Hence, we aim to extend the latency evaluation to larger models in our future research.

On-device Energy and Memory. Apart from latency, energy consumption is another important aspect of on-device execution, which is often overlooked by quantization research. To measure the energy requirements of different models, we run them on a number of identical mobile phones as used before continuously for 30 minutes. The phones are connected to the testing host machine via WiFi using an internal network without access to the internet, to avoid any undesired network activity. The phones are also not being charged and their screens are turned off. All phones begin each test at the same battery level and the final energy of running a model is calculated as the ratio of the total battery discharged over the duration of a test, minus reference discharge of a phone not running any model, divided by the number of times the model was run. We repeat measurements for different models 3 times, rotating the phones each time, and report the average. We also report peak memory required to run a model as the peak resident memory recorded for the benchmarking process by the Linux Kernel (the so-called Virtual Memory High Water Mark). From Table [6,](#page-7-0) the energy consumption of each model aligns well with the latency. Compared to *W8A16*, MobileQuant reduces 50% of the power usage for prompt encoding and 35% for autoregressive generation. The peak memory usage for all models are similar as it is dominated by the model weight.

## 6 Conclusion

We revisited LLM quantization from the perspective of deployment on edge devices such as mobile phones. We examined the limitations of current state-of-the-art models for on-device deployment and present MobileQuant, the first framework to facilitate compute-, and energy-efficient quantized LLMs with minimal performance loss. Mobile-Quant is drop-in compatible with today's edge device hardware and low-level runtimes.

## Limitations

The work explores reducing the overhead of ondevice deployment for Large Language Models by hardware-friendly quantization. Our current study focuses on established pretrained LLMs with 1 to 2 billion parameters, which limits the overall capacity of the quantized models. Also, the quantized models inherit the error of the pretrained models, e.g. hallucination, which may be corrected by extra guard models [\(Inan et al.,](#page-9-20) [2023\)](#page-9-20). For now, we demonstrate the efficiency and effectiveness of MobileQuant on specific high-end mobile phones. We plan to extend our research to more LLMs with different architectures, model sizes, capacities, as well as more edge devices in the future.

## References

- <span id="page-8-1"></span>Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. 2024. [QuaRot:](https://arxiv.org/abs/2404.00456) [Outlier-free 4-bit inference in rotated llms.](https://arxiv.org/abs/2404.00456) *Preprint*, arXiv:2404.00456.
- <span id="page-8-5"></span>Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. 2016. [Layer normalization.](https://arxiv.org/abs/1607.06450) *Preprint*, arXiv:1607.06450.
- <span id="page-8-6"></span>Marco Bellagente, Jonathan Tow, Dakota Mahan, Duy Phung, Maksym Zhuravinskyi, Reshinth Adithyan, James Baicoianu, Ben Brooks, Nathan Cooper, Ashish Datta, et al. 2024. [Stable LM 2 1.6 b technical](https://arxiv.org/abs/2402.17834) [report.](https://arxiv.org/abs/2402.17834) *Preprint*, arXiv:2402.17834.
- <span id="page-8-3"></span>Yelysei Bondarenko, Markus Nagel, and Tijmen Blankevoort. 2023. Quantizable transformers: Removing outliers by helping attention heads do nothing. In *Advances on Neural Information Processing Systems*.
- <span id="page-8-10"></span>Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. [Think you have solved question an](https://arxiv.org/abs/1803.05457)[swering? try ARC, the AI2 reasoning challenge.](https://arxiv.org/abs/1803.05457) *Preprint*, arXiv:1803.05457.
- <span id="page-8-4"></span>Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. 2022. LLM.int8(): 8-bit matrix multiplication for transformers at scale. In *Advances on Neural Information Processing Systems*.
- <span id="page-8-0"></span>Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2023. GPTQ: Accurate post-training compression for generative pretrained transformers. In *International Conference on Learning Representations*.
- <span id="page-8-8"></span>Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. 2020. [The Pile: An](https://arxiv.org/abs/2101.00027) [800GB dataset of diverse text for language modeling.](https://arxiv.org/abs/2101.00027) *Preprint*, arXiv:2101.00027.
- <span id="page-8-9"></span>Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. 2023. [A framework for few-shot language model](https://doi.org/10.5281/zenodo.10256836) [evaluation.](https://doi.org/10.5281/zenodo.10256836)
- <span id="page-8-2"></span>Google. 2021. Edge tpu. [https://](https://cloud.google.com/edge-tpu) cloud.google.[com/edge-tpu](https://cloud.google.com/edge-tpu).
- <span id="page-8-7"></span>Google. 2024. [Gemma: Open models based](https://arxiv.org/abs/2403.08295) [on Gemini research and technology.](https://arxiv.org/abs/2403.08295) *Preprint*, arXiv:2403.08295.

- <span id="page-9-18"></span>Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. *International Conference on Learning Representations*.
- <span id="page-9-13"></span>Dan Hendrycks and Kevin Gimpel. 2016. [Bridging non](https://arxiv.org/abs/1606.08415)[linearities and stochastic regularizers with gaussian](https://arxiv.org/abs/1606.08415) [error linear units.](https://arxiv.org/abs/1606.08415) *Preprint*, arXiv:1606.08415.
- <span id="page-9-20"></span>Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, and Madian Khabsa. 2023. [Llama guard: LLM](https://arxiv.org/abs/2312.06674)[based input-output safeguard for human-AI conver](https://arxiv.org/abs/2312.06674)[sations.](https://arxiv.org/abs/2312.06674) *Preprint*, arXiv:2312.06674.
- <span id="page-9-11"></span>Jung Hyun Lee, Jeonghoon Kim, Se Jung Kwon, and Dongsoo Lee. 2023. FlexRound: Learnable rounding based on element-wise division for post-training quantization. In *International Conference on Machine Learning*.
- <span id="page-9-0"></span>Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. 2024. AWQ: Activation-aware weight quantization for llm compression and acceleration. In *Conference on Machine Learning and Systems*.
- <span id="page-9-6"></span>Jing Liu, Ruihao Gong, Xiuying Wei, Zhiwei Dong, Jianfei Cai, and Bohan Zhuang. 2024. Qllm: Accurate and efficient low-bitwidth quantization for large language models. In *The Twelfth International Conference on Learning Representations*.
- <span id="page-9-5"></span>Zechun Liu, Barlas Oguz, Changsheng Zhao, Ernie Chang, Pierre Stock, Yashar Mehdad, Yangyang Shi, Raghuraman Krishnamoorthi, and Vikas Chandra. 2023. LLM-QAT: Data-free quantization aware training for large language models. *arXiv preprint arXiv:2307.06281*.
- <span id="page-9-3"></span>Haozheng Luo, Jerry Yao-Chieh Hu, Pei-Hsuan Chang, Hong-Yu Chen, Weijian Li, Wei-Po Wang, and Han Liu. 2024. [Outeffhop: A principled outlier-efficient](https://openreview.net/forum?id=ZCrRCICOkr) [attention layer from dense associative memory mod](https://openreview.net/forum?id=ZCrRCICOkr)[els.](https://openreview.net/forum?id=ZCrRCICOkr) In *Workshop on Efficient Systems for Foundation Models II @ ICML2024*.
- <span id="page-9-8"></span>E. Mahurin. 2023. [Qualocmm®](https://doi.org/10.1109/HCS59251.2023.10254715) hexagon™ NPU. In *IEEE Hot Chips Symposium*, pages 1–19. IEEE Computer Society.
- <span id="page-9-14"></span>Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016. [Pointer sentinel mixture mod](https://arxiv.org/abs/1609.07843)[els.](https://arxiv.org/abs/1609.07843) *CoRR*, abs/1609.07843.
- <span id="page-9-10"></span>Markus Nagel, Rana Ali Amjad, Mart Van Baalen, Christos Louizos, and Tijmen Blankevoort. 2020. Up or down? Adaptive rounding for post-training quantization. In *International Conference on Machine Learning*.

- <span id="page-9-9"></span>Markus Nagel, Mart van Baalen, Tijmen Blankevoort, and Max Welling. 2019. Data-free quantization through weight equalization and bias correction. In *IEEE International Conference on Computer Vision*.
- <span id="page-9-19"></span>Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. 2016. The LAMBADA dataset: Word prediction requiring a broad discourse context. In *Annual Meeting of the Association for Computational Linguistics*.
- <span id="page-9-7"></span>Qualcomm. 2024. Unlocking on-device generative ai with an npu and heterogeneous computing. https://www.qualcomm.[com/content/](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Unlocking-on-device-generative-AI-with-an-NPU-and-heterogeneous-computing.pdf) [dam/qcomm-martech/dm-assets/documents/](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Unlocking-on-device-generative-AI-with-an-NPU-and-heterogeneous-computing.pdf) [Unlocking-on-device-generative-AI-with](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Unlocking-on-device-generative-AI-with-an-NPU-and-heterogeneous-computing.pdf)[an-NPU-and-heterogeneous-computing](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Unlocking-on-device-generative-AI-with-an-NPU-and-heterogeneous-computing.pdf).pdf.
- <span id="page-9-4"></span>Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Peng Gao, Yu Qiao, and Ping Luo. 2024. OmniQuant: Omnidirectionally calibrated quantization for large language models. In *International Conference on Learning Representations*.
- <span id="page-9-16"></span>Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. [LLaMA: Open](https://arxiv.org/abs/2302.13971) [and efficient foundation language models.](https://arxiv.org/abs/2302.13971) *Preprint*, arXiv:2302.13971.
- <span id="page-9-2"></span>Shang Wu, Yen-Ju Lu, Haozheng Luo, Jerry Yao-Chieh Hu, Jiayi Wang, Jing Liu, Najim Dehak, Jesus Villalba, and Han Liu. 2024. [Fast adaptation and robust](https://openreview.net/forum?id=kZBM2UYiQh) [quantization of multi-modal foundation models from](https://openreview.net/forum?id=kZBM2UYiQh) [associative memory: A case study in speechLM.](https://openreview.net/forum?id=kZBM2UYiQh) In *Workshop on Efficient Systems for Foundation Models II @ ICML2024*.
- <span id="page-9-1"></span>Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. 2023. SmoothQuant: Accurate and efficient post-training quantization for large language models. In *International Conference on Machine Learning*.
- <span id="page-9-17"></span>Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. HellaSwag: Can a machine really finish your sentence? In *Annual Meeting of the Association for Computational Linguistics*.
- <span id="page-9-12"></span>Biao Zhang and Rico Sennrich. 2019. Root mean square layer normalization. In *Advances on Neural Information Processing Systems*.
- <span id="page-9-15"></span>Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. 2024. [TinyLlama: An open-source small](https://arxiv.org/abs/2401.02385) [language model.](https://arxiv.org/abs/2401.02385) *Preprint*, arXiv:2401.02385.

## A Appendix: On-device Experiments for W4A8

In this appendix, we provide further on-device evaluation for W4A8. The current on-device toolchains we use support only symmetric per-channel weight quantization. This, however, typically leads to performance degradation as the full quantization range may not be fully utilized if the weights are biased toward positive or negative. Here we first present extra W4A8 results with symmetric per-channel quantization. We then include on-device latency evaluation showcasing the advantages of using 4 bit integer representation for the weights.

## A.1 Symmetric vs Asymmetric W4A8 Quantization

We train extra W4A8 models with symmetric perchannel quantization. Table [7](#page-10-0) presents the performance of symmetric per-channel W4A8 models on Wikitext [\(Gao et al.,](#page-8-9) [2023\)](#page-8-9), confirming the performance degradation compared to the asymmetric counterparts.

<span id="page-10-0"></span>

| WikiText (↓)     | TinyLLaMA | StableLM-2 | Gemma |
|------------------|-----------|------------|-------|
|                  | 1.1B      | 1.6B       | 2B    |
| FP16             | 14.9      | 28.4       | 18.0  |
| MobileQuant-Asym | 17.1      | 33.6       | 21.4  |
| MobileQuant-Sym  | 17.5      | 36.4       | 24.7  |

Table 7: Evaluation of symmetric vs asymmetric W4A8 per-channel quantization on Wikitext [\(Gao et al.,](#page-8-9) [2023\)](#page-8-9).

<span id="page-10-1"></span>

| Method                                    | TinyLlaMA-1.1B                    | Gemma-2B |  |
|-------------------------------------------|-----------------------------------|----------|--|
|                                           | Prompt Encoding (Seq. Length 256) |          |  |
| W8A16                                     | 510                               | 1191     |  |
| MobileQuant (W8A8)                        | 276                               | 752      |  |
| full W8A8                                 | 89                                | 311      |  |
| W4A16                                     | 320                               | 617      |  |
| MobileQuant (W4A8)                        | 239                               | 460      |  |
| full W4A8                                 | 89                                | 98       |  |
| Autoregressive Gen. (Context Length 1024) |                                   |          |  |
| W8A16                                     | 54                                | 78       |  |
| MobileQuant (W8A8)                        | 46                                | 60       |  |
| full W8A8                                 | 42                                | 59       |  |
| W4A16                                     | 50                                | 56       |  |
| MobileQuant (W4A8)                        | 38                                | 40       |  |
| full W4A8                                 | 40                                | 40       |  |

Table 8: On-device latency (ms) for TinyLlaMA-1.1B [\(Zhang et al.,](#page-9-15) [2024\)](#page-9-15) and Gemma-2B [\(Google,](#page-8-7) [2024\)](#page-8-7) across different settings.

## A.2 On-device Latency for Symmetric W4A8 models

We further evaluate the on-device latency of the W4A8 models with symmetric quantization. Tabl[e8](#page-10-1) shows that, compared to W8A8, the W4A8 models demonstrate improved inference speed for both prompt encoding and autoregressive generation. For larger models like Gemma-2B, the improvements are more significant, *i.e.* reducing the latency of prompt encoding and autoregressive generation by 39% and 33%. Here, TinyLLaMA-1.1B achieves the same inference speed, *i.e.* 40 ms per token (25 tok/s). We conjecture that, in this setting, the autoregressive generation for these models is likely memory-bound. We plan to further investigate the performance bottleneck in future research.