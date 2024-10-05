---
title: Understanding RVC - Retrieval-based Voice Conversion
date: 2024-09-26 22:53:51
tags:
    - Music Signal Processing
    - Deep Learning
---
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML' async></script>

TLDR: This blog will discuss:
1 - Technical concepts in the RVC project
2 - Individual modules, such as VITS, RMVPE, HuBERT
3 - The `top-k` retrieval module, and how does it improve generation quality

## 1 - Introduction

AI cover songs have taken the internet by storm alongside the recent generative AI boom ignited by ChatGPT. If you still haven't experienced this piece of heavenly magic, have a listen [Fake Drake](https://www.nytimes.com/2023/04/19/arts/music/ai-drake-the-weeknd-fake.html), [AI Frank Sinatra](https://www.youtube.com/watch?v=HWsb7zTKplc&ab_channel=AiCovers), or [Stephanie Sun](https://www.youtube.com/watch?v=uPMXn7IbdXw&ab_channel=%E5%8D%8E%E8%AF%ADAI%E7%BF%BB%E5%94%B1) (who later posted a rather [gloomy response](https://www.straitstimes.com/life/entertainment/how-do-i-fight-with-that-stefanie-sun-issues-gloomy-response-to-popularity-of-ai-stefanie-sun) on the matter). To be honest, it's weird to feel exciting and scary at the same time after listening to them. 

In this blog post, I would like to provide a breakdown on (arguably) one of the most popular voice conversion project, which is the [**RVC project**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI). I personally think that RVC is one of the reasons why AI covers have gained momentum, given that RVC provides [a rather permissive license](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE) to use its source code and pretrained models for fine-tuning. I have come across many non-technical tutorials / videos about how to fine-tune RVC, but have yet to read an in-depth breakdown on the technical side of things, hence the motivation of writing this blog post. 

I will re-draw some of the diagrams based on my understanding, and (try to) justify the important steps in the RVC model. Another thing to note is that, RVC shares a lot of common concepts with its "predecessor", the [**So-VITS project**](https://github.com/svc-develop-team/so-vits-svc/tree/4.1-Stable), so I hope that this post provides enough details to help readers understand both projects. My crude understanding is that the only difference between So-VITS and RVC is the `top-k` retrieval module, so although the choices of some parts of the modules might be different, but the overall framework should stay similar.

## 2 - Architecture 
<figure>
  <img style="width:100%;" src="/img/rvc-train.png" alt=""/>
  <figcaption><br/>Figure 1: The architecture of RVC for training.</figcaption>
</figure>
<br/>

Voice conversion is essentially a disentanglement task which aims to separate the content and the speaker information. In general, a voice conversion model consists of a **content encoder** that extracts speaker-invariant content information (such as phonemes, text, intonation), and a **acoustic model** that reconstructs the converted voice based on the given content. 

We can breakdown RVC into the following modules:
- A **content feature extractor** to extract information such as phonemes, intonation, etc. Here, RVC chooses [HuBERT](), or more precisely, a variant of [ContentVec](). 
- A **pitch extractor** to get the coarse-level and fine-level F0 estimation. Pitch is also part of content information, especially in the context of singing voice. Here, RVC chooses [RMVPE]().

Figure 2 in VITS (https://arxiv.org/pdf/2310.05118) shows clearly the VITS architecture.
VITS original paper (https://arxiv.org/pdf/2106.06103).


## 3 - Inference 

<figure>
  <img style="width:100%;" src="/img/rvc-infer.png" alt=""/>
  <figcaption><br/>Figure 2: Infer RVC.</figcaption>
</figure>
<br/>

Generator is from DiffSinger's Harmonic Neural Source Filter HiFiGAN (https://github.com/openvpi/DiffSinger/blob/refactor/modules/nsf_hifigan/models.py#L176).


Figure 2 in VITS (https://arxiv.org/pdf/2310.05118) shows clearly the VITS architecture.
VITS original paper (https://arxiv.org/pdf/2106.06103).

VITS: variational inference, augmented with normalizing flows, adversarial training process