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

**AI cover songs** have taken the internet by storm alongside the recent generative AI boom ignited by ChatGPT. If you still haven't had any experience on this piece of heavenly magic, have a listen to [Fake Drake](https://www.nytimes.com/2023/04/19/arts/music/ai-drake-the-weeknd-fake.html), [AI Frank Sinatra](https://www.youtube.com/watch?v=HWsb7zTKplc&ab_channel=AiCovers), or [Stephanie Sun](https://www.youtube.com/watch?v=uPMXn7IbdXw&ab_channel=%E5%8D%8E%E8%AF%ADAI%E7%BF%BB%E5%94%B1) (who later posted a [gloomy response](https://www.straitstimes.com/life/entertainment/how-do-i-fight-with-that-stefanie-sun-issues-gloomy-response-to-popularity-of-ai-stefanie-sun) on the matter). To be honest, it's weird to feel exciting and scary at the same time after listening to them. 

In this blog post, I would like to provide a breakdown on (arguably) one of the most popular voice conversion project, which is the [**RVC project**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI). I personally think that RVC is one of the reasons why AI covers have gained huge momentum, given that RVC provides [a rather permissive license](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE) to use its source code and pretrained models, and an easy-to-use, beginner-friendly Web UI for fine-tuning. 

I have come across many non-technical tutorials / videos about how to fine-tune RVC, but have yet to read an in-depth breakdown on the technical side of things, hence the motivation of writing this blog post. I will re-draw some of the diagrams based on my understanding, and (try to) justify the important steps in the RVC model. 

Another thing to note is that, RVC shares a lot of common concepts with its "predecessor", the [**So-VITS project**](https://github.com/svc-develop-team/so-vits-svc/tree/4.1-Stable), so I hope that this post provides enough details to help readers understand both projects. My crude understanding is that the only difference between So-VITS and RVC is the `top-k` retrieval module, so although the choices of some parts of the modules might be different, but the overall framework should stay similar.

## 2 - Architecture 
<figure>
  <img style="width:100%;" src="/img/rvc-train.png" alt=""/>
  <figcaption><br/>Figure 1: The architecture of RVC for training.</figcaption>
</figure>
<br/>

Voice conversion is essentially a disentanglement task which aims to separate the content and the speaker information. Generally, a voice conversion model consists of a **content encoder** that extracts speaker-invariant content information (such as phonemes, text, intonation), and a **acoustic model** that reconstructs the target voice based on the given content. 

We can break-down RVC into the following modules:
- A **content feature extractor** to extract information such as phonemes, intonation, etc. from the source audio. Here, RVC chooses [**HuBERT**](), or more precisely, a variant of [**ContentVec**]() - this choice is similar to the early versions of So-VITS. 
- A **pitch extractor** to get the coarse-level and fine-level F0 estimation. Pitch is an important part of the content information, especially in the context of singing voice. Here, RVC chooses [**RMVPE**]().
- A **conditional acoustic model** to generate the target audio based on given conditions (i.e. speaker ID & content information). Here, RVC chooses [**VITS**](https://arxiv.org/pdf/2106.06103) as its generation framework, with some noticeble influence by [HiFi-GAN](https://arxiv.org/abs/2010.05646) on the [vocoder](https://github.com/openvpi/DiffSinger/tree/refactor/modules/nsf_hifigan) - this choice is also largely inherited from So-VITS.
- A **retrieval module** newly introduced by RVC. Here, RVC stores all content features of the same speaker into a vector index, which can be used later for similarity search during inference. With this, since the content features are from the training set instead of being extracted solely from the source audio, it could introduce more information about the target speaker, further helping the reconstruction to sound more like the target speaker.

Let's dive into each of the modules one-by-one.

## 3.1 - Content Feature Extraction

**HuBERT** is one of the most popular self-supervised speech representation, commonly used in speech and audio related tasks. First, audio chunks are converted into a series of tokens via an offline clustering method (e.g. a K-Means clustering on the MFCCs over a large dataset). Then, some tokens within a sequence are masked, and a Transformer is trained to predict the masked tokens, i.e. the **masked language modelling** method (there is a reason why BERT appears in the name). The features learnt seems to help reduce word-error rate in speech recognition tasks, showing its superiority in encoding content-related information. The clustering / masked-language-modelling parts are sometimes called the *teacher* / *student* modules.

Both RVC and So-VITS uses the same [hubert-base](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt) model, but despite its name, I don't think it's a purely pretrained HuBERT model, because from the [**ContentVec**](https://proceedings.mlr.press/v162/qian22b/qian22b.pdf) paper, it claims that HuBERT features could still achieve fairly good results on speaker identification, contradicting with the aim to disentangle speaker information. From the clues provided in So-VITS, I am more inclined that this "HuBERT" model resembles more with **ContentVec**, which requires more steps to reduce the source speaker information. ContentVec is basically HuBERT, with a few speaker-invariant tweaks:
- During offline-clustering, before converting the audio into MFCCs, the audio is randomly converted into other speaker's voices;
- xxx
- xxx

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