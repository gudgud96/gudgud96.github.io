---
title: ISMIR 2020 - Part 1
date: 2020-10-17 09:10:42
tags:
    - Music Information Retrieval
---
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML' async></script>

TLDR: This blog will discuss:
1 - Various exciting papers (sorted according to topics) and research directions of ISMIR 2020
2 - My own conference experience of this year's ISMIR
<br/>

## 1 - Introduction

Finally, ISMIR 2020 is around the corner! The conference was originally designated to take place in Montreal, Canada (which was a dream place for me to visit, because a lot of major conferences like NeurIPS, ICLR, ICML were hosted there before). But sadly due to COVID-19, the conference is changed into a virtual event. It is also my first ISMIR, and I have been so looking forward to it since day 1 of doing MIR research, so indeed it is a little bit disappointed for unable to travel and meet people physically this year.

Nevertheless, I can already feel that ISMIR is such a unique conference as compared to others, although in a virtual setting, and I can almost understand why so many ISMIR visitors have repeatedly emphasized that ISMIR is the best conference of all. The topics are super interesting, and the community is super friendly, always willing to share & exchange, and extremely fun to talk to.

Below I try to summarize the papers & posters that I have personally visited, sorted by relevant topics. The list will by no means be exhaustive, and will be very related to my own focus & familiarity (I am more familiar / interested in controllable music generation, music representation learning, music audio synthesis, and some popular MIR tasks e.g. pitch estimation, voice conversion, source separation etc.).

## 2 - Controllable Symbolic Music Generation

[**Attributes-Aware Deep Music Transformation**](https://program.ismir2020.net/poster_5-06.html)

<figure>
  <img style="width:100%;" src="/img/ismir_attr.png" alt=""/>
</figure>

This work uses a very similar architecture like [Fader Networks](https://arxiv.org/pdf/1706.00409.pdf) in the computer vision domain - a conditional VAE, with an additional adversarial component to ensure latent \\(z\\) does not incorporate condition information. Evaluation on controllability is done on monophonic music. I tried the same architecture on polyphonic music in [Music FaderNets](https://program.ismir2020.net/poster_1-13.html), but I found that it does not produce optimal results in terms of linearity as compared to other latent regularization methods.
One interesting thing is that the authors do not compare results on linear correlation with [GLSR-VAE](https://arxiv.org/pdf/1707.04588.pdf), because they argued that GLSR-VAE is not designed to enforce linear correlation between latent values and attributes. I agree this to a certain extent, but to me linear correlation between both is still the most intuitive way to achieve controllability on low-level attributes, hence measuring that is still important in the context of controllable generation.

[**BebopNet: Deep Neural Models for Personalized Jazz Improvisations (Best Paper Award)**](https://program.ismir2020.net/poster_6-08.html)

<figure>
  <img style="width:100%;" src="/img/ismir_bebop.png" alt=""/>
</figure>

Congrats on this paper getting the best research award of this year! Compared to other similar works, this work focuses on **personalization**. Within the pipeline, other than the generation component, a dataset personal to the user is collected to train personal preference metrics, very much like an active learning strategy. As the music plays, the user adjusts a meter to display the level of satisfaction of the currently heard jazz solo. Then a regression model is trained to predict the user's taste. Finally, a beam serach is employed by using the criterion of score predicted the user preference regression model. The output of beam search should result in a music piece most adhered to the user preference. A very simple idea, but could be widely adoptable to all kinds of generation models to add in more degree of personalization.

[**Connective Fusion: Learning Transformational Joining of Sequences with Application to Melody Creation**](https://program.ismir2020.net/poster_1-05.html)

<figure>
  <img style="width:100%;" src="/img/ismir_conn.png" alt=""/>
</figure>

This work proposes **connective fusion**, which is a generation scheme by transforming between two given music sequences. The architecture is inspired by the [Latent Constraint](https://arxiv.org/pdf/1711.05772.pdf) paper - firstly, we pretrain a VAE to learn latent code \\(z\\) for a music sequence. Then, using a GAN-like actor-critic method, we learn a generator \\(G\\) that generates latent code pair \\((z^\prime_L, z^\prime_R)\\) that is indistuingishable from the input pair\\((z_L, z_R)\\). During training, we also add in an additional style vector \\(s\\), hence also learning a style space which controls how the two sequences are connectively fused.
I was fortunate enough to discuss with the author Taketo Akama about several issues of using VAE for music generation. In general, we found a significant tradeoff between attribute controllability and reconstruction (identity preservation), and training to generate longer sequence seems to really be a hassle. [His work last year](http://archives.ismir.net/ismir2019/paper/000100.pdf) has also helped me a lot with Music FaderNets, so huge kudos to him!

[**Generating Music with a Self-Correcting Non-Chronological Autoregressive Model**](https://program.ismir2020.net/poster_6-16.html)

<figure>
  <img style="width:100%;" src="/img/ismir_edit.png" alt=""/>
</figure>

I spotted this work previously during ML4MD and find it interesting because it suggests a very different approach towards music generation, which is using **edit distance**. The two key differences with common music generation idea is that (i) music composition can be non-chronological in nature, and (ii) the generation process should allow adding and removing notes. The input representaion used is pixel-like piano roll, so the approach inherits the problem of not distinguishing long sustains and continuous short onsets. Also, the evaluation is done with comparison against [orderless NADE](https://www.jmlr.org/papers/volume17/16-272/16-272.pdf) and [CoCoNet](https://arxiv.org/pdf/1903.07227.pdf), but with several recent works suggesting that richer vocabulary of event tokens can improve generation results, it might me interesting to see how this work compares or even adds value on top of these works.

[**PIANOTREE VAE: Structured Representation Learning for Polyphonic Music**](https://program.ismir2020.net/poster_3-06.html)

<figure>
  <img style="width:60%;" src="/img/ismir_pianotree.png" alt=""/>
</figure>

This work proposes a new hierarchical representation for polyphonic music. Commonly, polyphonic music is either represented by piano rolls (which is commonly treated like pixels), or MIDI event tokens. The authors suggest a **tree-like structure**, where each beat is a tree node, and the notes played on the same beat are the childrens of the node. They also propose a VAE model structure which has one-to-one correspondence with the data structure, and the evaluation shows that as compared to previous representations, PianoTree VAE is superior in terms of reconstruction and downstream music generation.
I definitely think that PianoTree has the potential to be the *de facto* representation of polyphonic music, because indeed it is more reasonable to understand polyphonic music in terms of hierachical structure, as compared to a flat sequence of tokens. However, I personally think that the common usage of PianoTree will depend on two key factor: **the ease of usage** (e.g. open source of encoder components and examples of usage), and whether **the data structure is tightly coupled with the proposed VAE model**. Event tokens are used widespread because any kind of sequence models / NLP models can be ported on top of that representation. Can PianoTree be ported easily to other kinds of architectures, and will the performance on all aspects remain the same? This is a crucial point for whether the structure will replace event tokens and be adopted widely in my opinion.

[**Learning Interpretable Representation for Controllable Polyphonic Music Generation**](https://program.ismir2020.net/poster_5-05.html)

<figure>
  <img style="width:60%;" src="/img/ismir_interpretable.png" alt=""/>
</figure>

This work is a demonstration of the power of PianoTree VAE above. This time, the authors explore the **disentanglement of chords and texture** of a music piece. The architecture adopts a similar idea as their prior work called [EC\\(^2\\)-VAE](http://archives.ismir.net/ismir2019/paper/000072.pdf) (which inspires Music FaderNets a huge lot as well!), where a chord encoder and texture encoder is used for latent representation learning, and a chord decoder with the PianoTree VAE decoder is used for reconstruction. They evaluated the results on three practical generation tasks: compositional style transfer, texture variation via sampling, and accompaniment arrangement. And, their demo and quality of generation is really superb, so it seems like PianoTree could really work well.
Meeting the NYU Shanghai team has also been a great experience, especially the discussions with Ziyu Wang has been really enjoyable. Huge kudos to them!

[**Music FaderNets: Controllable Music Generation Based on High-level Features via Low-level Feature Modelling (My Own Work)**](https://program.ismir2020.net/poster_1-13.html)

<figure>
  <img style="width:100%;" src="/img/ismir_fadernets.png" alt=""/>
</figure>

My work on controllable polyphonic music generation! At first I wanted to work on controllable geneeration based on emotion, but I found that representations of high-level musical qualities are not easy to learn with supervised learning techniques, either because of the **insufficiency of labels**, or the **subjectiveness** (and hence large variance) in human-annotated labels. We propose to use low-level features as "bridges" to between the music and the high level features. Hence, the model consists of:
-  **faders**, where each fader controls a low-level attribute of the music sample independently in a continuous manner. This relies on latent regularization and feature disentanglement
-  **presets**, which learn the relationship between the levels of the sliding knobs of low-level features, and the selected high-level feature. This relies on Gaussian Mixture VAEs which imposes hierachical dependencies.

This method combines the advantages of **rule-based methods** and **data-driven machine learning**. Rule-based systems are good at interpretability (i.e. you can explicitly hear that some factors are obviously changing during generation), but it is not robust to all situations; whereas machine learning methods are the total opposite. Another interesting point is the usage of **semi-supervised learning**. Since we know that arousal labels are noisy, we can choose only the quality ones with lesser variance and higher representability for training. In this work we prove that lesser labels can be a good thing - using the semi-supervised setting of GM-VAE to train, with only 1% of labelled arousal data, we can learn well-separated, discriminative mixtures. This can provide a feasible approach to learn representations of other kinds of abstract high-level features.

[**Music SketchNet: Controllable Music Generation via Factorized Representations of Pitch and Rhythm**](https://program.ismir2020.net/poster_1-09.html)

<figure>
  <img style="width:80%;" src="/img/ismir_sketchnet.png" alt=""/>
</figure>

This work explores the application of music inpainting - given partial musical ideas (i.e. music segments), the model is able to "fill up the blanks" with sequences of similar style. An additional controllable factor is provided in this model on pitch and rhythm (pretty much inspired by [EC\\(^2\\)-VAE](http://archives.ismir.net/ismir2019/paper/000072.pdf) as well). There are 3 separate components: **SketchVAE** for latent representation learning, **SketchInpainter** for predicting missing measures based on previous and future contexts, and **SketchConnector** which finalizes the generation by simulating user controls with random unmasking (a common technique in training language generators).

[**The Jazz Transformer on the Front Line: Exploring the Shortcomings of AI-composed Music through Quantitative Measures**](https://program.ismir2020.net/poster_1-17.html)

<figure>
  <img style="width:80%;" src="/img/ismir_jazz.png" alt=""/>
</figure>

This is a really interesting work that tries to answer a lot of pressing questions related to Transformer-based music generation. Are Transformers really that good? If not, what are the culprits? Does structure-related labels help generation?

For me the real key contributions for this work are the findings concluded on the proposed objective metrics used to evaluate the generated music. There are so many objective metrics being proposed (I recall [this work](https://arxiv.org/pdf/1912.05537.pdf) suggesting several metrics for Transformer AE as well), but for Transformers which are often crowned for more structured generation, how do we evaluate structureness other than subjective tests? I find the idea of using [fitness scape plot](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4S3_ScapePlot.html) to quantify structureness super interesting. Although the field will never agree on a set of evaluation metrics, but understanding where Transformers are still short of in overall will definitely drive the community to pinpoint on certain areas to improve.

## 3 - Disentangled Representation Learning

[**Unsupervised Disentanglement of Pitch and Timbre for Isolated Musical Instrument Sounds**](https://program.ismir2020.net/poster_5-10.html)

<figure>
  <img style="width:80%;" src="/img/ismir_jyun.png" alt=""/>
</figure>

Work by my senpais, Yin-Jyun Luo and and Raven Cheuk, so definitely hands down! Jyun worked on [pitch-timbre disentanglement](https://arxiv.org/pdf/1906.08152.pdf) before, and in this work he decided to push it further - can we do such disentanglement in an unsupervised manner? 

This work employs a key idea: **moderate pitch shiftings will not change timbre**. Hence, even if we don't have any labels annotated on pitch and timbre, we can still achieve disentanglement by [contrastive learning paradigms](https://paperswithcode.com/task/contrastive-learning) - data augmentation by transposing the pitch, but enforce relations in \\(z_\textrm{pitch}\\) and \\(z_\textrm{timbre}\\). The authors propose 4 losses: regression loss, [contrastive loss](https://arxiv.org/pdf/2002.05709.pdf), [cycle consistency loss](https://arxiv.org/pdf/1703.10593v7.pdf) and a new **surrogate label loss**. I personally think the power of this framework is not just for disentangling timbre and pitch, but unsupervised representation learning as a whole. Can this unsupervised framework be applied on other harder problems (e.g. music sequences, and disentangling musical factors)? How would data augmentation happen in different problems, and would that affect the formulation of losses? These will be interesting questions that require much creativity to explore.

[**Metric learning VS classification for disentangled music representation learning**](https://program.ismir2020.net/poster_3-15.html)

<figure>
  <img style="width:105%;" src="/img/ismir_metric.png" alt=""/>
</figure>

This interesting work connects 3 things together: metric learning (learns similarity between examples), classification, and disentangled representation learning (which corresponds to [this work](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/lee_disentangledmusicsim_icassp2020.pdf)). Firstly, the authors connect classication and metric learning with **proxy-based metric learning**. Then, with all combinations of models and their disentangled version, evaluation is done on 4 types of tasks: training time, similarity retrieval, auto-tagging, and triplet-prediction. Results show that classification-based models are
generally advantageous for training time, similarity retrieval, and auto-tagging, while deep metric learning exhibits better performance for triplet-prediction. Disentanglement slightly improves the result on most settings.

[**dMelodies: A Music Dataset for Disentanglement Learning**](https://program.ismir2020.net/poster_1-15.html)

<figure>
  <img style="width:100%;" src="/img/ismir_dmel.png" alt=""/>
</figure>

This work proposes a new dataset which resembles [dSprites](https://github.com/deepmind/dsprites-dataset) in the computer vision domain, which is designed for learning and **evaluating disentangled representation learning algorithms for music**. The authors also ran benchmark experiments using common disentanglement methods (\\(\beta\\)-VAE, Annealed-VAE and Factor-VAE). Overall, the results suggest that disentanglement is comparable, but reconstruction accuracy is much worse, and the sensitivity on hyperparameters are much higher. This again proves the tradeoff between reconstruction and disentanglement / controllability using VAEs on music data.
I discussed with the author Ashis Pati on why not use real-world monophonic music dataset (e.g. [Nottingham dataset](https://ifdo.ca/~seymour/nottingham/nottingham.html)) with attribute annotations, but generating synthetic data instead. He suggests that it is to preserve the orthogonality and balanced composition of each attribute within the dataset. It seems like the balance between orthogonality and resemblance to real music is a lot more delicate that expected when creating a dataset like this. (Meanwhile, Ashis' work has been very crucial to Music FaderNets, and it is such a joy to finally meet him and chat in person. One of the coolest moment during the conference!)

## 4 - Singing Voice Conversion

[**Zero-Shot Singing Voice Conversion**](https://program.ismir2020.net/poster_1-08.html)

<figure>
  <img style="width:80%;" src="/img/ismir_singing.png" alt=""/>
</figure>

The most interesting part of this work is the **zero-shot** part, which largely incorporates ideas from the speech domain. Speaker embedding networks were found to be successful for enabling zero-shot voice conversion of speech, whereby the system can model and adapt to new unseen voices on the fly. The authors adopted the same idea for singing voice conversion by using a [pretrained speaker embedding network](https://github.com/CorentinJ/Real-Time-Voice-Cloning), and then using the WORLD vocoder with learnable parameters for synthesis. It seems like the "pre-trained fine-tune" idea from other domains has influenced much works in MIR, moreover this work shows that using relevant foreign-domain embeddings (speech) on music tasks (singing voice) can actually work.

## 5 - Audio Synthesis

[**DrumGAN: Synthesis of Drum Sounds with Timbral Feature Conditioning Using Generative Adversarial Networks**](https://program.ismir2020.net/poster_4-16.html)

<figure>
  <img style="width:60%;" src="/img/ismir_drumgan.png" alt=""/>
</figure>

Super cool and useful work (can't wait to use the plugin as a producer)! This work uses a **progressive growing GAN** (similar to the idea in [GANSynth]()) to synthesize different types of drum sounds. Moreover, to achieve user controllability, the model allows several factors to be changed during input time, including  brightness, boominess, hardness etc. to synthesize different kinds of drum sounds. To evaluate controllability, unlike using Spearman / Pearson correlation or [R-score in linear regressor](http://proceedings.mlr.press/v80/adel18a/adel18a.pdf), which are more popular in the music generation domain, this work evaluates against several other baseline scores as proposed in [a previous work using U-Net architecture](https://arxiv.org/pdf/1911.11853.pdf). This could probably shed light to a new spectrum of measurements in terms of factor controllability.

Another interesting thing is that this work uses **complex STFT spectrogram** as the audio representation. When I worked on piano audio synthesis, the common representation used is the magnitude Mel-spectrogram, which is why for the output a vocoder (e.g. WaveNet, WaveGAN, WaveGlow) is needed to invert Mel-spectrograms to audio. But in this work, the output directly reconstructs the real and imaginary parts of the spectrogram, and to reconstruct the audio we only need to do an inverse STFT. This can ensure better audio reconstruction quality, and phase information might also help audio representation learning.

*The remaining topics (source separation, transcription, model pruning and cover song detection) will be covered in [Part 2](/2020/10/17/ismir_2020_pt2/)*.