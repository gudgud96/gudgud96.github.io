---
title: ISMIR 2020 - Part 2
date: 2020-10-17 09:10:42
tags:
    - Music Information Retrieval
---
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML' async></script>

## 6 - Music Source Separation

[**Investigating U-Nets with various Intermediate Blocks for Spectrogram-based Singing Voice Separation**](https://program.ismir2020.net/poster_2-04.html)

<figure>
  <img style="width:70%;" src="/img/ismir_unets.png" alt=""/>
</figure>

U-Nets are very common in singing voice separation, with their prior success in image segmentation. This work further inspects the usage of various intermediate blocks by providing comparison and evaluations. 2 types of intermediate blocks are used, **Time-Distributed Blocks** which does not have inter-frame operations, and **Time-Frequency Blocks** which considers both time and frequency domain. The variants of each block are inspected (fully connected, CNN, RNN etc.). The [demo](https://www.youtube.com/watch?v=DuOvWpckoVE&feature=youtu.be&ab_channel=KU-Intelligence-Engineering-Lab) provided by this work is really superb - the best configuration found in this work yields a very clean singing voice separation.

[**Content based singing voice source separation via strong conditioning using aligned phonemes**](https://program.ismir2020.net/poster_6-07.html)

<figure>
  <img style="width:70%;" src="/img/ismir_phoneme1.png" alt=""/>
</figure>

This work explores **informed source separation** - utilizing prior knowledge about the mixture and target source. In this work, the conditioning information used is lyrics, which are further aligned in the granularity of phonemes. This work uses the [FiLM](https://arxiv.org/pdf/1709.07871.pdf) layer for conditioning, which the conditioning input is a 2D matrix of phonemes w.r.t. time. For weak conditioning, the same FiLM operation to the whole input patch; for strong conditioning, different FiLM operations are computed at different time frames.

[**Exploring Aligned Lyrics-informed Singing Voice Separation**](https://program.ismir2020.net/poster_5-08.html)

<figure>
  <img style="width:70%;" src="/img/ismir_phoneme2.png" alt=""/>
</figure>

Similar to the above work, this work also utilizes aligned lyrics / phonemes for improving singing voice separation. The architecture is different - this work takes the backbone from the state-of-the-art [Open Unmix](https://sigsep.github.io/open-unmix/) model, then the authors propose to use an additional **lyric encoder** to learn embeddings for conditioning on the backbone. This idea resembles much with the idea from [text-to-speech](https://paperswithcode.com/task/text-to-speech-synthesis) models, where the text information is encoded to condition on the speech synthesis component.

[**Multitask Learning for Instrument Activation Aware Music Source Separation**](https://program.ismir2020.net/poster_5-16.html)

<figure>
  <img style="width:50%;" src="/img/ismir_multitask.png" alt=""/>
</figure>

This work leverages multitask learning for source separation. Multitask learning states that by choosing a relevant subsidiary task, and allow it to train in line with the original task, can improve the performance of the original task. This work chooses to use **instrument activation detection** as the subsidary task, because it can intuitively suppress wrongly predicted activation by the source separation model at the supposed silent segments. By training on a larger dataset with multitask learning, the model can perform better on almost all aspects as compared to Open Unmix.

## 7 - Music Transcription / Pitch Estimation

[**Multiple F0 Estimation in Vocal Ensembles using Convolutional Neural Networks**](https://program.ismir2020.net/poster_2-18.html)

<figure>
  <img style="width:65%;" src="/img/ismir_vocal.png" alt=""/>
</figure>

This work is a direct adaptation of CNNs on F0 estimation, applying on vocal ensembles. The key takeaways for me in this work is of 3-fold: (i) **phase information does help** for F0 estimation tasks (would it also be the same for other tasks? this will be interesting to explore); (ii) deeper models will work better; (iii) late concatenation of magnitude and phase information works better than early concatenation of both.

[**Multi-Instrument Music Transcription Based on Deep Spherical Clustering of Spectrograms and Pitchgrams**](https://program.ismir2020.net/poster_3-01.html)

<figure>
  <img style="width:90%;" src="/img/ismir_spherical.png" alt=""/>
</figure>

This is a super interesting work! For previous music transcription works, the output will be of a pre-defined set of instruments, with activation predicted for each instrument. This work intends to transcribe arbitrary instruments, hence being able to transcribe undefined instruments that are not included in the training data. The key idea is also inspired by methods from the speech domain, where **deep clustering** separates a speech mixture to an arbitrary number of speakers based on the characteristics of voices. Hence, the spectrograms and pitchgrams (estimated by an [existing multi-pitch estimator](https://brianmcfee.net/papers/ismir2017_salience.pdf)) provide complementary information for timbre-based clustering and part separation.

[**Polyphonic Piano Transcription Using Autoregressive Multi-state Note Model**](https://program.ismir2020.net/poster_3-17.html)

<figure>
  <img style="width:70%;" src="/img/ismir_transcription.png" alt=""/>
</figure>

This work recognizes the problem of frame-level transcription: some frames might start after the onset events, which makes it harder to distinguish and transcribe. To solve this, the authors use an **autoregressive model** by utilizing the time-frequency and predicted transcription of the previous frame, and feeding them during the training of current step. Training of the autoregressive model is done via teacher-forcing. Results show that the model provides significantly higher accuracy on both note onset and offset estimation compared to its non-auto-regressive version. And just one thing to add: their [demo](https://program.ismir2020.net/lbd_444.html) is super excellent, such sleek and smooth visualization on real-time music transcription!

## 8 - Model Pruning

[**Ultra-light deep MIR by trimming lottery ticket**](https://program.ismir2020.net/poster_4-11.html)

<figure>
  <img style="width:70%;" src="/img/ismir_lottery.png" alt=""/>
</figure>

The [lottery ticket hypothesis](https://arxiv.org/pdf/1803.03635.pdf) paper is the best paper in ICLR 2020, which motivates me to looking into this interesting work. Also, model compression is a really useful technique in an industrial setting as it significantly reduces memory footprint when scaling up to large-scale applications. With the new proposed approach by the authors known as **structured trimming**, which remove units based on magnitude, activation and normalization-based criteria, model size can be even more lighter without trading off much in terms of accuracy. The cool thing of this paper is that it evaluates the trimmed model on various popular MIR tasks, and these efficient trimmed subnetworks, removing up to 85% of the weights in deep models, could be found.

## 9 - Cover Song Detection

[**Combining musical features for cover detection**](https://program.ismir2020.net/poster_2-15.html)

<figure>
  <img style="width:100%;" src="/img/ismir_doras.png" alt=""/>
</figure>

In previous cover song detection works, either the harmonic-related representation (e.g. [HPCP](https://www.upf.edu/web/mtg/hpcp), [cremaPCP](https://brianmcfee.net/papers/ismir2017_chord.pdf)) or the melody-related representation (e.g. [dominant melody](https://arxiv.org/pdf/1907.01824.pdf), [multi-pitch](https://arxiv.org/pdf/1910.09862.pdf)) is used. This work simply puts both together, and explores various fusion methods to inspect its improvement. The key intuition is that some cover songs are similar in harmonic content but not in dominant melody, and some are of the opposite. The interesting finding is that with only a simple average aggregation of \\(d_\textrm{melody}\\) and \\(d_\textrm{cremaPCP}\\), the model is able to yield the best improvement over individual models, and (strangely) it performs even better than a more sophisticated late fusion model.

[**Less is more: Faster and better music version identification with embedding distillation**](https://program.ismir2020.net/poster_6-15.html)

<figure>
  <img style="width:100%;" src="/img/ismir_furkan.png" alt=""/>
</figure>

In [a previous work](https://arxiv.org/pdf/1910.12551.pdf), the authors proposed a musically-motivated embedding learning model for cover song detection, but the required embedding size is pretty huge at around 16,000. In this work, the authors experimented with various methods to reduce the amount of dimension in the embedding for large-scale retrieval applications. The results show that with a **latent space reconfiguration** method, which is very similar to transfer learning methods by fine-tuning additional dense layers on a pre-trained model, coupling with a normalized softmax loss, the model can achieve the best performance even under an embedding size of 256. Strangely, this performs better than training the whole network + dense layers from scratch.


## 10 - Last Words on ISMIR 2020

That's all for my ISMIR 2020! I think the most magical moment for me would be when I could finally chat with some of the authors (and some are really big names!) of the works that I really like throughout my journey of MIR, and furthermore being able to exchange opinions with them. Just hope to be able to meet all of them physically some day!