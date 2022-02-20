---
title: MIR Papers 2021 (and ISMIR)
date: 2022-02-19 23:47:54
tags:
 - Music Information Retrieval
---
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML' async></script>

## 1 - Introduction

It's been almost a year since my last post, and of course a lot has changed. I am no longer working on cover song detection anymore, nor am I training models to solve MIR problems at the moment. I still do, like [trying something for MediaEval](https://github.com/gudgud96/noisy-student-emotion-training) or [reproducing papers I like](https://github.com/gudgud96/diff-wave-synth), but mostly after work which is, you know, not a lot of time left.

Which is also why I haven't been writing for long, because writing is slow and given the time available, I rather spend more time coding to maximize output (or procrastinate to minimize it...). But inspired by my recent co-worker, I still find blogging necessary to record and clear your mind, and most importantly, ISMIR 2021 is like N months ago and the idea to write a post about it was always dangling (which I really should have finished this N months ago...)

So here it comes!

## 2 - Differentiable Digital Signal Processing

I think 2021 is very much a year on DDSP and its applications, and it has also been my recent interest. Just to name a few works that I follow slightly closer:

[**DDSP: Differentiable Digital Signal Processing**](https://arxiv.org/abs/2001.04643)

The idea is neat - using WaveNet to synthesize audio is costing way too many parameters, so maybe we need to instill some domain knowledge as inductive bias. Perhaps we can make use of **synthesizers** (let's take additive synthesizers as an example). To learn the parameters of a synth, the synth itself should be "differentiable" so that gradients can be back-propagated. Hence each DSP component within a synth (F0, loudness, filter design, reverb, etc.) can be implemented as differentiable functions, to allow the learning of parameters via optimizing a (multi-level) spectrogram loss. The parameters needed is simply **10x lesser** to achieve high fidelity synthesis.

[**Differentiable Signal Processing with Black-Box Audio Effects**](https://arxiv.org/abs/2105.04752)

A neat idea with direct application usage. Given that third party audio FX plugins are non-differentiable, one most obvious solution is to bring in gradient approximation solutions (if you don't want to re-implement the differentiable version of the entire thing). This paper uses a stochastic gradient approximation method called [*simultaneous permutation stochastic approximation*](https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation) for the gradient from black box -> deep encoder. Various applications can span, e.g. automatic mixing / mastering / EQ etc.

*Side note*: the authors mention about the training process - it needs to be parallelized to a single-GPU, multi-CPU setup, so that model training runs on GPU but FX plugins run on CPU. This is pretty much the same training setup as in reinforcement learning (GPU for model, CPU for simulation). I am recently reading about [Ray](https://github.com/ray-project/ray), which is precisely designed for this kind of setup, and I think it might be a suitable use case.

[**Automatic Multitrack Mixing with A Differentiable Mixing Console of Neural Audio Effects**](https://arxiv.org/abs/2010.10291)

Also on auto-mixing, this paper basically trains a (TCN) model to learn the parameters on the mixing console, which also needs the components on the mixing console (EQ, Compressor, Reverb) to be differentiable. Also, a stereo loss function is introduced because spatialization is important for mixing.

[**Synthesizer Soundmatching with Differentiable DSP**](https://archives.ismir.net/ismir2021/paper/000053.pdf)

This work focuses more on estimating synth parameters (presets). It needs an estimator network (for the params) and a differentiable (additive/subtractive) synth. Two mode of trainings can be supported - either *supervised* with synth parameter labels,  or *unsupervised* by reproducing input waveform.

[**Differentiable Wavetable Synthesis**](https://arxiv.org/abs/2111.10003#:~:text=Differentiable%20Wavetable%20Synthesis%20(DWTS)%20is,end%2Dto%2Dend%20training.)

Simply implementing the similar idea of DDSP on wavetable synthesis. Since I am a Serum user, wavetable synthesis have always been my favourite, so I tried to [reproduce this paper](https://github.com/gudgud96/diff-wave-synth) as well, with my own implementation of wavetable synth in PyTorch.

[**MIDI-DDSP**](https://arxiv.org/abs/2112.09312)

I worked in this direction a short while before (on piano synthesis, [short paper](https://github.com/gudgud96/piano-synthesis) here), and I only managed to model dynamics and articulation, and I have to use WaveGlow for synthesis which doesn't perform very well at that time, not to mention the amount of time it takes to train WaveGlow. This work extends to model *timbre* and *vibrato*, as well as using DDSP for the synthesis layer. Much more extensive and optimized work.


## 3 - Sound Synthesis

[**RAVE: A variational autoencoder for fast and high-quality neural audio synthesis**](https://arxiv.org/abs/2111.05011)

A VAE + adversarial fine tuning method for audio synthesis, main advantage is that it can synthesize on CPU + **20x** faster than before. VAE + adversarial fine tuning is a pretty common training technique to improve audio synthesis quality, one paper that I read before is [Latent Constraints](https://arxiv.org/abs/1711.05772). 

[**Neural Waveshaping Synthesis**](https://arxiv.org/abs/2107.05050)

One of my favourite papers in ISMIR. To achieve efficient CPU synthesis, this work proposes *Neural Waveshaping Unit* (and a fast version) to produce complex timbre evolution. Combining with a differentiable noise synthesizer, only 260k model parameters are needed. I probably will write a bit more to delve into this work further in future.

[**Towards Lightweight Controllable Audio Synthesis with Conditional Implicit Neural Representations**](https://arxiv.org/abs/2111.08462)

A starting work on using implicit neural representations for audio synthesis. The advantage is that INR is itself a compact representation, hence it can learn faster and generally produce quantitatively better audio reconstructions with equal parameter counts (I think there's much more value in this direction and I should probably delve deeper)

## 4 - General Audio Representations

[**Towards Learning Universal Audio Representations**](https://arxiv.org/abs/2111.12124)

A very ambitious work to provide universal audio representation for various downstream tasks. The architecture used is a [*Slow-Fast*](https://arxiv.org/abs/1812.03982) [*Normalizer-Free*](https://arxiv.org/abs/2102.06171) *F0 Net* given observations in previous work, and transfers well among a wide range of audio-reated tasks.

[**MuseBERT: Pre-training Music Representation for Music Understanding and Controllable Generation**](https://archives.ismir.net/ismir2021/paper/000090.pdf)

Convert symbolic music into note-level token representation, and apply BERT-like pretraining. The model is capable of texture generation, chord analysis, accompaniment refinement.

[**Contrastive Learning of Musical Representations**](https://arxiv.org/abs/2103.09410)

[SimCLR](https://arxiv.org/abs/2002.05709) on raw music audio, hence the augmentations are specific to music (which the author provides a [repo](https://github.com/Spijkervet/torchaudio-augmentations) on it). This self-supervised method can work well on MTAT and MSD tagging.


## 5 - On Music Tagging

[**Semi-supervised Music Tagging Transformer**](https://arxiv.org/abs/2111.13457)

Probably one of the hottest paper during ISMIR. The idea is pretty simple - which is to apply semi-supervised learning on music tagging, and it is super reasonable because tagged music data is super scarce. I tried a [similar idea](https://github.com/gudgud96/noisy-student-emotion-training) during the same time for MediaEval, hence working with emotion data, and noisy student training does not give very conclusive results in the end.

## 6 - Some music open source projects

The most familiar one I know would still be [nnAudio](https://github.com/KinWaiCheuk/nnAudio) by Kin Wai Cheuk, it's a cool PyTorch audio processing library with sweet CQT implementations. I helped a little on Griffin-Lim and VQT, and I probably would try inverse-CQT next.

Recently I am also looking into [pedalboard](https://github.com/spotify/pedalboard) by Spotify, which is quite a cool project on chaining audio effects - it's simple and it just works. I also find [DawDreamer](https://github.com/DBraun/DawDreamer) an interesting one, basically it's a DAW on Python. The backend still uses JUCE but it provides a Python user interface.





