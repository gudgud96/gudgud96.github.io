---
title: Understanding Neural Vocoders
date: 2025-05-31 22:53:51
tags:
    - Music Signal Processing
    - Deep Learning
---
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML' async></script>

TLDR: This blog will discuss:
1 - Technical concepts in neural vocoders
2 - DDSP, NSF, GAN variants

<div style="outline: 2px #100100 round; border-radius: 25px;
    background: #eafaff;
    padding-left: 30px;
    padding-right: 30px;
    padding-top: 10px;
    padding-bottom: 20px;
    width: 40%">
    <h3>Table of Contents</h3>
    <a href="#">1 - Introduction</a><br/>
    <a href="#">2 - Architecture</a><br/>
    <a href="#">3 - Deep Dive</a><br/>
    <a href="#">&emsp;&emsp;&emsp;3.1 - Content Feature Extraction</a><br/>
    <a href="#">&emsp;&emsp;&emsp;3.2 - Pitch Extraction</a><br/>
    <a href="#">&emsp;&emsp;&emsp;3.3 - Acoustic Model (VITS)</a><br/>
    <a href="#">&emsp;&emsp;&emsp;3.4 - Retrieval Module</a><br/>
    <a href="#">4 - Inference</a><br/>
    <a href="#">5 - Related Work</a><br/>
</div>
