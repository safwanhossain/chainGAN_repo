# ChainGAN
Code for the https://arxiv.org/pdf/1811.08081.pdf

Abstract:
We propose a new architecture and training methodology for generative adversar-ial networks. Current approaches attempt to learn the transformation from a noisesample to a generated data sample in one shot.  Our proposed generator architec-ture, calledChainGAN, uses a two-step process.  It first attempts to transform anoise vector into a crude sample, similar to a traditional generator.  Next, a chainof networks, callededitors, attempt to sequentially enhance this sample. We traineach of these units independently, instead of with end-to-end backpropagation onthe entire chain.  Our model is robust, efficient, and flexible as we can apply it tovarious network architectures.  We provide rationale for our choices and experi-mentally evaluate our model, achieving competitive results on several datasets.
