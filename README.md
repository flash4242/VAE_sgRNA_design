# My bachelor's thesis: sgRNA Design for CRISPR/Cas9 Gene Editing Technology with Deep Learning

Deep learning is a subset of machine learning within the field of artificial intelligence. It
offers a powerful and efficient approach for analyzing and modeling vast amounts of data,
using neural networks and advanced computational capabilities. This methodology excels
in tasks requiring complex pattern recognition, particularly when working with large-scale
datasets.

CRISPR/Cas9 is a cutting-edge genome editing technology that has gained significant
attention in medical research due to its versatility and ease of use compared to other
DNA editing techniques. It holds promise across various fields, including agriculture
and healthcare. A key element of this system is the single guide RNA (sgRNA), which
directs the Cas proteins to specific DNA sequences. These proteins then precisely cut
the double-stranded DNA, enabling targeted gene modifications. A vital task for the
successful application of CRISPR systems is to design sgRNAs that maximize on-target
efficiency while minimizing off-target effects. Achieving this balance is essential for reliable
gene editing.

The goal of this work is to explore the possibilities of introducing deep neural networks
to enhance sgRNA generation. The focus is on three neural networks: one for on-target
efficiency prediction, one for off-target profile prediction and one for generating sgRNAs.
I evaluate the generated sgRNAs using the two predictor networks. I design a custom
loss function for the generator network to shift the generated sgRNA distribution to more
efficient ones. With this approach, I aim to contribute to the practical application of
CRISPR/Cas9 technology in various fields.
