
**Questions**
*General*
1. What is the key difference between the maximum likelihood estimation used in AR and flow-models versus that used in Variational Inference (VI) and diffusion models?

2. Why is estimating p and q for computing divergence a challenging task in the context of Probability Divergence?

3. In the context of Integral Probability Metrics (IPM), how does a non-smooth function f(x) impact the calculated IPM?

4. What is the role of the discriminator function in the standard GAN loss, and why are its values significant for real and generated samples?

5. What is the goal of adversarial training in GAN Training?

6. How does sampling in GAN models work and what does it involve?

7. What is the relationship between the GAN loss and JS-Divergence when the optimal solution to GAN is achieved?

8. Why do existing divergences fail when Q and P have different supports, and how can a smoothly varying discriminator function address this issue?

9. How does the Wasserstein GAN (WGAN) ensure smoothness in the discriminator function?

10. What is the role of Wasserstein distance or Earth Mover's Distance in the context of the WGAN algorithm?

11. How do BigGAN and StyleGAN, as notable GAN variants, effectively disentangle between layers?

*Losses*
1. What is Jensen-Shannon Divergence and how does it relate to the training of Generative Adversarial Networks (GANs)?

2. How does the Kullback-Leibler Divergence measure the difference between two probability distributions and what are its limitations?

3. What is the main characteristic of the Total Variation Distance and how does it compare probability distributions?

4. Explain how the Wasserstein Distance or Earth Mover's Distance is calculated and why it is beneficial for training GANs?

5. What are Integral Probability Metrics and how do they differ from other metrics in measuring the difference between distributions?

6. In what circumstances would one prefer to use Wasserstein Distance over the Kullback-Leibler Divergence in comparing two distributions?

7. How might the choice of different probability metrics affect the training and performance of generative models?

8. Why is it important to consider both near and far samples when using Integral Probability Metrics to compare distributions?

**Metrics**

In the context of comparing probability distributions, especially for generative models, several common metrics or divergences are used,
including Jensen-Shannon (JS) Divergence,
Kullback-Leibler (KL) Divergence,
Total Variation (TV) Distance,
and Wasserstein (or Earth Mover's) Distance. 

Let's discuss some of their key characteristics:

1. **Jensen-Shannon (JS) Divergence**: This is a symmetrized and smoothed version of the KL divergence.
    It is always finite and has the value of 0 if and only if the two distributions being compared are the same.
    It's commonly used in training GANs, but its main drawback is that it doesn't provide a gradient everywhere, which can cause issues with model training.

4. **Kullback-Leibler (KL) Divergence**: Also known as relative entropy, KL divergence measures how one probability distribution diverges from a second,
    expected probability distribution. However, it's not symmetric, meaning KL(P||Q) is not equal to KL(Q||P).
    It can be infinite if there are points where Q(x) is zero and P(x) is non-zero.
    In the context of GANs, this can contribute to the vanishing gradients problem.

5. **Total Variation (TV) Distance**: The TV distance is a measure of the difference between two probability distributions.
    It is symmetric and lies between 0 and 1. However, it only considers the maximum difference in probabilities between outcomes,
    so it can be insensitive to changes in the distribution that don't affect the maximum difference.

6. **Wasserstein (Earth Mover's) Distance**: Also known as the first Wasserstein metric or Earth Mover's distance,
    this distance measure considers the total difference between two distributions in terms of the minimum cost that must be
    paid to transform one distribution into the other. It's more robust to changes in the location of the distribution and provides
    meaningful gradients almost everywhere, making it particularly useful for training GANs.
    The downside is that it can be more computationally intensive to compute.

7. **Integral Probability Metrics (IPM)**: IPMs are another class of metrics used to compare distributions. They have a representation in terms of a supremum over a class of functions. Examples include the Wasserstein distance, Energy distance, and Maximum Mean Discrepancy (MMD). IPMs can capture differences between distributions in a more comprehensive manner, taking both near and far samples into account.

The choice of the appropriate probability metric depends on the specific context, including the characteristics of the data, the generative model, and the computational resources available.




- **Adversarial Models or Integral Probability Measures:**
  - So far, models have used maximum likelihood estimation.
  - AR and flow-models provide precise likelihood estimation, but their scaling behavior is worse.
  - Variational Inference (VI) and diffusion models estimate likelihoods approximately.
  - Considering a new type of model that doesn't require implicit likelihood estimation.

- **Two-Sample Test:**
  - A statistical test to identify differences between two distributions, p(x) and q(y).

- **Probability Divergence:**
  - A method to measure the difference between two distributions, with various known divergences like KL, JS, etc.
  - Estimating p and q for computing divergence is challenging.

- **Integral Probability Metrics (IPM):**
  - A method to compute the distance between two samples without estimating p,q.
  - The difference is large when P,Q are different and small when near.
  - Non-smooth f(x) could lead to IPM having arbitrarily small values even for very different P,Q.

- **Generative Adversarial Models (GAN):**
  - Measures the similarity between the distribution of generated samples Q and true samples P.
  - The discriminator function f is used in the standard GAN loss, it should have high values for real samples and low values for generated samples.

- **GAN Training:**
  - The adversarial training involves training f to discriminate against G and training G to fool f.
  - The goal is to optimize G to minimize the IPM.

- **Sampling Using GAN Models:**
  - Sampling involves generating random noise vector from a Gaussian distribution and mapping it to an image using generator G.

- **Optimal Solution to GAN & JS-Divergence:**
  - The GAN loss equates to JS-Divergence when the optimal solution is achieved.
  - f-GAN provides a lower bound for the f-Divergence using IPM.

- **Probability Distance Measures & Divergences:**
  - Existing divergences fail when Q and P have different supports and don't vary smoothly, indicating the need for a more expressive measure.
  - A solution is to enforce a smoothly varying discriminator function.

- **Wasserstein GAN (WGAN):**
  - The WGAN ensures smoothness in the discriminator function through techniques like weight clipping, gradient penalty, and spectral normalization.
  - WGAN measures the Wasserstein distance or Earth Mover's Distance for optimal matching between samples.

- **GAN Variants:**
  - BigGAN and StyleGAN are notable GAN variants with effective disentanglement between layers.




**Answers**

*General*

1. In AR and flow-models, maximum likelihood estimation is used for precise likelihood estimation. They provide a detailed probabilistic description of the data but often have poor scaling behavior. Variational Inference (VI) and diffusion models, on the other hand, use maximum likelihood estimation to approximate the likelihoods. They are generally more scalable but can have certain approximations or assumptions, which may result in a less precise estimation.

2. Estimating p and q for computing divergence is challenging because estimating the underlying distribution from samples is a complex task. It often requires making certain assumptions about the form of the distribution or using non-parametric estimation techniques, which can be computationally intensive. Moreover, these estimates may still not be accurate, especially in high-dimensional settings.

3. In the context of Integral Probability Metrics (IPM), if the function f(x) is not smooth, the calculated IPM can end up having arbitrarily small values. This can happen even when the two distributions P and Q being compared are significantly different. This happens because non-smooth functions can fail to capture the differences between distributions accurately.

4. In standard GAN loss, the discriminator function's role is to differentiate between real and generated samples. It should assign high values to real samples and low values to generated samples. The aim is to make the discriminator as good as possible at distinguishing real from fake, while the generator tries to fool the discriminator.

5. The goal of adversarial training in GAN Training is to optimize the generator G to minimize the Integral Probability Metric (IPM). This involves training the discriminator to maximally discriminate against the generator and simultaneously training the generator to maximally fool the discriminator.

6. Sampling in GAN models involves generating a random noise vector from a Gaussian distribution and then mapping this noise to an image using the generator function G. This process generates new, 'fake' samples that the GAN attempts to make as close as possible to the real data distribution.

7. The optimal solution to the GAN problem corresponds to the point where the GAN loss is equal to the Jensen-Shannon (JS) divergence. This divergence is a measure of the difference between the true data distribution and the generated data distribution. At the optimal point, the generated data is indistinguishable from the real data according to the JS divergence.

8. Existing divergences such as KL or JS divergence fail when Q and P have different supports because they can give infinite or undefined values. The issue of not varying smoothly comes from these divergences not providing useful gradients for learning everywhere in the space. A smoothly varying discriminator function addresses this by being Lipschitz-1, which ensures a useful gradient at all points.

9. Wasserstein GAN (WGAN) ensures smoothness in the discriminator function by introducing constraints on the function, ensuring that it is 1-Lipschitz. Techniques such as weight clipping, gradient penalty, and spectral normalization are used to enforce this property.

10. In the context of the WGAN algorithm, the Wasserstein distance or Earth Mover's Distance provides a measure of the cost of transporting mass to convert one distribution into another. It helps in providing optimal matching between the samples of two distributions, which is useful for evaluating and optimizing the performance of the GAN.

11. GAN variants such as BigGAN and StyleGAN effectively disentangle between layers by implementing specific architectural choices. In StyleGAN, for instance, the styles (long-term factors of variation like identity in a face) are injected into all layers of the generator, enabling control over both coarse and fine details. The noise, on the other hand, is added only at the output level of each layer, allowing for instance-level variation, resulting in effective disentanglement.

*Losses*
1. **Jensen-Shannon (JS) Divergence** is a symmetrized and smoothed version of the KL divergence. It measures the similarity between two probability distributions and is widely used in the training of Generative Adversarial Networks (GANs). However, it doesn't provide a gradient everywhere, which can lead to challenges in training the model.

2. The **Kullback-Leibler (KL) Divergence** measures how one probability distribution diverges from a second, expected probability distribution. It quantifies the amount of information lost when the expected distribution is used to approximate the actual one. However, KL divergence is not symmetric (KL(P||Q) is not equal to KL(Q||P)) and can be infinite if there are points where Q(x) is zero and P(x) is non-zero.

3. The **Total Variation (TV) Distance** is a measure of the difference between two probability distributions. It is symmetric and its values lie between 0 and 1. The TV distance considers the maximum difference in probabilities between outcomes, making it somewhat insensitive to changes in the distribution that don't affect the maximum difference.

4. The **Wasserstein Distance** or Earth Mover's Distance considers the total difference between two distributions in terms of the 'cost' that must be paid to transform one distribution into the other. It's more robust to changes in the location of the distribution and provides meaningful gradients almost everywhere, making it particularly useful for training GANs.

5. **Integral Probability Metrics (IPM)** are a class of metrics used to compare probability distributions. They differ from other metrics by providing a more comprehensive comparison, taking both near and far samples into account. They represent the metric in terms of a supremum over a class of functions, including the Wasserstein distance, Energy distance, and Maximum Mean Discrepancy (MMD).

6. One would prefer to use **Wasserstein Distance** over the Kullback-Leibler Divergence when the distributions being compared do not overlap or only partially overlap. This is because the KL Divergence can be infinite in these cases, while the Wasserstein Distance provides meaningful gradients even in such scenarios.

7. The choice of different **probability metrics** can significantly affect the training and performance of generative models. Some metrics may provide more useful gradients for training, while others may be more suitable for certain data types or distributions. Therefore, the choice of the metric should align with the characteristics of the data, the generative model, and the available computational resources.

8. It's crucial to consider both **near and far samples** when using Integral Probability Metrics to compare distributions because it provides a more comprehensive comparison of the distributions. Considering both near and far samples allows the IPM to account for global structure in the distributions rather than just local differences, resulting in a more accurate and meaningful comparison.
