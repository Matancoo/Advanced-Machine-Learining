

**Definition and Basics of Deep Learning**
- Deep Learning is about constructing networks of parameterized functional modules & training them from examples using gradient-based optimization.
- Deep learning architectures combine simple operations with learned parameters. Operations can be in series or parallel.

**Expressivity and Optimization**
- The network's operation must be expressible; small networks may not approximate our desired task. 
- Balancing maximum expressivity and minimum unrequired parameters is essential for function approximation and reducing overfitting.
- The weights in a deep learning network are obtained through optimization techniques like Stochastic Gradient Descent (SGD) with momentum.

**Challenges with Fully Connected Networks**
- For large inputs such as images, Fully Connected (FC) networks do not scale well due to high computational and memory cost.
- Options include keeping the same input dimension (high cost), or severely compressing dimensions (loss of information).

**Improving Approaches: Tokenization and Context**
- Tokenization: Instead of treating input as one vector, it's divided into multiple "tokens."
- Context: For each token, relevant context is gathered from other tokens and combined with original token information.

**Convolutional Layers and Context Limitations**
- Convolutional layers can learn complex patterns by stacking and analyzing local contexts.
- However, local contexts can be limited in scope, and stacking convolutional layers increases the receptive field slowly.

**Advanced Techniques: Self-Attention and Transformers**
- Self-Attention: A method to capture global context efficiently by mapping each token into key, query, and value.
- Transformers: Use adaptive global context with weighted averaging. Limitations can be mitigated by using multiple attention heads.

**Overcoming Optimization Issues**
- Deep networks face optimization issues. Residual (skip) connections are used to provide direct feedback from the output to each weight.
- Normalization techniques like Batch or Layer normalization can help with SGD optimization.

**Global Context Modeling**
- Two ways to reach global context with few parameters: CNN (local patches and multi-scale for larger context) and Transformer (self-attention to encode relevant context).

**Expressivity vs. Inductive Bias**
- Transformers are very expressive, while CNNs are a special case of transformers.
- Overfitting can result from too much expressivity, so the right kind of expressivity is essential for positive inductive bias.

**Choosing Neural Architectures in Practice**
- It's often better to use existing architectures from similar problems than to invent new ones.
- Only a few architectures have stood the test of time.





**Conclusion**
- Deep learning architectures are converging, enabling global context without too many parameters.
- Hyperparameter tuning remains a complex task, and new tasks often require innovative tokenization approaches.