
**Questions**
1. How is Deep Learning defined in the context of this text?
2. What are the key components of deep learning architectures?
3. Why is it important to balance maximum expressivity and minimum unrequired parameters in a deep learning network?
4. How are the weights in a deep learning network obtained?
5. What are the challenges with Fully Connected (FC) networks, especially when dealing with large inputs such as images?
6. What is the concept of tokenization and how does it help improve deep learning approaches?
7. How does the use of context improve the performance of deep learning models?
8. What are the limitations of using convolutional layers in a deep learning model?
9. How do self-attention and transformers improve the ability of a model to capture global context?
10. How are residual (skip) connections and normalization techniques used to overcome optimization issues in deep learning?
11. What are two ways to reach global context with few parameters?
12. What is the relationship between expressivity and inductive bias in deep learning?
13. Why might it be more beneficial to use existing neural architectures from similar problems than to invent new ones?
14. Despite the advancements in deep learning architectures, what remains a complex task?
15. How have deep learning architectures evolved, and what does this mean for future research and applications?

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

**Answers**
1. Deep Learning is defined in the context of this text as the construction and training of networks of parameterized functional modules using gradient-based optimization, with the modules trained from examples.

2. The key components of deep learning architectures are networks of parameterized functional modules that are trained using examples. These modules perform simple operations that are combined with learned parameters, either in series or parallel.

3. Balancing maximum expressivity and minimum unrequired parameters is essential in a deep learning network to ensure an efficient approximation of functions while reducing the risk of overfitting. Maximum expressivity allows the model to capture complex patterns in the data, while minimizing unrequired parameters reduces the complexity of the model and helps to prevent it from fitting too closely to the training data.

4. The weights in a deep learning network are obtained through optimization techniques, specifically mentioned in the text is Stochastic Gradient Descent (SGD) with momentum. These techniques aim to minimize the error or loss function by adjusting the weights of the network.

5. Fully Connected (FC) networks face challenges when dealing with large inputs such as images due to high computational and memory costs. This could lead to either keeping the same high dimensional input (which incurs a high cost) or severely compressing the dimensions (which may result in loss of information).

6. Tokenization is a concept where the input is divided into multiple "tokens" instead of treating it as a single vector. This approach helps to manage and process large inputs more efficiently.

7. Context is used to gather relevant information from other tokens and combine it with the original token information. This enhances the performance of deep learning models by allowing them to consider additional relevant information when processing each token.

8. Convolutional layers, while able to learn complex patterns by stacking and analyzing local contexts, face the limitation of their local contexts being limited in scope. Stacking more layers only increases the receptive field slowly, making it challenging to capture broader contextual patterns efficiently.

9. Self-Attention and Transformers improve the ability of a model to capture global context. Self-Attention enables each token to capture relevant global context by mapping each token into key, query, and value. Transformers use adaptive global context with weighted averaging, mitigating some of the limitations of self-attention by using multiple attention heads.

10. Deep networks often face optimization issues due to the complexity and depth of the network. Residual (skip) connections are used to provide direct feedback from the output to each weight, helping to alleviate vanishing gradient problems. Normalization techniques like Batch or Layer normalization help with SGD optimization by ensuring consistent distribution of inputs across layers, aiding in the training process.

11. Two ways to reach global context with few parameters are through Convolutional Neural Networks (CNN), which use local patches and multi-scale techniques for larger context, and Transformers, which use self-attention to encode relevant context.

12. Expressivity and inductive bias are intertwined in deep learning. While Transformers are very expressive (able to represent complex functions), overfitting can occur due to too much expressivity. So, the right kind of expressivity is essential to ensure a positive inductive bias, which guides the learning algorithm towards better generalization.

13. It is often more beneficial to use existing neural architectures from similar problems than to invent new ones because these architectures have already been optimized and tested, reducing the risk of issues and speeding up the development process.

14. Despite advancements in deep learning architectures, hyperparameter tuning remains a complex task. It often involves trial-and-error, as there is no one-size-fits-all solution, and every problem might need a different set of optimal hyperparameters.

15. Deep learning architectures have been evolving to enable global context without needing many parameters. This progression suggests a future with more efficient models capable of capturing complex patterns in data with reduced computational resources. However, new tasks often require innovative tokenization approaches, indicating that there is still much room for further development and refinement in deep learning.
  
