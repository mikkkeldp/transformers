# Transformers explained

This is a short explanation to how transformers work. 

## Background
Transformers was introduced in [Attention is all you need](https://arxiv.org/abs/1706.03762) back in 2017 as an alternative to Recurrent Neural Networks (RNN’s).RNN's treat sequences sequentially to keep the order of the sentence in place. To satisfy that design, each RNN component (layer) needs the previous (hidden) output. As such, stacked LSTM computations were performed sequentially. This is where transformers differ. The fundamental building block of a transformer is self-attention. Transformers feed the entire input sequence at once.

## Representing the input sentence
Before the input is fed to the model, 3 processing  steps are performed. 

### 1. Sets and tokenization
<p align="center">
    <img src="https://theaisummer.com/static/c9a851690a62f1faaf054430ca35ab20/c7dcc/tokenization.png" style="width:50%">
</p>
The input is first tokenized (as shown above). So, instead of a sequence of elements, we have a set. Sets are a collection of distinct elements, where the arrangement of the elements in the set does not matter. We denote the input set as 
<img src="https://render.githubusercontent.com/render/math?math=X = x_1,x_2. \ldots, x_N"> where 
<img src="https://render.githubusercontent.com/render/math?math=x \in R^{N \times d_{in}}">. The elements of the sequence <img src="https://render.githubusercontent.com/render/math?math=x_i"> are referred to as tokens.
After tokenization, we project words in a distributed geometrical space, or simply build word embeddings.


### 2. Word Embeddings
In general, an embedding is a representation of a symbol (word, character, sentence) in a distributed low-dimensional space of continuous-valued vectors. Ideally, an embedding captures the semantics of the input by placing semantically similar inputs close together in the embedding space.


<p align="center">
    <img src="https://media2.giphy.com/media/p5MMLRVr2Om34WU9pg/giphy.gif?cid=790b7611128d6740994480011c0c936275431bd1844f4e51&rid=giphy.gif&ct=g" style="width:50%">
</p>

We will now provide some notion of order in the set through positional encodings.

### 3. Positional encodings
When you convert a sequence into a set (tokenization), you lose the notion of order. Since transformers process sequences as sets, they are, in theory, permutation invariant. Officially, positional encoding is a set of small constants, which are added to the word embedding vector before the first self-attention layer. 

We try to provide some context to each word (token). o if the same word appears in a different position, the actual representation will be slightly different, depending on where it appears in the input sentence.

<p align="center">
    <img src="https://theaisummer.com/static/257848131da90edbf099aa8c4bf392c4/27524/input-processing-tokenization-embedding.png" style="width:40%">
</p>

In the transformer paper, the authors came up with the sinusoidal function for the positional encoding. The sine function tells the model to pay attention to a particular wavelength <img src="https://render.githubusercontent.com/render/math?math=\lambda">. Given a signal <img src="https://render.githubusercontent.com/render/math?math=y(x) = \sin (k x)y(x)=sin(kx)"> the wavelength will be <img src="https://render.githubusercontent.com/render/math?math=k = \frac{2 \pi}{\lambda}">. In our case the λ will be dependent on the position in the sentence. i is used to distinguish between odd and even positions.

Mathematically:
<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=PE(pos,2i)=\sin(\frac{pos}{10000^{2i/512}})" style="width:30%">
</p>
<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=PE(pos,{2*i} *plus* {1})=\cos(\frac{pos}{10000^{2i/512}})" style="width:35%">
</p>
where  <img src="https://render.githubusercontent.com/render/math?math=512 = d_{model}">, which is the dimensionality of the embedding vectors. Below is a @D visualization of a positional encoding.


<p align="center">
    <img src="https://theaisummer.com/static/a662e9c10a5401d1bd1ccdce52dfdbd6/c1b63/positional-encoding.png" style="width:60%">
</p>

This is in contrast to recurrent models, where we have an order but we are struggling to pay attention to tokens that are not close enough.

## Fundamental concepts of the Transformer
We now provide some necessary background information needed to understand some ideas introduced within the transformer architecture. 
### Feature-based attention: The Key, Value, and Query
Key-value-query concepts come from information retrieval systems.This concept is important to grasp. Let's take youtube search as an example.

When you search (**query**) for a particular video, the search engine will map your query against a set of video titles, descriptions, etc. (**keys**) associated with stored videos. The algorithm then returns the best matched videos (**values**). This is the foundation of content/**feature-based lookup**.

<p align="center">
    <img src="https://theaisummer.com/static/2e000851b686eb35c6c3c06522437715/26a94/attention-as-database-query.png" style="width:60%">
</p>

In the single video retrieval example, the attention is the choice of the video with a maximum relevance score. But we can relax this idea. To this end, the main difference between attention and retrieval systems is that we introduce a more abstract and smooth notion of ‘retrieving’ an object. By defining a degree of similarity (weight) between our representations (videos for youtube) we can weight our query. Instead of choosing where to look according to the position within a sequence, we now attend to the content that we wanna look at. 

We use the **keys** to define the attention weights to look at the data and the values as the information that we will actually get. For the so-called mapping, we need to quantify similarity, that we will be seeing next.


### Vector similarity in high dimensional spaces
In geometry, the **inner vector product** is interpreted as a vector projection. One way to define vector similarity is by computing the normalized inner product. In low dimensional space, like the 2D example below, this would correspond to the cosine value.
<p align="center">
    <img src="https://theaisummer.com/static/ebfe1b1dbab018e608a77f85457e52db/16caa/vector-similarity.png" style="width:24%">
</p>
Mathematically:
<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=sim(a,b)=\cos(a,b)=\frac{a b}{|a||b|}=\frac{1}{s}*a b" style="width:35%">
</p>
We can associate the similarity between vectors that represent anything (i.e. animals) by calculating the scaled dot product, namely the cosine of the angle.

In transformers, this is the most basic operation and is handled by the self-attention layer as we’ll see.


## Self-Attention: The Transformer encoder
''Self-attention, sometimes called intra-attention, is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.” ~ Ashish Vaswani et al. [2] from Google Brain''

Self-attention enables us to find correlations between different words of the input indicating the syntactic and contextual structure of the sentence.

Let’s take the input sequence “Hello I love you” for example. A trained self-attention layer will associate the word “love” with the words ‘I” and “you” with a higher weight than the word “Hello”. From linguistics, we know that these words share a subject-verb-object relationship and that’s an intuitive way to understand what self-attention will capture.

<p align="center">
    <img src="https://theaisummer.com/static/4022cf02281d234e0e85fa44ad08b4e2/9f933/self-attention-probability-score-matrix.png" style="width:30%">
</p>

In practice, the Transformer uses 3 different representations: the Queries, Keys and Values of the embedding matrix. This can easily be done by multiplying our input <img src="https://render.githubusercontent.com/render/math?math=X \in R^{N \times d_k}"> with 3 different weight matrices <img src="https://render.githubusercontent.com/render/math?math=W_Q, W_K  \text{ and } W_V \in R^{d_k \times d_{model}}">. This is just a matrix multiplication in the original word embeddings. The resulted dimension will be smaller: <img src="https://render.githubusercontent.com/render/math?math=d_k > d_{model}">.


<p align="center">
    <img src="https://theaisummer.com/static/56773616d30b9dcb31aa792f2d701276/c1b63/key-query-value.png" style="width:70%">
</p>

Having the Query, Value and Key matrices, we can now apply the self-attention layer as:

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\text{Attention}(Q,K,V)=\softmax(\frac{QK^T}{\sqrt{d_k}})V" style="width:35%">
</p>

In the original paper, the scaled dot-product attention was chosen as a scoring function to represent the correlation between two words (the attention weight). Note that we can also utilize another similarity function. The <img src="https://render.githubusercontent.com/render/math?math=\sqrt{d_{k}}"> is here simply as a scaling factor to make sure that the vectors won’t explode.


Following the database-query paradigm we introduced before, this term simply finds the similarity of the searching query with an entry in a database. Finally, we apply a softmax function to get the final attention weights as a probability distribution.

Remember that we have distinguished the Keys (K) from the Values (V) as distinct representations. Thus, the final representation is the self-attention matrix <img src="https://render.githubusercontent.com/render/math?math=\softmax\left(\frac{\textbf{Q} \textbf{K}^{T}}{\sqrt{d_{k}}}\right)"> multiplied with the Value (V) matrix. 

You can think of the attention matrix as where to look and the value matrix as what I actually want to get.

In the context of vector similarity, we have matrices instead of vectors and as a result matrix multiplications. We also don;t scale down by vector magnitude but by matrix size (d_k), which is the number of words in the sentence.

The next steps are normalization and skip connections, similar to processing a tensor after convolution or recurrency. 

## Short residual skip connections
Skip connections give a transformer a tiny ability to allow the representations of different levels of processing to interact. With the forming of multiple paths, we can “pass” our higher-level understanding of the last layers to the previous layers.  This allows us to re-modulate how we understand the input. For more information on [skip connections](https://theaisummer.com/skip-connections/).

## Layer normalization
In Layer Normalization (LN), the mean and variance are computed across channels and spatial dims. In language, each word is a vector. Since we are dealing with vectors we only have one spatial dimension.

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\mu_{n}=\frac{1}{K} \sum_{k=1}^{K} x_{nk}" style="width:18%">
</p>
<br>
<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\sigma_{n}^{2}=\frac{1}{K} \sum_{k=1}^{K}\left(x_{nk}-\mu_{n}\right)^{2}" style="width:25%">
</p>
<br>
<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\hat{x}_{nk}= \frac{x_{nk}-\mu_{n}}{\sqrt{\sigma_{n}^{2} *plus* \epsilon}}, \hat{x}_{nk} \in R" style="width:25%">
</p>

<br>
<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\mathrm{LN}_{\gamma, \beta}\left(x_{n}\right) =\gamma \hat{x}_{n} *plus*\beta ,x_{n} \in R^{K}" style="width:30%">
</p>
<br>
In a 4D tensor with merged spatial dimensions, we can visualize this with the following figure:

<p align="center">
    <img src="https://theaisummer.com/static/3ed7199184645f3e632d17ab6441244f/63a68/layer-norm.png" style="width:40%">
</p>

After applying a [normalization](https://theaisummer.com/normalization/) layer and forming a residual skip connection we are here:

<p align="center">
    <img src="https://theaisummer.com/static/f6068bcb3559a017af003c2bde071bcf/e3b18/encoders-attention-with-normalizarion.png" style="width:30%">
</p>

Even though this could be a stand-alone building block, the creators of the transformer add another linear layer on top and renormalize it along with another skip connection.

## Linear layer
linear layer (PyTorch), dense layer (Keras), feed-forward layer (old ML books), fully connected layer. We will simply say linear layer which is:

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=y = xW^T+b" style="width:10%">
</p>

Where **W** is a matrix and **x,y,b** are vectors. They add two linear layers with dropout and non-linearities in between.

```
import torch.nn as nn
dim = 512
dim_linear_block = 512*4

linear = nn.Sequential(
    nn.Linear(dim, dim_linear_block),
    nn.ReLU,
    nn.Dropout(dropout),
    nn.Linear(dim_linear_block, dim),
    nn.Dropout(dropout))
```

The main intuition is that they project the output of self-attention in a higher dimensional space (X4 in the paper). This solves bad initializations and rank collapse. We will depict it in the diagrams simply as Linear.

This (almost) concludes encoder part of the transformer with N such building blocks, as depicted below:

<p align="center">
    <img src="https://theaisummer.com/static/dc71435f329458ee5cc09cb2ea09ebf8/7bc0b/encoder-without-multi-head.png" style="width:30%">
</p>

The final aspect to explore is multi-head attention

## The core building block: Multi-head attention and parallel implementation

In the original paper, the authors expand on the idea of self-attention to multi-head attention. In essence, we run through the attention mechanism several times.

Each time, we map the independent set of Key, Query, Value matrices into different lower dimensional spaces and compute the attention there (the output is called a “head”). The mapping is achieved by multiplying each matrix with a separate weight matrix, denoted as <img src="https://render.githubusercontent.com/render/math?math=W_i^K, W_i^Q, W_i^V \in R^{d_model \times d_k}">.

To compensate for the extra complexity, the output vector size is divided by the number of heads. Specifically, in the vanilla transformer, they use <img src="https://render.githubusercontent.com/render/math?math=d_{model}=512d"> model and h=8 heads, which gives us vector representations of 64. Now, the model has multiple independent paths (ways) to understand the input.

The heads are then concatenated and transformed using a square weight matrix <img src="https://render.githubusercontent.com/render/math?math=W^O \in R^{d_{model} \times d_{model}}"> since <img src="https://render.githubusercontent.com/render/math?math=d_{model} = hd_k">

Putting this together we get:

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\text{MultiHead}(Q,K,V) = \text{concat}(head_1, head_2, \ldots, head_h)W^O" style="width:47%">
</p>
where
<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=head_i = \text{Attention}(QW_i^Q,KW_i^K,VW_i^V)" style="width:30%">
</p>
where again
<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=QW_i^Q,KW_i^K,VW_i^V \in R^{d_{model}\times d_k}" style="width:25%">
</p>

Since heads are independent from each other, we can perform the self-attention computation in parallel on different workers:

<p align="center">
    <img src="https://theaisummer.com/static/9dc2e417714211a5166ece483b862d75/442cb/parallel-multi-head-attention.png" style="width:60%">
</p>

The intuition behind multihead attention is that it allows us to attend to different parts of the sequence differently each time. This practically means that:
- The model can better capture **positional information** because each head will attend to different segments of the input. The combination of them will give us a more robust representation.
- Each head will capture different contextual information as well, by correlating words in a unique manner.


''Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.''

We will depict Multi-head self-attention in our diagrams like this:

<p align="center">
    <img src="https://theaisummer.com/static/bba48bd14e38ede88ac1cacd8a638d6d/a4078/multi-head-attention.png
" style="width:30%">
</p>

Helpful resource on multihead attention: [Pytorch implementation using the einsum notation.](https://theaisummer.com/einsum-attention/)

## Summuary of encoder
Input is processed in 3 steps:
1. Word embeddings of the input sentence are computed simultaneously.
2. Positional encodings are then applied to each embedding resulting in word vectors that also include positional information.
3. The word vectors are passed to the first encoder block.

Each block consists of the following layers in the same order:
1. A multi-head self-attention layer to find correlations between each word

2. A normalization layer

3. A residual connection around the previous two sublayers

4. A linear layer

5. A second normalization layer

6. A second residual connection

Note that the above block can be replicated several times to form the Encoder. In the original paper, the encoder composed of 6 identical blocks.

<p align="center">
    <img src="https://theaisummer.com/static/18072c01858310b080b3b6d9b4950175/e45a9/encoder.png
" style="width:30%">
</p>

## The decoder
The decoder consists of all the aforementioned components plus two novel ones. As before:

1. The output sequence is fed in its entirety and word embeddings are computed

2. Positional encoding are again applied

3. And the vectors are passed to the first Decoder block

Each decoder block includes:

1. A Masked multi-head self-attention layer

2. A normalization layer followed by a residual connection

3. A new multi-head attention layer (known as Encoder-Decoder attention)

4. A second normalization layer and a residual connection

5. A linear layer and a third residual connection

The decoder block appears again 6 times. The final output is transformed through a final linear layer and the output probabilities are calculated with the standard softmax function.

<p align="center">
    <img src="https://theaisummer.com/static/7d6c2aa7af90f14cf44d533cbf88726e/8ff13/decoder.png
" style="width:30%">
</p>

The output probabilities predict the next token in the output sentence. How? In essence, we assign a probability to each word in the French language and we simply keep the one with the highest score.

To put things into perspective, the original model was trained on the WMT 2014 English-French dataset consisting of 36M sentences and 32000 tokens.

While most concepts of the decoder are already familiar, there are two more that we need to discuss. Let’s start with the Masked multi-head self-attention layer.

## Masked Multi-head attention

In case you haven’t realized, in the decoding stage, we predict one word (token) after another. In such NLP problems like machine translation, sequential token prediction is unavoidable. As a result, the self-attention layer needs to be modified in order to consider only the output sentence that has been generated so far.

In our translation example, the input of the decoder on the third pass will be “Bonjour”, “je” … …”.

As you can tell, the difference here is that we don’t know the whole sentence because it hasn’t been produced yet. That’s why we need to disregard the unknown words. Otherwise, the model would just copy the next word! To achieve this, we mask the next word embeddings (by setting them to −inf).

Mathematically we have:

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\text{MaskedAttention}(Q,K,V) = \softmax(\frac{QK^T*plus*M}{\sqrt{d_k}})V" style="width:45%">
</p>

where the matrix M (mask) consists of zeros and -inf. Zeros will become ones with the exponential while infinities become zeros. 

This effectively has the same effect as removing the corresponding connection. The remaining principles are exactly the same as the encoder’s attention. And once again, we can implement them in parallel to speed up the computations.

Obviously, the mask will change for every new token we compute.

## Encoder-Decoder attention:
This is actually where the decoder processes the encoded representation. The attention matrix generated by the encoder is passed to another attention layer alongside the result of the previous Masked Multi-head attention block.

The intuition behind the encoder-decoder attention layer is to combine the input and output sentence. The encoder’s output encapsulates the final embedding of the input sentence. It is like our database. So we will use the encoder output to produce the Key and Value matrices. On the other hand, the output of the Masked Multi-head attention block contains the so far generated new sentence and is represented as the Query matrix in the attention layer. Again, it is the “search” in the database.

''The encoder-decoder attention is trained to associate the input sentence with the corresponding output word.''
It will eventually determine how related each English word is with respect to the French words. This is essentially where the mapping between English and French is happening.


Notice that the output of the last block of the encoder will be used in each decoder block.


## Why the transformer architecture works so well
1. **Distributed and independent representations at each block**: Each transformer block has h=8 contextualized representations. Intuitively, you can think of it as the multiple feature maps of a convolution layer that capture different features from the image. The difference with convolutions is that here we have multiple views (linear reprojections) to other spaces. This is of course possible by initially representing words as vectors in a euclidean space (and not as discrete symbols).
2. **The meaning heavily depends on the context**: This is exactly what self-attention is all about! We associate relationships between word representation expressed by the attention weights. There is no notion of locality since we naturally let the model make global associations.
3. **Multiple encoder and decoder blocks**: With more layers, the model makes more abstract representations. Similar to stacking recurrent or convolution blocks we can stack multiple transformer blocks. The first block associates word-vector pairs, the second pairs of pairs, the third of pairs of pairs of pairs, and so on. In parallel, the multiple heads focus on different segments of the pairs. This is analogous to the receptive field but in terms of pairs of distributed representations.
4. **Combination of high and low-level information**: Skip-connections enable top-down understanding to flow back with the multiple gradient paths that flow backward.

## Self-attention VS linear layers VS convolutions
What is the difference between attention and a feedforward layer? Don't linear layers do exactly the same operations to an input vector as attention? No, You see the values of the self-attention weights are computed on the fly. They are data-dependent dynamic weights because they change dynamically in response to the data (fast weights).

For example, each word in the translated sequence (Bonjour, je t’aime) will attend differently with respect to the input.

On the other hand, the weights of a feedforward (linear) layer change very slowly with stochastic gradient descent. In convolutions, we further constrict the (slow) weight to have a fixed size, namely the kernel size.




## Resources
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). [Attention is all you need](https://scholar.google.com/scholar_url?url=https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf&hl=en&sa=T&oi=gsb-gga&ct=res&cd=0&d=2960712678066186980&ei=RSB_X4-XNeiOy9YP_6W9oAo&scisig=AAGBfm1BT0D5TrbyR7oR4v3lmmVXXipXoA). In Advances in neural information processing systems (pp. 5998-6008).
- DeepMind’s deep learning videos 2020 with UCL, Lecture: [Deep Learning for Natural Language Processing, Felix Hill](https://www.youtube.com/watch?v=8zAP2qWAsKg&t=2410s&ab_channel=DeepMind)
- [How Transformers work in deep learning and NLP: an intuitive introduction](https://theaisummer.com/transformer/)
  
