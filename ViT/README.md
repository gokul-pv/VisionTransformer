## Explanation for the Vision Transformer classes
The key  contribution of the ViT paper was the application of an existing  architecture (Transformers, introduce in Attention is all you need) to  the field of computer vision. It is the  training method and the dataset used to pre-train the network, that was  key for ViT to get excellent results compared to SOTA on ImageNet. 

 As an overall method, from the paper:

> *We split an image into fixed-size patches, linearly embed each of them,  add positional embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder. In order to perform classification, we  use the standard approach of adding an extra learnable "classification  token" to the sequence.*

The overall architecture can be described easily in five simple steps:

1. Split an input image into patches
2. Get linear embeddings (representations) from each patch referred to as Patch Embeddings
3. Add positional embeddings and a [CLS] token to each of the Patch Embeddings
4. Pass through a Transformer Encoder and get the output values for each of the [CLS] tokens.
5. Pass the representations of [CLS] tokens through an MLP Head to get final class predictions. 

![](https://github.com/gokul-pv/VisionTransformer/blob/main/ViT/Images/vit-01.png)

## Step 1: Patch embedding

Let's look at the code for Patch embedding

```python
class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.

    """

    def __init__(self, image_size=224, patch_size=16, num_channels=3, embed_dim=768):
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        # FIXME look at relaxing size constraints
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x
```

Let us start with a 224x224x3 image. We divide the image into 14 patches of 16x16x3 size. The model is not limited to this 14 patch restriction, and neither we need to send a square image. It just needs to be a  multiple of 16. Hence all the below images are of acceptable size:

- 224x160, 160x160, 480x480, 1920x1280, ...

`to_2tuple()` will check if the input is inerrable or not. If it is not, then it will make it a tuple and return it. Then we pass these patches through the **same** linear projection layer to get 1x768. We use convolutions for this

```
nn.Conv2d(3, 768, kernel_size=16, stride=16)

The point is to convert that 16x16x3 into 768. 

224x224x3 → 14x14x16x16x3 → 196x16x16x3 | Conv2D(3, 768, 16, 16) → 196x768 → 768x196

```



## Step2: Position and CLS Embeddings

```python
class ViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = PatchEmbeddings(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=config.hidden_size,
        )
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        embeddings = self.patch_embeddings(pixel_values)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
```

This process can be visualized as below:

![img](https://github.com/gokul-pv/VisionTransformer/blob/main/ViT/Images/vit-03.png)

So [CLS] token is a vector of size 1x768, and nn.Parameter makes it a  learnable parameter. We prepend it to the Patch Embedding and add  Positional Embeddings. 

```
224x224x3 → 14x14x16x16x3 → 196x16x16x3 | Conv2D(3, 768, 16, 16) → 196x768 → 768x196

PatchEmbedding (768x196) + CLS_TOKEN (768X1) → Intermediate_Value (768x197)

Positional Embedding (768x197) + Intermediate_Value (768x197) → Combined Embedding (768x197)
```

 It is interesting to see what these position embeddings look like after training:

![](https://github.com/gokul-pv/VisionTransformer/blob/main/ViT/Images/visualizing-positional-encodings-vit.png)



## Step 3: Transformer Encoder

![](https://github.com/gokul-pv/VisionTransformer/blob/main/ViT/Images/vit-07.png)



- The **Combined Embedding** (768x197) is sent as the input to the **first** transformer
- The first layer of the Transformer encoder accepts Combined Embedding of shape 197x768 as input. **For all subsequence layers, the inputs are the output matrix of shape 197x768**. Note that we are maintaining that additional CLS Token embedding dimension. 
- There are 12 such layers in the ViT-Base architecture. 

```python
self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.num_hidden_layers)])
```

- Inside the Layer, inputs are first passed through a Layer Norm, and then fed to a multi-head attention block.
- Next we have a fc layer to expand the dimension to:  torch.Size([197, 2304])
- The vectors are divided into query, key and value after expanded by an fc layer. QKV are further divided into H (=12) and fed to the parallel attention heads.

```
QKV → 197x768 | QKV_LINEAR_LAYER (197x768x3) | QKV-Vector (197x2304)

QKV-Vector (197x2304) → 12Head-QKV-Vector(197x3x12x64)

DESTACK

Q-Vector (12x197x64) | K-Vector (12x197x64) | V-Vector (12x197x64)

SoftMax(Q×K')= 12×197×197 = Attention_Matrix

Attention_Matrix(12x197x197) × V(12x197x64) → Output(12x197×64) → Output(197×768)
```



Transformer encoder works in the following steps:

- Send the embeddings to the layer normalization
- Generate key, value & queries from the embeddings
- To get the attention scores multiply query with tanspose of key
- Divide the attention scores with square root of attention head size  and pass it through softmax layer to get the attention probabilities
- Multiply the attention probablities with the values to get the context layer



```python
class ViTSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
```

A linear layer is added to the output of  context layer to transform the context layer so that input and output are of the same dimension. **ViTSelfOutput** class is used for this purpose

```python
class ViTSelfOutput(nn.Module):
  """
  This is just a Linear Layer Block
  """
  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)

    return hidden_states
   
```

**ViTAttention** combines two classes ViTSelfAttention and ViTSelfOutput.  the input states of embeddings are first send to the ViTSelfAttention to get the context layer and then to ViTSelfOutput

```python
class ViTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = ViTSelfAttention(config)
        self.output = ViTSelfOutput(config)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs
```



## STEP 4- MLP Block

The MLP block consists of two linear Layers and a GELU non-linearity. 

- we start with 768 and expand the dimension (i.e 768 x 4)
- add GELU, which is sent to the next linear layer
- the linear layer takes in 768x4 (i.e 3072) and converts that into 768

```python
class ViTIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states):

        hidden_states = self.dense(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states)

        return hidden_states
    
class ViTOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states    
```



The class **ViTLayer** combines all the block and returns the final output as  shown in this [image](https://github.com/gokul-pv/VisionTransformer/blob/main/ViT/Images/vit-07.png)

```python
class ViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config):
        super().__init__()
        self.seq_len_dim = 1
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)

        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs

```



```python
class ViTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        
        return hidden_states,all_hidden_states,all_self_attentions
```



The **whole process** would be:

![](https://github.com/gokul-pv/VisionTransformer/blob/main/ViT/Images/vit-06.png)

Notice that the 12th block returns 197x768, but the MLP head receives 768 as input! This is the last line of the code of the Vision Transformer's forward function!

```python
return x[:,0]
```

 [CLS] Token is used to maintain what each transformer block thinks about the image,  and finally, use only this vector for prediction! 

The class **ViTPooler** is an optional class to add more capacity at the end if required. It is simple class having a dense layer and Tanh as the activation function. This pooled output is then sent to the classifier (which is again a linear layer) to get the final output/prediction

```python
class ViTPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
```



```python
class ViTModel():
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = ViTEmbeddings(config)
        self.encoder = ViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings


    def forward(
        self,
        pixel_values=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:

        Examples::

            >>> from transformers import ViTFeatureExtractor, ViTModel
            >>> from PIL import Image
            >>> import requests

            >>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
            >>> model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

            >>> inputs = feature_extractor(images=image, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return sequence_output,pooled_output,encoder_outputs.hidden_states,encoder_outputs.attentions
```



```python
sequence_output = encoder_output[0]
layernorm = nn.LayerNorm(config.hidden_size, eps=0.00001)
sequence_output = layernorm(sequence_output)
# VitPooler
dense = nn.Linear(config.hidden_size, config.hidden_size)
activation = nn.Tanh()
first_token_tensor = sequence_output[:, 0]
pooled_output = dense(first_token_tensor)
pooled_output = activation(pooled_output)

classifier = nn.Linear(config.hidden_size, 100)
logits = classifier(pooled_output)
```



## References

- [https://theaisummer.com/vision-transformer/#how-the-vision-transformer-works-in-a-nutshell](https://theaisummer.com/vision-transformer/#how-the-vision-transformer-works-in-a-nutshell)
- [https://jacobgil.github.io/deeplearning/vision-transformer-explainability](https://jacobgil.github.io/deeplearning/vision-transformer-explainability)
- [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
- [https://github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
