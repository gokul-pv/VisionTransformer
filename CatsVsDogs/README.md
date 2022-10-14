## Hands-on Vision Transformers

The objective is to implement this [blog](https://analyticsindiamag.com/hands-on-vision-transformers-with-pytorch/) and train the ViT model for Cats vs Dogs classification using [this](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data) dataset. Transfer learning can be used if needed.

## Dataset Description

The train folder contains 25,000 images of dogs and cats. Each image in  this folder has the label as part of the filename. The test folder  contains 12,500 images, named according to a numeric id. For each image  in the test set, you should predict a probability that the image is a  dog (1 = dog, 0 = cat).

## Data Visualization

![](https://github.com/gokul-pv/VisionTransformer/blob/main/CatsVsDogs/Images/train.png)



## Model using ViT-pytorch and Linformer

- Link to [notebook](https://github.com/gokul-pv/VisionTransformer/blob/main/CatsVsDogs/VisionTranformer.ipynb) 

## Model Parameters

```
dim=128,
seq_len=49+1,  # 7x7 patches + 1 cls-token
depth=12,
heads=8,
k=64
image_size=224,
patch_size=32,
num_classes=2,
channels=3
```

## Model

```
ViT(
  (to_patch_embedding): Sequential(
    (0): Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=32, p2=32)
    (1): Linear(in_features=3072, out_features=128, bias=True)
  )
  (transformer): Linformer(
    (net): SequentialSequence(
      (layers): ModuleList(
        (0): ModuleList(
          (0): PreNorm(
            (fn): LinformerSelfAttention(
              (to_q): Linear(in_features=128, out_features=128, bias=False)
              (to_k): Linear(in_features=128, out_features=128, bias=False)
              (to_v): Linear(in_features=128, out_features=128, bias=False)
              (dropout): Dropout(p=0.0, inplace=False)
              (to_out): Linear(in_features=128, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
          (1): PreNorm(
            (fn): FeedForward(
              (w1): Linear(in_features=128, out_features=512, bias=True)
              (act): GELU()
              (dropout): Dropout(p=0.0, inplace=False)
              (w2): Linear(in_features=512, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
        )
        (1): ModuleList(
          (0): PreNorm(
            (fn): LinformerSelfAttention(
              (to_q): Linear(in_features=128, out_features=128, bias=False)
              (to_k): Linear(in_features=128, out_features=128, bias=False)
              (to_v): Linear(in_features=128, out_features=128, bias=False)
              (dropout): Dropout(p=0.0, inplace=False)
              (to_out): Linear(in_features=128, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
          (1): PreNorm(
            (fn): FeedForward(
              (w1): Linear(in_features=128, out_features=512, bias=True)
              (act): GELU()
              (dropout): Dropout(p=0.0, inplace=False)
              (w2): Linear(in_features=512, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
        )
        (2): ModuleList(
          (0): PreNorm(
            (fn): LinformerSelfAttention(
              (to_q): Linear(in_features=128, out_features=128, bias=False)
              (to_k): Linear(in_features=128, out_features=128, bias=False)
              (to_v): Linear(in_features=128, out_features=128, bias=False)
              (dropout): Dropout(p=0.0, inplace=False)
              (to_out): Linear(in_features=128, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
          (1): PreNorm(
            (fn): FeedForward(
              (w1): Linear(in_features=128, out_features=512, bias=True)
              (act): GELU()
              (dropout): Dropout(p=0.0, inplace=False)
              (w2): Linear(in_features=512, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
        )
        (3): ModuleList(
          (0): PreNorm(
            (fn): LinformerSelfAttention(
              (to_q): Linear(in_features=128, out_features=128, bias=False)
              (to_k): Linear(in_features=128, out_features=128, bias=False)
              (to_v): Linear(in_features=128, out_features=128, bias=False)
              (dropout): Dropout(p=0.0, inplace=False)
              (to_out): Linear(in_features=128, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
          (1): PreNorm(
            (fn): FeedForward(
              (w1): Linear(in_features=128, out_features=512, bias=True)
              (act): GELU()
              (dropout): Dropout(p=0.0, inplace=False)
              (w2): Linear(in_features=512, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
        )
        (4): ModuleList(
          (0): PreNorm(
            (fn): LinformerSelfAttention(
              (to_q): Linear(in_features=128, out_features=128, bias=False)
              (to_k): Linear(in_features=128, out_features=128, bias=False)
              (to_v): Linear(in_features=128, out_features=128, bias=False)
              (dropout): Dropout(p=0.0, inplace=False)
              (to_out): Linear(in_features=128, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
          (1): PreNorm(
            (fn): FeedForward(
              (w1): Linear(in_features=128, out_features=512, bias=True)
              (act): GELU()
              (dropout): Dropout(p=0.0, inplace=False)
              (w2): Linear(in_features=512, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
        )
        (5): ModuleList(
          (0): PreNorm(
            (fn): LinformerSelfAttention(
              (to_q): Linear(in_features=128, out_features=128, bias=False)
              (to_k): Linear(in_features=128, out_features=128, bias=False)
              (to_v): Linear(in_features=128, out_features=128, bias=False)
              (dropout): Dropout(p=0.0, inplace=False)
              (to_out): Linear(in_features=128, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
          (1): PreNorm(
            (fn): FeedForward(
              (w1): Linear(in_features=128, out_features=512, bias=True)
              (act): GELU()
              (dropout): Dropout(p=0.0, inplace=False)
              (w2): Linear(in_features=512, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
        )
        (6): ModuleList(
          (0): PreNorm(
            (fn): LinformerSelfAttention(
              (to_q): Linear(in_features=128, out_features=128, bias=False)
              (to_k): Linear(in_features=128, out_features=128, bias=False)
              (to_v): Linear(in_features=128, out_features=128, bias=False)
              (dropout): Dropout(p=0.0, inplace=False)
              (to_out): Linear(in_features=128, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
          (1): PreNorm(
            (fn): FeedForward(
              (w1): Linear(in_features=128, out_features=512, bias=True)
              (act): GELU()
              (dropout): Dropout(p=0.0, inplace=False)
              (w2): Linear(in_features=512, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
        )
        (7): ModuleList(
          (0): PreNorm(
            (fn): LinformerSelfAttention(
              (to_q): Linear(in_features=128, out_features=128, bias=False)
              (to_k): Linear(in_features=128, out_features=128, bias=False)
              (to_v): Linear(in_features=128, out_features=128, bias=False)
              (dropout): Dropout(p=0.0, inplace=False)
              (to_out): Linear(in_features=128, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
          (1): PreNorm(
            (fn): FeedForward(
              (w1): Linear(in_features=128, out_features=512, bias=True)
              (act): GELU()
              (dropout): Dropout(p=0.0, inplace=False)
              (w2): Linear(in_features=512, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
        )
        (8): ModuleList(
          (0): PreNorm(
            (fn): LinformerSelfAttention(
              (to_q): Linear(in_features=128, out_features=128, bias=False)
              (to_k): Linear(in_features=128, out_features=128, bias=False)
              (to_v): Linear(in_features=128, out_features=128, bias=False)
              (dropout): Dropout(p=0.0, inplace=False)
              (to_out): Linear(in_features=128, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
          (1): PreNorm(
            (fn): FeedForward(
              (w1): Linear(in_features=128, out_features=512, bias=True)
              (act): GELU()
              (dropout): Dropout(p=0.0, inplace=False)
              (w2): Linear(in_features=512, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
        )
        (9): ModuleList(
          (0): PreNorm(
            (fn): LinformerSelfAttention(
              (to_q): Linear(in_features=128, out_features=128, bias=False)
              (to_k): Linear(in_features=128, out_features=128, bias=False)
              (to_v): Linear(in_features=128, out_features=128, bias=False)
              (dropout): Dropout(p=0.0, inplace=False)
              (to_out): Linear(in_features=128, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
          (1): PreNorm(
            (fn): FeedForward(
              (w1): Linear(in_features=128, out_features=512, bias=True)
              (act): GELU()
              (dropout): Dropout(p=0.0, inplace=False)
              (w2): Linear(in_features=512, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
        )
        (10): ModuleList(
          (0): PreNorm(
            (fn): LinformerSelfAttention(
              (to_q): Linear(in_features=128, out_features=128, bias=False)
              (to_k): Linear(in_features=128, out_features=128, bias=False)
              (to_v): Linear(in_features=128, out_features=128, bias=False)
              (dropout): Dropout(p=0.0, inplace=False)
              (to_out): Linear(in_features=128, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
          (1): PreNorm(
            (fn): FeedForward(
              (w1): Linear(in_features=128, out_features=512, bias=True)
              (act): GELU()
              (dropout): Dropout(p=0.0, inplace=False)
              (w2): Linear(in_features=512, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
        )
        (11): ModuleList(
          (0): PreNorm(
            (fn): LinformerSelfAttention(
              (to_q): Linear(in_features=128, out_features=128, bias=False)
              (to_k): Linear(in_features=128, out_features=128, bias=False)
              (to_v): Linear(in_features=128, out_features=128, bias=False)
              (dropout): Dropout(p=0.0, inplace=False)
              (to_out): Linear(in_features=128, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
          (1): PreNorm(
            (fn): FeedForward(
              (w1): Linear(in_features=128, out_features=512, bias=True)
              (act): GELU()
              (dropout): Dropout(p=0.0, inplace=False)
              (w2): Linear(in_features=512, out_features=128, bias=True)
            )
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
    )
  )
  (to_latent): Identity()
  (mlp_head): Sequential(
    (0): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (1): Linear(in_features=128, out_features=2, bias=True)
  )
)
```



## Result

- Model was trained for 20 epochs with a batch size of 64 
- Training accuracy = 68.41% 
- Validation accuracy = 66.85%
- Train loss = 0.5866
- Validation loss = 0.5992 

**Training Logs for last 5 epochs**

```
100% 313/313 [01:55<00:00, 3.04it/s]
Epoch : 16 - loss : 0.5980 - acc: 0.6729 - val_loss : 0.6108 - val_acc: 0.6634

100% 313/313 [01:56<00:00, 3.11it/s]
Epoch : 17 - loss : 0.5938 - acc: 0.6776 - val_loss : 0.5982 - val_acc: 0.6648

100% 313/313 [01:56<00:00, 3.01it/s]
Epoch : 18 - loss : 0.5930 - acc: 0.6763 - val_loss : 0.6123 - val_acc: 0.6620

100% 313/313 [01:56<00:00, 3.10it/s]
Epoch : 19 - loss : 0.5909 - acc: 0.6823 - val_loss : 0.5960 - val_acc: 0.6748

100% 313/313 [01:56<00:00, 3.14it/s]
Epoch : 20 - loss : 0.5866 - acc: 0.6841 - val_loss : 0.5992 - val_acc: 0.6685

```

**Loss and Accuracy plot**

![](https://github.com/gokul-pv/VisionTransformer/blob/main/CatsVsDogs/Images/loss.png)



## Predictions on few test images



| ![](https://github.com/gokul-pv/VisionTransformer/blob/main/CatsVsDogs/Images/test_04.PNG) | ![](https://github.com/gokul-pv/VisionTransformer/blob/main/CatsVsDogs/Images/test_01.PNG)                       Misclassified |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![](https://github.com/gokul-pv/VisionTransformer/blob/main/CatsVsDogs/Images/test_02.PNG) | ![](https://github.com/gokul-pv/VisionTransformer/blob/main/CatsVsDogs/Images/test_03.PNG) |



## Reference

- [https://analyticsindiamag.com/hands-on-vision-transformers-with-pytorch/](https://analyticsindiamag.com/hands-on-vision-transformers-with-pytorch/)
- [https://arxiv.org/abs/2006.04768](https://arxiv.org/abs/2006.04768)
- [https://www.kaggle.com/general/74235](https://www.kaggle.com/general/74235)

