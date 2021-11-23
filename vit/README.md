## Pytorch implementation of Vision Transformer (ViT)

Implementation of the Vision Transformer proposed in ''An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale''. We re-create the model architecture of the model and transfer the pre-trained weights given by the ``timm`` python package. 

### Usage
#### Install timm package
``
pip install timm
``

#### Transfer weights from pre-trained model to our model
``
python3 transfer_weights.py
``
#### Run inference on ``cat.png``
``
python3 pred.py
``

### Input Image


<img src="cat.png" style="width:25%">



### Output
Top 10 ImageNet class predictions 
```console
Top 10 predictions are: 
----------------------------------------------------------
0: tabby, tabby_cat                              --- 0.8001
1: tiger_cat                                     --- 0.1752
2: Egyptian_cat                                  --- 0.0172
3: lynx, catamount                               --- 0.0018
4: Persian_cat                                   --- 0.0011
5: Siamese_cat, Siamese                          --- 0.0002
6: bow_tie, bow-tie, bowtie                      --- 0.0002
7: weasel                                        --- 0.0001
8: lens_cap, lens_cover                          --- 0.0001
9: remote_control, remote                        --- 0.0001
```


#### References:
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Official Github repo](https://github.com/google-research/vision_transformer)
- [Pytorch implementation](https://github.com/lucidrains/vit-pytorch)
- [Weight transfer tutorial](https://www.youtube.com/watch?v=ovB0ddFtzzA&t=142s&ab_channel=mildlyoverfitted)
