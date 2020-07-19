# IrisSegBenchmark

<img src='iris.jpg' width="400px">

### Introduction
This repository contains some codes for **iris segmentation**. 

### Citation
If you use the codes for your research, please cite our paper.

```
王财勇, 孙哲南. 虹膜分割算法评价基准[J]. 计算机研究与发展, 2020, 57(2): 395-412.
Wang Caiyong, Sun Zhenan. A Benchmark for Iris Segmentation[J]. Journal of Computer Research and Development, 2020, 57(2): 395-412.
```
Anyone is permitted to use, distribute and change this program for any non-commercial usage. However each of such usage and/or publication must include above citation of this paper.

### CNN models
The used models includes:
- [FCN](https://arxiv.org/abs/1411.4038)
- [Deeplab V1,V2,V3](http://liangchiehchen.com/projects/DeepLab.html)
- [ParseNet](https://arxiv.org/abs/1506.04579)
- [PSPNet](https://arxiv.org/abs/1612.01105)
- [SegNet](http://mi.eng.cam.ac.uk/projects/segnet/)
- [U-Net](https://arxiv.org/abs/1505.04597)

We use the [extended caffe](https://github.com/xiamenwcy/extended-caffe) to implement these models.

If you want to train your model, you can use the data from  [IrisParseNet](https://github.com/xiamenwcy/IrisParseNet)

The trained model can be downloaded via https://pan.baidu.com/s/1t0fSgVZKzSGtAcQhVEP8jg       . 

### Traditional methods
- TVM: [https://www4.comp.polyu.edu.hk/~csajaykr/tvmiris.htm](https://www4.comp.polyu.edu.hk/~csajaykr/tvmiris.htm)
-  PRL_Haindl_Krupicka: [Unsupervised detection of non-iris occlusions](https://www.sciencedirect.com/science/article/pii/S0167865515000604) , [code](https://ars.els-cdn.com/content/image/1-s2.0-S0167865515000604-mmc1.zip)

### Evaluation protocols
  please see [IrisParseNet](https://github.com/xiamenwcy/IrisParseNet)

### Questions
Please contact caiyong.wang@cripac.ia.ac.cn

