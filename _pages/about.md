---
permalink: /
title: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---
Hi, my name is Jun Dan.

I'm a 5th-year Ph.D. student at [Zhejiang University](https://www.zju.edu.cn/).\
Previously, I graduated with a bachelor’s degree in 2020 from the School of Microelectronics and Communication Engineering at [Chongqing University](https://www.cqu.edu.cn/).

**Research Interests:**
* Transfer Learning
* Face Perception and Understanding
* LMMs
* Data-Centric AI

Publications
======
* **TFGDA: Exploring Topology and Feature Alignment in Semi-supervised Graph Domain Adaptation through Robust Clustering**. [[PDF]](https://openreview.net/forum?id=26BdXIY3ik&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2024%2FConference%2FAuthors%23your-submissions))\
**Dan J**, Liu W, Tan Y, et al.
<em>Accepted by NeurIPS 2024, Main Track.</em>

* **TopoFR: A Closer Look at Topology Alignment on Face Recognition**. [[PDF]](https://openreview.net/forum?id=KVAx5tys2p&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2024%2FConference%2FAuthors%23your-submissions)) [[CODE]](https://github.com/modelscope/facechain/tree/main/face_module/TopoFR)\
**Dan J**, Liu Y, Deng J, et al. 
<em>Accepted by NeurIPS 2024, Main Track.</em>

* **TransFace: Calibrating Transformer Training for Face Recognition from a Data-Centric Perspective**. [[PDF]](https://openaccess.thecvf.com/content/ICCV2023/html/Dan_TransFace_Calibrating_Transformer_Training_for_Face_Recognition_from_a_Data-Centric_ICCV_2023_paper.html) [[CODE]](https://github.com/DanJun6737/TransFace)\
**Dan J**, Liu Y, Xie H, et al. 
<em>Proceedings of the IEEE/CVF international conference on computer vision (ICCV). 2023: 20642-20653.</em>

* **HOGDA: Boosting Semi-supervised Graph Domain Adaptation via High-Order Structure-Guided Adaptive Feature Alignment**. [[PDF]](https://openreview.net/forum?id=2KjnPzj8gf)\
**Dan J**, Liu W, Liu M, et al. 
<em>Accepted by ACM Multimedia (ACM MM) 2024.</em>

* **PIRN: Phase Invariant Reconstruction Network for Infrared Image Super-Resolution**. [[PDF]](https://www.sciencedirect.com/science/article/abs/pii/S0925231224009925)\
**Dan J**, Jin T, Chi H, et al. 
<em>Neurocomputing, 2024, 599: 128221.</em>

* **Similar Norm More Transferable: Rethinking Feature Norms Discrepancy in Adversarial Domain Adaptation**. [[PDF]](https://www.sciencedirect.com/science/article/abs/pii/S0950705124005422)\
**Dan J**, Liu M, Xie C, et al. 
<em>Knowledge-Based Systems, 2024, 296: 111908.</em>

* **Trust-aware Conditional Adversarial Domain Adaptation with Feature Norm Alignment**. [[PDF]](https://www.sciencedirect.com/science/article/abs/pii/S0893608023005543)\
**Dan J**, Jin T, Chi H, et al. 
<em>Neural Networks, 2023, 168: 518-530.</em>

* **HOMDA: High-Order Moment-Based Domain Alignment for unsupervised domain adaptation**. [[PDF]](https://www.sciencedirect.com/science/article/abs/pii/S0950705122013016)\
**Dan J**, Jin T, Chi H, et al. 
<em>Knowledge-Based Systems, 2023, 261: 110205.</em>

* **Uncertainty-guided Joint Unbalanced Optimal Transport for Unsupervised Domain Adaptation**. [[PDF]](https://link.springer.com/article/10.1007/s00521-022-07976-x)\
**Dan J**, Jin T, Chi H, et al. 
<em>Neural Computing and Applications, 2023, 35(7): 5351-5367.</em>

* **CM-UNet: Hybrid CNN-Mamba UNet for Remote Sensing Image Semantic Segmentation**. [[PDF]](https://arxiv.org/abs/2405.10530) [[CODE]](https://github.com/XiaoBuL/CM-UNet)\
Liu M, **Dan J**, Lu Z, et al. 
<em>arXiv preprint arXiv:2405.10530, 2024.</em>

News
======
- 🚀🚀🚀 TransFace is integrated in [FaceChain-FACT](https://github.com/modelscope/facechain) as a key identity-preserved module to assist Stable Diffusion in generating human portraits with fine-grained facial details and diverse styles.
In the newest FaceChain-FACT (Face Adapter with deCoupled Training) version, with only 1 photo and 10 seconds, you can generate personal portraits in different settings (multiple styles now supported!). (May 28th, 2024 UTC)

<a href='https://facechain-fact.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/DHqEl0qwi-M?si=y6VpInXdhIX0HpbI)

 The entire framework of [FaceChain-FACT](https://github.com/modelscope/facechain) is shown in the figure below.

![image](docs/framework.png)

## ModelScope
You can quickly experience and invoke our TransFace model on the [ModelScope](https://modelscope.cn/models/damo/cv_vit_face-recognition/summary).

* Quickly utilize our model as a feature extractor to extract facial features from the input image.
```
# Usage: Input aligned facial images (112x112) to obtain a 512-dimensional facial feature vector.
# For convenience, the model integrates the RetinaFace model for face detection and keypoint estimation.
# Provide two images as input, and for each image, the model will independently perform face detection,
# select the largest face, align it, and extract the corresponding facial features.
# Finally, the model will return a similarity score indicating the resemblance between the two faces.

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import numpy as np

face_mask_recognition_func = pipeline(Tasks.face_recognition, 'damo/cv_vit_face-recognition')
img1 = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/face_recognition_1.png'
img2 = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/face_recognition_2.png'
emb1 = face_mask_recognition_func(img1)[OutputKeys.IMG_EMBEDDING]
emb2 = face_mask_recognition_func(img2)[OutputKeys.IMG_EMBEDDING]
sim = np.dot(emb1[0], emb2[0])
print(f'Face cosine similarity={sim:.3f}, img1:{img1}  img2:{img2}')
```

Awards & Honors
======
* National Scholarship, 2024.
* 3PEAK Corporate Scholarship, 2024.
* Five-Star Graduate Student of Zhejiang University, 2023 & 2024.
* Outstanding Graduate Student of Zhejiang University. 2021 & 2023 & 2024.
* Outstanding Graduate of Chongqing City, 2020. (1% students awarded)
* Outstanding Graduate of Chongqing University, 2020.
* National Encouragement Scholarship, 2017 & 2018 & 2019. 
* Comprehensive Scholarship for Outstanding Students of Chongqing University, 2017 & 2018 & 2019 & 2020.



