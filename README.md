# MultiNet: An Image Classification DCNN for Wrinkle Detection

This repository contains the pieces of code used in our paper titled "Robot-assisted composite manufacturing based on machine learning applied to multi-view computer vision". This paper introduces an automated wrinkle detection methodon semi-finished fiber products in the  aerospace manufacturing industry. Machine learning, computer vision techniques, and evidential reasoning are combined to detect wrinkles during the draping process of fibre-reinforced materials with an industrial robot. A well-performing Deep Convolutional Neural Network (DCNN) was developed based on a preliminary, hand-labelled dataset captured on a functioning robotic system used in a composite manu-facturing facility. Generalization of this model to different, unlearned wrinkle features naturally compromises detection accuracy. To alleviate this problem, the proposed method employs computer vision techniques and belief functions to enhance defect detection accuracy. Co-temporal views of the same fabric are extracted, and individual detection results obtained from the DCNN are fused using the Dempster-Shafer Theory (DST). By the application of the DST rule of combination, the overall wrinkle detection accuracy for the generalized case is greatly improved in this composite manufacturing facility.


<p align="center">
    <img src="imgs/Cam_1_Pic_19-07-52.jpg" width="500px"></br>
</p>

## Citation
If you use this code or the MultiNet architecture for your research, please cite our paper.
```
@inproceedings{djavadifar2019smart,
  title={Robot-assisted composite manufacturing based on machine learning applied to multi-view computer vision},
  author={Djavadifar, Abtin and Graham-Knight, John Brandon and Gupta, Kashish and Körber, Marian and Lasserre, Patricia and Najjaran, Homayoun},
  booktitle={International Conference on Smart Multimedia},
  year={2019},
  organization={Springer}
}
```

## Prerequisites
- Python 2 or 3
- CPU or NVIDIA GPU + CUDA CuDNN
