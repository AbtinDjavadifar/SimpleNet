# MultiNet: An Image Classification DCNN for Wrinkle Detection

This repository contains the pieces of code used in our paper titled "Robot-assisted composite manufacturing based on machine learning applied to multi-view computer vision".

This paper introduces an automated wrinkle detection methodon  semi-finished  fiber  products  in  the  aerospace  manufacturing  industry.Machine learning, computer vision techniques, and evidential reasoning arecombined to detect wrinkles during the draping process of fibre-reinforcedmaterials with an industrial robot. A well-performing Deep ConvolutionalNeural Network (DCNN) was developed based on a preliminary, hand-labelleddataset captured on a functioning robotic system used in a composite manu-facturing facility. Generalization of this model to different, unlearned wrinklefeatures naturally compromises detection accuracy. To alleviate this prob-lem,  the  proposed  method  employs  computer  vision  techniques  and  belieffunctions  to  enhance  defect  detection  accuracy.  Co-temporal  views  of  thesame fabric are extracted, and individual detection results obtained from theDCNN are fused using the Dempster-Shafer Theory (DST). By the applica-tion of the DST rule of combination, the overall wrinkle detection accuracyfor the generalized case is greatly improved in this composite manufacturingfacility.

<p align="center">
    <img src="imgs/Cam_1_Pic_19-07-52.jpg" width="700px"></br>
</p>

## Citation
If you use this code or the MultiNet architecture for your research, please cite our paper.
```
@inproceedings{
}
```

## Prerequisites
- Python 2 or 3
- CPU or NVIDIA GPU + CUDA CuDNN
