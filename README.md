# WHU-Stereo
This repository contains:

I. A large-scale dataset named WHU-Stereo for stereo matching of high-resolution satellite imagery.

II. Several deep learning methods (as well as the tool for disparity accuracy evaluation) for stereo matching.

## Dataset
This work is done by the team of Prof. Wanshou Jiang in State Key Laboratory of Information Engineering in Surveying, Mapping and Remote Sensing, Wuhan University, Wuhan, China. Please see: http://openrs.whu.edu.cn/md/members/jws/jws.html.

#### Links

The dataset can be downloaded from:

Baidu drive: https://pan.baidu.com/s/1SF2RRIRJeP8TbKMoDSL0OQ?pwd=xbyx

or

Google drive: https://drive.google.com/drive/folders/1mw6PrPRidDxP1OtS3_fgblv4T5x44I_k

#### Directory

The directory of the data is as follows:

    with ground truth
      test
        disp
        left
        right
      train
        disp
        left
        right
      val
        disp
        left
        right
    without ground truth
      left
      right

Stereo pairs with ground-truth disparity maps are stored in the directory "with ground truth", we have splitted them into three subsets. Satellite images with ground-truth labels are collected from six cities, namely Wuhan, Hengyang, Shaoguan, Kunming, Yingde, and Qichun. Images and labels are prefixed with abbreviations of city names. Stereo pairs without ground-truth disparity maps are stored in the directory "without ground truth". For details, please refer to readme.xlsx.

## Deep learning methods
The methods include:

i. "StereoNet: Guided Hierarchical Refinement for Real-Time Edge-Aware Refinement for Real-Time Edge-Aware" by Sameh Khamis, Sean Fanello, Christoph Rhemann, Adarsh Kowdle, Julien Valentin, and Shahram Izadi, in ECCV 2018.

ii. "Pyramid Stereo Matching Network" by Jia-Ren Chang and Yong-Sheng Chen, in CVPR 2018.

iii. "HMSM-Net: Hierarchical multi-scale matching network for disparity estimation of high-resolution satellite stereo images" by Sheng He, Shenhong Li, San Jiang, and Wanshou Jiang, in ISPRS Journal of Photogrammetry and Remote Sensing, 2022.

Note that the code is completed by referring to the original open-source code and may be a little different from the original papers.

The file "readme.xlsx" describes the numbers of samples of each city.

Development environment: CUDA 11.2, TensorFlow 2.5.0, Python 3.7.

## Citation
If you find this work helpful to your research, please cite:

@ARTICLE{10044710,

  author={Li, Shenhong and He, Sheng and Jiang, San and Jiang, Wanshou and Zhang, Lin},
  
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  
  title={WHU-Stereo: A Challenging Benchmark for Stereo Matching of High-Resolution Satellite Images}, 
  
  year={2023},
  
  volume={},
  
  number={},
  
  pages={1-1},
  
  doi={10.1109/TGRS.2023.3245205}}
