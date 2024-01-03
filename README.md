Depth Estimation Papers
===

A collection of depth estimation papers.

# Outline

- [Monocular Depth Estimation](#1-Monocular-Depth-Estimation)
  - [Supervised](#11-Supervised)
  - [Self-Supervised](#12-Self-Supervised)
- [Multi-Focus Image Based Depth Estimation](#2-Multi-Focus-Image-Based-Depth-Estimation)
- [Survey](#3-Survey)
- [Datasets](#4-Datasets)
  - [Outdoor](#41-Outdoor)
  - [Indoor](#42-Indoor)
  - [Both](#43-Both)

# 1. Monocular Depth Estimation

## 1.1 Supervised

| Paper | Source      | Resource | Comment |
| --- |-------------| --- | --- |
| [Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation](https://arxiv.org/abs/2312.02145) | arXiv 2023 | [[Code]](https://github.com/prs-eth/marigold) | Zero-Shot |
| [iDisc: Internal Discretization for Monocular Depth Estimation](https://openaccess.thecvf.com/content/CVPR2023/papers/Piccinelli_iDisc_Internal_Discretization_for_Monocular_Depth_Estimation_CVPR_2023_paper.pdf) | CVPR 2023 | [[Code]](https://github.com/SysCV/idisc) | High-Level Patterns |
| [Trap Attention: Monocular Depth Estimation with Manual Traps](https://openaccess.thecvf.com/content/CVPR2023/papers/Ning_Trap_Attention_Monocular_Depth_Estimation_With_Manual_Traps_CVPR_2023_paper.pdf) | CVPR 2023 | [[Code]](https://github.com/ICSResearch/TrapAttention) | Transformer, Lightweight |
| [NDDepth: Normal-Distance Assisted Monocular Depth Estimation](https://openaccess.thecvf.com/content/ICCV2023/papers/Shao_NDDepth_Normal-Distance_Assisted_Monocular_Depth_Estimation_ICCV_2023_paper.pdf) | ICCV 2023 | [[Code]](https://github.com/ShuweiShao/NDDepth) | Scene Geometry Constraint |
| [Learning Depth Estimation for Transparent and Mirror Surfaces](https://openaccess.thecvf.com/content/ICCV2023/papers/Costanzino_Learning_Depth_Estimation_for_Transparent_and_Mirror_Surfaces_ICCV_2023_paper.pdf) | ICCV 2023 | [[Code]](https://github.com/CVLAB-Unibo/Depth4ToM-code#-learning-depth-estimation-for-transparent-and-mirror-surfaces-iccv-2023-) | Transparent and Mirror Surfaces |
| [Towards Zero-Shot Scale-Aware Monocular Depth Estimation](https://openaccess.thecvf.com/content/ICCV2023/papers/Guizilini_Towards_Zero-Shot_Scale-Aware_Monocular_Depth_Estimation_ICCV_2023_paper.pdf) | ICCV 2023 | [[Code]](https://github.com/tri-ml/vidar) | Zero-Shot |
| [GEDepth: Ground Embedding for Monocular Depth Estimation](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_GEDepth_Ground_Embedding_for_Monocular_Depth_Estimation_ICCV_2023_paper.pdf) | ICCV 2023 | [[Code]](https://github.com/qcraftai/gedepth) | Zero-Shot |
| [IEBins: Iterative Elastic Bins for Monocular Depth Estimation](https://arxiv.org/pdf/2309.14137.pdf) | NIPS 2023 | [[Code]](https://github.com/ShuweiShao/IEBins) | Classification-Regression-Based |
| [The Surprising Effectiveness of Diffusion Models for Optical Flow and Monocular Depth Estimation](https://arxiv.org/pdf/2306.01923.pdf) | NIPS 2023 |  | Diffusion Model |
| [Deep Ordinal Regression Network for Monocular Depth Estimation](https://openaccess.thecvf.com/content_cvpr_2018/papers/Fu_Deep_Ordinal_Regression_CVPR_2018_paper.pdf) | CVPR 2018 |  | First Work to Turn Regression to Classification |
| [Multi-Scale Continuous CRFs as Sequential Deep Networks for Monocular Depth Estimation](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xu_Multi-Scale_Continuous_CRFs_CVPR_2017_paper.pdf) | CVPR 2017 |  |  |
| [Predicting Depth, Surface Normals and Semantic Labels With a Common Multi-Scale Convolutional Architecture](https://openaccess.thecvf.com/content_iccv_2015/papers/Eigen_Predicting_Depth_Surface_ICCV_2015_paper.pdf) | ICCV 2015 |  |  |
| [Depth map prediction from a single image using a multi-scale deep network](https://proceedings.neurips.cc/paper_files/paper/2014/file/7bccfde7714a1ebadf06c5f4cea752c1-Paper.pdf) | NIPS 2014 |  | First Work to Utilize DL for Monocular Depth Estimation |

## 1.2 Self-Supervised

| Paper | Source      | Resource | Comment |
| --- |-------------| --- | --- |
| [Lite-Mono: A Lightweight CNN and Transformer Architecture for Self-Supervised Monocular Depth Estimation](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Lite-Mono_A_Lightweight_CNN_and_Transformer_Architecture_for_Self-Supervised_Monocular_CVPR_2023_paper.pdf) | CVPR 2023 | [[Code]](https://github.com/noahzn/Lite-Mono) | CNN+Transformer, Lightweight |
| [Deep Digging into the Generalization of Self-Supervised Monocular Depth Estimation](https://ojs.aaai.org/index.php/AAAI/article/view/25090) | AAAI 2023 |  | CNN+Transformer |
| [Self-supervised Monocular Depth Estimation: Let’s Talk About The Weather](https://openaccess.thecvf.com/content/ICCV2023/papers/Saunders_Self-supervised_Monocular_Depth_Estimation_Lets_Talk_About_The_Weather_ICCV_2023_paper.pdf) | ICCV 2023 | [[Code]](https://github.com/kieran514/robustdepth) | Adverse Weather |
| [Robust Monocular Depth Estimation under Challenging Conditions](https://openaccess.thecvf.com/content/ICCV2023/papers/Gasperini_Robust_Monocular_Depth_Estimation_under_Challenging_Conditions_ICCV_2023_paper.pdf) | ICCV 2023 | [[Code]](https://md4all.github.io/) | Adverse Weather |
| [Self-Supervised Monocular Depth Estimation by Direction-aware Cumulative Convolution Network](https://openaccess.thecvf.com/content/ICCV2023/papers/Han_Self-Supervised_Monocular_Depth_Estimation_by_Direction-aware_Cumulative_Convolution_Network_ICCV_2023_paper.pdf) | ICCV 2023 |  | Explore Direction Sensitivity in Self-Supervision |
| [Robust Geometry-Preserving Depth Estimation Using Differentiable Rendering](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Robust_Geometry-Preserving_Depth_Estimation_Using_Differentiable_Rendering_ICCV_2023_paper.pdf) | ICCV 2023 |  | Geometry Consistency |
| [3D Distillation:Improving Self-Supervised Monocular Depth Estimation on Reflective Surfaces](https://openaccess.thecvf.com/content/ICCV2023/papers/Shi_3D_Distillation_Improving_Self-Supervised_Monocular_Depth_Estimation_on_Reflective_Surfaces_ICCV_2023_paper.pdf) | ICCV 2023 |  | Reflective Surfaces |
| [Out-of-Distribution Detection for Monocular Depth Estimation](https://openaccess.thecvf.com/content/ICCV2023/papers/Hornauer_Out-of-Distribution_Detection_for_Monocular_Depth_Estimation_ICCV_2023_paper.pdf) | ICCV 2023 | [[Code]](https://github.com/jhornauer/mde_ood) | Out-of-Distribution Detection |
| [MonoViT: Self-Supervised Monocular Depth Estimation with a Vision Transformer](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10044409) | 3DV 2022 | [[Code]](https://github.com/zxcqlf/MonoViT) | Transformer |
| [Transformers in Self-Supervised Monocular Depth Estimation with Unknown Camera Intrinsics](https://arxiv.org/abs/2202.03131) | arXiv 2022 |  | Transformer |
| [Channel-Wise Attention-Based Network for Self-Supervised Monocular Depth Estimation](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9665890) | 3DV 2021 | [[Code]](https://github.com/kamiLight/CADepth-master) | Channel Attention |
| [R-MSFM: Recurrent Multi-Scale Feature Modulation for Monocular Depth Estimating](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_R-MSFM_Recurrent_Multi-Scale_Feature_Modulation_for_Monocular_Depth_Estimating_ICCV_2021_paper.pdf) | ICCV 2021 | [[Code]](https://github.com/jsczzzk/R-MSFM) | Feature Modulation Module |
| [Digging Into Self-Supervised Monocular Depth Estimation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Godard_Digging_Into_Self-Supervised_Monocular_Depth_Estimation_ICCV_2019_paper.pdf) | ICCV 2019 | [[Code]](https://github.com/nianticlabs/monodepth2) | Monodepth2 |
| [Unsupervised Monocular Depth Estimation With Left-Right Consistency](https://openaccess.thecvf.com/content_cvpr_2017/papers/Godard_Unsupervised_Monocular_Depth_CVPR_2017_paper.pdf) | CVPR 2017 | [[Code]](https://github.com/mrharicot/monodepth) | Monodepth, First Stereo-Based Self-Supervised |
| [Unsupervised Learning of Depth and Ego-Motion from Video](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_Unsupervised_Learning_of_CVPR_2017_paper.pdf) | CVPR 2017 | [[Code]](https://github.com/tinghuiz/SfMLearner) | First Video-Based Self-Supervised |
| [Unsupervised CNN for Single View Depth Estimation: Geometry to the Rescue](https://link.springer.com/chapter/10.1007/978-3-319-46484-8_45) | ECCV 2016 |  |  |
---

# 2. Multi-Focus Image Based Depth Estimation

| Paper | Source      | Resource |
| --- |-------------| --- |
| [Fully Self-Supervised Depth Estimation from Defocus Clue](https://openaccess.thecvf.com/content/CVPR2023/papers/Si_Fully_Self-Supervised_Depth_Estimation_From_Defocus_Clue_CVPR_2023_paper.pdf) | CVPR 2023 | [[Code]](https://github.com/Ehzoahis/DEReD) |
| [Bridging Unsupervised and Supervised Depth From Focus via All-in-Focus Supervision](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Bridging_Unsupervised_and_Supervised_Depth_From_Focus_via_All-in-Focus_Supervision_ICCV_2021_paper.pdf) | ICCV 2021 | [[Code]](https://github.com/albert100121/AiFDepthNet) |
---

# 3. Survey

| Paper | Source    | Comment |
| --- |-----------| --- |
| [RoboDepth: Robust Out-of-Distribution Depth Estimation under Corruptions](https://arxiv.org/pdf/2310.15171.pdf) | NIPS 2023 | Investigation and Benchmark of Out-of-Distribution in Monocular Depth Estimation |

---

# 4. Datasets

## 4.1 Outdoor
- [KITTI](https://www.cvlibs.net/datasets/kitti/index.php)
- [nuScenes](https://www.nuscenes.org/)
- [Make3D](http://make3d.cs.cornell.edu/data.html#make3d)
- [DDAD](https://github.com/TRI-ML/DDAD?tab=readme-ov-file)
- [Oxford RobotCar](https://robotcar-dataset.robots.ox.ac.uk/)

## 4.2 Indoor
- [NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
- [7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
- [ScanNet](http://www.scan-net.org/)

## 4.3 Both
- [ETH3D](https://www.eth3d.net/datasets)
- [DIODE](https://diode-dataset.org/)
