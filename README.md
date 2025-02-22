# Robotic Task Learning from Human Demonstrations using Spherical Representations

Full project paper and work available [here](https://drive.google.com/file/d/13vBX1J4_3KLHbya9zJw0oxTyxLq4XuAt/view).

This repository contains two modular deep neural networks (DNNs) designed for robotic task learning from human demonstrations using spherical representations. The models work in tandem as shown in the figure below:

<img src="https://github.com/KryptixOne/Thesis-Learning-from-Human-Demonstration/blob/main/Images/Pipeline%20General.PNG" width="800" />

## Model Part I

<img src="https://github.com/KryptixOne/Thesis-Learning-from-Human-Demonstration/blob/main/Images/FinalModelPartI.PNG" width="800" />

Model Part I predicts likelihood maps representing the probability distribution of grasp positions. The model takes a 2D image as input, generated via a hemispherical transformation of a 3D object mesh, and outputs likelihood estimates of where a grasp is most likely to occur.

### Visual Results
Predictions from Model Part I are depicted below:

<img src="https://github.com/KryptixOne/Thesis-Learning-from-Human-Demonstration/blob/main/Images/FinalModelPart1_VisualResults.PNG" width="800" />

## Model Part II

<img src="https://github.com/KryptixOne/Thesis-Learning-from-Human-Demonstration/blob/main/Images/ModelForPartII.PNG" width="800" />

Model Part II is a meta-learned model trained using First-Order MAML (FOMAML). It refines the likelihood maps produced by Model Part I based on human demonstration data. Additionally, it outputs maximum likelihood grasp angles, including azimuth, zenith, and a rotational angle (γ). Model Part II takes as input both the spherically transformed mesh image and the likelihood priors from Model Part I.

### Task Augmentation and Training Insights
While training Model Part II with FOMAML, we explored different task augmentation strategies:
- **Effective Augmentation:** Adding discrete noise to angular data improved adaptability without degrading performance.
- **Ineffective Augmentation:** Modifying labeled likelihood maps negatively impacted the model’s flexibility.

#### Effect of Noise on MAML Training

<img src="https://github.com/KryptixOne/Thesis-Learning-from-Human-Demonstration/blob/main/Images/EffectOfNoiseOnMAML.png" width="800" />
*(a) Labeled data vs. (b) Predicted data*

## Data Generation
The dataset used for training these models was generated using a custom pipeline. Details can be found in the following repository:
[Spherical Data Generation for 3D Meshes](https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes).

---

This repository provides an approach to robotic grasp learning through human demonstrations, leveraging spherical representations and meta-learning techniques. Contributions, issues, and discussions are welcome!

