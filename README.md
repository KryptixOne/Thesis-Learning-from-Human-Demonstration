# Robotic Task Learning From Human Demonstrations using Spherical Representations

The repository at hand is designed into two modular DNNs such that they can ultimately be use used as in the following figure

<img src="https://github.com/KryptixOne/Thesis-Learning-from-Human-Demonstration/blob/main/Images/Pipeline%20General.PNG" width="600" />


## Model Part I


<img src="https://github.com/KryptixOne/Thesis-Learning-from-Human-Demonstration/blob/main/Images/FinalModelPartI.PNG" width="600" />

Model Part 1 is the name given to the model used to determine the likelihood maps asssociated with the probability
of the absolute position of a grasp occuring. This prediction is given based off of an input 2-D image which was created via a hemispherical transform on a 3-D object mesh.

Visual results of such predictions using Model Part I are as depicted:

<img src="https://github.com/KryptixOne/Thesis-Learning-from-Human-Demonstration/blob/main/Images/FinalModelPart1_VisualResults.PNG" width="600" />

## Model Part II

<img src="https://github.com/KryptixOne/Thesis-Learning-from-Human-Demonstration/blob/main/Images/ModelForPartII.PNG" width="600" />

Model Part 2 is the meta-learned (using MAML) model that was tasks with updating the likelihood maps produced in Model Part 1 according to demonstrated human grasps.
Model Part 2 also outputs the maximum liklihood angles (azimuth, zenith and our rotational angle gamma).
As input Model Part 2 takes both the spherically transformed mesh image and the priors produced in Model part 1.


## Data Generator:
For data Generation used for these models, see my other repository:
https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes

