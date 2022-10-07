Goal of this repository is to create determine feasible grasp positions and orientations using a spherically transformed dataset.

Model Part 1 is the model used to determine the likelihood maps asssociated with the probability of the absolute position of a grasp occuring. This is predicted given a 2-D input of the spherically transformed mesh.

Model Part 2 is the meta-learned (using MAML) model that was tasks with updating the likelihood maps produced in Model Part 1 according to demonstrated human grasps.
Model Part 2 also outputs the maximum liklihood angles (azimuth, zenith and our rotational angle gamma).
As input Model Part 2 takes both the spherically transformed mesh image and the priors produced in Model part 1.

For data Generation used for these models, see my other repository:
https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes

