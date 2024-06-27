Image Tiling
============

Introduction
############

Motivation
**********
Autonomous driving and aerial drone photography are two very important applications for computer vision. 
Autonomous vehicles, if possible, will reduce traffic fatalities and allow for more comfortable transit, and aeiral drone photography is used extensively in spying and warfare.

Thus, with large financial incentives behind them, these two sources of data have shaped a small niche of computer vision research due to one factor that they both share: *objects in these kinds of datasets are very small relative to the frame*.

.. image:: ../assets/aerial_photo.png
   :width: 100%

(Example of object detection used in aerial photography. Credit to [#]_)

The Technical Issue of Small Objects
************************************
Object detectors, in general, struggle with small object detection [#]_. 
The reasons for this are due to varying factors depending on the exact approach the object detector employs.  

For any object detector which uses convolution as a part of the neural network architecture, their convolutional layers will shrink the size of objects as they pass their outputs to the next layers.
This causes already small objects to shrink even further, which further causes later convolutional kernels to struggle to learn meaningful features.  

A second problem emerges for single shot multibox detectors (SSDs), arguably the most commonly used object detectors [#]_. 

Rather than using a region proposal algorithm, SSDs impose a grid of small squares on the image.
Each grid tile can be responsible for the center of one object. 
Thus, if objects are too small and there are two or more objects in one grid tile, only one will recieve a bounding box, causing accuracy to be poor.  

.. image:: ../assets/yolo_v1_diagram.png
    :width: 100%

(A visual explanation of the YOLO single shot detector's algorithm. A bounding box's center must be 'owned' by one and only one square in the SxS grid. Credit to [#]_.)

Sources
#######
.. [#] https://github.com/SOTIF-AVLab/SinD
.. [#] https://doi.org/10.1155/2020/3189691
.. [#] https://arxiv.org/abs/1512.02325
.. [#] https://leimao.github.io/blog/YOLOs/
.. [#] https://doi.org/10.1109/CVPRW.2019.00084
