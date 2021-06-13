# Rotation-Invariant-Object-Detector

Object detection has become one of the key computer technologies in the 21st century. From face 
detection to self-driving cars, humans are using object detection way more and are showing no way 
back. But the existing technologies have a limitation that they cannot detect properly when the objects 
are tilted. In real-life situations, we can't expect the images to be perfectly oriented because of 
positional disturbance of the camera during capture, the image might not be taken by a photographer. 
Moreover, few scenes obtain significant views while capturing in the landscape while some in portrait. 
When all those images are passed through the same object detector it might not extract features 
properly resulting in incorrect predictions.
 
We have come up with 2 different solutions implementing 2 papers for the above problem.
Our first solution proposes training a pre-trained Resnet50 with perfectly oriented images 
rotated at random angles by replacing the model’s top layer with another dense layer which classifies 
into 360 classes, each class covering one degree. We solved this as a classification problem, where 
our model produces a vector of 360 values, each representing the probability at which the image is 
oriented at that particular angle. Finally, we used the Object detector powered by the Resnet50 model 
to identify the objects.
In the second solution, we have come up with a novel two-step architecture, which efficiently 
detects multiple objects at any angle in an image efficiently. We utilize eigenvector analysis on the 
input image based on bright pixel distribution. The vertical and horizontal vectors are used as a 
reference to detect the deviation of an image from the original orientation. This analysis gives four 
orientations of the input image which pass through a pre-trained Object Detector using Resnet50 with 
proposed decision criteria. Our approach, referred to as “Eigen Vectors based Rotation Invariant 
Multi-Object Deep Detector” (EVRI-MODD), produces rotation invariant detection without any 
additional training on augmented data and also determines actual image orientation without any prior 
information. The proposed network achieves high performance on Pascal-VOC 2012 dataset. We 
evaluate our network performance by rotating input images at random angles between 90° and 270°, 
and achieve a significant gain in accuracy by 43% over Resnet50.
