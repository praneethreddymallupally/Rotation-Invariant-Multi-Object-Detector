import cv2 
import numpy as np
from numpy import linalg as LA
from sympy import Matrix, pretty
import math
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from keras.utils.data_utils import GeneratorEnqueuer
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import math, os
from imageai.Detection import ObjectDetection
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions


def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h),borderValue=(0,128,0))

    return rotated

def covarince(x,y):
    ag=0
    for i in range(len(x)):
        ag+=x[i]*y[i]
    ag/=(len(x)-1)    
    return ag

def CorrectionMain(inputImg):
    image = cv2.imread('/content/drive/MyDrive/MajorProjectBP1/Dataset/Good Dog Images/DogImageForEigenVector.jpg') 
    image=inputImg
    image=rotate(image,60)
    gi = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    ret, bi = cv2.threshold(gi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    mat=[]
    x=[]
    y=[]
    for i in range(len(bi)):
        for j in range(len(bi[0])):
            if(bi[i][j]==255):
                x.append(i)
                y.append(j)
                mat.append([i,j])
    xmean=0
    ymean=0
    xmax=-1
    ymax=-1
    xmin=9999999999999
    ymin=9999999999999

    for row in mat:
        xmean+=row[0]
        xmax=max(xmax,row[0])
        xmin=min(xmin,row[0])
        ymean+=row[1]
        ymax=max(ymax,row[1])
        ymin=min(ymin,row[1])


    xmean=xmean/len(mat)
    ymean=ymean/len(mat)

    mat2=[]
    xut=[]
    yut=[]
    for row in mat:
        mat2.append([(row[0]-xmean)/(xmax-xmin),(row[1]-ymean)/(ymax-ymin)])
        xut.append((row[0]-xmean)/(xmax-xmin))
        yut.append((row[1]-ymean)/(ymax-ymin))

    covxy=covarince(xut,yut)
    covxx=covarince(xut,xut)
    covyy=covarince(yut,yut)

    covar=[[covxx,covxy],[covxy,covyy]]
    # print("Printing eigen values and vectors")
    w, v = LA.eig(covar)
    # print(w)
    # print(v)
    CorrectionAngle=math.atan(v[0][0]/v[1][0])*57.2958
    # print ("Correction angle :",CorrectionAngle)
    alpha = math.atan(v[0][0]/v[1][0])*57.295
    beta = abs(90 + alpha)
    image1=rotate(image,alpha)
    image2=rotate(image,180+alpha)
    image3=rotate(image,beta)
    image4=rotate(image,180+beta)
    return CorrectionAngle




def test_eigenvectorsmodel(InputImage):
    image=cv2.imread(InputImage)
    originalImg=image.copy()
    rangle=np.random.randint(90,270)
    rangle=184
    image=rotate(image, rangle)
    rotatedImage=image.copy()
    CorrectionAngle=CorrectionMain(image)
    alpha=CorrectionAngle
    beta=abs(90+alpha)
    image1=rotate(image,alpha)
    image2=rotate(image,180+alpha)
    image3=rotate(image,beta)
    image4=rotate(image,180+beta)

    cv2.imwrite("/content/drive/MyDrive/MajorProjectBP1/Dataset/sampletestoutputs/image1.jpeg", image1)
    cv2.imwrite('/content/drive/MyDrive/MajorProjectBP1/Dataset/sampletestoutputs/image2.jpeg', image2)
    cv2.imwrite('/content/drive/MyDrive/MajorProjectBP1/Dataset/sampletestoutputs/image3.jpeg', image3)
    cv2.imwrite('/content/drive/MyDrive/MajorProjectBP1/Dataset/sampletestoutputs/image4.jpeg', image4)

    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath('/content/drive/MyDrive/MajorProjectBP1/resnet50_coco_best_v2.1.0.h5')
    detector.loadModel()
    min_prob=30
    detections=[]
    detections.append(detector.detectObjectsFromImage(input_image="/content/drive/MyDrive/MajorProjectBP1/Dataset/sampletestoutputs/image1.jpeg", output_image_path="/content/drive/MyDrive/MajorProjectBP1/Dataset/sampletestoutputs/image5.jpg", minimum_percentage_probability=min_prob))
    detections.append(detector.detectObjectsFromImage(input_image="/content/drive/MyDrive/MajorProjectBP1/Dataset/sampletestoutputs/image2.jpeg", output_image_path="/content/drive/MyDrive/MajorProjectBP1/Dataset/sampletestoutputs/image6.jpg", minimum_percentage_probability=min_prob))
    detections.append(detector.detectObjectsFromImage(input_image="/content/drive/MyDrive/MajorProjectBP1/Dataset/sampletestoutputs/image3.jpeg", output_image_path="/content/drive/MyDrive/MajorProjectBP1/Dataset/sampletestoutputs/image7.jpg", minimum_percentage_probability=min_prob))
    detections.append(detector.detectObjectsFromImage(input_image="/content/drive/MyDrive/MajorProjectBP1/Dataset/sampletestoutputs/image4.jpeg", output_image_path="/content/drive/MyDrive/MajorProjectBP1/Dataset/sampletestoutputs/image8.jpg", minimum_percentage_probability=min_prob))
    image5 = cv2.imread('/content/drive/MyDrive/MajorProjectBP1/Dataset/sampletestoutputs/image5.jpg') 
    image6 = cv2.imread('/content/drive/MyDrive/MajorProjectBP1/Dataset/sampletestoutputs/image6.jpg') 
    image7 = cv2.imread('/content/drive/MyDrive/MajorProjectBP1/Dataset/sampletestoutputs/image7.jpg') 
    image8 = cv2.imread('/content/drive/MyDrive/MajorProjectBP1/Dataset/sampletestoutputs/image8.jpg') 
    countDict=dict()
    for detection in detections:
        for object in detection:
            if object.get('name') not in countDict.keys():
                countDict[object.get('name')]=1
            else:
                countDict[object.get('name')]+=1
    MostFreqClass=max(countDict,key=countDict.get)
    maxFreqClassImg=0
    maxProb=0
    i=1
    for detection in detections:
        for object in detection:
            if object.get('name')==MostFreqClass:
                temp=object.get('percentage_probability')
                if temp!=None and temp>maxProb:
                    maxProb=temp
                    maxFreqClassImg=i
        i+=1
    finalCorrectedImage= cv2.imread('/content/drive/MyDrive/MajorProjectBP1/Dataset/sampletestoutputs/image'+str(maxFreqClassImg)+'.jpeg') 
    cv2.imwrite("/content/drive/MyDrive/MajorProjectBP1/Dataset/sampletestoutputs/image9.jpeg",originalImg)
    OriginalDetection=detector.detectObjectsFromImage(input_image="/content/drive/MyDrive/MajorProjectBP1/Dataset/sampletestoutputs/image9.jpeg", output_image_path="/content/drive/MyDrive/MajorProjectBP1/Dataset/sampletestoutputs/image11.jpg", minimum_percentage_probability=min_prob)
    maxOriginalProb=0
    for object in OriginalDetection:
        if object.get('name')==MostFreqClass:
            temp=object.get('percentage_probability')
            if temp!=None and temp>maxOriginalProb:
                maxOriginalProb=temp
    cv2.imwrite("/content/drive/MyDrive/MajorProjectBP1/Dataset/sampletestoutputs/image10.jpeg",rotatedImage)
    RotatedDetection=detector.detectObjectsFromImage(input_image="/content/drive/MyDrive/MajorProjectBP1/Dataset/sampletestoutputs/image10.jpeg", output_image_path="/content/drive/MyDrive/MajorProjectBP1/Dataset/sampletestoutputs/image12.jpg", minimum_percentage_probability=min_prob)
    maxRotatedProb=0
    for object in RotatedDetection:
        if object.get('name')==MostFreqClass:
            temp=object.get('percentage_probability')
            if temp!=None and temp>maxRotatedProb:
                maxRotatedProb=temp
    
    num_images=1
    plt.figure(figsize=(10.0, 2 * num_images))

    title_fontdict = {
        'fontsize': 14,
        'fontweight': 'bold'
    }
    options = {}
    fig_number = 1
    ax = plt.subplot(num_images, 3, fig_number)
    if fig_number == 1:
        plt.title('Original\n', fontdict=title_fontdict)
    plt.imshow(np.squeeze(originalImg).astype('uint8'), **options)
    plt.axis('off')

    fig_number += 1
    ax = plt.subplot(num_images, 3, fig_number)
    if fig_number == 2:
        plt.title('Rotated\n', fontdict=title_fontdict)
    ax.text(
        0.5, 1.03, 'Angle: {0}'.format(rangle),
        horizontalalignment='center',
        transform=ax.transAxes,
        fontsize=11
    )
    plt.imshow(np.squeeze(rotatedImage).astype('uint8'), **options)
    plt.axis('off')

    fig_number += 1
    ax = plt.subplot(num_images, 3, fig_number)
    # corrected_angle = angle_difference(rangle, true_angle)
    if fig_number == 3:
        plt.title('Corrected\n', fontdict=title_fontdict)
    ax.text(
        0.5, 1.03, '',
        horizontalalignment='center',
        transform=ax.transAxes,
        fontsize=11
    )
    plt.imshow(np.squeeze(finalCorrectedImage).astype('uint8'), **options)
    plt.axis('off')

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    return maxProb,maxOriginalProb,maxRotatedProb


if __name__==__main__:
    # !pip install imageai --upgrade
    test_image_filepath='/content/drive/MyDrive/MajorProjectBP1/Dataset/Good Dog Images/dog3.jpeg'
    maxCorrectedProb,maxOriginalProb,maxRotatedProb=test_eigenvectorsmodel(test_image_filepath)
    print("Accuracy for original image:",maxOriginalProb)
    print("Accuracy for rotated image:",maxRotatedProb)
    print("Accuracy for corrected image:",maxCorrectedProb)