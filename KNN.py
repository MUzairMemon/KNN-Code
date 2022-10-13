
import cv2
import math
import glob
import numpy as np


CATS = glob.glob("D:\RIME\Sem 3\DL\Project\Ass 1\Cat\*.jpg")    #folder for cat images
DOGS = glob.glob("D:\RIME\Sem 3\DL\Project\Ass 1\Dog\*.jpg")    #folder for dog images

# fuction for resizing, Feature 1 (Gray Image), Feature 2 (Edge detection) //////////////
def FeatureExtraction (data):
    GrayImages = []
    EdgeImages = []
    for img in data:
        oimg = cv2.imread(img)
        rimg = cv2.resize(oimg,(300 ,400))
        gimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
        bimg = cv2.GaussianBlur(gimg, (3,3), 0)
        eimg = cv2.Canny(image=bimg, threshold1=70, threshold2=200)
        
        EdgeFeature = np.sum(eimg)/120000
        GrayFeaure = np.sum(gimg)/120000
        GrayImages.append(GrayFeaure)
        EdgeImages.append(EdgeFeature)
    
    return (GrayImages,EdgeImages)

# /////////////////////////////////////////////////////////////////////////////////////////

# Finding Euclidean Distance //////////////////////////////////////////////////////////////
def euclidean (Gimage,Eimage,G,E):
    for g in Gimage:
        Ig=[]
        for i in G:
            Ig.append((g-i)**2)
            
    for e in Eimage:
        Ie=[]
        for  j in E:
            Ie.append((e-j)**2)
    I=[]
    for i in range (0,10,1):
        
        a = Ig[i]+Ie[i]
        I.append(math.sqrt(a))
        
    return (I)

# ///////////////////////////////////////////////////////////////////////////////////////////

#    Prediction Fuction   /////////////////////////////////////////////////////////////////// 
def predict(Distances,K,GCat, ECat,GDog, EDog,dCat,dDog):
    Cidx = []
    Didx =[]
    GNN = []
    ENN = []
    Cvote = 0
    Dvote = 0
    for k in range (0,K,1):
        minDist  = min(Distances)
        if minDist in dCat:
            idx = dCat.index(minDist)
            GNN.append(GCat[idx])
            ENN.append(ECat[idx])
            Cidx.append(idx)
            Cvote = Cvote+1
            
        elif minDist in dDog:
            idx = dDog.index(minDist)
            GNN.append(GDog[idx])
            ENN.append(EDog[idx])
            Didx.append(idx)
            Dvote =Dvote+1
            
        Distances.remove(minDist)
    return (GNN,ENN,Cvote,Dvote )

GCat,ECat = FeatureExtraction(CATS)
GDog,EDog = FeatureExtraction(DOGS)

#  To save the values of training data
# np.savez('Points' , GCat=GCat,ECat=ECat,GDog=GDog,EDog=EDog)  