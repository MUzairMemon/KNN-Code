import matplotlib.pyplot as plt
import glob
import numpy as np
import KNN

 # Test folde place image it i-------  Give one Image at a time else it will predict for only last Image
image = glob.glob("D:\RIME\Sem 3\DL\Project\Ass 1\image\*.jpg")

# loading Saved training values
Points = np.load('Points.npz')

GCat = Points['GCat']
ECat = Points['ECat']
GDog = Points['GDog']
EDog = Points['EDog']


Gimage,Eimage = KNN.FeatureExtraction(image)    # for preprocessing of New image

dCat = KNN.euclidean(Gimage, Eimage, GCat, ECat) # finding distance with cat images
dDog = KNN.euclidean(Gimage, Eimage, GDog, EDog) # finding distance with dog images

Distances = dCat+dDog                            # Merging both distances

K=3         # Value of Change if needed

GNN,ENN,Cvote,Dvote = KNN.predict(Distances,K,GCat, ECat,GDog, EDog,dCat,dDog)


if Cvote > Dvote:
    print("This Image is of CAT ")
    
elif Dvote > Cvote:
    print("This Image is of Dog ")

# Plots
plt.scatter(GCat, ECat, label = 'cat')    # all Cat Images
plt.scatter(GDog, EDog, label = 'Dog')    # All Dog Images
plt.scatter(Gimage, Eimage, label = 'image')   # New Image Given by User
plt.scatter(GNN,ENN,label = 'Nearest Neighbour',marker = 'X')   # Nearest Neighbour Images
plt.legend(loc="upper right")
plt.xlabel(" Gray Scaled Feature ")
plt.ylabel(" Edge Detection Image ")

plt.show()



