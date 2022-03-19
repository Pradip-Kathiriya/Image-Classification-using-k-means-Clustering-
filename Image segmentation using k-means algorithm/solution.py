import cv2 as cv
import numpy as np

# function to compute euclidean distance
def euclidean_dist(X,Y):
    return np.sqrt(np.sum((X-Y)**2))

# read image
image = cv.imread("Q4image.png")

# make the copy of the image to work on
img = np.copy(image)
# convert image from uint8 to float32 data type
img = np.float32(img)

# compute number of rows and colums in the image
rows = img.shape[1]
cols = img.shape[0]

# select four point randomaly
x = np.random.choice(cols,4)
y = np.random.choice(rows,4)

# initialize four (r,g,b) mean from random points
centroid = [[img[x[0],y[0],0],img[x[0],y[0],1],img[x[0],y[0],2]],
            [img[x[1],y[1],0],img[x[1],y[1],1],img[x[1],y[1],2]],
            [img[x[2],y[2],0],img[x[2],y[2],1],img[x[2],y[2],2]],
            [img[x[3],y[3],0],img[x[3],y[3],1],img[x[3],y[3],2]],          
           ]

centroid = np.array(centroid)

# initilise clustes list
clusters = [[],[],[],[]]


while True:
    #re initialize cluster list after each iteration
    clusters = [[],[],[],[]]
    
    #iterate throgh each pixel in the image to compute its distace from mean of each cluster
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            X = img[i][j]
            dist_1 = euclidean_dist(X,centroid[0])
            dist_2 = euclidean_dist(X,centroid[1])
            dist_3 = euclidean_dist(X,centroid[2])
            dist_4 = euclidean_dist(X,centroid[3])
            
            # find the nearest cluster and append points into that cluster
            min_dist = min(dist_1,dist_2,dist_3,dist_4)
            
            if min_dist == dist_1:
                clusters[0].append([[i,j],img[i][j]])
                
            elif min_dist == dist_2:
                clusters[1].append([[i,j],img[i][j]])
                
            elif min_dist == dist_3:
                clusters[2].append([[i,j],img[i][j]])
                
            else:
                clusters[3].append([[i,j],img[i][j]])
    
    # convert the clister list into numpy array          
    clusters[0] = np.array(clusters[0])
    clusters[1] = np.array(clusters[1])
    clusters[2] = np.array(clusters[2])
    clusters[3] = np.array(clusters[3])
    
    # make the copy of previous centroid list
    old_centroid = np.copy(centroid)
    
    # compute new mean from the updated cluster
    centroid[0] = np.mean(clusters[0][:,1],axis=0)
    centroid[1] = np.mean(clusters[1][:,1],axis=0)
    centroid[2] = np.mean(clusters[2][:,1],axis=0)
    centroid[3] = np.mean(clusters[3][:,1],axis=0)
    
    # check the convergenace of the algorithm
    d1 = euclidean_dist(old_centroid[0],centroid[0])
    d2 = euclidean_dist(old_centroid[1],centroid[1])
    d3 = euclidean_dist(old_centroid[2],centroid[2])
    d4 = euclidean_dist(old_centroid[3],centroid[3])
    distances = [d1,d2,d3,d4]
    
    # break the loop when algorithm converge
    # when algorithm converge, new mean is same as old mean 
    if np.sum(distances) == 0:
        break       


centroid = centroid.astype(int)

# change the pixel intesity of each pixel to the mean pixel intesity of the cluster the pixel is falling on
for i in range(len(clusters)):
    for j in range(len(clusters[i])):
        idx = clusters[i][j][0]
        image[idx[0]][idx[1]][0] = centroid[i][0]
        image[idx[0]][idx[1]][1] = centroid[i][1]
        image[idx[0]][idx[1]][2] = centroid[i][2]
        

# generate segmented image
cv.imwrite('output_Q4.png',image)
final_image = cv.imread("output_Q4.png")
cv.imshow('segmented_image',final_image)
cv.waitKey(0)  



                
