import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn import preprocessing

imagePath = "./assets/img07/"
image="Img07.png"
gt_image = "mask.png"
peer_gt = "PeerMask.png"
algorithm = 1
n_clusters = 2
bandwidth = 20
print("\033c")
# For Building gabor kernels to filter image
orientations = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
#orientations2 = [0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8, np.pi ]
wavelengths = [3, 6, 12, 24, 48, 96]

def calculateImg(img, gtImg):
    from sklearn.metrics import precision_recall_fscore_support as score
    
    precision, recall, fscore, _ = score(y_true = gtImg, y_pred= img, average="macro")

    print()
    print()
    print()
    print('precision: ' + str(precision))
    print('recall: ' + str(recall))
    print('fscore: ' + str(fscore))

def build_gabor_kernels():
    filters = []
    ksize = 40
    for rotation in orientations:
        for wavelength in wavelengths:
            kernel = cv.getGaborKernel((ksize, ksize), 4.25, rotation, wavelength, 0.5, 0, ktype=cv.CV_32F)
            filters.append(kernel)

    return filters

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))

def meanShift(image):
    print("started Mean Shift Algorithm")
    originShape = image.shape  
    flatImg=np.reshape(image, [-1, 3])
   
    #bandwidth = estimate_bandwidth(flatImg, quantile=0.1, n_samples=100)    
    ms = MeanShift(bandwidth = bandwidth, bin_seeding=True)
    ms.fit(flatImg)
    labels=ms.labels_ 
    cluster_centers = ms.cluster_centers_    
    # Finding and diplaying the number of clusters    
    labels_unique = np.unique(labels)    
    n_clusters_ = len(labels_unique)    
    print("number of estimated clusters : %d" % n_clusters_) 
    segmentedImg = cluster_centers[np.reshape(labels, originShape[:2])]
    segmentedImg = segmentedImg.astype(np.uint8)
    return segmentedImg, labels

def gaborFilter(image):
    # Resizing the image
    #image = cv.resize(image, (int(cols * 0.5), int(rows * 0.5)))
    rows, cols, channels = image.shape
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    saveImage(gray, "convertedToGray")

    gaborKernels = build_gabor_kernels()
    print("Length of Gabor Kernel: " + str(len(gaborKernels)))

    gaborFilters = []
    for (i, kernel) in enumerate(gaborKernels):
        filteredImage = cv.filter2D(gray, cv.CV_8UC1, kernel)
        # Blurring the image
        sigma = int(3*0.5*wavelengths[i % len(wavelengths)])
        # Sigma needs to be odd
        if sigma % 2 == 0:
            sigma = sigma + 1

        blurredImage = cv.GaussianBlur(filteredImage,(int(sigma),int(sigma)),0)
        gaborFilters.append(blurredImage)

    print("Length of Gabor Filters: " + str(len(gaborFilters)))

    # numberOfFeatures = 1 (gray color) + number of gabor filters + 2 (x and y)
    numberOfFeatures = 1  + len(gaborKernels) + 2
    # Empty array that will contain all feature vectors
    featureVectors = []

    for i in range(0, rows, 1):
        for j in range(0, cols, 1):
            vector = [gray[i][j]]
            for k in range(0, len(gaborKernels)):
                vector.append(gaborFilters[k][i][j])
            vector.extend([i+1, j+1])
            featureVectors.append(vector)

    print("Length of Feature Vector: " + str(len(featureVectors)))
    # Normalizing the feature vectors
    scaler = preprocessing.StandardScaler()
    scaler.fit(featureVectors)
    featureVectors = scaler.transform(featureVectors)

    #bandwidth = estimate_bandwidth(image, quantile=0.1, n_samples=100)    
    if(algorithm == 2):
        print("started Mean Shift Algorithm")
        ms = MeanShift(bandwidth = bandwidth, bin_seeding=True)
        ms.fit(featureVectors)
        labels=ms.labels_ 
        centers = ms.cluster_centers_  
        result = centers[labels]
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)    
        print("number of estimated clusters : %d" % n_clusters_)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=170)
        kmeans.fit(featureVectors)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        result = centers[labels]
    # Only keep first 3 columns to make it easy to plot as an RGB image
    result = np.delete(result, range(3, numberOfFeatures), 1)

    pic = normalize(result.reshape(rows, cols, 3) * 100)

    return pic, labels

def kMeans(image, n_clusters):
    rows, cols, channels = image.shape
    all_pixels = image.reshape(-1, channels)

    km = KMeans(n_clusters=n_clusters)
    km.fit(all_pixels)

    centers = np.array(km.cluster_centers_, dtype='uint8')
    new_img = np.zeros((rows * cols, channels), dtype='uint8')
    
    labels = km.labels_
    new_img = centers[labels]

    new_img = new_img.reshape((image.shape))

    return new_img, labels

def conImgToColorSpace(src, code):
    cvtImg= cv.cvtColor(src,code)    
    return cvtImg

def saveImage(image, name):
    #plt.imshow(cvtImg)
    plt.imsave(imagePath + name + '.png', image)

def showImage(image, name):
    cv.imshow(name + ".png",image)
    cv.waitKey(0)

def menu():
    #print("\033c")
    print()
    print("==================================================================")
    print("       Khulna University of Engineering & Technology (KUET)       ")
    print("            Dept. of Computer Science and Engineering             ")
    print("    Programming Assignment on Advanced Digital Image Processing   ")
    print("                     Topic: Image Segmentation                    ")
    print("              Submitted To: Dr. Sk. Md. Masudul Ahsan             ")
    print()
    print(" ImagePath: '" + imagePath + "'")
    print(" Image: '"+ imagePath + image + "'")
    print(" GT Image: '"+ imagePath + gt_image + "'")
    print()
    print("                        Select your choice                        ")
    if image == "":
        print(" [1] Select Images")
    else:
        print(" [1] change Images")
    print(" [2] Segment using K-means")
    print(" [3] Segment using Mean Shift")
    print(" [0] Exit")
    print("===================================================================")

def filterMenu():
    print("\033c")
    print()
    print("========================Select feature space========================")
    if algorithm == 2:
        print("[1] Set Bandwidth Value (Currently Bandwidth = " + str(bandwidth) + ")")
    else:
        print("[1] Set Cluster Value (Currently k = " + str(n_clusters) + ")")
    print("[2] Segment Using RGB")
    print("[3] Segment Using LAB")
    print("[4] Segment Using Texture and Spatial Information")
    print("[0] Back")
    print("=====================================================================")

menu()
option = int(input("Enter your choice: "))

while option !=0:
    if option == 1:
        imagePath = input("Enter image Path: ")
        print("you have selected image path: " + imagePath)
        image = input("Enter image name: ")
        print("you have selected image: " + image)
        gt_image = input("Enter GT image: ")
        print("you have selected image: " + gt_image)
    elif option == 2:
        algorithm = 1;
        print("Selected algorithm " + str(algorithm))
        filterMenu()
        option = int(input("Enter filter: "))
        while option !=0:
            if option == 1:
                n_clusters = int(input("Enter Cluster Value: "))
                break
            if option == 2:
                img = cv.imread(imagePath + image)
                gt_img = cv.imread(imagePath + gt_image)
                peer_img = cv.imread(imagePath + peer_gt)
                img = conImgToColorSpace(img,cv.COLOR_BGR2RGB)

                segImg, segLebels = kMeans(img, n_clusters)
                showImage(segImg, "segmentedRGB")
                saveImage(segImg, "segmentedRGB")

                _, tLebels = kMeans(gt_img, n_clusters)
                calculateImg(segLebels, tLebels)

                #_, pLebels = kMeans(peer_img, n_clusters)
                #calculateImg(segLebels, pLebels)
                break
            elif option == 3:
                img= cv.imread(imagePath + image)
                gt_img = cv.imread(imagePath + gt_image)
                peer_img = cv.imread(imagePath + peer_gt)
                img = conImgToColorSpace(img,cv.COLOR_BGR2LAB)

                segImg, labels = kMeans(img, n_clusters)
                showImage(segImg, "segmentedLAB")
                saveImage(segImg, "segmentedLAB")

                _, tLebels = kMeans(gt_img, n_clusters)
                calculateImg(segLebels, tLebels)

                #_, pLebels = kMeans(peer_img, n_clusters)
                #calculateImg(segLebels, pLebels)
                break
            elif option == 4:
                img= cv.imread(imagePath + image)
                gt_img = cv.imread(imagePath + gt_image)
                peer_img = cv.imread(imagePath + peer_gt)

                segImg, segLebels = gaborFilter(img)
                showImage(segImg, "SegmentedWithGabor")
                saveImage(segImg, "SegmentedWithGabor")

                tImg, tLebels = kMeans(gt_img, n_clusters)
                calculateImg(segLebels, tLebels)

                #_, pLebels = kMeans(peer_img, n_clusters)
                #calculateImg(segLebels, pLebels)
                break
            else:
                break
    elif option == 3:
        algorithm = 2;
        print("Selected algorithm " + str(algorithm))
        filterMenu()
        option = int(input("Enter filter: "))
        while option !=0:
            if option == 1:
                bandwidth = int(input("Enter Bandwidth Value: "))
                break
            if option == 2:
                img = cv.imread(imagePath + image)
                gt_img = cv.imread(imagePath + gt_image)
                peer_img = cv.imread(imagePath + peer_gt)
                img = conImgToColorSpace(img,cv.COLOR_BGR2RGB)

                segImg, segLebels = meanShift(img)
                showImage(segImg, "segmentedRGBusingMeanShift")
                saveImage(segImg, "segmentedRGBusingMeanShift")

                _, tLebels = kMeans(gt_img, n_clusters)
                calculateImg(segLebels, tLebels)

                #_, pLebels = kMeans(peer_img, n_clusters)
                #calculateImg(segLebels, pLebels)
                break
            elif option == 3:
                img= cv.imread(imagePath + image)
                gt_img = cv.imread(imagePath + gt_image)
                peer_img = cv.imread(imagePath + peer_gt)
                img = conImgToColorSpace(img,cv.COLOR_BGR2LAB)

                segImg, segLebels = meanShift(img)
                showImage(segImg, "segmentedLABusingMeanShift")
                saveImage(segImg, "segmentedLABusingMeanShift")

                tImg, tLebels = kMeans(gt_img, n_clusters)
                calculateImg(segLebels, tLebels)

                #_, pLebels = kMeans(peer_img, n_clusters)
                #calculateImg(segLebels, pLebels)
                break
            elif option == 4:
                img= cv.imread(imagePath + image)
                gt_img = cv.imread(imagePath + gt_image)
                peer_img = cv.imread(imagePath + peer_gt)

                segImg, segLebels = gaborFilter(img)
                showImage(segImg, "SegmentedWithGaborusingMeanShift")
                saveImage(segImg, "SegmentedWithGaborusingMeanShift")

                tImg, tLebels = kMeans(gt_img, n_clusters)
                calculateImg(segLebels, tLebels)

                #_, pLebels = kMeans(peer_img, n_clusters)
                #calculateImg(segLebels, pLebels)
                break
            else:
                break
    else:
        print("Invalid choice.")
    
    print()
    menu()
    option = int(input("Enter your choice: "))

print("Thanks for using this program.")