import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

for picNum in range(1000):
    img = np.array(cv.imread("cityDataset/" + str(picNum) + ".jpg"), dtype=np.double)


    imgSz = img.shape

    X = np.array([img[0:, 0:, 0].copy().flatten(), 
              img[0:, 0:, 1].copy().flatten(), 
              img[0:, 0:, 2].copy().flatten(), ]).transpose()

    print(X.shape)

    scores = [];
    plottingX = [];

    kmeans = KMeans(n_clusters = 100, random_state=42)
    y_pred = kmeans.fit_predict(X)
    scores.append(-kmeans.score(X))
    plottingX.append(100)
    
    slopes = [];
    for k in range(105, 200, 5):
        kmeans = KMeans(n_clusters = k, random_state=42)
        y_pred = kmeans.fit_predict(X)
        scores.append(-kmeans.score(X))
        plottingX.append(k)

        slopes.append(scores[-1] - scores[-2])
        print(str(k) + " : " + str(len(scores)) + " : m = " + str(slopes[-1]))

    plt.plot(plottingX, scores);
    plt.xlabel("k")
    plt.ylabel("Intertia")
    plt.show()

    print("Begin DBSCAN stuff")
    dbscan = DBSCAN(eps = 0.2, min_samples=155, n_jobs = -1, p = 2)
    dbscan.fit(X)
    print("End DBSCAN")

    print("Begin KNN Classifier")
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])
    print("End K Neighbors")

    print("Do Prediction...")
    A = knn.predict(X)

    print("Reshape and show")
    A = A / A.max()
    newImg = 255 * A.reshape([imgSz[0], imgSz[1]])
    cv.imshow("a", np.uint8(newImg))
    cv.waitKey()














