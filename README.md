# Introduction
Image segmentation is used to separate an image into several “meaningful” parts. It is an old research topic, which started around 1970, but there is still no robust solution toward it. There are two main reasons, the first is that the content variety of images is too large, and the second one is that there is no benchmark standard to judge the performance.

# Feature Extraction using Gabor Filters
From experimentation, it is known that Gabor filters are a reasonable model of simple cells in the mammalian vision system. Because of this, Gabor filters are thought to be a good model of how humans distinguish texture, and are, therefore, a good model to use when designing algorithms to recognize texture.

![Gabor Filters](https://github.com/mrjthedifferent/ImageSegmentationUsingKmeansMeanShift/blob/master/assets/img07/gaborfilters.png?raw=true)

# Image Segmentation using K-means
K-means clustering is a simple and elegant approach for partitioning a data set into K distinct, non-overlapping clusters. To perform K-means clustering, we must first specify the desired number of clusters K; then the K-means algorithm will assign each observation to exactly one of the K clusters.

Segmented using K-means with Gabor Filters where k=2

![Image](https://github.com/mrjthedifferent/ImageSegmentationUsingKmeansMeanShift/blob/master/assets/img07/Img07.png?raw=true)      ![Segmented Using K-means with Gabor Filters](https://github.com/mrjthedifferent/ImageSegmentationUsingKmeansMeanShift/blob/master/assets/img07/SegmentedWithGabor.png?raw=true)

