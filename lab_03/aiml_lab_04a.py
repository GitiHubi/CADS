#!/usr/bin/env python
# coding: utf-8

# <img align="right" style="max-width: 200px; height: auto" src="hsg_logo.png">
# 
# ##  Lab 04 - "Unsupervised Machine Learning"
# 
# Introduction to AI and ML, University of St. Gallen, Autumn Term 2020

# In the last lab you learned about how to utilize **supervised** learning classification techniques namely (1) the Gaussian Naive-Bayes (Gaussian NB) classifier, (2) the k Nearest-Neighbor (kNN) classifier and (3) the Logistic Regression classifer. 
# 
# In this lab we will learn about an **unsupervised** machine learning technique referred to as **k-Means Clustering**. We will use this technique to classify un-labelled data (i.e., data without defined categories or groups). In general, clustering-based techniques are widely used in **unsupervised machine learning**.
# 
# <img align="center" style="max-width: 500px" src="machinelearning.png">
# 
# (Courtesy: Intro to AI & ML lecture, Prof. Dr. Borth, University of St. Gallen)
# 
# The **k-Means Clustering** algorithm is one of the most popular clustering algorithms used in machine learning. The goal of k-Means Clustering is to find groups (clusters) in a given dataset. It can be used (1) to **confirm business assumptions** about what types of groups exist or (2) to **identify unknown groups** in complex data sets. Some examples of business-related use cases are:
# 
# >- Segment customers by purchase history;
# >- Segment users by activities on an application or a website;
# >- Group inventory by sales activity; or,
# >- Group inventory by manufacturing metrics.
# 
# (Source: https://www.datascience.com/blog/k-means-clustering)
# 
# Once the algorithm has been run and the groups are defined, any new data can be easily assigned to the correct group.

# As always, pls. don't hesitate to ask all your questions either during the lab, post them in our CANVAS (StudyNet) forum (https://learning.unisg.ch), or send us an email (using the course email).

# ## 1. Lab Objectives:

# After today's lab, you should be able to:
# 
# > 1. Know how to setup a **notebook or "pipeline"** that solves a simple unsupervised clustering task.
# > 2. Understand how a **k-Means Clustering** algorithm can be trained and evaluated.
# > 3. Know how to select an **optimal number of clusters** or cluster means.
# > 4. Know how to Python's **sklearn library** to perform unsupervised clustering.
# > 5. Understand how to **evaluate** and **interpret** the obtained clustering results.

# ## 2. Setup of the Analysis Environment

# Suppress potential warnings:

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# Similarly to the previous labs, we need to import a couple of Python libraries that allow for data analysis and data visualization. In this lab will use the `Pandas`, `Numpy`, `Scikit-Learn (sklearn)`, `Matplotlib` and the `Seaborn` library. Let's import the libraries by the execution of the statements below:

# In[2]:


# import the pandas data science library
import pandas as pd
import numpy as np

# import the scipy spatial distance capability
from scipy.spatial.distance import cdist

# import sklearn data and data pre-processing libraries
from sklearn import datasets

# import sklearn k-means classifier library
from sklearn.cluster import KMeans

# import matplotlib data visualization library
import matplotlib.pyplot as plt
import seaborn as sns

# import matplotlibs 3D plotting capabilities
from mpl_toolkits.mplot3d import Axes3D


# Enable inline Jupyter notebook plotting:

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# Use the 'Seaborn' plotting style in all subsequent visualizations:

# In[4]:


plt.style.use('seaborn')


# Set random seed of all our experiments:

# In[5]:


random_seed = 42


# ## 3. k-Means Clustering

# ### 3.1. Dataset Download and Data Assessment

# The **Iris Dataset** is a classic and straightforward dataset often used as a "Hello World" example in multi-class classification. This data set consists of measurements taken from three different types of iris flowers (referred to as **Classes**),  namely the Iris Setosa, the Iris Versicolour and the Iris Virginica, and their respective measured petal and sepal length (referred to as **Features**).

# <img align="center" style="max-width: 700px; height: auto" src="iris_dataset.png">
# 
# (Source: http://www.lac.inpe.br/~rafael.santos/Docs/R/CAP394/WholeStory-Iris.html)

# In total, the dataset consists of **150 samples** (50 samples taken per class) as well as their corresponding **4 different measurements** taken for each sample. Please, find below the list of the individual measurements:
# 
# >- `Sepal length (cm)`
# >- `Sepal width (cm)`
# >- `Petal length (cm)`
# >- `Petal width (cm)`
# 
# Further details on the dataset can be obtained from the following publication: *Fisher, R.A. "The use of multiple measurements in taxonomic problems" Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to Mathematical Statistics" (John Wiley, NY, 1950)."*
# 
# Let's load the dataset and conduct a preliminary data assessment: 

# In[6]:


iris = datasets.load_iris()


# Print and inspect feature names of the dataset:

# In[7]:


iris.feature_names


# Print and inspect the class names of the dataset:

# In[8]:


iris.target_names


# Print and inspect the top 5 feature rows of the dataset:

# In[9]:


pd.DataFrame(iris.data).head(5)


# Print and inspect the top 5 labels of the dataset:

# In[10]:


pd.DataFrame(iris.target).head(5)


# Determine and print the feature dimensionality of the dataset:

# In[11]:


iris.data.shape


# Determine and print the label dimensionality of the dataset:

# In[12]:


iris.target.shape


# Let's briefly envision how the feature information of the dataset is collected and presented in the data:

# <img align="center" style="max-width: 900px; height: auto" src="featurecollection.png">

# Let's now conduct a more in depth data assessment. Therefore, we plot the feature distributions of the Iris dataset according to their respective class memberships as well as the features pairwise relationships.

# Pls. note that we use Python's **Seaborn** library to create such a plot referred to as **Pairplot**. The Seaborn library is a powerful data visualization library based on the Matplotlib. It provides a great interface for drawing informative statstical graphics (https://seaborn.pydata.org). 

# In[13]:


plt.figure(figsize=(10, 10))
iris_plot = sns.load_dataset("iris")

# supervised scenario
# sns.pairplot(iris_plot, diag_kind='hist', hue='species');

# unsupervised scenario
sns.pairplot(iris_plot, diag_kind='hist');


# It can be observed from the created Pairplot, that most of the feature measurements correspond to at least two to three clusters that exhibit a nice **linear seperability**. Now imagine that we are not in possession of the `species` (class) label associated with each observation in the iris dataset. **How could we distinguish or infer the three iris flower species in the dataset?**

# ### 3.2. The k-Means Clustering Algorithm

# As explained in the lecture, the **k-Means Clustering** algorithm is one of the most popular "first choice" unsupervised clustering algorithms to find groups (clusters) in a given multidimensional dataset $X$.
# 
# <img align="center" style="max-width: 400px; height: auto" src="kmeans.png">
# 
# Thereby, the basic form of k-Means Clustering makes the following **two assumptions**:
# 
# - Each observation is closer to its own cluster center than to the center of the other clusters.
# - A cluster center is the arithmetic mean of all the points that belong to the cluster.
# 
# Let's briefly revisit the distinct step of the algorithm before applying it to the iris dataset. Therefore, let's assume:
# 
# - We have dataset $X$ consisting records $x_1, x_2, x_3, ..., x_n \in \mathcal{R}^d$; 
# - That samples are clustered around $k$ centers (the "$k$ means") denoted by $\mu_1, \mu_{2}, ..., \mu_{k} \in \mathcal{R}^d$; and,
# - Each sample $x_{i}$ belongs to its closest mean $\mu_{i}$.
# 
# We can then iteratively perform the following steps that comprise the **k-Means Clustering** algorithm:
# 
# >- **Step 1** - Pick $k$ random points $\mu_{i}$ as cluster centres called 'means'.
# >- **Step 2** - Assign each $x_i$ to its to nearest cluster mean by calculating its distance to each mean.
# >- **Step 3** - Determine the new cluster centres by calculating the average of the assigned points in each cluster.
# >- **Step 4** - Repeat Step 2 and 3 until none of the cluster assignments change.
# 
# Note, that a single execution of all the four steps outlined above is usually referred to as 'iteration'.

# ### 3.3. k-Means Clustering in a 2-Dimensional Feature Space

# Now, let's see how we can apply it to the iris dataset. We will start with an introductory example of detecting the classes of the iris dataset based on two of its features namely the (1) `Petal length (cm)` and (2) `Petal width (cm)`. Let's first gain an intuition of those two features as well as their distribution by visualizing them accordingly:

# In[14]:


# init the plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)

# add grid
ax.grid(linestyle='dotted')

# plot petal length (3rd feature in the dataset) vs. petal width (4th feature in the dataset)
ax.scatter(iris.data[:,2], iris.data[:,3])

# add axis legends
ax.set_xlabel("[petal_length]", fontsize=14)
ax.set_ylabel("[petal_width]", fontsize=14)

# add plot title
plt.title('Petal Length vs. Petal Width Feature Distribution', fontsize=14);


# Let's now define the parameters of the k-Means Clustering. We will start by specifying the **number of clusters** $k$ we aim to detect in the iris dataset. We hypothesize that our observations are drawn from an unknown distributions of three iris flower species (each distribution corresponding to a different mean $\mu_1$, $\mu_2$, and, $\mu_3$). Therefore, we set the number of clusters to be detected to $k=3$:

# In[15]:


no_clusters = 3


# Next, we need to define a corresponding number of **initial 'means' $\mu_{i}$** (the initial random cluster centers) that will be used as 'starting points' in the first iteration of the clustering process. In our case we will specify $k=3$ cluster means each of dimension 2, since we aim to retrieve 3 clusters based on the 2 features `Petal length (cm)` and `Petal width (cm)`:

# In[16]:


init_means = np.array([[1, 3], [2, 6], [1, 7]])


# Finally, we will define a **maximum number of iterations** that we want to run the k-Means Clustering algorithm. Please, note that the clustering terminates once there will be no further changes in the cluster assignments. However, it's good practice to define an upper bound of the iterations applied in the clustering (especially when analyzing datasets that exhibt a high-dimensional feature space):

# In[17]:


max_iterations = 10


# Now, we are ready to initialize an instance of the **k-Means Clustering** algorithm using Python's `sklearn` library of data science algorithms. Please note again, that for each classifier, available in the `sklearn` library, a designated and detailed documentation is provided. It often also includes a couple of practical examples and use cases. The documentation of the **k-Means Clustering** algorithm can be obtained from the following url: 
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

# Let's use `Scikit-Learn` and instantiate the **k-Means Clustering** algorithm:

# In[18]:


kmeans = KMeans(n_clusters=no_clusters, init=init_means, max_iter=max_iterations)


# Let's run the k-Means Clustering to learn a model of the `Petal length (cm)` and  `Petal width (cm)` features. Pls. note that we are using columns 2 and 3 to extract the values of the two features from the iris dataset:

# In[19]:


kmeans.fit(iris.data[:,2:4]) # note that we are using column 2 (petal length) and 3 (petal width) 


# Now that we have conducted the clustering, let's inspect the distinct cluster labels that have been assigned to the individual records of the iris dataset. This can be achieved by calling the `labels_` attribute of the fitted model: 

# In[20]:


labels = kmeans.labels_ # obtain the assigned cluster labels
print(labels)           # print the cluster labels


# Furthermore, we want to inspect the coordinates of the cluster means (sometimes also referred to as "centroids") assigned by the algorithm. This can be achieved by calling the `cluster_centers_`attribute of the fitted model:

# In[21]:


means = kmeans.cluster_centers_ # obtain the assigned cluster means 
print(means)                    # print the cluster center coordinates


# Let's now plot the iris dataset records using the two features `Petal length (cm)` and `Petal width (cm)` as well as the labels and cluster means determined by the **k-Means Clustering** algorithm:

# In[22]:


# init the plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)

# add grid
ax.grid(linestyle='dotted')

# plot petal length vs. petal width and corresponding classes
scatter = ax.scatter(iris.data[:,2], iris.data[:,3], c=labels.astype(np.float), cmap=plt.cm.Set1)

# prepare data legend
legend = ax.legend(*scatter.legend_elements(), loc='upper left', title='Cluster')

# add legend to plot
ax.add_artist(legend)

# plot cluster means
ax.scatter(means[:,0], means[:,1], marker='x', c='black', s=100)

# iterate over distinct cluster means
for i, mean in enumerate(means):
    
    # determine max cluster point distance
    cluster_radi = cdist(iris.data[:, 2:4][labels==i], [mean]).max()
    
    # plot cluster size
    ax.add_patch(plt.Circle(mean, cluster_radi, fc='darkgrey', edgecolor='slategrey', lw=1, alpha=0.1, zorder=1))

# add axis legends
ax.set_xlabel("[petal_length]", fontsize=14)
ax.set_ylabel("[petal_width]", fontsize=14)

# add plot title
plt.title('Petal Length vs. Petal Width - Clustering Results', fontsize=14);


# To build an even better intuition about the k-Means clustering let's have look at the animation of the distinct clustering iterations shown below:

# In[23]:


get_ipython().run_cell_magic('HTML', '', '<div align="middle">\n<video width="60%" controls>\n<source src="kmeansvideo.mp4" type="video/mp4">\n</video></div>\n')


# It can be observed that, upon convergence, the **k-Means Clustering algorithm** nicely found three clusters in the dateset. Let's inspect to which extend this corresponds to the true 'species' class labels 'verginica', 'setosa', and 'versicolor' of the iris dataset to obtain an idea of the quality of the clusterin result:

# In[24]:


# init the plot
fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(16, 8))

#### plot true iris class labels

# add grid
ax[0].grid(linestyle='dotted')

# iterate over distinct species
for species in np.unique(iris.target):
    
    # obtain iris petal length and petal width
    iris_features = iris.data[iris.target == species,:]
    
    # obtain iris species name
    iris_target_name = iris.target_names[species]
    
    # plot petal length vs. petal width as well as the true labels
    ax[0].scatter(iris_features[:,2], iris_features[:,3], c='C{}'.format(str(species)), label=iris_target_name)

# prepare data legend
ax[0].legend(loc='upper left', title='Classes')

# set axis range
ax[0].set_xlim([-1.5, 7.5])
ax[0].set_ylim([-0.5, 3.5])

# add axis legends
ax[0].set_xlabel("[petal_length]", fontsize=14)
ax[0].set_ylabel("[petal_width]", fontsize=14)

#### plot clustering results

# add grid
ax[1].grid(linestyle='dotted')

# plot petal length vs. petal width and corresponding classes
scatter = ax[1].scatter(iris.data[:,2], iris.data[:,3], c=labels.astype(np.float), cmap=plt.cm.Set1)

# prepare data legend
ax[1].legend(*scatter.legend_elements(), loc='upper left', title='Cluster')

# plot cluster means
ax[1].scatter(means[:,0], means[:,1], marker='x', c='black', s=100)

# iterate over distinct cluster means
for i, mean in enumerate(means):
    
    # determine max cluster point distance
    cluster_radi = cdist(iris.data[:, 2:4][labels==i], [mean]).max()
    
    # plot cluster size
    ax[1].add_patch(plt.Circle(mean, cluster_radi, fc='darkgrey', edgecolor='slategrey', lw=1, alpha=0.1, zorder=1))

# set axis range
ax[1].set_xlim([-1.5, 7.5])
ax[1].set_ylim([-0.5, 3.5])
    
# add axis legends
ax[1].set_xlabel("[petal_length]", fontsize=14)
ax[1].set_ylabel("[petal_width]", fontsize=14)

# add plot title
plt.suptitle('Petal Length vs. Petal Width - True Class Labels (left) and Clustering Results (right)', fontsize=14);


# Ok, it seems that our clustering did a quite good job.

# In addition, let's inspect the distance of all dataset records $X$ to their nearest means $\mu_{i}$. Let's recall that k-Means Clustering conducts a local optimization of the sum of "squared errors", as expressed by:
# 
# $$E(\mu_{1}, \mu_{2}, ..., \mu_{k}) = \sum_{i=1}^{n}(x_{i}-\mu_{k(i)})^{2},$$
# 
# were $x_{i}$ denotes a feature vector (or observation) of the dataset and $\mu_{k(i)}$ its closest mean in the feature space $\mathcal{R}^{d}$.
# 
# We can obtain the sum of those squared distances $E(\mu_{1}, \mu_{2}, ..., \mu_{k})$ by calling the `inertia_` attribute of the learned k-Means Clustering model. It will return the sum of squared distances of each sample to its closest cluster center:

# In[25]:


distances = kmeans.inertia_
print(distances)


# ### Exercises:

# We recommend you try the following exercises as part of the lab:
# 
# **1. Train and evaluate the k-Means Clustering algorithm and obtain its assigned labels as well as squared distances $E(\mu_{1}, \mu_{2}, ..., \mu_{k})$ for distinct max iterations.**
# 
# > Continuously increase the number of training iterations $i$ of the k-Means Clustering starting with 1 and up to 5 iterations ($i=1,...,5$) and repeat the clustering accordingly. What can be observed in terms of the cluster means as well as the sum of squared cluster distances with increasing $i$.

# In[26]:


# ***************************************************
# INSERT YOUR CODE HERE
# ***************************************************


# **2. Determine if the k-Means Clustering algorithm always converges to the same result.**
# 
# > Carefully review the k-Means algorithm and answer to the following question: Does the k-Means algorithm always converge to the same result? Please, explain your reasoning.

# In[27]:


# ***************************************************
# INSERT YOUR CODE HERE
# ***************************************************


# **3. Application of the k-Means Clustering algorithm to distinct data distributions.**
# 
# > Consider the following data distributions. Determine which are suitable for a k-Means clustering and what $k$ value should be applied in the clustering. Please, explain your reasoning.
# 
# <img align="center" style="max-width: 600px; height: auto" src="clustering.png">

# In[28]:


# ***************************************************
# INSERT YOUR CODE HERE
# ***************************************************


# ### 3.4. k-Means Clustering in a 3-Dimensional Feature Space

# Let's see now understand how we can conduct a **k-Means Clustering** in higher dimensional feature spaces. We will now aim detecting the classes of the iris dataset based on three of its features namely the (1) `Petal length (cm)`, (2) `Petal width (cm)`, and (3) `Sepal length (cm)`. Let's first gain an intuition of those two features as well as their distribution by visualizing them accordingly in an interactive 3-dimensional plot.
# 
# To enable 3-dimensional plotting we re-load the `matplotlib` library and import its 3D plotting capabilities. 

# In[29]:


get_ipython().run_line_magic('matplotlib', 'notebook')
from importlib import reload
reload(plt)

# import the seaborn plotting library
import seaborn as sns

# import matplotlibs 3D plotting capabilities
from mpl_toolkits.mplot3d import Axes3D


# Let's now extend the k-Means Clustering to a 3-dimensional features space $\mathcal{R}^{3}$ space by clustering the first three features of the Iris dataset. Upon successful plotting let's visually inspecting the 3-dimensional cluster distributions in the feature space:

# In[30]:


# init the plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)

# init 3D plotting
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=30, azim=120)

# add grid
ax.grid(linestyle='dotted')

# plot petal length (3rd feature in the dataset) vs. petal width (4th feature in the dataset)
ax.scatter(iris.data[:,0], iris.data[:,1], iris.data[:,2], s=40)

# add axis legends
ax.set_xlabel("[sepal_length]", fontsize=14)
ax.set_ylabel("[sepal_width]", fontsize=14)
ax.set_zlabel("[petal_length]", fontsize=14)

# add plot title
plt.title('Sepal Length vs. Sepal Width vs. Petal Length', fontsize=14)

# show the 3-dimensional plot
plt.show();


# In order to conduct the clustering in the 3D feature space we will start again by defining a **max. number of clustering iterations** of the `sklearn` **k-Means Clustering** algorithm:

# In[31]:


max_iterations = 10


# Let's also initialize a corresponding number $k$ **initial random cluster 'means' $\mu_{i}$**. This time will specify $k=3$ cluster means each of dimension 3, since we aim to retrieve 3 clusters based on the 3 features `Sepal length (cm)`, `Sepal width (cm)`, and `Petal length (cm)`:

# In[32]:


init_means = np.array([[1.0, 3.0, 3.0], [2.0, 6.0, 5.0], [1.0, 7.0, 2.0]])


# Now, we ready to initialize an instance of the **k-Means Clustering** algorithm using Python's `sklearn` library of data science algorithms:

# In[33]:


kmeans = KMeans(n_clusters=no_clusters, init=init_means, max_iter=max_iterations)


# Let's run the k-Means Clustering to now learn a model of the `Sepal length (cm)`, `Sepal width (cm)`, and `Petal length (cm)` features. Pls. note that we are using columns 0, 1, and 2 to extract the values of the three features from the iris dataset:

# In[34]:


kmeans.fit(iris.data[:,0:3]) # note that we are using column 1 (sepal length), 2 (sepal width) and 3 (petal length) 


# Let's again inspect the labels assigned to each individual record:

# In[35]:


labels = kmeans.labels_ # obtain the assigned cluster labels
print(labels)           # print the cluster labels


# As well as the determined cluster means:

# In[36]:


means = kmeans.cluster_centers_ # obtain the assigned cluster means 
print(means)                    # print the cluster center coordinates


# Let's also plot the iris dataset records using the three features `Sepal length (cm)`, `Sepal width (cm)` and `Petal length (cm)`, their corresponding learned labels as well as the cluster means as determined by the **k-Means Clustering** algorithm:

# In[37]:


# init the plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)

# init 3D plotting
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=30, azim=120)

# add grid
ax.grid(linestyle='dotted')

# plot petal length vs. petal width and corresponding classes
ax.scatter(iris.data[:,0], iris.data[:,1], iris.data[:,2], c=labels.astype(np.float), cmap=plt.cm.Set1, s=40)

# plot cluster means
ax.scatter(means[:,0], means[:,1], means[:,2], marker='x', c='black', s=100)

# add axis legends
ax.set_xlabel("[sepal_length]", fontsize=14)
ax.set_ylabel("[sepal_width]", fontsize=14)
ax.set_zlabel("[petal_length]", fontsize=14)

# add plot title
plt.title('Sepal Length vs. Sepal Width vs. Petal Length', fontsize=14);

# show the 3-dimensional plot
plt.show();


# Inspecting the clustering results we can observe that the means of the nicely converged to three separating cluster in the three dimensional feature space. Finally, let's again inspect the distance of all dataset records $X$ to their nearest means $\mu_{i}$ by calling the `inertia_` attribute of the learned k-Means Clustering model:

# In[38]:


distances = kmeans.inertia_
print(distances)


# Enable inline Jupyter notebook plotting:

# In[39]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ### 3.5. Drawbacks of the k-Means Clustering Algorithm 

# Even though the **k-Means Clustering** algorithm is one of the most popular clustering algorithms used in machine learning. There are three major drawbacks associated with it, which are explained hereafter:
# 
# 1. The k-Means Clustering is guaranteed to improve the result in each iteration but there are **no guarantees** that it will find the **global best of clusters**. Recall the example, where we initalize the algorithm with a different seed of cluster means. 
# 
# > **Practical solution:** Run the algorithm with multiple random initializations. This also done per default when using the `scikit` learn of machine learning algorithms.
# 
# 2. The k-Means Clustering **cannot learn** the **optimal number of clusters** from the provided data. E.g., if we ask the algorithm for 20 clusters it will find 20 clusters, which may or may not be meaningful. 
# 
# > **Practical solution:** Use the "Elbow" technique as explained in the next section of the notebook. Another option might be the usage of a more complex clustering algorithm like Gaussian Mixture Models, or one that can choose a suitable number of clusters, e.g., the DBSCAN clustering algorithm.
# 
# 3. The k-Means Clustering **doesn't work well** in instances of a **non-linear seperable** dataset. This is caused by its assumption that points will be closer to their own cluster center than to others.
# 
# > **Practical solution:** Transform (if possible) the dataset into a higher dimension where a linear separation becomes possible, e.g., by using a spectral clustering algorithm.

# ## 4. Optimal Cluster Number Selection

# Recall that, one of the basic ideas behind unsupervised machine learning methods, such as **k-Means clustering**, is to define clusters for which the total intra-cluster variation (usually measured by the total sum of squared distances) is minimized:

# $$k^{*} =\underset{k}{\arg \min} \sum_{i=1}^{n}(x_{i}-\mu_{k(i)})^{2},$$

# were $x_{i}$ denotes a single feature vector (or observation) in the dataset and $\mu_{k(i)}$ its closest mean in the feature space $\mathcal{R}^{d}$. Challenge: What is the optimal number of clusters $k$ for a given dataset? Selection of the right $k$ may result in the following issues:
# 
# - if $k$ too small (under-segmentation), then the clusters are too diverse; and;
# - if $k$ too high (over-segmentation), then the clusters are too fine-grain.
# 
# Examples: 

# <img align="center" style="max-width: 800px; height: auto" src="kselection.png">

# Solution: We can then use the sum of "squared errors" $E(\mu_{1}, \mu_{2}, ..., \mu_{k})$ metric to find an optimal number of clusters $k$! This can be achieved by the execution of the so-called **'Elbow'** technique defined by the following algorithm:
# 
# >- **Step 1** - Compute the k-Means clustering algorithm for different number of clusters $k$.
# >- **Step 2** - For each $k$ calculate the sum of the within-cluster sum of squared distances $E(\mu_{1}, \mu_{2}, ..., \mu_{k})$.
# >- **Step 3** - For each $k$ plot the $k$ value vs. its corresponding sum of within-cluster sum of squared distances $E$. 
# >- **Step 4** - Inspect the plot and determine the location of a bend (appropriate number of clusters).

# Let's utilize the **'Elbow'** technique by first defining a max. number of iterations that we aim to apply at each k-Means clustering run:

# In[40]:


max_iterations = 100


# Now we can implement the and run the 'elbow' technique:

# In[41]:


# init the list of squared distances
sum_of_squared_distances = []

# define the range of k-values to investigate
K = range(1,30)

# iterate over all k-values
for k in K:
    
    # init the k-Means clustering algorithm of the current k-value
    kmeans = KMeans(n_clusters=k, init='random', max_iter=max_iterations)
    
    # run the k-Means clustering of sepal-length and sepal-width features
    kmeans = kmeans.fit(iris.data[:,0:2])
    
    # collect the sum of within squared distances of the current k-value
    sum_of_squared_distances.append(kmeans.inertia_)


# Upon completion of the loop above let's inspect the distinct within-cluster sum of squared distances $E$:

# In[42]:


# print the collected sum of squared distances of each k
sum_of_squared_distances


# Furthermore, let's plot the cluster number $k$ vs. the within-cluster sum of squared distances $E$:

# In[43]:


# init the plot
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111)

# add grid
ax.grid(linestyle='dotted')

# plot petal length (3rd feature in the dataset) vs. petal width (4th feature in the dataset)
ax.plot(K, sum_of_squared_distances)

# add axis legends
ax.set_xlabel("[number of clusters $k$]", fontsize=14)
ax.set_ylabel("[within-cluster distance $E$]", fontsize=14)

# add plot title
plt.title('Cluster Number $k$ vs. Within-Cluster Distance $E$', fontsize=14);


# ### Exercises:

# We recommend you to try the following exercises as part of the lab:
# 
# **1. Apply the k-Means Clustering algorithm to all four features contained in the Iris dataset.**
# 
# > Use the k-Means classifier to learn a model of all four features contained in the Iris dataset, namely `Sepal length (cm)`, `Sepal width (cm)`, `Petal length (cm)` and `Petal width (cm)`.

# In[44]:


# ***************************************************
# INSERT YOUR CODE HERE
# ***************************************************


# **2. Determine the optimal number of cluster values $k$ of all four features contained in the iris dataset.**
# 
# > Determine the optimal number of clusters $k$ needed to cluster the observations of all four features contained in the iris dataset using the **'Elbow'** technique outlined above.

# In[45]:


# ***************************************************
# INSERT YOUR CODE HERE
# ***************************************************


# ### Lab Summary:

# In this lab, a step by step introduction into the unsupervised **k-Means Clustering** algorithms was presented. The code and exercises presented in this lab may serve as a starting point for more complex and tailored programs.

# You may want to execute the content of your lab outside of the Jupyter notebook environment, e.g. on a compute node or a server. The cell below converts the lab notebook into a standalone and executable python script. Pls. note that to convert the notebook, you need to install Python's **nbconvert** library and its extensions:

# In[ ]:


# installing the nbconvert library
get_ipython().system('pip3 install nbconvert')
get_ipython().system('pip3 install jupyter_contrib_nbextensions')


# Let's now convert the Jupyter notebook into a plain Python script:

# In[ ]:


get_ipython().system('jupyter nbconvert --to script aiml_lab_04a.ipynb')

