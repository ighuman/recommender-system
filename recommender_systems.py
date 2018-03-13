
# coding: utf-8

# # **Basic Introduction to Numpy**

# In[2]:

import numpy as np


# # Initialize array

# In[2]:

#np.zeros(5)


# In[3]:

#np.zeros((5,5))


# In[4]:

#np.ones((5,5))


# In[5]:

#np.array([3,4,3,4,3,4])


# # Properties

# In[6]:

#tmpArr = np.zeros((5,5))


# In[7]:

# shape - size of each dimension
#tmpArr.shape


# In[8]:

# number of dimenstions
#tmpArr.ndim


# In[9]:

# type of array
#tmpArr.dtype.name


# # Accessing

# In[10]:

#tmpArr[2,3]


# In[11]:

#tmpArr[4,2] = -1


# In[12]:

#how does 4,2 correlate to 5,3 probably starts 
#tmpArr


# # Slicing

# In[13]:

#tmpArr[4, :]


# In[14]:

#tmpArr[:, 2]


# In[15]:

#tmpArr2 = np.array([100,200,300,400,500])
#tmpArr2[[2,4]]


# In[16]:

#tmpArr


# In[17]:

tmpArr.nonzero()


# In[18]:

tmpArr[tmpArr.nonzero()]


# ## Basic Manipulations

# In[19]:

tmpArr2 = np.array([1,2,3,4,5,6,7,8,9])
tmpArr2


# In[20]:

# reshape 1D array to matrix
tmpMat = tmpArr2.reshape((3,3))
tmpMat


# In[21]:

# transpose matrix
tmpMatTranspose = tmpMat.T
tmpMatTranspose


# In[22]:

tmpMat.dot(tmpMatTranspose)


# In[23]:

# convert array to float32
np.asarray(tmpMat, np.float32)


# In[24]:

# Print original tmpArr2
print(tmpArr2)

# Define a function: if item is even - keep its value, otherwise change it to 0
vf = np.vectorize(lambda x: x if x % 2 == 0 else 0)

# Apply it over tmpArr2
newTmpArr2 = vf(tmpArr2)

# Print the new array
print(newTmpArr2)


# ## Matrix Calculations

# In[25]:

tmpArr3 = np.array([1,2,3])
tmpArr4 = np.array([4,5,6])


# In[26]:

tmpArr3 + tmpArr4


# In[27]:

tmpArr3 * tmpArr4


# In[28]:

tmpArr3 / tmpArr4


# In[29]:

# 1*4 + 2*5 + 3*6
tmpArr3.dot(tmpArr4)


# In[30]:

tmpArr3.sum()


# In[31]:

tmpArr3.mean()


# In[32]:

tmpArr3.var()


# # Recommender for MovieLens

# ### Getting MovieLens data

# * Download the movielens 100k dataset from this link: [ml-100k.zip](http://files.grouplens.org/datasets/movielens/ml-100k.zip)
# 
# * Upload ml-100k.zip using "My Data" to /resources/data 
# 
# * Extract using the following cell:

# In[33]:

#!unzip /resources/data/ml-100k.zip -d /resources/data


# ### Building the recommender

# In[3]:

# import required libraries
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from heapq import nlargest
from sklearn.metrics import mean_squared_error
from math import sqrt
import os.path
import scipy.stats as st
from scipy import stats


# In[4]:

# define constant for movie lends 100K directory
MOVIELENS_DIR = "/resources/data/ml-100k/"


# ## Loading the data

# First, we inspect the directory content

# In[5]:

#5 splits provided, 
get_ipython().system(u'ls $MOVIELENS_DIR')


# We then load the full MovieLens 100K dataset to find the number of users and items

# In[6]:

fields = ['userID', 'itemID', 'rating', 'timestamp']
ratingDF = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u.data'), sep='\t', names=fields)

ratingDF.head()


# In[7]:

numUsers = len(ratingDF.userID.unique())
numItems = len(ratingDF.itemID.unique())

print("Number of users:", numUsers)
print("Number of items:", numItems)


# In[8]:

fieldsMovies = ['movieID', 'movieTitle', 'releaseDate', 'videoReleaseDate', 'IMDbURL', 'unknown', 'action', 'adventure',
          'animation', 'childrens', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'filmNoir', 'horror',
          'musical', 'mystery', 'romance','sciFi', 'thriller', 'war', 'western']
moviesDF = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u.item'), sep='|', names=fieldsMovies, encoding='latin-1')

moviesDF.head()


# Then, we load a train-test split

# In[9]:

trainDF = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u1.base'), sep='\t', names=fields)
testDF = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u1.test'), sep='\t', names=fields)

# test number of records (total should be 100K)
print("# of lines in train:", trainDF.shape[0])
print("# of lines in test:", testDF.shape[0])


# ## Building User-to-Item Rating Matrix

# In[10]:

def buildUserItemMatrix(dataset, numUsers, numItems):
    # Initialize a of size (numUsers, numItems) to zeros
    matrix = np.zeros((numUsers, numItems), dtype=np.int8)
    
    # Populate the matrix based on the dataset
    for (index, userID, itemID, rating, timestamp) in dataset.itertuples():
        matrix[userID-1, itemID-1] = rating
    return matrix


# In[11]:

trainUserItemMatrix = buildUserItemMatrix(trainDF, numUsers, numItems)
testUserItemMatrix = buildUserItemMatrix(testDF, numUsers, numItems)


# In[12]:

# Inspect the train matrix
trainUserItemMatrix


# ## Baseline solution - Average User Rating

# In[13]:

def predictByUserAverage(trainSet, numUsers, numItems):
    # Initialize the predicted rating matrix with zeros
    predictionMatrix = np.zeros((numUsers, numItems))
    
    for (user,item), rating in np.ndenumerate(trainSet):
        # Predict rating for every item that wasn't ranked by the user (rating == 0)
        if rating == 0:
            # Extract the items the user already rated, take the row of the user and use it to rate everything that he didnt watch.
            userVector = trainSet[user, :]
            ratedItems = userVector[userVector.nonzero()]
            
            # If not empty, calculate average and set as rating for the current item
            if ratedItems.size == 0:
                itemAvg = 0
            else:
                itemAvg = ratedItems.mean()
            predictionMatrix[user, item] = itemAvg
            
        # report progress every 100 users
        if (user % 100 == 0 and item == 1):
            print ("calculated %d users" % (user,))
    
    return predictionMatrix


# In[45]:

userAvgPreiction = predictByUserAverage(trainUserItemMatrix, numUsers, numItems)


# In[46]:

userAvgPreiction


# ## How well did we do?

# In[14]:

def rmse(pred, test):
    # calculate RMSE for all the items in the test dataset
    predItems = pred[test.nonzero()].flatten() 
    testItems = test[test.nonzero()].flatten()
    return sqrt(mean_squared_error(predItems, testItems))


# In[48]:

rmse(userAvgPreiction, testUserItemMatrix)


# ## User-User Similarity

# In[49]:

userSimilarity = 1 - pairwise_distances(trainUserItemMatrix, metric='cosine')


# In[50]:

userSimilarity


# In[15]:

def predictByUserSimilarity(trainSet, numUsers, numItems, similarity):
    # Initialize the predicted rating matrix with zeros
    predictionMatrix = np.zeros((numUsers, numItems))
    
    for (user,item), rating in np.ndenumerate(trainSet):
        # Predict rating for every item that wasn't ranked by the user (rating == 0)
        if rating == 0:
            # Extract the users that provided rating for this item
            itemVector = trainSet[:,item]
            usersRatings = itemVector[itemVector.nonzero()]
            
            # Get the similarity score for each of the users that provided rating for this item
            usersSim = similarity[user,:][itemVector.nonzero()]
            
            # If there no users that ranked this item, use user's average
            if len(usersSim) == 0:
                userVector = trainSet[user, :]
                ratedItems = userVector[userVector.nonzero()]
                
                # If the user didnt rated any item use 0, otherwise use average
                if len(ratedItems) == 0:
                    predictionMatrix[user,item] = 0
                else:
                    predictionMatrix[user,item] = ratedItems.mean()
            else:
                # predict score based on user-user similarity
                predictionMatrix[user,item] = (usersRatings*usersSim).sum() / usersSim.sum()
        
        # report progress every 100 users
        if (user % 100 == 0 and item == 1):
            print ("calculated %d users" % (user,))
    
    return predictionMatrix


# In[52]:

userSimPreiction = predictByUserSimilarity(trainUserItemMatrix, numUsers, numItems, userSimilarity)


# In[53]:

rmse(userSimPreiction, testUserItemMatrix)


# ## Precision@k and Recall@k

# In[16]:

def avgPrecisionAtK(testSet, prediction, k):
    # Initialize sum and count vars for average calculation
    sumPrecisions = 0
    countPrecisions = 0
    
    # Define function for converting 1-5 rating to 0/1 (like / don't like)
    vf = np.vectorize(lambda x: 1 if x >= 4 else 0)
    
    for userID in range(numUsers):
        # Pick top K based on predicted rating
        userVector = prediction[userID,:]
        topK = nlargest(k, range(len(userVector)), userVector.take)
        
        # Convert test set ratings to like / don't like
        userTestVector = vf(testSet[userID,:]).nonzero()[0]
        
        # Calculate precision
        precision = len([item for item in topK if item in userTestVector])/len(topK)
        
        # Update sum and count
        sumPrecisions += precision
        countPrecisions += 1
        
    # Return average P@k
    return sumPrecisions/countPrecisions


# In[17]:

def avgRecallAtK(testSet, prediction, k):
    # Initialize sum and count vars for average calculation
    sumRecalls = 0
    countRecalls = 0
    
    # Define function for converting 1-5 rating to 0/1 (like / don't like)
    vf = np.vectorize(lambda x: 1 if x >= 4 else 0)
    
    for userID in range(numUsers):
        # Pick top K based on predicted rating
        userVector = prediction[userID,:]
        topK = nlargest(k, range(len(userVector)), userVector.take)
        
        # Convert test set ratings to like / don't like
        userTestVector = vf(testSet[userID,:]).nonzero()[0]
        
        # Ignore user if has no ratings in the test set
        if (len(userTestVector) == 0):
            continue
        
        # Calculate recall
        recall = len([item for item in topK if item in userTestVector])/len(userTestVector)
        
        # Update sum and count
        sumRecalls += recall
        countRecalls += 1
    
    # Return average R@k
    return sumRecalls/countRecalls


# In[56]:

print("k\tP@k\tR@k")
for k in [25, 50, 100, 250, 500, 940]:
    print("%d\t%.3lf\t%.3lf" % (k, avgPrecisionAtK(testUserItemMatrix, userSimPreiction, k), avgRecallAtK(testUserItemMatrix, userSimPreiction, k)))


# ## Popularity Based Recommendations

# In[18]:

def predictByPopularity(trainSet, numUsers, numItems):
    # Initialize the predicted rating matrix with zeros
    predictionMatrix = np.zeros((numUsers, numItems))
    
    # Define function for converting 1-5 rating to 0/1 (like / don't like)
    vf = np.vectorize(lambda x: 1 if x >= 4 else 0)
    
    # For every item calculate the number of people liked (4-5) divided by the number of people that rated
    itemPopularity = np.zeros((numItems))
    for item in range(numItems):
        numOfUsersRated = len(trainSet[:, item].nonzero()[0])
        numOfUsersLiked = len(vf(trainSet[:, item]).nonzero()[0])
        if numOfUsersRated == 0:
            itemPopularity[item] = 0
        else:
            itemPopularity[item] = numOfUsersLiked/numOfUsersRated
    
    for (user,item), rating in np.ndenumerate(trainSet):
        # Predict rating for every item that wasn't ranked by the user (rating == 0)
        if rating == 0:
            predictionMatrix[user, item] = itemPopularity[item]
            
        # report progress every 100 users
        if (user % 100 == 0 and item == 1):
            print ("calculated %d users" % (user,))
    
    return predictionMatrix


# In[58]:

popPreiction = predictByPopularity(trainUserItemMatrix, numUsers, numItems)


# In[59]:

print("k\tP@k\tR@k")
for k in [25, 50, 100, 250, 500]:
    print("%d\t%.3lf\t%.3lf" % (k, avgPrecisionAtK(testUserItemMatrix, popPreiction, k), avgRecallAtK(testUserItemMatrix, popPreiction, k)))


# ## Making Recommendations for a User

# In[19]:

def userTopK(prediction, moviesDataset, userID, k):
    # Pick top K based on predicted rating
    userVector = prediction[userID+1,:]
    topK = nlargest(k, range(len(userVector)), userVector.take)
    namesTopK = list(map(lambda x: moviesDataset[moviesDataset.movieID == x+1]["movieTitle"].values[0], topK))
    return namesTopK


# In[61]:

# recommend for userID 350 according to popularity recommender
userTopK(popPreiction, moviesDF, 350, 10)


# In[62]:

# recommend for userID 350 according to average rating recommender
userTopK(userAvgPreiction, moviesDF, 350, 10)


# In[63]:

# recommend for userID 350 according to user similarity recommender
userTopK(userSimPreiction, moviesDF, 350, 10)


# ## Evaluating the other datasets

# In[20]:

datasetsFileNames = [('u1.base', 'u1.test'),
                     ('u2.base', 'u2.test'),
                     ('u3.base', 'u3.test'),
                     ('u4.base', 'u4.test'),
                     ('u5.base', 'u5.test')]


# In[65]:

rmseList = []
for trainFileName, testFileName in datasetsFileNames:
    curTrainDF = pd.read_csv(os.path.join(MOVIELENS_DIR, trainFileName), sep='\t', names=fields)
    curTestDF = pd.read_csv(os.path.join(MOVIELENS_DIR, testFileName), sep='\t', names=fields)
    curTrainUserItemMatrix = buildUserItemMatrix(curTrainDF, numUsers, numItems)
    curTestUserItemMatrix = buildUserItemMatrix(curTestDF, numUsers, numItems)
    
    curUserAvgPreiction = predictByUserAverage(curTrainUserItemMatrix, numUsers, numItems)
    avgRMSE = rmse(curUserAvgPreiction, curTestUserItemMatrix)
    
    curUserSimilarity = 1 - pairwise_distances(curTrainUserItemMatrix, metric='cosine')
    curUserSimPreiction = predictByUserSimilarity(curTrainUserItemMatrix, numUsers, numItems, curUserSimilarity)
    simRMSE = rmse(curUserSimPreiction, curTestUserItemMatrix)
    
    rmseList.append((avgRMSE, simRMSE))


# In[66]:

print("Avg\tSim")
for avgScore, simScore in rmseList:
    print("%.3lf\t%.3lf" % (avgScore, simScore))


# #  Assignment 3
# ## 1
# #### Q1. Similarity in User-User Collaborative Filtering
# (a) In the user-user collaborative filtering example, report comparative results for both ‘euclidean’ distance and ‘cosine’ distance on RMSE. Be careful how you convert ‘euclidean’ distance to a [0, 1] similarity for use in the recommender. Which metric works better? Why?
# 
# A. Cosine similarity performed better than the eucledian distance as excpected. This is because euclidean distance takes into consideration the lenght whereas the cosine distance depends on the angle and doesnt care about the distance. 
# 
# (b) Try an additional third metric and justify the results observed with your choice. (Google for “pairwise distance scikit learn” for a list of distance metrics, more Googling will tell you what they mean.)
# 

# In[21]:

datasetsFileNames = [('u1.base', 'u1.test'),
                     ('u2.base', 'u2.test'),
                     ('u3.base', 'u3.test'),
                     ('u4.base', 'u4.test'),
                     ('u5.base', 'u5.test')]


# In[94]:

rmseList_cosine = []
    
    
for trainFileName, testFileName in datasetsFileNames:
    curTrainDF = pd.read_csv(os.path.join(MOVIELENS_DIR, trainFileName), sep='\t', names=fields)
    curTestDF = pd.read_csv(os.path.join(MOVIELENS_DIR, testFileName), sep='\t', names=fields)
    curTrainUserItemMatrix = buildUserItemMatrix(curTrainDF, numUsers, numItems)
    curTestUserItemMatrix = buildUserItemMatrix(curTestDF, numUsers, numItems)
    
    #curUserAvgPreiction = predictByUserAverage(curTrainUserItemMatrix, numUsers, numItems)
    #avgRMSE = rmse(curUserAvgPreiction, curTestUserItemMatrix)
    
    curUserSimilarity = 1 - pairwise_distances(curTrainUserItemMatrix, metric='cosine')
    curUserSimPreiction = predictByUserSimilarity(curTrainUserItemMatrix, numUsers, numItems, curUserSimilarity)
    simRMSE = rmse(curUserSimPreiction, curTestUserItemMatrix)
    
   
    rmseList_cosine.append((simRMSE))


# In[95]:


print("Sim_cosine")
for simScore in rmseList_cosine:
    print("%.3lf" % (simScore))
print ("the average is ", np.mean(rmseList_cosine))
print("The 95% CI for cosine is",(st.t.interval(0.95, len(rmseList_cosine)-1, loc=np.mean(rmseList_cosine), scale=st.sem(rmseList_cosine))))


# In[91]:

rmseList_eucd = []
    
    
for trainFileName, testFileName in datasetsFileNames:
    curTrainDF = pd.read_csv(os.path.join(MOVIELENS_DIR, trainFileName), sep='\t', names=fields)
    curTestDF = pd.read_csv(os.path.join(MOVIELENS_DIR, testFileName), sep='\t', names=fields)
    curTrainUserItemMatrix = buildUserItemMatrix(curTrainDF, numUsers, numItems)
    curTestUserItemMatrix = buildUserItemMatrix(curTestDF, numUsers, numItems)
    
    
    
    curUserSimilarity_eucd = pairwise_distances(curTrainUserItemMatrix, metric='euclidean')
    curUserSimPreiction_eucd = predictByUserSimilarity(curTrainUserItemMatrix, numUsers, numItems, curUserSimilarity_eucd)
    simRMSE_eucd = rmse(curUserSimPreiction_eucd, curTestUserItemMatrix)
    rmseList_eucd.append((simRMSE_eucd))


# In[93]:

print("Sim_eucd")
for simScore in rmseList_eucd:
    print("%.3lf" % (simScore))
print ("the average is ", np.mean(rmseList_eucd))
print("The 95% CI for euclidean is",(st.t.interval(0.95, len(rmseList_eucd)-1, loc=np.mean(rmseList_eucd), scale=st.sem(rmseList_eucd))))


# In[77]:

rmseList_cityblock = []
    
    
for trainFileName, testFileName in datasetsFileNames:
    curTrainDF = pd.read_csv(os.path.join(MOVIELENS_DIR, trainFileName), sep='\t', names=fields)
    curTestDF = pd.read_csv(os.path.join(MOVIELENS_DIR, testFileName), sep='\t', names=fields)
    curTrainUserItemMatrix = buildUserItemMatrix(curTrainDF, numUsers, numItems)
    curTestUserItemMatrix = buildUserItemMatrix(curTestDF, numUsers, numItems)
    
    
    
    curUserSimilarity_cityblock = pairwise_distances(curTrainUserItemMatrix, metric='cityblock')
    curUserSimPreiction_cityblock = predictByUserSimilarity(curTrainUserItemMatrix, numUsers, numItems, curUserSimilarity_cityblock)
    simRMSE_cityblock = rmse(curUserSimPreiction_cityblock, curTestUserItemMatrix)
    rmseList_cityblock.append((simRMSE_cityblock))


# In[78]:

print("Sim_cityblock")
for simScore in rmseList_cityblock:
    print("%.3lf" % (simScore))
print("The 95% CI for cityblock is",(st.t.interval(0.95, len(rmseList_cityblock)-1, loc=np.mean(rmseList_cityblock), scale=st.sem(rmseList_cityblock))))


# ## 2
# #### Q2. Item-Item Collaborative Filtering
# (a) Leveraging the user-user collaborative filtering example, implement an item-item based approach in a Python function called:
# def predictByItemSimilarity(trainSet, numUsers, numItems, similarity).
# Show your solution in your IPython notebook.
# (b) Report comparative RMSE results between user-user and item-item based collaborative filtering for cosine similarity. Can you explain why one method may have performed better?
# Consider the average number of ratings per user and the average number of ratings per item.
# 
# B. more data per user. The User-user similarity performs better than the item-item similarity but not by much. This might be the case because of the varied dataset that has been provided to us. It covers movies from all genres hence the item item might have problems tryin to judge the similarities between them.data is biased towards the users as compared to the items. on average a user rated a 100 items. more users than items. 
# 
# Sim_cosine
# 1.026
# 1.021
# 1.013
# 1.009
# 1.016
# the average is  1.01735412166
# The 95% CI for cosine is (1.0090130802261479, 1.0256951630950137)
# 
# Item_Item_Sim_cosine
# 1.038
# 1.021
# 1.010
# 1.014
# 1.018
# the average is  1.02008290011
# The 95% CI for cosine is (1.006824268625073, 1.0333415315874224)

# In[22]:

def predictByItemSimilarity(trainSet, numUsers, numItems, similarity):
    # Initialize the predicted rating matrix with zeros
    predictionMatrix = np.zeros((numItems, numUsers))
    
    for (user,item), rating in np.ndenumerate(trainSet):
        # Predict rating for every item that wasn't ranked by the user (rating == 0)
        if rating == 0:
            # Extract the users that provided rating for this item
            itemVector = trainSet[:,item]
            usersRatings = itemVector[itemVector.nonzero()]
            
            # Get the similarity score for each of the users that provided rating for this item
            usersSim = similarity[user,:][itemVector.nonzero()]
            
            # If there no users that ranked this item, use user's average
            if len(usersSim) == 0 or usersSim.sum()== 0:
                userVector = trainSet[user, :]
                ratedItems = userVector[userVector.nonzero()]
                
                # If the user didnt rated any item use 0, otherwise use average
                if len(ratedItems) == 0:
                    predictionMatrix[user,item] = 0
                else:
                    predictionMatrix[user,item] = ratedItems.mean()
            #elif (usersSim.sum() == 0):
                #predictionMatrix[user,item]=0
            else:
                # predict score based on user-user similarity
                predictionMatrix[user,item] = (usersRatings*usersSim).sum() / usersSim.sum()
        
        # report progress every 100 users
        if (user % 100 == 0 and item == 1):
            print ("calculated %d users" % (user,))
    
    return predictionMatrix


# In[80]:

rmseList_item = []
for trainFileName, testFileName in datasetsFileNames:
    curTrainDF = pd.read_csv(os.path.join(MOVIELENS_DIR, trainFileName), sep='\t', names=fields)
    curTestDF = pd.read_csv(os.path.join(MOVIELENS_DIR, testFileName), sep='\t', names=fields)
    curTrainUserItemMatrix = buildUserItemMatrix(curTrainDF, numUsers, numItems)
    curTestUserItemMatrix = buildUserItemMatrix(curTestDF, numUsers, numItems)
    curItemSimilarity = 1 - pairwise_distances(curTrainUserItemMatrix.T, metric='cosine')
    curItemSimPreiction = predictByItemSimilarity(curTrainUserItemMatrix.T, numUsers, numItems, curItemSimilarity)
    itemRMSE = rmse(curItemSimPreiction, curTestUserItemMatrix.T)
    
    rmseList_item.append((itemRMSE))


# In[81]:


print("Item_Item_Sim_cosine")
for simScore in rmseList_item:
    print("%.3lf" % (simScore))
print ("the average is ", np.mean(rmseList_item))
print("The 95% CI for cosine is",(st.t.interval(0.95, len(rmseList_item)-1, loc=np.mean(rmseList_item), scale=st.sem(rmseList_item))))


# ## 3
# #### Q3. Performance Comparison
# (a) Compare all the recommenders in the lab and from Q1 and Q2 (using cosine similarity) on
# RMSE, P@k, and R@k. Show the results.
# (b) Some baselines cannot be evaluated with some metrics? Which ones and why?
# RMSE for popularity would be useless as this method aims to rank items relative to eachother and ignores the actual rank itself, which will cause an inflated RMSE.
# P@k and R@k for averaging cannot be evaluated as we cannot rank items if they all have similar values
# 
# (c) What is the best algorithm for each of RMSE, P@k, and R@k? Can you explain why this
# may be?
# for rmse is best user-user.
# for p@k and r@k its hard to determine which ones better
# 
# (d) Does good performance on RMSE imply good performance on ranking metrics and vice versa?
# Why / why not?
# Not necessarily rmse only gives an overall ranking, but does not consider context. rmse is good for ratings but not rankings. 

# In[33]:

#popularity recommender 
curTrainDF_PP_1 = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u1.base'), sep='\t', names=fields)
curTestDF_PP_1 = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u1.test'), sep='\t', names=fields)
curTrainUserItemMatrix_PP_1 = buildUserItemMatrix(curTrainDF_PP_1, numUsers, numItems)
curTestUserItemMatrix_PP_1 = buildUserItemMatrix(curTestDF_PP_1, numUsers, numItems)
curUserSimilarity_PP_1 = 1 - pairwise_distances(curTrainUserItemMatrix_PP_1, metric='cosine')
curPopPreiction_PP_1 = predictByPopularity(curTrainUserItemMatrix_PP_1, numUsers, numItems)

    
print("k\tP@k\tR@k")
for k in [25, 50, 100, 250, 500, 940]:
    print("%d\t%.3lf\t%.3lf" % (k, avgPrecisionAtK(curTestUserItemMatrix_PP_1, curPopPreiction_PP_1 , k), avgRecallAtK(curTestUserItemMatrix_PP_1 ,curPopPreiction_PP_1 , k)))


# In[82]:

curTrainDF1 = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u1.base'), sep='\t', names=fields)
curTestDF1 = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u1.test'), sep='\t', names=fields)
curTrainUserItemMatrix1 = buildUserItemMatrix(curTrainDF1, numUsers, numItems)
curTestUserItemMatrix1 = buildUserItemMatrix(curTestDF1, numUsers, numItems)
curUserSimilarity1 = 1 - pairwise_distances(curTrainUserItemMatrix1, metric='cosine')
curUserSimPreiction1 = predictByUserSimilarity(curTrainUserItemMatrix1, numUsers, numItems, curUserSimilarity1)

    
print("k\tP@k\tR@k")
for k in [25, 50, 100, 250, 500, 940]:
    print("%d\t%.3lf\t%.3lf" % (k, avgPrecisionAtK(curTestUserItemMatrix1, curUserSimPreiction1 , k), avgRecallAtK(curTestUserItemMatrix1 ,curUserSimPreiction1 , k)))


# In[83]:

curTrainDF2 = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u2.base'), sep='\t', names=fields)
curTestDF2 = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u2.test'), sep='\t', names=fields)
curTrainUserItemMatrix2 = buildUserItemMatrix(curTrainDF2, numUsers, numItems)
curTestUserItemMatrix2 = buildUserItemMatrix(curTestDF2, numUsers, numItems)  
curUserSimilarity2 = 1 - pairwise_distances(curTrainUserItemMatrix2, metric='cosine')
curUserSimPreiction2 = predictByUserSimilarity(curTrainUserItemMatrix2, numUsers, numItems, curUserSimilarity2)

    
print("k\tP@k\tR@k")
for k in [25, 50, 100, 250, 500, 940]:
    print("%d\t%.3lf\t%.3lf" % (k, avgPrecisionAtK(curTestUserItemMatrix2, curUserSimPreiction2 , k), avgRecallAtK(curTestUserItemMatrix2 ,curUserSimPreiction2 , k)))


# In[84]:

curTrainDF3 = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u3.base'), sep='\t', names=fields)
curTestDF3 = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u3.test'), sep='\t', names=fields)
curTrainUserItemMatrix3 = buildUserItemMatrix(curTrainDF3, numUsers, numItems)
curTestUserItemMatrix3 = buildUserItemMatrix(curTestDF3, numUsers, numItems)
curUserSimilarity3 = 1 - pairwise_distances(curTrainUserItemMatrix3, metric='cosine')
curUserSimPreiction3 = predictByUserSimilarity(curTrainUserItemMatrix3, numUsers, numItems, curUserSimilarity3)

    
print("k\tP@k\tR@k")
for k in [25, 50, 100, 250, 500, 940]:
    print("%d\t%.3lf\t%.3lf" % (k, avgPrecisionAtK(curTestUserItemMatrix3, curUserSimPreiction3 , k), avgRecallAtK(curTestUserItemMatrix3 ,curUserSimPreiction3 , k)))


# In[85]:

curTrainDF4 = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u4.base'), sep='\t', names=fields)
curTestDF4 = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u4.test'), sep='\t', names=fields)
curTrainUserItemMatrix4 = buildUserItemMatrix(curTrainDF4, numUsers, numItems)
curTestUserItemMatrix4 = buildUserItemMatrix(curTestDF4, numUsers, numItems)    
curUserSimilarity4 = 1 - pairwise_distances(curTrainUserItemMatrix4, metric='cosine')
curUserSimPreiction4 = predictByUserSimilarity(curTrainUserItemMatrix4, numUsers, numItems, curUserSimilarity4)

    
print("k\tP@k\tR@k")
for k in [25, 50, 100, 250, 500, 940]:
    print("%d\t%.3lf\t%.3lf" % (k, avgPrecisionAtK(curTestUserItemMatrix4, curUserSimPreiction4 , k), avgRecallAtK(curTestUserItemMatrix4 ,curUserSimPreiction4 , k)))


# In[86]:

curTrainDF5 = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u5.base'), sep='\t', names=fields)
curTestDF5 = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u5.test'), sep='\t', names=fields)
curTrainUserItemMatrix5 = buildUserItemMatrix(curTrainDF5, numUsers, numItems)
curTestUserItemMatrix5 = buildUserItemMatrix(curTestDF5, numUsers, numItems)
curUserSimilarity5 = 1 - pairwise_distances(curTrainUserItemMatrix5, metric='cosine')
curUserSimPreiction5 = predictByUserSimilarity(curTrainUserItemMatrix5, numUsers, numItems, curUserSimilarity5)

    
print("k\tP@k\tR@k")
for k in [25, 50, 100, 250, 500, 940]:
    print("%d\t%.3lf\t%.3lf" % (k, avgPrecisionAtK(curTestUserItemMatrix5, curUserSimPreiction5 , k), avgRecallAtK(curTestUserItemMatrix5 ,curUserSimPreiction5 , k)))


# ## Item-Item

# In[88]:

curTrainDF_i_1 = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u1.base'), sep='\t', names=fields)
curTestDF_i_1 = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u1.test'), sep='\t', names=fields)
curTrainUserItemMatrix_i_1 = buildUserItemMatrix(curTrainDF_i_1, numUsers, numItems)
curTestUserItemMatrix_i_1 = buildUserItemMatrix(curTestDF_i_1, numUsers, numItems)    
curItemSimilarity_i_1 = 1 - pairwise_distances(curTrainUserItemMatrix_i_1.T, metric='cosine')
curItemSimPreiction_i_1 = predictByItemSimilarity(curTrainUserItemMatrix_i_1.T, numUsers, numItems, curItemSimilarity_i_1)

    
print("k\tP@k\tR@k")
for k in [25, 50, 100, 250, 500, 940]:
    print("%d\t%.3lf\t%.3lf" % (k, avgPrecisionAtK(curTestUserItemMatrix_i_1.T, curItemSimPreiction_i_1 , k), avgRecallAtK(curTestUserItemMatrix_i_1.T ,curItemSimPreiction_i_1 , k)))


# In[89]:

curTrainDF_i_2 = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u2.base'), sep='\t', names=fields)
curTestDF_i_2 = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u2.test'), sep='\t', names=fields)
curTrainUserItemMatrix_i_2 = buildUserItemMatrix(curTrainDF_i_2, numUsers, numItems)
curTestUserItemMatrix_i_2 = buildUserItemMatrix(curTestDF_i_2, numUsers, numItems)    
curItemSimilarity_i_2 = 1 - pairwise_distances(curTrainUserItemMatrix_i_2.T, metric='cosine')
curItemSimPreiction_i_2 = predictByItemSimilarity(curTrainUserItemMatrix_i_2.T, numUsers, numItems, curItemSimilarity_i_2)

    
print("k\tP@k\tR@k")
for k in [25, 50, 100, 250, 500, 940]:
    print("%d\t%.3lf\t%.3lf" % (k, avgPrecisionAtK(curTestUserItemMatrix_i_2.T, curItemSimPreiction_i_2 , k), avgRecallAtK(curTestUserItemMatrix_i_2.T ,curItemSimPreiction_i_2 , k)))


# In[90]:

curTrainDF_i_3 = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u3.base'), sep='\t', names=fields)
curTestDF_i_3 = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u3.test'), sep='\t', names=fields)
curTrainUserItemMatrix_i_3 = buildUserItemMatrix(curTrainDF_i_3, numUsers, numItems)
curTestUserItemMatrix_i_3 = buildUserItemMatrix(curTestDF_i_3, numUsers, numItems)   
curItemSimilarity_i_3 = 1 - pairwise_distances(curTrainUserItemMatrix_i_3.T, metric='cosine')
curItemSimPreiction_i_3 = predictByItemSimilarity(curTrainUserItemMatrix_i_3.T, numUsers, numItems, curItemSimilarity_i_3)

    
print("k\tP@k\tR@k")
for k in [25, 50, 100, 250, 500, 940]:
    print("%d\t%.3lf\t%.3lf" % (k, avgPrecisionAtK(curTestUserItemMatrix_i_3.T, curItemSimPreiction_i_3 , k), avgRecallAtK(curTestUserItemMatrix_i_3.T ,curItemSimPreiction_i_3 , k)))


# In[91]:

curTrainDF_i_4 = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u4.base'), sep='\t', names=fields)
curTestDF_i_4 = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u4.test'), sep='\t', names=fields)
curTrainUserItemMatrix_i_4 = buildUserItemMatrix(curTrainDF_i_4, numUsers, numItems)
curTestUserItemMatrix_i_4 = buildUserItemMatrix(curTestDF_i_4, numUsers, numItems)    
curItemSimilarity_i_4 = 1 - pairwise_distances(curTrainUserItemMatrix_i_4.T, metric='cosine')
curItemSimPreiction_i_4 = predictByItemSimilarity(curTrainUserItemMatrix_i_4.T, numUsers, numItems, curItemSimilarity_i_4)

    
print("k\tP@k\tR@k")
for k in [25, 50, 100, 250, 500, 940]:
    print("%d\t%.3lf\t%.3lf" % (k, avgPrecisionAtK(curTestUserItemMatrix_i_4.T, curItemSimPreiction_i_4 , k), avgRecallAtK(curTestUserItemMatrix_i_4.T ,curItemSimPreiction_i_4 , k)))


# In[92]:

curTrainDF_i_5 = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u5.base'), sep='\t', names=fields)
curTestDF_i_5 = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u5.test'), sep='\t', names=fields)
curTrainUserItemMatrix_i_5 = buildUserItemMatrix(curTrainDF_i_5, numUsers, numItems)
curTestUserItemMatrix_i_5 = buildUserItemMatrix(curTestDF_i_5, numUsers, numItems)
curItemSimilarity_i_5 = 1 - pairwise_distances(curTrainUserItemMatrix_i_5.T, metric='cosine')
curItemSimPreiction_i_5 = predictByItemSimilarity(curTrainUserItemMatrix_i_5.T, numUsers, numItems, curItemSimilarity_i_5)

    
print("k\tP@k\tR@k")
for k in [25, 50, 100, 250, 500, 940]:
    print("%d\t%.3lf\t%.3lf" % (k, avgPrecisionAtK(curTestUserItemMatrix_i_5.T, curItemSimPreiction_i_5 , k), avgRecallAtK(curTestUserItemMatrix_i_5.T ,curItemSimPreiction_i_5 , k)))


# ## 4
# ### Q4. Similarity Evaluation
# (a) Go through the list of movies and pick three not-so-popular movies that you know well. I.e.,do not choose “Star Wars” and note that we expect everyone in the class to have chosen different movies. For each of these three movies, list the top 5 most similar movie names according to item-item cosine similarity (you might use a function like numpy argsort).
# 
# (b) Can you justify these similarities? Why or why not? Consider that similarity is determined
# indirectly by users who rated both items.
# (b ans) It was good for the more popular movies but not so good with the obscure movies. 

# In[25]:

curTrainDF_i_K = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u1.base'), sep='\t', names=fields)
curTestDF_i_K = pd.read_csv(os.path.join(MOVIELENS_DIR, 'u1.test'), sep='\t', names=fields)
curTrainUserItemMatrix_i_K = buildUserItemMatrix(curTrainDF_i_K, numUsers, numItems)
curTestUserItemMatrix_i_K = buildUserItemMatrix(curTestDF_i_K, numUsers, numItems) 
curItemSimilarity_i_K = 1 - pairwise_distances(curTrainUserItemMatrix_i_K.T, metric='cosine')
curItemSimPreiction_i_K = predictByItemSimilarity(curTrainUserItemMatrix_i_K.T, numUsers, numItems, curItemSimilarity_i_K)


# In[57]:

curItemSimilarity_i_K[1,:]


# In[62]:

userVector = curItemSimilarity_i_K[0,:]


# In[63]:

userVector


# In[77]:

topK = userVector.argsort()[::-1][0:5]


# In[78]:

topK


# In[26]:


def userTopK(sim, moviesDataset, itemID, k):
    # Pick top K based on predicted rating
    userVector = sim[itemID,:]
    topK = userVector.argsort()[::-1][1:k+1]
    namesTopK = list(map(lambda x: moviesDataset[moviesDataset.movieID == x+1]["movieTitle"].values[0], topK))
    return namesTopK


# In[27]:

#Top K using rumble in the bronx
userTopK(curItemSimilarity_i_K, moviesDF, 23, 5)


# In[28]:

#Top K using Billy Madison
userTopK(curItemSimilarity_i_K, moviesDF, 40, 5)


# In[29]:

#Top K using The Godfather
userTopK(curItemSimilarity_i_K, moviesDF, 126, 5)


# # 5
# #### (MIE1513 Only) Q5. Testing with different user types
# (a) Look at a histogram of the number of ratings per user. (Google for “scipy histogram”.) Pick a threshold τ that you believe divides users with few ratings and those with a moderate to
# large number of ratings. What τ did you choose? Repeat Q3, but in each of the following
# two cases testing on only users that meet the following criteria:
# 
# (i) Above threshold τ of liked items
# (ii) Below threshold τ of liked items
# 
# Are there any differences between recommender performance for (i) and (ii)? Can you explain
# them?

# In[30]:

get_ipython().magic(u'matplotlib inline')
ratingDF_histogram = ratingDF[['userID',  'rating']]
th = ratingDF_histogram.groupby(['userID']).count()
th



# In[31]:

th['rating'].hist()


# In[ ]:



