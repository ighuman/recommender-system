
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

