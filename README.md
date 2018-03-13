# recommender-system

#####Using the ML100 movie sample built a recommender system and answered a few questions that came up while building the model. 


 Q1. Similarity in User-User Collaborative Filtering
(a) In the user-user collaborative filtering example, report comparative results for both ‘euclidean’ distance and ‘cosine’ distance on RMSE. Be careful how you convert ‘euclidean’ distance to a [0, 1] similarity for use in the recommender. Which metric works better? Why?
 
 A. Cosine similarity performed better than the eucledian distance as excpected. This is because euclidean distance takes into consideration the lenght whereas the cosine distance depends on the angle and doesnt care about the distance. 
 
 (b) Try an additional third metric and justify the results observed with your choice. (Google for “pairwise distance scikit learn” for a list of distance metrics, more Googling will tell you what they mean.)

Q2. Item-Item Collaborative Filtering

 (a) Report comparative RMSE results between user-user and item-item based collaborative filtering for cosine similarity. Can you explain why one method may have performed better?
 Consider the average number of ratings per user and the average number of ratings per item.
 
 a. more data per user. The User-user similarity performs better than the item-item similarity but not by much. This might be the case because of the varied dataset that has been provided to us. It covers movies from all genres hence the item item might have problems tryin to judge the similarities between them.data is biased towards the users as compared to the items. on average a user rated a 100 items. more users than items. 
 
 Sim_cosine
 1.026
 1.021
 1.013
 1.009
 1.016
 the average is  1.01735412166
 The 95% CI for cosine is (1.0090130802261479, 1.0256951630950137)
 
 Item_Item_Sim_cosine
1.038
 1.021
 1.010
 1.014
 1.018
 the average is  1.02008290011
 The 95% CI for cosine is (1.006824268625073, 1.0333415315874224)


Q3. Performance Comparison
 (a) Compare all the recommenders from Q1 and Q2 (using cosine similarity) on
 RMSE, P@k, and R@k. Show the results.
 
 (b) Some baselines cannot be evaluated with some metrics? Which ones and why?
 
 RMSE for popularity would be useless as this method aims to rank items relative to eachother and ignores the actual rank itself, which will cause an inflated RMSE.
 P@k and R@k for averaging cannot be evaluated as we cannot rank items if they all have similar values
 
 (c) What is the best algorithm for each of RMSE, P@k, and R@k? Can you explain why this
 may be?
 for rmse is best user-user.
 for p@k and r@k its hard to determine which ones better
 
 (d) Does good performance on RMSE imply good performance on ranking metrics and vice versa?
 Why / why not?
 Not necessarily rmse only gives an overall ranking, but does not consider context. rmse is good for ratings but not rankings. 


Q4. Similarity Evaluation
(a) Go through the list of movies and pick three not-so-popular movies that you know well. I.e.,do not choose “Star Wars” and note that we expect everyone in the class to have chosen different movies. For each of these three movies, list the top 5 most similar movie names according to item-item cosine similarity (you might use a function like numpy argsort).

(b) Can you justify these similarities? Why or why not? Consider that similarity is determined
indirectly by users who rated both items.
(b ans) It was good for the more popular movies but not so good with the obscure movies. 


