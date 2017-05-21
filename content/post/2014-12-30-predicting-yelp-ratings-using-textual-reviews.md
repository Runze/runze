---
author: Runze
comments: true
date: 2014-12-30 02:04:42+00:00
link: http://www.runzemc.com/2014/12/predicting-yelp-ratings-using-textual-reviews.html
slug: predicting-yelp-ratings-using-textual-reviews
title: Predicting Yelp ratings using textual reviews
wordpress_id: 346
categories:
- Data Analysis
tags:
- Python
- Text mining
---

Internet is truly full of free and fascinating datasets! I found this [Yelp Dataset Challenge](http://www.yelp.com/dataset_challenge) the other day that includes, among others, over 1 million reviews (most of which are recent) along with their respective 5-star ratings - excellent text mining material! Although to enter the competition (which ends on 12/31/14), you have to be a current student (which I'm not), but everyone is welcome to play around with the data. Hence, during my recent break, I tried my hands on it and here is my first attempt :-)

In this first attempt, I tried to use the reviews' text alone to predict the ratings by first computing the reviews' sentiment scores in a supervised fashion and then using the estimated scores to predict the 5-class outcome. Currently the model achieved a mean absolute error of 0.66 and an accuracy score of 0.48. In the future, I'm planing to incorporate more relevant information to improve the prediction power.

The IPython notebooks are rendered by nbviewer [here](http://nbviewer.ipython.org/github/Runze/yelp_data_challenge/tree/master/) and the individual files can be accessed and viewed directly below:



	
  1. [Exploratory analysis](http://nbviewer.ipython.org/github/Runze/yelp_data_challenge/blob/master/1.%20explore_data.ipynb)

	
  2. [Analyzing reviews](http://nbviewer.ipython.org/github/Runze/yelp_data_challenge/blob/master/2.%20parse_reviews.ipynb)

	
  3. [Training models using the sentiment scores](http://nbviewer.ipython.org/github/Runze/yelp_data_challenge/blob/master/3.%20train_models.ipynb)

	
  4. [Applying the model to the test set](http://nbviewer.ipython.org/github/Runze/yelp_data_challenge/blob/master/4.%20test_models.ipynb)





