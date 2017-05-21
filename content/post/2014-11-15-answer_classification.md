---
author: ernuzwang@gmail.com
comments: true
date: 2014-11-15 22:15:22+00:00
link: http://www.runzemc.com/2014/11/answer_classification.html
slug: answer_classification
title: Quora challenge - answer classification
wordpress_id: 319
categories:
- Data Analysis
tags:
- Machine learning
- Quora
- R
---

Last night I tried my hands on a [Quora challenge](http://www.quora.com/challenges#answer_classifier) that classifies user-submitted answers into 'good' and 'bad.' All the information is anonymized, including the variable names, but you can tell by looking at their values what some of them may represent. For example, some appear to be count data or some summary statistics based on them, and, given that many of the values are 0 and heavily right-skewed, they seem to be some measure of the writers' reputations, the number of upvotes an answer received, or the follow-up comments. There are also a lot of dummy variables whose meaning are harder to guess, but I imagine they may be related to the writers' profile including their education level and some measure of the answer qualities too (e.g., whether an answer has referenced an external source).

Anyway, I found this challenge quite interesting and, based on my exploratory plots, some of the variables are indeed quite correlated with the classification outcome. As my first attempt, I tried 4 simple models: logistic regression, SVM, random forest, and gradient boost trees, of which, I found the last two outperform the first. By combining the two together, I scored an ROC of .897 on the test data! Needless to say, I'm quite surprised by the result although I still wish I could take a peak at the actual variable names to see what the best indicators really are.

My code is hosted on my [github](https://github.com/Runze/quora_classification_challenge) and below is the knitr output hosted on [RPubs](http://rpubs.com/runzemc/quora_classification_challenge):

[iframe src="http://rpubs.com/runzemc/quora_classification_challenge" width="100%" height="17350"]

Fun! :-)
