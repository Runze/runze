---
author: ernuzwang@gmail.com
comments: true
date: 2014-07-10 07:49:59+00:00
link: http://www.runzemc.com/2014/07/how-californians-reacted-to-the-brazil-vs-germany-game.html
slug: how-californians-reacted-to-the-brazil-vs-germany-game
title: How Californians reacted to the Brazil vs. Germany game
wordpress_id: 82
categories:
- Data Analysis
tags:
- R
- World Cup
---

Unless you've been living under a rock, there is no way you haven't heard about the <insert appropriate adjective> game between Brazil and Germany yesterday (7/8/14), and chances are you have also seen this twitter heat map [video](http://cartodb.com/v/worldcup/match/?TC=x&vis=31ed2e2a-06ce-11e4-8c1d-0e230854a1cb&h=t&t=Brazil,FFCC00%7CGermany,B40903&m=7%2F8%2F2014%2017:00:00%20GMT,7%2F8%2F2014%2018:52:00GMT&g=108%7C11,23,24,26,29,87,97#/2/-17.5/-5.8/0) that went viral today. While watching it, I realized I have all the data to do such an analysis myself too (as I've been collecting tweets for another project).  So I did.

The caveat is that the tweets that I collected are only those originated from California (and thus my twist). It would have been much more exciting if I had tweets from Brazil directly, but alas, I can't turn back time! Nevertheless, I did find some interesting trends in them (interesting is probably not the most pc word to use here...).

**Number of tweets throughout the day and the game**

I have collected about 250,000 geo-coded tweets yesterday from 8 am to 10 pm. After counting them in the 5-minute intervals, I got the following chart showing the total number of tweets over the day:

[![num](http://www.runzemc.com/wp-content/uploads/2014/08/num-1024x977.jpeg)](http://www.runzemc.com/wp-content/uploads/2014/08/num.jpeg)

Not surprisingly, twitter exploded during the time of the game (indicated by the area between the two dotted lines). As twitter reported itself, there were over 35 million tweets created in total during the game - THIRTY-FIVE MILLION! I wonder if the final will break the record.

Since the majority of the tweets were generated during the game, let's zoom in and focus on 1 pm - 3 pm:

[![num_game](http://www.runzemc.com/wp-content/uploads/2014/08/num_game-1024x977.jpeg)](http://www.runzemc.com/wp-content/uploads/2014/08/num_game.jpeg)[
](http://www.runzemc.com/wp-content/uploads/2014/07/num_game.jpeg)To see how the activities on twitter corresponded to the activities at the game, I added seven purple dotted lines to indicate the seven goals scored by Germany (<insert appropriate adjective>). As we can see, the tweeting intensified after the second goal, which, <insert appropriate adverb>, was followed by two other goals right within the same 5-minute interval. No wonder people went wild. Given how worked up viewers from California were, I can't possibly (and do not particularly want to) imagine how people were reacting right at the game.

**Tweet Sentiment**

The number of tweets aside, what were these tweets actually saying? Were the Californian viewers reacting to the game positively or negatively? Which side were they on? To answer these questions, I calculated the sentiment scores for each tweet using the polarity function in the [qdap](http://cran.r-project.org/web/packages/qdap/qdap.pdf) package, which provides a more sophisticated algorithm to estimate sentiment than simply counting positive and negative words. Since I was only interested in the polarized tweets, I removed all the neutral tweets and charted the number of remaining tweets as follows (in addition, I've only kept the tweets that actually mentioned Brazil or Germany):

[![sentiment](http://www.runzemc.com/wp-content/uploads/2014/08/sentiment-1024x977.jpeg)](http://www.runzemc.com/wp-content/uploads/2014/08/sentiment.jpeg)[
](http://www.runzemc.com/wp-content/uploads/2014/07/sentiment.jpeg)Looking at the whole day, before the game started, there were roughly equal amount of positive and negative tweets. Although they both expanded during the game on the similar manners, we see the number of negative tweets quickly exceeded that of the positive ones. However, they eventually subsided somehow as the game progressed and as the initial shock went away. Now let's focus again on the game:

[![sentiment_game](http://www.runzemc.com/wp-content/uploads/2014/08/sentiment_game-1024x977.jpeg)](http://www.runzemc.com/wp-content/uploads/2014/08/sentiment_game.jpeg)[
](http://www.runzemc.com/wp-content/uploads/2014/07/sentiment_game.jpeg)Ominously, the negative sentiment started right after the game started and greatly exceeded the positive ones during the first three times when Germany scored. However, it is probably too simplistic to conclude that people who expressed a negative sentiment are all fans of Brazil and those that stayed positive are all those of Germany. To really understand what people were feeling, we still need to look at their actual tweets. Here, I created a word cloud showing the most used words during the game (excluding stop words and any neutral words such as the country names, soccer, world cup, and so on):

[![word](http://www.runzemc.com/wp-content/uploads/2014/08/word-300x286.jpeg)](http://www.runzemc.com/wp-content/uploads/2014/08/word.jpeg)

And the most used word is...I'll just let it speak for itself.
