---
author: ernuzwang@gmail.com
comments: true
date: 2014-07-01 06:40:44+00:00
link: http://www.runzemc.com/2014/07/no-more-finding-out-about-a-concert-too-late.html
slug: no-more-finding-out-about-a-concert-too-late
title: No more finding out about a concert too late
wordpress_id: 58
categories:
- Data Analysis
tags:
- Pitchfork
- R
- Shiny
---

<blockquote>The R code used to create this app has been uploaded to [github](https://github.com/Runze/pitchfork).</blockquote>


After the numerous times of finding out about a concert too late and ending up either paying a premium or not being able to go, I finally decided to do something about it, and this is what I came up with.

[https://runzemc.shinyapps.io/pitchfork/](https://runzemc.shinyapps.io/pitchfork/)

This R Shiny app I made pulls data from [pitchfork](http://pitchfork.com/) automatically every time it's open and shows the upcoming shows per city and per artist (indie artist, to be precise).  Specifically, it pulls data from the website's [tours](http://pitchfork.com/news/tours/) page, which is frequently updated to include all the new announcement made by the artists with regard to their upcoming shows. After gathering the raw html data, the app retrieves the relevant tour information from it (unfortunately, readHTMLTable doesn't work here), cleans and aggregates it, and filters out past events. The key part of this app is the reactive function that gathers new data per each run and updates the table output as well as the UI input accordingly so that the user is only presented with the most recent options from the two drop-down boxes (i.e., the available cities and artists to choose from on the side panel are only those that currently have a show in the future). You can learn to do it from the shiny [tutorial](http://shiny.rstudio.com/tutorial/lesson6/) and this very helpful demo [code](https://github.com/wch/testapp/blob/master/setinput/server.R) on using the observe function to reactively updating the input.

Screenshot time!

Here is a snapshot of the upcoming concerts in LA:

[![Screen Shot 2014-08-18 at 7.47.35 PM](http://www.runzemc.com/wp-content/uploads/2014/07/Screen-Shot-2014-08-18-at-7.47.35-PM-1024x600.png)](http://www.runzemc.com/wp-content/uploads/2014/07/Screen-Shot-2014-08-18-at-7.47.35-PM.png)

Each link will open the search result of the artist on [Songkick](http://www.songkick.com/), where one can buy tickets or follow the artist (you are welcome).

Below is the artist tab for Lorde, who is coming to LA in October (not shown in the snapshot page)!

[![Screen Shot 2014-08-18 at 7.49.40 PM](http://www.runzemc.com/wp-content/uploads/2014/07/Screen-Shot-2014-08-18-at-7.49.40-PM-1024x598.png)](http://www.runzemc.com/wp-content/uploads/2014/07/Screen-Shot-2014-08-18-at-7.49.40-PM.png)

Enjoy the concert :-)
