---
author: Runze
comments: true
date: 2014-09-02 02:05:07+00:00
link: http://www.runzemc.com/2014/09/what-mining-my-own-emails-told-me-about-myself.html
slug: what-mining-my-own-emails-told-me-about-myself
title: What mining my own emails told me about myself
wordpress_id: 238
categories:
- Data Analysis
tags:
- gmail
- Python
- R
---

On Tuesday last week, I attended a data visualization meetup organized by [Data Science LA](http://datascience.la) and the topic was about the most recent [Eyeo Festival](http://eyeofestival.com). Of all the talks that [Amelia](https://twitter.com/AmeliaMN) shared with us, what impressed me and inspired me the most was [Nicholas Felton](http://feltron.com)'s personal data projects. In case you are not familiar with him, every year he publishes an annual report that documents his personal data projects / experiments conducted throughout the year. In 2013, he spent a whole year recording and logging all his communications with people, which include emails, text messages, and conversations. Imagine diligently and painstakingly doing that for an entire year! With all those data, he did many interesting analyses and visualizations like the one [here](http://feltron.com/FAR13_06.html).

The reason it left me with such a strong impression is that, after seeing so many examples of people using data to understand the world and the society better, this is the first example I've seen of someone using data to understand himself better. Yet it just seems such a natural thing to do. Every day as we are bombarded with knowledge and information, it is so easy to lose ourselves in this race. Did I change from what I was a year ago? Who are the important people in my life now? And who am I losing? True, compared with the bigger problems in the world, on a grand scheme of things, these seem so trivial. But without understanding yourself, how can you be you?

Another reason that makes me think about these things is the latest book by Haruki Murakami, titled _[Colorless Tsukuru Tazaki and His Years of Pilgrimage](http://www.amazon.com/Colorless-Tsukuru-Tazaki-Years-Pilgrimage/dp/0385352107)_, which I just finished reading. Without turning this post into a book review (and really, despite how much I want to, I can't possibly put my thoughts upon reading this book in words), I'll just mention the main theme of this book is about prying open the lid that sealed the unspeakable past and revisiting the history in order to move on. On a lighter note, this book explores past friendships and how those friends are these days after everyone went separate ways years ago. While reading it, I couldn't help but think the old friends that I used to be close to but don't talk to much these days anymore, mostly due to geographic reasons.

After mulling these questions over and thinking about what I can do with the available resources I have, I came up with the idea of mining my own emails because they gave me an honest picture of who I've been actively in touch with over the years. Of course, email is not the sole means of communication for me, and, without any log of conversations like Felton's, text messages are an ideal supplement. However, I did not save most of my texts (and most of the old ones were sent and received on "dumb" phones anyway). Therefore, I settled down on emails.

**Retrieving emails in Python**

Luckily, 7 years ago, with some tremendous foresight, I switched my email agent to gmail to happily discover 7 years later that there are an abundance of libraries written for it. After some researching and playing around, I found this unofficial [package](https://github.com/charlierguo/gmail) written by Charlie Guo the most easy to use. You can see my actual code on my [github](https://github.com/Runze/mining_gmail) page but essentially all you need to do is connect to the email server using your credentials (note if you have the 2-step verification turned on, this won't work), select a mailbox to download, and use `fetch()` to retrieve the actual message. What I struggled initially is decoding and encoding the emails written in Chinese. Not that I can do much with them (to my knowledge, the text mining in R does not work too well with Chinese characters), I do need them to be properly decoded and encoded again in 'utf-8' to write to json. The solution I came up with is first use `chardet.detect()` to first detect the underlying encoding of the message, and, after decoding it using the detection result, encode it in 'utf-8'. For each email, I fetched its subject, body, recipient, and timestamp.

But wait, what mailbox should I retrieve messages from? I decided to go with the sent box because my inbox just includes too many emails that I don't bother to reply to, most of which are mass emails that passed the spam test. Sent box, on the other hand, contains emails that I evidently wrote myself and the conversations that I was actually engaged in. Hence, the emails saved in there should give me a better understanding of my active communications over the years.

**What I have found**

After gathering all the emails I sent over the years (2,572 in total since my registration in October 2008), the first thing I did is plotting the number of emails per year and what I found is rather surprising.

<img src="https://raw.githubusercontent.com/Runze/mining_gmail/master/msg_yr.jpeg" alt="alt text" width="600">

What was surprising to me is that, before doing this analysis, I thought the emails I sent per year should be somewhat constant, but it turned out it varied a lot over the years. Specifically, it seems that I was particularly sociable in 2010 and very introverted in 2012 (2008 should be ignored since it only represents 3 months). Hmm, what happened in those years? Upon reviewing the emails from back then (and trust me, it was actually quite fun reading what you wrote and remembering what you were doing years ago), I realized the reason I was very active in 2010-2011 is that I was in grad school and the vast majority of these additional emails were communications with fellow students, professors, and, later, recruiting companies. After all this were settled, I guess I returned to my normal ~200-emails-a-year lifestyle with communications with close friends only (and as we'll see shortly, most of these communications were with a set group of close friends). In 2014, there is another upward trend again, and that's because I just reentered the job market and my emails with recruiters make up the margin.

Now who am I sending all these emails to? To answer this question, I created the following bar plot showing the top 10 recipients of my emails each year. To keep anonymity, I hid their actual email addresses and assigned aliases to them.

<img src="https://raw.githubusercontent.com/Runze/mining_gmail/master/dest.jpeg" alt="alt text" width="600">

Looking at this alone brought back so many memories, especially because the last 6-7 years were quite eventful to me. Over those years, I packed up my bag and moved from China to the U.S., attended grad school for 2 years, and was lucky enough to stay afterwards and has been working until now. All this has been faithfully documented in my emails: in the last 3 months of 2008, my life was occupied by school applications, thesis project, and graduation; 2009-2011 were full of grad school emails, conversations about classes and exams, internship and full-time job applications and interviews; starting from 2012, those emails have almost completely disappeared and were replaced by communications with new friends made at workplace and services such as car dealership and insurance agency; this trend continued until the present time when emails with recruiters recently surged. Yep, that's the last 7 years of my life (still can't believe it's been so long already)!

Looking at this graph closely, there are 2 things that stand out to me: 1) I've lost in touch with a lot of friends along the way, most of them were friends from school, and 2) nowadays I mainly communicate with a small group of close friends (at least via emails). The first discovery is rather sad. Take my No. 1 recipient in 2008 for example (whom I dubbed as "childhood friend; applied to school and came to the US together"), although not evident in this 7-year email history, we go way back. We went to the same elementary school, middle school, and then high school. Although not in the same college, we both took the TOEFL and GRE tests and applied to the U.S. grad school together, and ultimately both ended up in USC. Our emails in 2008 were mostly about school applications and essay editing and commenting. They later changed to school selection, visa application, housing application, and, eventually, flight booking and travel planning. Since we studied in different programs at USC, our communications gradually subsided, and, after settling down in different cities after graduation, we seldom talk nowadays. Quantitatively, here is how our email communications evolved (it was reduced to 0 completely after 2011):

<img src="https://raw.githubusercontent.com/Runze/mining_gmail/master/friend.jpeg" alt="alt text" width="600">

Sad, isn't it? I don't even know what she is up to these days anymore. We did meet up over a year ago but we no longer had much to talk about. Similarly, I'm not in touch with any of my college friends, whom I was used to be very close with. After graduation, we all moved to different parts of the world and had our own lives. I guess this is just life then - sad but also strangely natural.

On the other hand, my communications with people nowadays are closely tied to a small focus group that is made up of mostly friends made from work (e.g., "former coworker 1 / great friend" topped my list for 3 consecutive years). It's true my social circle has become much smaller compared to my early student years, but at least I'm glad I'm able to make these new friends who all have very good influence on me :-)

Now that I found out whom I've been talking to over the years, what have we been talking about? As much as I want to, I don't have time to read through all the old emails, so I decided to do some topic modeling to get a general idea. To do that, I cleaned the body of all the emails, picked out nouns, constructed a document-term matrix, and ran LDA on them. To determine the optimal number of topics, I used 10-fold cross-validation and, as shown below, the performance (as indicated by the perplexity measurement) leveled off at around 16 topics - apparently the algorithm thinks I'm not diverse enough to make up 16 topics to talk about!

<img src="https://raw.githubusercontent.com/Runze/mining_gmail/master/perplex.jpg" alt="alt text" width="600">

Now all the topics are clear cut though and there are some recurring themes. To save space, I'm only showing 4 of the most apparent topics here:

<img src="https://raw.githubusercontent.com/Runze/mining_gmail/master/lda1_student.png" alt="alt text" width="600">

<img src="https://raw.githubusercontent.com/Runze/mining_gmail/master/lda5_school.png" alt="alt text" width="600">

<img src="https://raw.githubusercontent.com/Runze/mining_gmail/master/lda3_job app.png" alt="alt text" width="600">

<img src="https://raw.githubusercontent.com/Runze/mining_gmail/master/lda15_banking & consulting.png" alt="alt text" width="600">

The 2 on the top are clearly related to my experience as a student (oh how I miss those days!) and the bottom 2 are related to my job hunting experience (from applications to interviews), which reconfirms the fact that these 2 buckets made up the overarching theme of my emails.

Another question or hypothesis that I'm interested in testing is whether I have become happier or moodier over the years. I kind of have this feeling that I'm heading towards the latter but I don't have any evidence to back it up, and who is a better judge than data and statistical tests?

To assess the sentiment of the messages, I relied on the `polarity` from the [`qdap`](http://cran.r-project.org/web/packages/qdap/qdap.pdf) package. After running it on all my emails throughout the years, this is what I found out about how my average sentiment change over the time:

<img src="https://raw.githubusercontent.com/Runze/mining_gmail/master/sent.jpeg" alt="alt text" width="600">

Hmm, what happened in 2012? Looks like I had a really rough year. Whatever happened, I'm glad I was able to bounce back and move my sentiment up. It's also interesting to divide the 7 years into 2 groups: pre-2011 and post-2011 because I finished my master's program and ended my student years in 2011 and entered the workforce for the first time. Hence, the hypothesis is whether I was happier as a student or as a working professional. To find out, I performed a t-test and a wilcoxon ranksum test. The former assumes normality of the distribution while the latter does not. To test the normality assumption, I made 2 quantile-quantile plot for the 2 subgroups (as compared to the theoretical normal distribution), and as you can see, although close, they are not exactly normal:

<img src="https://raw.githubusercontent.com/Runze/mining_gmail/master/qq.jpeg" alt="alt text" width="600">

I ran both tests anyway and the results I got are unanimous:

<img src="https://raw.githubusercontent.com/Runze/mining_gmail/master/t_test.png" alt="alt text" width="600">

Both the p-values are abysmally small, indicating I am indeed getting grumpier after college!

**Conclusion**

I think mining one's own data is a very interesting, necessary, and often overlooked exercise. Perhaps the idea sounds a little silly and unintuitive at first since, presumably, we know ourselves well enough to need any data to back it up. But if you happen to have any historical record of yourself, whether they are emails, text messages, journals, or letters, I highly recommend going through them, as farther back as possible, if only just to be amazed by how much you've changed over the years (and trust me, you will be amazed). For example, as I was going through the emails I sent when I first came to the U.S., I couldn't believe I once talked like that (and I surely hope no one ever found out). On the other end of the spectrum, doing so also enabled me to discover what I have lost over the years, whether they are friends, prior knowledge, or perspectives. Hence, it's a very valuable introspective practice, not to mention also a ton of fun. Happy digging :-)
