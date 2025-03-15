HW2:

In this homework I started to get into some common NLP tools, specifically, sentiment scoring. 

I've seen sentiment scoring used primarily in finance to gauge how investors feel about markets and the economy, but it is also quite useful for quickly analyzing large amounts of survey data. 

In this assignment I created a function called gen_senti which tokenizes some input text against a pre-uploaded dictionary of positive and negative lexicons. 

It then generates a sentiment score between -1 and 1. 

The example article I used is from yahoo finance, and the topic centered on the difficulties of owning a home in the American southwest. 

The function generated a sentiment score of -0.5, citing some positive words like 'boom' (economic boom) and more negative words like 'cruel', 'desert', or 'struggled'. 

The function was then applied to a folder containing hundreds of different text files and generated sentiment scores for each based on the positive and negative dictionaries provided to it. 

I also experimented with VADER, which proved to be more flexible then the key-word simple-senti function.
