---
layout:     post
title:      "Word2Vec2Graph"
subtitle:   "About Connections"
date:       2018-05-01 12:00:00
author:     "Melenar"
header-img: "img/modern53.jpg"
---

<p><h3>Word2Vec2Graph Model and New Associations</h3>
<p>In<i><a href="http://sparklingdataocean.com/2017/12/24/word2vec2graphPsychoanalysis/"> our previous post</a></i> we proved that high cosine similarity text topics are already well known and topics with low cosine similarities give us new associations and new ideas. </p>
<p>In this post we will show another Word2Vec2Graph method to find new word to word associations. This method is similar to psychoanalysis Free Association techniques as well as methods of finding text topics. As a text file in this post we will use data about Creativity and Aha Moments.
</p>

<p>We will look at word neighbors using 'find' GraphFrames function that we described in <i><a href="http://sparklingdataocean.com/2017/10/03/word2vec2graphNeighbors/">"Word2Vec2Graph Model - Neighbors"</a></i> post.

<h3>Data Preparation</h3>
Data preparation process for Word2Vec2Graph model in described in several post and summarized in the  <i><a href="http://spa.rklingdataocean.com/2017/12/24/word2vec2graphPsychoanalysis/">"Quick Look at Text"</a></i> post. Here we used the same data preparation process of text data about Creativity and Aha Moments:
<ul>
<li>Read text file</li>
<li>Tokenize</li>
<li>Remove stop words</li>
<li>Transform text file to next to each other pairs of words</li>
<li>Retrain Word2Vec model</li>
<li>Calculate cosine similarities within pairs of words</li>
<li>Build a direct graph on pairs of words using words as nodes and cosine similarities as edges weights</li>
<li>Save graph vertices and edges</li>
</ul>
<p></p>

<h3>Build a Graph</h3>


<p>Read graph vertices and edges saved as Parquet to Databricks locations and build the graph for Word2Vec2Graph model.</p>

{% highlight scala %}

import org.graphframes.GraphFrame
val graphInsightVertices = sqlContext.read.parquet("graphInsightVertices")
val graphInsightEdges = sqlContext.read.parquet("graphInsightEdges")

val graphInsight = GraphFrame(graphInsightVertices, graphInsightEdges)

{% endhighlight %}





<a href="#">
    <img src="{{ site.baseurl }}/img/word2vec2graph.jpg" alt="Post Sample Image" width="1400" height="2000">
</a>
