---
layout:     post
title:      "Introduction to Word2Vec2Graph Model"
subtitle:   "Connecting Word2Vec Model with Graph"
date:       2017-09-17 12:00:00
author:     "Melenar"
header-img: "img/sdo14.jpg"
---

<p><h3>Graph and Word2Vec Model </h3>
Word2Vec model maps words to vectors which gives us an opportunity to calculate cosine similarities within pairs of words then translate pairs of words to graph using words as nodes, word pairs as edges and cosine similarities as edge weights.</p>
<p>We are running a small AWS cluster
<i><a href="https://databricks.com/blog/2016/02/17/introducing-databricks-community-edition-apache-spark-for-all.html">on Databricks community edition</a></i>
and for the Word2Vec2Graph model we will use a small size text file with data about stress taken from a Wikipedia article. We will call this text file Stress Data File.</p>
<p>As Word2VecModel we will use the model that was trained on News and Wiki data about psychology corpus. <i><a href="https://sparklingdataocean.github.io/gh-pages/2017/09/06/w2vTrain/"> We described this model in our previous post.</a></i>
<p>
<h3>Read and Clean Stress Data File </h3>
Read Wiki data file: </p>
{% highlight scala %}
val inputStress=sc.
   textFile("/FileStore/tables/stressWiki.txt").
   toDF("charLine")
inputStress.count//--247
{% endhighlight %}

<p>Tokenize Stress data file:</p>
{% highlight java %}
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
val tokenizer = new RegexTokenizer().
   setInputCol("charLine").
   setOutputCol("value").
   setPattern("[^a-z]+").
   setMinTokenLength(5).
   setGaps(true)
val tokenizedStress = tokenizer.
   transform(inputStress)

tokenizedStress.count//--274
{% endhighlight %}

<p>Remove stop words from Stress data file: </p>
{% highlight java %}
val remover = new StopWordsRemover().
   setInputCol("value").
   setOutputCol("stopWordFree")
val removedStopWordsStress = remover.
   setStopWords(Array("none","also","nope","null")++remover.getStopWords).
   transform(tokenizedStress)

{% endhighlight %}

<p> </p>
<p>Explode Stress word arrays to words:</p>
{% highlight scala %}

import org.apache.spark.sql.functions.explode
val slpitCleanWordsStress = removedStopWordsStress.
   withColumn("cleanWord",explode($"stopWordFree")).
   select("cleanWord").
   distinct

slpitCleanWordsStress.count//--1233

{% endhighlight %}
<p> </p>
<p>
<h3>Exclude Words that are not in the Word2Vec Model </h3>
We will use our trained Word2Vec model for word pairs cosine similarities. First, we will read our trained Word2VecModel:</p>
{% highlight scala %}

import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml._
import org.apache.spark.ml.feature.Word2VecModel
import org.apache.spark.sql.Row
val word2vec= new Word2Vec().
   setInputCol("value").
   setOutputCol("result")
val modelNewsWiki=Word2VecModel.
   read.
   load("w2vNewsWiki")
{% endhighlight %}

<p>Next we will get the list of all words from the Word2Vec model:</p>
{% highlight scala %}
val modelWords=modelNewsWiki.
   getVectors
{% endhighlight %}

<p>To be able to use this Word2Vec model for Stress Data file cosine similarities, we will filter out words from Stress Data file that are not in the Word2Vec list of words:</p>
{% highlight scala %}
val stressWords=slpitCleanWordsStress.
   join(modelWords,'cleanWord === 'word).
   select("word","vector").distinct

stressWords.count//--1125

{% endhighlight %}
<p></p>

<p>Finally we will create word to word matrix:</p>
{% highlight scala %}
val stressWords2=stressWords.
   toDF("word2","vector2")
val w2wStress=stressWords.
   join(stressWords2,'word=!='word2)

w2wStress.count//--1264500
{% endhighlight %}

<p>
<h3>Word2Vec Cosine Similarity Function</h3>
Now we want to use Word2Vec cosine similarity to see how words are connected with other words. We will create a function to calculate cosine similarity between vectors from the Word2Vec model</p>
{% highlight scala %}
import org.apache.spark.ml.linalg.Vector
def dotVector(vectorX: org.apache.spark.ml.linalg.Vector,
   vectorY: org.apache.spark.ml.linalg.Vector): Double = {
   var dot=0.0
   for (i <-0 to vectorX.size-1) dot += vectorX(i) * vectorY(i)
   dot
}
def cosineVector(vectorX: org.apache.spark.ml.linalg.Vector,
   vectorY: org.apache.spark.ml.linalg.Vector): Double = {
   require(vectorX.size == vectorY.size)
   val dot=dotVector(vectorX,vectorY)
   val div=dotVector(vectorX,vectorX) * dotVector(vectorY,vectorY)
   if (div==0)0
   else dot/math.sqrt(div)
}

{% endhighlight %}



<p></p>

<p></p>
<p>
<h3>Cosine Similarity between Stress Data File Words</h3>
Now we can calculate word to word cosine similarities between word pairs from Stress Data File and save the results.</p>
{% highlight scala %}
val w2wStressCosDF=w2wStress.
   map(r=>(r.getAs[String](0),r.getAs[String](2),
    cosineVector(r.getAs[org.apache.spark.ml.linalg.Vector](1),
    r.getAs[org.apache.spark.ml.linalg.Vector](3)))).
   toDF("word1","word2","cos")
{% endhighlight %}

<p>Example: Word combinations with high Cosine Similarities:</p>
{% highlight scala %}
display(w2wStressCosDF.
   select('word1,'word2,'cos).
   filter('cos>0.8).limit(7))
word1,word2,cos
disorders,chronic,0.8239098331266418
strategies,processes,0.8079603436193109
loans,mortgage,0.8055626753867968
reduction,increase,0.8029783072858347
capabilities,processes,0.8165733928557892
second,third,0.8717226080244964
second,first,0.8096815780218063
{% endhighlight %}

<p>Example: Word combinations with low Cosine Similarity:</p>
{% highlight scala %}
display(w2wStressCosDF.
   select('word1,'word2,'cos).
   filter('cos<(0.65)).
   filter('cos>(0.6)).limit(7))
word1,word2,cos
interaction,disorders,0.6114415840642784
persist,affect,0.6126901072184042
recognize,affect,0.6309318473017483
interaction,empathy,0.6406613207655409
persist,perceptions,0.6048191825219467
everyday,communicate,0.6137230335862902
recognize,respond,0.6024905770721792
{% endhighlight %}


<p>Store and read Stress Data File word pairs with cosine similarities between them:</p>
{% highlight scala %}
w2wStressCosDF.write.parquet("w2wStressCos")
val w2wStressCos2 = sqlContext.read.parquet("w2wStressCos")
{% endhighlight %}

<p><h3>Graph of Combinations of Stress Data File Words </h3>
Now we can build a graph using words as nodes, {word1, word2} word combinations as edges and cosine similarities between the words as edge weights:</p>
{% highlight scala %}
import org.graphframes.GraphFrame
val graphNodes=w2wStressCosDF.
   select("word1").
   union(w2wStressCosDF.select("word2")).
   distinct.
   toDF("id")
val graphEdges=w2wStressCosDF.
   select("word1","word2","cos").
   distinct.
   toDF("src","dst","edgeWeight")
val graph1 = GraphFrame(graphNodes,graphEdges)

{% endhighlight %}

<p>We will save graph vertices and edges in Parquet format to use them for future posts:</p>

{% highlight scala %}

graph1.vertices.
   write.
   parquet("graphStressNodes")
graph1.edges.
   write.
   parquet("graphStressEdges")

{% endhighlight %}

<p> Load vertices and edges and rebuild the same graph:</p>

{% highlight scala %}
import org.graphframes.GraphFrame
val graphStressNodes = sqlContext.
   read.
   parquet("graphStressNodes")
val graphStressEdges = sqlContext.
   read.
   parquet("graphStressEdges")
val graphStress = GraphFrame(graphStressNodes,graphStressEdges)

{% endhighlight %}
<p></p>

<p>
<h3>Connected Components</h3>
They are many interesting things we can do with Spark GraphFrames. In this post we will play with connected components.
</p>
{% highlight scala %}
sc.setCheckpointDir("/FileStore/")
val resultStressCC = graphStress.
   connectedComponents.
   run()
val ccStressCount=resultStressCC.
   groupBy("component").
   count.
   toDF("cc","ccCt")

display(ccStressCount.orderBy('ccCt.desc))
cc,ccCt
0,1125
{% endhighlight %}

<p>This graph was built on all {word1, word2} combinations of Stress Data File so all word pairs are in the same large connected component. We will look at connected components of subgraphs with different edge weight thresholds.

</p>
<p></p>
<p></p>
<p>
<h3>Connected Components with High Cosine Similarity</h3>
For this post we will use edge weight threshold 0.75, i.e. we will use only word pairs with cosine similarity higher than 0.75.
</p>
{% highlight scala %}

val edgeHightWeight = graphStress.edges.
   filter("edgeWeight > 0.75")
val graphHightWeight = GraphFrame(graphStress.vertices, edgeHightWeight)

{% endhighlight %}

<p>Run connected components for graph with high cosine similarity:
</p>
{% highlight scala %}

val graphHightWeightCC = graphHightWeight.
   connectedComponents.
   run()
val graphHightWeightCcCount=graphHightWeightCC.
   groupBy("component").
   count.
   toDF("cc","ccCt")

display(graphHightWeightCcCount.
   orderBy('ccCt.desc).
   limit(11))
cc,ccCt
60129542144,17
60129542145,9
240518168580,8
575525617665,4
901943132160,4
558345748482,3
901943132166,3
214748364800,3
1108101562370,3
618475290630,2
532575944709,2

{% endhighlight %}

<p>Words in the biggest component:</p>
{% highlight scala %}

display(graphHightWeightCC.
   filter("component=60129542144").
   select('id))
id
humans
harmful
function
illnesses
digestive
chronic
disorder
hormones
symptoms
behavioral
anxiety
cardiovascular
syndrome
prevention
disorders
tumors
acute
{% endhighlight %}

<p>Words in the second component:</p>
{% highlight scala %}

display(graphHightWeightCC.
   filter("component=60129542145").
   select('id))
id
capabilities
governmental
practices
minimize
enhance
strategies
facilitates
functions
processes
{% endhighlight %}
<p></p>

<p>And of course some components are not very interesting:</p>
{% highlight scala %}

display(graphHightWeightCC.
   filter("component=240518168580").
   select('id))
id
increased
increase
decreasing
reduction
decreases
versus
decrease
decreased
{% endhighlight %}
<p></p>
<p>
<h3>Next Post - Word2Vec2Graph Page Rank</h3>
Spark GraphFrames library has many interesting functions. In the next post we will look at Page Rank for Word2Vec2Graph.
</p>
