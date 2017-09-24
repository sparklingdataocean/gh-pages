---
layout:     post
title:      "Word2Vec2Graph for Pairs of Words"
subtitle:   "Connecting Word2vec Model with Graph"
date:       2017-09-29 12:00:00
author:     "Melenar"
header-img: "img/home_sdo8.jpg"
---

<p><h3>Graph and Word2Vec Model </h3>
Word2Vec model maps words to vectors. It allows to calculate cosine similarity within pairs of words. Therefore we can translate pairs of words to graph: words to nodes, pairs to edges and cosine similarities to edge weights.</p>

<p>We are running a small AWS cluster so we will play with a small text file with Wiki data about psychology.
As the Word2VecModel model we will use the model that was trained on the corpus of combined News data and Wiki data files.</p>

<p><h3>Read and Clean Wiki File Data</h3>
Read Wiki input file: </p>
{% highlight scala %}
val inputWiki=sc.textFile("/FileStore/tables/opef4cqb1504567762417/WikiTest.txt").toDF("charLine")
{% endhighlight %}

<p>Tokenize the file:</p>
{% highlight java %}
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
val tokenizer = new RegexTokenizer().
setInputCol("charLine").setOutputCol("value").setPattern("[^a-z]+").
setMinTokenLength(3).setGaps(true)
val tokenizedWiki = tokenizer.transform(inputWiki)
{% endhighlight %}

<p>Remove stop words: </p>
{% highlight java %}
val remover = new StopWordsRemover().setInputCol("value").setOutputCol("stopWordFree")
val removedStopWordsWiki = remover.setStopWords(Array("none","also","nope","null")++remover.getStopWords).transform(tokenizedWiki)
{% endhighlight %}

<p><h3>Transform it to Pairs of Words</h3>
Get pairs of words - ngram(2)</p>
{% highlight scala %}
val ngram = new NGram().setInputCol("stopWordFree").setOutputCol("ngrams").setN(2)
val ngramCleanWordsWiki = ngram.transform(removedStopWordsWiki)
{% endhighlight %}

<p>Explode ngrams:</p>
{% highlight scala %}

import org.apache.spark.sql.functions.explode
val slpitNgramsWiki=ngramCleanWordsWiki.withColumn("ngram",explode($"ngrams")).select("ngram").
map(s=>(s(0).toString,s(0).toString.split(" ")(0),s(0).toString.split(" ")(1))).//
toDF("ngram","ngram1","ngram2").filter('ngram1=!='ngram2)

{% endhighlight %}

<p><h3>Exclude Word Pairs that are not in the Word2Vec Model </h3>

Read the trained Word2VecModel</p>
{% highlight scala %}

import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml._
import org.apache.spark.ml.feature.Word2VecModel
import org.apache.spark.sql.Row
val word2vec= new Word2Vec()
  .setInputCol("value")
  .setOutputCol("result")
val modelNewsWiki=Word2VecModel.read.load("w2vNewsWiki")

{% endhighlight %}

<p>Get all words from the Word2Vec model</p>
{% highlight scala %}
val modelWords=modelNewsWiki.getVectors.select("word")
{% endhighlight %}

<p>Filter out word pairs that are not in words of Word2Vec model</p>
{% highlight scala %}
val ngramW2Vwiki=slpitNgramsWiki.join(modelWords,'ngram1==='word).join(modelWords.toDF("word2"),'ngram2==='word2).
select("ngram","ngram1","ngram2").distinct
{% endhighlight %}

<p><h3>Word2Vec Cosine Similarity Function</h3>
Now we want to see how pairs of words are connected. We will create a new function to calculate cosine similarity between words from the Word2Vec model</p>
{% highlight scala %}
import org.apache.spark.ml.linalg.DenseVector
def dotDouble(x: Array[Double], y: Array[Double]): Double = {
    (for((a, b) <- x zip y) yield a * b) sum
  }
def magnitudeDouble(x: Array[Double]): Double = {
    math.sqrt(x map(i => i*i) sum)
  }
def cosineDouble(x: Array[Double], y: Array[Double]): Double = {
    require(x.size == y.size)
    dotDouble(x, y)/(magnitudeDouble(x) * magnitudeDouble(y))
}

val modelMap = sc.broadcast(modelNewsWiki.getVectors.map(r=>(r.getString(0),r.getAs[DenseVector](1).toArray)).collect.toMap)

def w2wCosine(word1: String, word2: String): Double = {
    cosineDouble(modelMap.value(word1),modelMap.value(word2))
}
{% endhighlight %}

<p>Example: Word2Vec cosine similarity</p>
{% highlight scala %}
w2wCosine("stress","idea")

res1: Double = 0.2533538702772619
{% endhighlight %}

<p><h3>How Word Pairs are Connected?</h3>
Next step: calculate cosine similarity of word pairs:</p>
{% highlight scala %}
val ngBroadcastWiki=sc.broadcast(ngramW2Vwiki.collect)
val ngramWord2Vec=ngBroadcastWiki.value.map(s=>(s(0).toString,s(1).toString,s(2).toString,
w2wCosine(s(1).toString,s(2).toString)))
val ngramWord2VecDF=sc.parallelize(ngramWord2Vec).toDF("ngram","ngram1","ngram2","cos")
{% endhighlight %}

<p>Example: Word Pairs with High Cosine Similarity:</p>
{% highlight scala %}
display(ngramWord2VecDF.select('ngram,'cos).filter('cos>0.8))

ngram,cos
buenos aires,0.9164193562252616
johns hopkins,0.82613953851904
hundreds thousands,0.8022228190557517
psychology harvard,0.8081958719357202
psychology sociology,0.8133461799616877
soldiers civilians,0.8432130270116076
{% endhighlight %}

<p>Example: Word Pairs with Low Cosine Similarity:</p>
{% highlight scala %}
display(ngramWord2VecDF.select('ngram,'cos).filter('cos<(-0.33)))

ngram,cos
motivation according,-0.334148527691187
james defined,-0.3599664443972162
spirituality sexual,-0.33271384384143715
funding alfred,-0.37083478956090055
described sizeable,-0.3927395029617375
assignment different,-0.34060692231352563
{% endhighlight %}

<p><h3>Graph on Word Pairs</h3>
Now we can build a graph on word pairs:</p>
{% highlight scala %}
import org.graphframes.GraphFrame

val graphNodes=ngramWord2VecDF.select("ngram1").
  union(ngramWord2VecDF.select("ngram2")).distinct.toDF("id")
val graphEdges=ngramWord2VecDF.select("ngram1","ngram2","cos").
  distinct.toDF("src","dst","edgeWeight")
val graph = GraphFrame(graphNodes,graphEdges)

{% endhighlight %}

<p>We will Save graph vertices and edges as Parquet to some locations</p>

{% highlight scala %}

graph.vertices.write.parquet("graphWikiNodes")
graph.edges.write.parquet("graphWikiEdges")

{% endhighlight %}

<p> Load vertices and edges and rebuild the same graph back</p>

{% highlight scala %}
val graphWikiNodes = sqlContext.read.parquet("graphWikiNodes")
val graphWikiEdges = sqlContext.read.parquet("graphWikiEdges")

val graphWiki = GraphFrame(graphWikiNodes, graphWikiEdges)

{% endhighlight %}

<p><h3>Connected Components</h3>
It's a lot of interesting things we can do with Spark GraphFrames. Let's start with connected components.
</p>
{% highlight scala %}

sc.setCheckpointDir("/FileStore/")
val resultWikiCC = graphWiki.connectedComponents.run()

val ccWikiCount=resultWikiCC.groupBy("component").count.toDF("cc","ccCt")
display(ccWikiCount.orderBy('ccCt.desc))

{% endhighlight %}

<p>Most of word pairs are in the same large connected component and only a few pairs are separate:
</p>

<p><h3>Connected Components with High Cosine Similarity</h3>
Now we will look at connected components of subgraph with edge weight threshold > 0.6.
</p>
{% highlight scala %}

val edgeHightWeight = graphWiki.edges.filter("edgeWeight > 0.6")
val graphHightWeight = GraphFrame(graphWiki.vertices, edgeHightWeight)

{% endhighlight %}

<p>Run connected components for graph with high cosine similarity
</p>
{% highlight scala %}
val graphHightWeightCC = graphHightWeight.connectedComponents.run()
val graphHightWeightCcCount=graphHightWeightCC.groupBy("component").count.toDF("cc","ccCt")
display(graphHightWeightCcCount.orderBy('ccCt.desc).limit(7))

{% endhighlight %}

<p>Words in the biggest component:</p>
{% highlight scala %}

display(graphHightWeightCC.filter("component=103079215114"))

id,component
institute,103079215114
harvard,103079215114
cornell,103079215114
psychiatry,103079215114
maryland,103079215114
hopkins,103079215114
yale,103079215114
michigan,103079215114
consultant,103079215114
university,103079215114
johns,103079215114
engineering,103079215114
psychology,103079215114
sociology,103079215114
{% endhighlight %}

<p>Words in the second component:</p>
{% highlight scala %}

display(graphHightWeightCC.filter("component=171798691847"))

id,component
therapy,171798691847
functions,171798691847
cognitive,171798691847
clinical,171798691847
behavioral,171798691847
practices,171798691847
educational,171798691847
maps,171798691847
genetic,171798691847
{% endhighlight %}

<p><h3>Next Post - Page Rank</h3>
Spark GraphFrames library has many interesting functions. In the next post we will look at Page Rank for Word2Vec2Graph.
</p>
