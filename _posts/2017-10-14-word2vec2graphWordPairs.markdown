---
layout:     post
title:      "Word2Vec2Graph for Pairs of Words"
subtitle:   "Connecting Word2vec Model with Graph"
date:       2017-10-14 12:00:00
author:     "Melenar"
header-img: "img/klee2.jpg"
---

<p><h3>Word2Vec2Graph Model - Direct Graph</h3>
In previous posts we introduced
<i><a href="https://sparklingdataocean.github.io/gh-pages/2017/09/17/word2vec2graph/">Word2Vec2Graph model in Spark</a></i>.
Word2Vec2Graph model connects Word2Vec model with Spark GraphFrames library and gives us new opportunities to use graph approach to text mining.</p>


<p>
As Word2Vec model we will use the same
<i><a href="https://sparklingdataocean.github.io/gh-pages/2017/09/06/w2vTrain/">Word2Vec model </a></i> that was trained on the corpus of combined News data and Wiki data files.</p>

<p>
As a text file we will use the same Stress Data file - a small text file with data about stress that was copied from Wikipedia article. In previous posts we looked at graph for all pairs of words from Stress Data file. Now we will look at pairs of words that stay next to each other in text file and will use these pairs as graph edges.</p>

<h3>Read and Clean Stress Data File </h3>
Read Stress Data file:
{% highlight scala %}
val inputStress=sc.textFile("/FileStore/tables/cjzokasj1506175253652/stressWiki.txt").
   toDF("charLine")

{% endhighlight %}

<p>Using Spark ML functions tokenize and remove stop words from Stress Data file:</p>
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
val remover = new StopWordsRemover().
   setInputCol("value").
   setOutputCol("stopWordFree")
val removedStopWordsStress = remover.
   setStopWords(Array("none","also","nope","null")++
   remover.getStopWords).
   transform(tokenizedStress)

{% endhighlight %}


<p><h3>Transform the results to Pairs of Words</h3>
Get pairs of words - use Spark ML library ngram function:</p>
{% highlight scala %}
val ngram = new NGram().
   setInputCol("stopWordFree").
   setOutputCol("ngrams").
   setN(2)
val ngramCleanWords = ngram.
   transform(removedStopWordsStress)
{% endhighlight %}

<p>Explode ngrams:</p>
{% highlight scala %}

import org.apache.spark.sql.functions.explode
val slpitNgrams=ngramCleanWords.
   withColumn("ngram",explode($"ngrams")).
   select("ngram").
   map(s=>(s(0).toString,
      s(0).toString.split(" ")(0),
      s(0).toString.split(" ")(1))).
   toDF("ngram","ngram1","ngram2").
   filter('ngram1=!='ngram2)


{% endhighlight %}

<p><h3>Exclude Word Pairs that are not in the Word2Vec Model </h3>

Read the trained Word2Vec model</p>
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

<p>Get a set of all words from the Word2Vec model</p>
{% highlight scala %}
val modelWords=modelNewsWiki.
   getVectors.
   select("word")
{% endhighlight %}

<p>Filter out word pairs with words that are not in the set of words from the Word2Vec model</p>
{% highlight scala %}

val ngramW2V=slpitNgrams.
   join(modelWords,'ngram1==='word).
   join(modelWords.toDF("word2"),'ngram2==='word2).
   select("ngram","ngram1","ngram2").
   distinct

{% endhighlight %}



<p><h3>How Word Pairs are Connected?</h3>
Now we will calculate cosine similarities of words within word pairs.

We already introduced Word2Vec Cosine Similarity Function in the  
<i><a href="https://sparklingdataocean.github.io/gh-pages/2017/09/17/word2vec2graph/">Word2Vec2Graph model Introduction post.</a></i>
</p>
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
val modelMap = sc.broadcast(modelNewsWiki.
   getVectors.
   map(r=>(r.getString(0),r.getAs[DenseVector](1).toArray)).
      collect.toMap)
def w2wCosine(word1: String, word2: String): Double = {
   cosineDouble(modelMap.value(word1),modelMap.value(word2))
}
{% endhighlight %}

Next step: calculate cosine similarity within ngrams:</p>
{% highlight scala %}

val ngBroadcast=sc.broadcast(ngramW2V.collect)
val ngW2V=ngBroadcast.
   value.
   map(s=>(s(0).toString,
      s(1).toString,
      s(2).toString,
      w2wCosine(s(1).toString,s(2).toString)))
val ngramWord2VecDF=sc.parallelize(ngW2V).
   toDF("ngram","ngram1","ngram2","cos")
display(ngramWord2VecDF.limit(7))
ngram,ngram1,ngram2,cos
humans chronic,humans,chronic,0.4952597099545433
individuals exposed,individuals,exposed,0.18908252800993414
general suppression,general,suppression,0.18860520433779804
response partial,response,partial,0.4066760138993952
environmental stimulus,environmental,stimulus,0.21930354206811606
stress subject,stress,subject,0.4080685672179879
hospitalized children,hospitalized,children,0.31194855489247675
{% endhighlight %}

<p>Example: Word Pairs with High Cosine Similarity:</p>
{% highlight scala %}
display(ngramWord2VecDF.
   select('ngram,'cos).
   filter('cos>0.7).
   orderBy('cos.desc))
ngram,cos
acute chronic,0.7848571640793651
governmental organizations,0.7414504735574394
realistic helpful,0.730824091817287
disease chronic,0.7064366889098306
feelings thoughts,0.7000105635150229
thoughts feelings,0.7000105635150229


{% endhighlight %}

<p>Example: Word Pairs with Cosine Similarity close to 0:</p>
{% highlight scala %}
display(ngramWord2VecDF.
   select('ngram,'cos).
   filter('cos>(-0.002)).filter('cos<(0.002)).orderBy('cos))
ngram,cos
researchers interested,-0.0019752767768097153
defense mechanisms,-0.0014974826488316265
whether causes,-0.0008734112750530817
share others,0.0002295526607795157
showed direct,0.00045697478567580015
individual takes,0.0017983474881583593
{% endhighlight %}

<p><h3>Graph on Word Pairs</h3>
Now we can build a graph on word pairs: words will be nodes, ngrams - edges and cosine similarities - edge weights.</p>
{% highlight scala %}
import org.graphframes.GraphFrame
val graphNodes1=ngramWord2VecDF.
   select("ngram1").
   union(ngramWord2VecDF.select("ngram2")).
   distinct.
   toDF("id")
val graphEdges1=ngramWord2VecDF.
   select("ngram1","ngram2","cos").
   distinct.
   toDF("src","dst","edgeWeight")
val graph1 = GraphFrame(graphNodes1,graphEdges1)

{% endhighlight %}

<p>To use this graph in several posts we will save graph vertices and edges as Parquet to Databricks locations.</p>

{% highlight scala %}

graph1.vertices.write.
   parquet("graphNgramVertices")
graph1.edges.write.
   parquet("graphNgramEdges")

{% endhighlight %}

<p> Load vertices and edges and rebuild the same graph back</p>

{% highlight scala %}
val graphNgramStressVertices = sqlContext.read.
   parquet("graphNgramVertices")
val graphNgramStressEdges = sqlContext.read.
   parquet("graphNgramEdges")
val graphNgramStress = GraphFrame(graphNgramStressVertices, graphNgramStressEdges)

{% endhighlight %}


<p><h3>Page Rank</h3>
Calculate Page Rank: </p>
{% highlight scala %}
val graphNgramStressPageRank = graphNgramStress.
   pageRank.
   resetProbability(0.15).
   maxIter(11).
   run()
display(graphNgramStressPageRank.vertices.
   distinct.
   sort($"pagerank".desc).
   limit(11))

   id,pagerank
   stress,36.799029843873065
   social,8.794399876715186
   individual,8.756866689676286
   person,8.466242702036295
   stressful,7.9825617601531444
   communication,7.274847096155088
   health,6.398223040310048
   situation,5.924707831050667
   events,5.7227621841425975
   changes,5.642126628136843
   chronic,5.2918611240572755
{% endhighlight %}


<p><h3>Next Post - Connected Word Pairs</h3>
In the next post we will play with Spark GraphFrames library and run Connected Components and Label Propagation functions for direct Word2Vec2Graph model.</p>
