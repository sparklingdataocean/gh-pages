---
layout:     post
title:      "Word2Vec2Graph for Pairs of Words"
subtitle:   "Binding Pairs of Words"
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
val inputStress=sc.textFile("/FileStore/tables/stressWiki.txt").
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

display(slpitNgrams)
ngram,ngram1,ngram2
psychological stress,psychological,stress
wikipedia encyclopedia,wikipedia,encyclopedia
kinds stress,kinds,stress
stress disambiguation,stress,disambiguation
video explanation,video,explanation
psychology stress,psychology,stress
stress feeling,stress,feeling
feeling strain,feeling,strain
strain pressure,strain,pressure
pressure small,pressure,small
small amounts,small,amounts
amounts stress,amounts,stress
stress desired,stress,desired
desired beneficial,desired,beneficial
beneficial healthy,beneficial,healthy
healthy positive,healthy,positive
positive stress,positive,stress
{% endhighlight %}

<p><h3>Exclude Word Pairs that are not in the Word2Vec Model </h3>

In the post where we
<i><a href="https://sparklingdataocean.github.io/gh-pages/2017/09/17/word2vec2graph/"> introduced Word2Vec2Graph model</a></i>, we calculated cosine similarities of all word-to-word combinations of
Stress Data File based on Word2Vec model and saved the results.
</p>
{% highlight scala %}

val w2wStressCos = sqlContext.read.parquet("w2wStressCos")
display(w2wStressCos.filter('cos< 0.1).filter('cos> 0.0).limit(7))
word1,word2,cos
conducted,contribute,0.08035969605150468
association,contribute,0.06940379539008698
conducted,crucial,0.0254494353390933
conducted,consequences,0.046451274237478545
exhaustion,ideas,0.08462263299060188
conducted,experience,0.05733563656740034
conducted,inflammation,0.09058846853618428

{% endhighlight %}



<p>Filter out word pairs with words that are not in the set of words from the Word2Vec model</p>
{% highlight scala %}

val ngramW2V=slpitNgrams.
   join(w2wStressCos,'ngram1==='word1 && 'ngram2==='word2).
   select("ngram","ngram1","ngram2","cos").distinct
{% endhighlight %}

<p>Example: Word Pairs with high Cosine Similarity >0.7:</p>
   {% highlight scala %}
display(ngramW2V.
   select('ngram,'cos).
   filter('cos>0.7).orderBy('cos.desc))
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
val graphNodes1=ngramW2V.
   select("ngram1").
   union(ngramW2V.select("ngram2")).
   distinct.toDF("id")
val graphEdges1=ngramW2V.  
   select("ngram1","ngram2","cos").
   distinct.toDF("src","dst","edgeWeight")
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
