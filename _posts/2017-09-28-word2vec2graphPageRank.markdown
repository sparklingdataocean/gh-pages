---
layout:     post
title:      "Word2Vec2Graph Page Rank"
subtitle:   "Connecting Word2vec Model with Graph"
date:       2017-09-28 12:00:00
author:     "Melenar"
header-img: "img/sdo11.jpg"
---

<p><h3>Word2Vec2Graph Model</h3>
In the previous post we explained how to build Word2Vec2Graph model as combination of Word2vec model and Graph engine in Spark.
Spark GraphFrames library has many interesting functions. In this post we will look at Page Rank for Word2Vec2Graph.
</p>



<p>We will use the Word2Vec2Graph model based on a small Stress data file - text file about stress that we extracted from Wikipedia. As edge weights we used cosine similarity based on Word2VecModel model trained on the corpus of combined News data and Wiki data.</p>

<p><h3>Get Data from Data Storage</h3>
Read word to word Stress Data File combinations with Word2Vec cosine similarities: </p>
{% highlight scala %}
val w2wStressCos = sqlContext.read.parquet("w2wStressCos")
{% endhighlight %}

Read vertices and edges of Word2Vec2Graph and build a graph: </p>
{% highlight scala %}
import org.graphframes.GraphFrame
val graphStressNodes = sqlContext.read.parquet("graphStressNodes")
val graphStressEdges = sqlContext.read.parquet("graphStressEdges")
val graphStress = GraphFrame(graphStressNodes,graphStressEdges)
{% endhighlight %}

<p><h3>Page Rank</h3>
Calculate Page Rank: </p>
{% highlight scala %}
val stressPageRank = graphStress.pageRank.resetProbability(0.15).maxIter(11).run()
display(stressPageRank.vertices.distinct().sort($"pagerank".asc).limit(11))
{% endhighlight %}

<p>Our graph is built on the full matrix so all words pairs are connected therefore we are getting all Page Ranks equal to 1. Now we will look at Page Rank of a subgraph based on the edge weight threshold. We will use the same threshold (>.075) as we used in the previous post when we calculated graph  connected components.</p>
<p>Build subgraph: </p>
{% highlight java %}
val edgeHightWeight = graphStress.edges.filter("edgeWeight > 0.75")
val graphHightWeight = GraphFrame(graphStress.vertices, edgeHightWeight)
{% endhighlight %}

<p>Calculate Page Rank: </p>
{% highlight java %}
val stressHightWeightPageRank = graphHightWeight.pageRank.
resetProbability(0.15).maxIter(11).run()
display(stressHightWeightPageRank.vertices.distinct().sort($"pagerank".desc).limit(11))

id,pagerank
hormones,11.925990421899789
processes,11.314750484908657
disorders,10.58017031766379
necessarily,9.314082346458498
decreased,7.890450585933449
decrease,7.7083413274183785
worries,6.90389557706388
methods,6.90389557706388
interact,6.90389557706388
second,6.90389557706388
cardiovascular,6.061954567119659
{% endhighlight %}

<p><h3>Page Rank and Degrees</h3>
Graph that we use now is indirect so high Page Rank vertices are similar to high in-degree vertices: </p>
{% highlight scala %}
val vertexInDegrees= graphHightWeight.inDegrees
display(vertexInDegrees.orderBy('inDegree.desc).limit(11))
id,inDegree
processes,6
disorders,6
decrease,5
decreased,5
hormones,5
necessarily,3
chronic,3
cardiovascular,3
decreasing,3
tumors,3
strategies,3
{% endhighlight %}

<p>In the next post we will look at direct Word2Vec2Graph and the results will be different.</p>

<p><h3>Page Rank and Connected Components</h3>
Connected components: </p>
{% highlight scala %}
sc.setCheckpointDir("/FileStore/")
val graphHightWeightCC = graphHightWeight.connectedComponents.run()
val graphHightWeightCcCount=graphHightWeightCC.groupBy("component").count.toDF("cc","ccCt")
display(graphHightWeightCcCount.orderBy('ccCt.desc).limit(11))
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
532575944709,2
206158430214,2
{% endhighlight %}

<p>Combine two connected components with Page Rank. Biggest component: </p>
{% highlight java %}
val cc1=graphHightWeightCC.filter('component==="60129542144").select("id").toDF("word")
display(cc1.join(stressHightWeightPageRank.vertices,'word==='id).
select('word,'pagerank).orderBy('pagerank.desc))
word,pagerank
hormones,11.925990421899789
disorders,10.58017031766379
cardiovascular,6.061954567119659
chronic,5.6632316778829574
tumors,5.03612633319159
illnesses,5.03612633319159
disorder,4.164508550059325
digestive,3.821437671845854
behavioral,3.736942075966831
symptoms,3.575537714033525
syndrome,2.513757392508552
humans,2.4015901395309998
function,2.4015901395309998
harmful,2.4015901395309998
anxiety,2.4015901395309998
acute,2.3538115732380214
prevention,2.3170945436519768
{% endhighlight %}

<p>Second component: </p>
{% highlight java %}
val cc1=graphHightWeightCC.filter('component==="60129542145").select("id").toDF("word")
display(cc1.join(stressHightWeightPageRank.vertices,'word==='id).
select('word,'pagerank).orderBy('pagerank.desc))
word,pagerank
processes,11.314750484908657
strategies,5.968773769006186
enhance,5.9687737690061855
capabilities,3.960742150950389
functions,3.960742150950389
governmental,2.3657865118684
minimize,2.3657865118684
facilitates,2.269011960232378
practices,2.269011960232378
{% endhighlight %}



<p><h3>Next Post - Word Neighbors</h3>
In the next post we will look at word neighbors for Word2Vec2Graph.
</p>
