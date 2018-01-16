---
layout:     post
title:      "Word2Vec2Graph Model - Neighbors"
subtitle:   "Review Friends-of-Friends for Words"
date:       2017-10-03 12:00:00
author:     "Melenar"
header-img: "img/pic17.jpg"
---

<p><h3>Word2Vec2Graph Model - How to Find Neighbors</h3>
Two posts before we introduced
<i><a href="https://sparklingdataocean.github.io/gh-pages/2017/09/17/word2vec2graph/">Word2Vec2Graph model</a></i>.
In the previous post we played with
<i><a href="https://sparklingdataocean.github.io/gh-pages/2017/09/28/word2vec2graphPageRank/">Page Rank for Word2Vec2Graph</a></i>.

<p>In this post we will look at different ways to find neighbors via the Word2Vec2Graph model.</p>


<p><h3>Two Connected Components with Page Rank </h3>
Here are the results from the previous post. We combined two large connected components with Page Rank. </p>
<p>Biggest component: </p>
{% highlight java %}
val cc1=graphHightWeightCC.
   filter('component==="60129542144").
   select("id").
   toDF("word")
display(cc1.join(stressHightWeightPageRank.vertices,
   'word==='id).
   select('word,'pagerank).
   orderBy('pagerank.desc))
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
val cc1=graphHightWeightCC.
   filter('component==="60129542145").
   select("id").
   toDF("word")
display(cc1.join(stressHightWeightPageRank.vertices,
   'word==='id).
   select('word,'pagerank).
   orderBy('pagerank.desc))
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

<p><h3>Find Neighbors via GraphFrames "Find" Function</h3>
We will look at neighbors of the word with highest Page Rank for each of these connected components. To find word neighbors we will use 'find' GraphFrames function. </p>
<p>Word "hormones":</p>
{% highlight scala %}

val neighbor1=graphHightWeight.
   find("(a) - [ab] -> (b)").
   filter($"a.id"==="hormones").
   select("ab.src","ab.dst","ab.edgeWeight")
display(neighbor1.
   orderBy('edgeWeight.desc))
src,dst,edgeWeight
hormones,function,0.7766509520166169
hormones,harmful,0.7686013469021604
hormones,digestive,0.767779980189556
hormones,anxiety,0.7622259604849809
hormones,humans,0.7537222776853296
{% endhighlight %}

<p>Word "processes":</p>
{% highlight scala %}

val neighbor1=graphHightWeight.
   find("(a) - [ab] -> (b)").
   filter($"a.id"==="processes").
   select("ab.src","ab.dst","ab.edgeWeight")
display(neighbor1.
   orderBy('edgeWeight.desc))
src,dst,edgeWeight
processes,functions,0.846240488718132
processes,capabilities,0.8165733928557892
processes,strategies,0.8079603436193109
processes,enhance,0.7657391985684395
processes,facilitates,0.7605530303717686
processes,practices,0.7548515830286028
{% endhighlight %}

<p><h3>Find Neighbors via Word2Vec Model</h3>
Another way to find word neighbors is similar to 'findSynonyms' in Word2Vec. Here is a function for a matrix based on words from text file (Stress Data File) with Word2Vec cosine similarities. We will have two parameters: cosine similarity threshold and number of similar words to find. </p>

{% highlight scala %}
import org.apache.spark.sql.DataFrame

def findSimilarWords(w2wCos: DataFrame, word: String, cosine: Double, number: Int):
   DataFrame = {
     w2wCos.
     filter('word1===word).
     filter('cos>cosine).
     select('word2,'cos).
     orderBy('cos.desc).limit(number)
}
{% endhighlight %}

<p>Word "processes" neighbors - we use the same threshold as we used to build a graph. We are getting the same results as using GraphFrames 'find' function:</p>

{% highlight scala %}
display(findSimilarWords(w2wStressCos,
   "processes",0.75,11))
word2,cos
functions,0.846240488718132
capabilities,0.8165733928557892
strategies,0.8079603436193109
enhance,0.7657391985684395
facilitates,0.7605530303717686
practices,0.7548515830286028
{% endhighlight %}

<p>Word "hormones" neighbors:</p>

{% highlight scala %}
display(findSimilarWords(w2wStressCos,
   "hormones",0.75,11))
word2,cos
function,0.7766509520166169
harmful,0.7686013469021604
digestive,0.767779980189556
anxiety,0.7622259604849809
humans,0.7537222776853296
{% endhighlight %}


<p><h3>Finding Neighbors of Neighbors</h3>
Now let's say we need to find neighbors of neighbors, i.e. words with two degrees of separation. Doing this via functions similar to Word2Vec 'findSynonyms' function is not easy. But GraphFrames has elegant solutions to such problems via 'find' function.</p>

<p>Word "processes" neighbors of neighbors:</p>

{% highlight scala %}
val neighbor2=graphHightWeight.
   find("(a) - [ab] -> (b); (b) - [bc] -> (c)").
   filter($"a.id"=!=$"c.id").
   filter($"a.id"==="processes").
   select("ab.src","ab.dst","ab.edgeWeight","bc.dst","bc.edgeWeight")
display(neighbor2)
src,dst,edgeWeight,dst,edgeWeight
processes,strategies,0.8079603436193109,governmental,0.7553409742807539
processes,strategies,0.8079603436193109,capabilities,0.7789548334064621
processes,capabilities,0.8165733928557892,strategies,0.7789548334064621
processes,enhance,0.7657391985684395,functions,0.7894137410909503
processes,enhance,0.7657391985684395,minimize,0.7743199181186822
processes,functions,0.846240488718132,enhance,0.7894137410909503
{% endhighlight %}

<p>Word "hormones" neighbors of neighbors:</p>

{% highlight scala %}
val neighbor2=graphHightWeight.
   find("(a) - [ab] -> (b); (b) - [bc] -> (c)").
   filter($"a.id"=!=$"c.id").
   filter($"a.id"==="hormones").
   select("ab.src","ab.dst","ab.edgeWeight","bc.dst","bc.edgeWeight")
display(neighbor2)
src,dst,edgeWeight,dst,edgeWeight
hormones,digestive,0.767779980189556,disorders,0.7784715813141609
{% endhighlight %}

<p>The word "hormones" has only one second degree neighbor and the word "processes" has several: some word combinations are appeared twice:</p>
{% highlight scala %}
{processes, strategies, capabilities} and {processes, capabilities, strategies}
{processes, enhance, functions} and {processes, functions, enhance}
{% endhighlight %}
<p>This shows that two triangles are attached to the word "processes".</p>

<p><h3>Triangles</h3>
First we will look GraphFrames 'triangleCount' function  </p>

{% highlight scala %}
val graphTriangles=graphHightWeight.
   triangleCount.
   run()
display(graphTriangles.
   select('id,'count).
   filter('count>0).
   orderBy('id))
id,count
capabilities,1
decrease,3
decreased,3
decreases,1
decreasing,2
disorders,1
enhance,1
functions,1
illnesses,2
increase,1
processes,2
reduction,1
strategies,1
symptoms,1
tumors,2
versus,1
{% endhighlight %}

<p>To see triangle word combinations will use 'find' function:</p>
{% highlight scala %}
val triangles=graphHightWeight.
   find("(a) - [ab] -> (b); (b) - [bc] -> (c); (c) - [ca] -> (a)").
   filter($"a.id"<$"b.id").
   filter($"b.id"<$"c.id").
   select("ab.src","ab.dst","bc.dst").
   toDF("word1","word2","word3")
display(triangles.
   orderBy('word1))
word1,word2,word3
capabilities,processes,strategies
decrease,decreased,versus
decrease,decreased,decreasing
decrease,increase,reduction
decreased,decreases,decreasing
disorders,illnesses,tumors
enhance,functions,processes
illnesses,symptoms,tumors
{% endhighlight %}

<p><h3>Next Post - Direct Graph</h3>
In the next post we will look at direct Word2Vec2Graph graphs.
</p>
