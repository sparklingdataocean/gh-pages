---
layout:     post
title:      "Word2Vec2Graph - Psychoanalysis Topics"
subtitle:   "Quick Look at Text"
date:       2017-12-24 12:00:00
author:     "Melenar"
header-img: "img/modern36.jpg"
---

<p><h3>Word2Vec2Graph Model and Free Associations</h3>
In this post we will show the process of converting text files to Word2Vec2Graph model. As a text file we will use text data about Psychoanalysis from Wikipedia. </p>
<p>Word2Vec2Graph technique to find topics is similar to Free Association technique used in psychoanalysis. "The importance of free association is that the patients spoke for themselves, rather than repeating the ideas of the analyst; they work through their own material, rather than parroting another's suggestions" (Freud).
</p>

<h3>Read and Clean Psychoanalysis Data File </h3>
Read Psychoanalysis Data file, tokenize and remove stop words:
{% highlight scala %}
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
val inputPsychoanalysis=sc.textFile("/FileStore/tables/psychoanalisys1.txt").
   toDF("charLine")

val tokenizer = new RegexTokenizer().
   setInputCol("charLine").
   setOutputCol("value").
   setPattern("[^a-z]+").
   setMinTokenLength(5).
   setGaps(true)
val tokenizedPsychoanalysis = tokenizer.
   transform(inputPsychoanalysis)
val remover = new StopWordsRemover().
   setInputCol("value").
   setOutputCol("stopWordFree")
val removedStopWordsPsychoanalysis = remover.
   setStopWords(Array("none","also","nope","null")++
   remover.getStopWords).
   transform(tokenizedPsychoanalysis)

{% endhighlight %}

<p> </p>
<p>Explode Psychoanalysis word arrays to words:</p>
{% highlight scala %}

import org.apache.spark.sql.functions.explode
val slpitCleanPsychoanalysis = removedStopWordsPsychoanalysis.
   withColumn("cleanWord",explode($"stopWordFree")).
   select("cleanWord").
   distinct
slpitCleanPsychoanalysis.count//--4030

{% endhighlight %}
<p> </p>



<p><h3>Are Word Pairs in Trained Word2Vec Model? </h3>
Read trained Word2Vec model that was trained and described in
<i><a href="https://sparklingdataocean.github.io/gh-pages/2017/09/06/w2vTrain/"> our first post</a></i>. </p>
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

<p>Get a set of all words from the Word2Vec model and compare Psychoanalysis file word pairs with words from the Word2Vec model </p>
{% highlight scala %}

val cleanPsychoW2V=slpitCleanPsychoanalysis.
   join(modelWords,'cleanWord==='word).
   select("cleanWord").
   distinct
cleanPsychoW2V.count//--3318

{% endhighlight %}

<p>The Word2Vec model was trained on corpus based on News and Wikipedia data about psychology but only 82% of Psychoanalysis File word pairs are in the model. To increase this percentage we will include Psychoanalysis file data to training corpus and retrain the Word2Vec model. </p>

<p><h3>Retrain Word2Vec Model</h3>


{% highlight scala %}

val inputNews=sc.
  textFile("/FileStore/tables/newsTest.txt").
  toDF("charLine")
val inputWiki=sc.textFile("/FileStore/tables/WikiTest.txt").
   toDF("charLine")
val tokenizedNewsWikiPsychoanalysis = tokenizer.
   transform(inputNews.
   union(inputWiki).
   union(inputPsychoanalysis))

val w2VmodelNewsWikiPsychoanalysis=word2vec.
      fit(tokenizedNewsWikiPsychoanalysis)
   w2VmodelNewsWikiPsychoanalysis.
      write.
      overwrite.
      save("w2VmodelNewsWikiPsychoanalysis")
val modelNewsWikiPsychoanalysis=Word2VecModel.
      read.
      load("w2VmodelNewsWikiPsychoanalysis")   
{% endhighlight %}

<p>Get a set of all words from the new Word2Vec model and compare them with Psychoanalysis file words:</p>
{% highlight scala %}

val modelNewsWikiPsychoanalysis=Word2VecModel.
   read.
   load("w2VmodelNewsWikiPsychoanalysis")
val modelWordsPsychoanalysis=modelNewsWikiPsychoanalysis.
    getVectors.
    select("word","vector")
val cleanPsychoNewW2V=slpitCleanPsychoanalysis.
    join(modelWordsPsychoanalysis,'cleanWord==='word).
    select("word","vector").
    distinct
cleanPsychoNewW2V.count//--3433    
{% endhighlight %}

<p>This new Word2Vec model works better: 85% of Psychoanalysis File words are in the model. </p>

<p><h3>How Word Pairs are Connected?</h3>
Now we will calculate cosine similarities of words within word pairs.
We introduced Word2Vec Cosine Similarity Function in the  
<i><a href="https://sparklingdataocean.github.io/gh-pages/2017/09/17/word2vec2graph/">Word2Vec2Graph model Introduction post.</a></i>
</p>
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

val cleanPsychoNewW2V2=cleanPsychoNewW2V.
   toDF("word2","vector2")
val w2wPsycho=cleanPsychoNewW2V.
   join(cleanPsychoNewW2V2,'word=!='word2)
val w2wPsychoCosDF=w2wPsycho.
   map(r=>(r.getAs[String](0),r.getAs[String](2),
    cosineVector(r.getAs[org.apache.spark.ml.linalg.Vector](1),
    r.getAs[org.apache.spark.ml.linalg.Vector](3)))).
   toDF("word1","word2","cos")
{% endhighlight %}

<p><h3>Transform to Pairs of Words</h3>
Get pairs of words and explode ngrams:</p>

{% highlight scala %}
import org.apache.spark.sql.functions.explode
val ngram = new NGram().
   setInputCol("stopWordFree").
   setOutputCol("ngrams").
   setN(2)
val ngramCleanWords = ngram.
   transform(removedStopWordsPsychoanalysis)
val slpitNgrams=ngramCleanWords.
    withColumn("ngram",explode($"ngrams")).
    select("ngram").
    map(s=>(s(0).toString,
         s(0).toString.split(" ")(0),
         s(0).toString.split(" ")(1))).
    toDF("ngram","ngram1","ngram2").
    filter('ngram1=!='ngram2)   
{% endhighlight %}

Cosine similarities for pairs of words:</p>

{% highlight scala %}
val ngramCos=slpitNgrams.
   join(w2wPsychoCosDF,'ngram1==='word1 && 'ngram2==='word2)
{% endhighlight %}

<p><h3>Graph on Word Pairs</h3>
Now we can build a graph on word pairs: words will be nodes, ngrams - edges and cosine similarities - edge weights.</p>
<p>We will save graph vertices and edges as Parquet to Databricks locations, load vertices and edges and rebuild the same graph.</p>

{% highlight scala %}

import org.graphframes.GraphFrame
val graphNodes1=ngramCos.
   select("ngram1").
   union(ngramCos.select("ngram2")).
   distinct.
   toDF("id")
val graphEdges1=ngramCos.
   select("ngram1","ngram2","cos").
   distinct.
   toDF("src","dst","edgeWeight")
val graph1 = GraphFrame(graphNodes1,graphEdges1)

graph1.vertices.write.
   parquet("graphPsychoVertices")
graph1.edges.write.
   parquet("graphPsychoEdges")

val graphPsychoanalysisVertices = sqlContext.read.parquet("graphPsychoVertices")
val graphPsychoanalysisEdges = sqlContext.read.parquet("graphPsychoEdges")

val graphPsychoanalysis = GraphFrame(graphPsychoanalysisVertices, graphPsychoanalysisEdges)

{% endhighlight %}


<p><h3>Page Rank</h3>
Calculate Page Rank : </p>
{% highlight scala %}
val graphPsychoanalysisPageRank = graphPsychoanalysis.
   pageRank.
   resetProbability(0.15).
   maxIter(11).
   run()
display(graphPsychoanalysisPageRank.vertices.
   distinct.
   sort($"pagerank".desc).
   limit(10))

   id,pagerank
   freud,94.16935233039906
   psychoanalysis,30.977656078470016
   psychoanalytic,22.478475163400674
   theory,16.603352488179016
   unconscious,16.420744218061404
   patients,13.99147342505276
   sexual,13.563442527065638
   patient,12.591079870941268
   analyst,11.662939169635427
   treatment,11.566778056932069

{% endhighlight %}


<p><h3>Finding Topics </h3>
In the previous post we described
<i><a href="https://sparklingdataocean.github.io/gh-pages/2017/12/09/word2vec2graphWordPairsContinue/"> how to find document topics via Word2Vec2Graph model</a></i>.

We created a function to calculate connected components with cosine similarly and component size parameters and a function to transform subgraph edges to DOT language:
</p>
{% highlight scala %}
import org.apache.spark.sql.DataFrame
def w2v2gConnectedComponents(graphVertices: DataFrame,
   graphEdges: DataFrame,
   cosineMin: Double, cosineMax: Double,
   ccMin: Int, ccMax: Int): DataFrame = {
val graphEdgesSub= graphEdges.
   filter('edgeWeight>cosineMin).
   filter('edgeWeight<cosineMax)
 sc.setCheckpointDir("/FileStore/")
 val resultCC = graphSub.
    connectedComponents.
    run()
 val resultCCcount=resultCC.
    groupBy("component").
    count.
    toDF("cc","ccCt")
 val sizeCC=resultCC.join(resultCCcount,'component==='cc).
    filter('ccCt<ccMax).filter('ccCt>ccMin).
    select("id","component","ccCt").distinct
    graphEdges.join(sizeCC,'src==='id).
    union(graphEdges.join(sizeCC,'dst==='id)).
    select("component","ccCt","src","dst","edgeWeight").distinct
}

def roundAt(p: Int)(n: Double): Double = {
   val s = math pow (10, p); (math round n * s) / s
}
def component2dot(graphComponents: DataFrame,
   componentId: Long, cosineMin: Double, cosineMax: Double): DataFrame = {
  graphComponents.
    filter('component===componentId).
    filter('edgeWeight>cosineMin).
    filter('edgeWeight<cosineMax).
    select("src","dst","edgeWeight").distinct.
    map(s=>("\""+s(0).toString +"\" -> \""
      +s(1).toString +"\""+" [label=\""+roundAt(2)(s(2).toString.toDouble)+"\"];")).
      toDF("dotLine")  
}
{% endhighlight %}

<p>To select parameters we analyzed cosine similarity distribution. </p>
{% highlight scala %}
val weightDistribution=graphPsychoanalysisEdges.
   map(s=>(roundAt(2)(s(2).toString.toDouble))).
   toDF("cosine1").groupBy("cosine1").count.toDF("cosine1","cosine1Count")
display(weightDistribution.orderBy('cosine1))
{% endhighlight %}

<a href="#">
    <img src="{{ site.baseurl }}/img/weightDistribution2.jpg" alt="Post Sample Image" width="700" height="700">
</a>

<p>Based on cosine similarity distribution we'll look at topics with high, medium and low cosine similarities. </p>

<p><h3>Psychoanalysis Topics with High Cosine Similarities</h3>
Connected components with edge weights greater than 0.7: </p>
{% highlight scala %}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
val partitionWindow = Window.
   partitionBy($"component").
   orderBy($"edgeWeight".desc)
val result=w2v2gConnectedComponents(graphPsychoanalysisVertices,
   graphPsychoanalysisEdges,0.7,1.0,3,30)
result.persist
val result3edges = result.
    withColumn("rank", rank().over(partitionWindow))
display(result3edges.filter("rank<4").orderBy("component","rank"))

component,ccCt,src,dst,edgeWeight,rank
94489280524,8,patient,symptoms,0.7865587769877727,1
94489280524,8,causes,symptoms,0.7812604197062254,2
94489280524,8,psychological,symptoms,0.7336816137236032,3
103079215116,6,university,professor,0.814475286620036,1
103079215116,6,medicine,university,0.7513358180212253,2
103079215116,6,cornell,university,0.7424065353940977,3
137438953473,7,france,italy,0.8368846158082526,1
137438953473,7,italy,netherlands,0.8187021693660411,2
137438953473,7,italy,switzerland,0.8021272522224945,3
541165879312,5,treatments,therapy,0.8333818545810646,1
541165879312,5,therapy,schizophrenia,0.7847859437652135,2
541165879312,5,behavioral,therapy,0.744693173351612,3

{% endhighlight %}

<p>We selected component '94489280524'. First we'll create a graph with the same cosine similarity parameters them we used to look at connected components, i.e. for word pairs with cosine similarity >0.7:</p>
{% highlight scala %}
val resultDot=component2dot(result,94489280524L,0.7,1.0)
display(resultDot)
"patient" -> "symptoms" [label="0.79"];
"causes" -> "illness" [label="0.7"];
"mental" -> "physical" [label="0.71"];
"causes" -> "symptoms" [label="0.78"];
"psychological" -> "symptoms" [label="0.73"];
"mental" -> "illness" [label="0.73"];
"patient" -> "outcomes" [label="0.72"];
{% endhighlight %}


<a href="#">
    <img src="{{ site.baseurl }}/img/graph21d.jpg" alt="Post Sample Image" width="400" height="500">
</a>


<p>Next we'll expand the topic graph for word pairs with cosine similarity >0.6:</p>

<a href="#">
    <img src="{{ site.baseurl }}/img/graph21f.jpg" alt="Post Sample Image" width="600" height="500">
</a>

<p>Then we'll expand the same connected component to cosine similarity >0.5:</p>

<a href="#">
    <img src="{{ site.baseurl }}/img/graph21a.jpg" alt="Post Sample Image" width="700" height="500">
</a>

<p><h3>Psychoanalysis Topics with Medium Cosine Similarities</h3>
Connected components parameters: edge weights in (0.17, 0.2): </p>
<p>Graph picture parameters: edge weights in (0.1, 0.2):</p>

<a href="#">
    <img src="{{ site.baseurl }}/img/graph27.jpg" alt="Post Sample Image" width="600" height="500">
</a>

<p><h3>Psychoanalysis Topics with Low Cosine Similarities</h3>
Connected components with edge weights in (-0.5, 0.0): </p>
<p>Graph picture with no parameters: edge weights in (-1.0, 1.0):</p>


<p>Example1:</p>

<a href="#">
    <img src="{{ site.baseurl }}/img/graph24.jpg" alt="Post Sample Image" width="500" height="500">
</a>


<p>Example2:</p>

<a href="#">
    <img src="{{ site.baseurl }}/img/graph23.jpg" alt="Post Sample Image" width="560" height="500">
</a>

<p>Example3:</p>

<a href="#">
    <img src="{{ site.baseurl }}/img/graph22.jpg" alt="Post Sample Image" width="700" height="500">
</a>

<p>This post example topics with high cosine similarity word pairs are more expected then topics with low cosine similarity word pairs. Lowly correlated word pairs give us more interesting and unpredicted results. The last example shows that within Psychoanalysis text file the word 'association' is associated with unexpected words...</p>

<p><h3>Next Post - Associations</h3>
In the next several posts we will deeper look at data associations.</p>
