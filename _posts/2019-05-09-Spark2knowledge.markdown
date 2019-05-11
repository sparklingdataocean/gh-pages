---
layout:     post
title:      "Explainable AI Mind Mapping"
subtitle:   "Bridges between convolutional neural network, brain science and modern art"
date:       2019-05-09 12:00:00
author:     "Melenar"
header-img: "img/modern147.jpg"
---
<p><h3>Connections between AI, Science and Art</h3>
<p>
Two books of Eric Kandel, Nobel Prize winner in Physiology or Medicine, motivated us for this post.
Eric Kandel is building bridges between Art and Science, in particularly between Modern Art and Brain Science.
His book <i><a href="
https://www.amazon.com/Age-Insight-Understand-Unconscious-Present/dp/1400068711">"The Age of Insight: The Quest to Understand the Unconscious in Art, Mind, and Brain, from Vienna 1900 to the Present"</a></i>
 inspired us to think about Explainable Artificial Intelligence via Science and Art and his other book - <i><a href="
https://www.amazon.com/Reductionism-Art-Brain-Science-Bridging/dp/0231179626/ref=pd_lpo_sbs_14_t_0?_encoding=UTF8&psc=1&refRID=EWB292R76ZSKZW6JYGZR">"Reductionism in Art and Brain Science: Bridging the Two Cultures"</a></i> clearly explained how Brain Science and Abstract Art are connected.
</p>
<p>Both Artificial Intelligence and Neuroscience are fields of <a href="
https://en.wikipedia.org/wiki/Cognitive_science">Cognitive Science</a> - interdisciplinary study of cognition in humans, animals, and machines.  Cognitive Science should help us to open "Artificial Intelligence Black Box" and move from unexplainable AI to XAI (Explainable AI).

</p>
<p>
In this post we will attempt to connect AI with Science and with Art. We will use the approach described by Kandel in his 'Reductionism in Art and Brain Science' book where he shows how reductionism, applied to the most complex puzzles can been used by science to explore these complexities.

</p>
<p><h3>CNN Deep Learning Lower Layers and 'Simple Cells'</h3>

<p>Convolutional Neural Network breakthrough moment happened in 2012. Soon after
<i><a href="
https://medium.com/technologymadeeasy/the-best-explanation-of-convolutional-neural-networks-on-the-internet-fbb8b1ad5df8">"The best explanation of Convolutional Neural Networks on the Internet!"</a></i> article explained in pictures that the first few layers of CNN are very simple colored pieces and edges.
</p>
<p>In the middle of 20th century neuroscientists discovered that Human Visualization process works very similar: some neurons are specialized on very simple images. These <i><a href="https://en.wikipedia.org/wiki/Simple_cell">"simple cells"</a></i> were discovered by Torsten Wiesel and David Hubel in 1958. Based on these simple cells in 1989 Yann LeCun developed Convolutional Neural Network.
</p>
There are two general processes involved in perception: top-down and bottom-up. Getting simple pieces (bottom-up processing) is combined with existing knowledge (top-down processing). This combination of two processes is similar to popular approach in deep learning where pre-trained models are used as the starting point for training.

<p>

</p>
<p>
The article about CNN history:
<i><a href="
https://hackernoon.com/a-brief-history-of-computer-vision-and-convolutional-neural-networks-8fe8aacc79f3
">"A Brief History of Computer Vision (and Convolutional Neural Networks)"</a></i>
describes in more details how neuroscience affected CNN.
</p>
<p>
<p><h3>Simple Images in Abstract Art</h3>
<p>
Art perception is also based on two strategies: bottom-up and top-down processes. Understanding of these processes appeared in art much earlier than in science: in 1907 Pablo Picasso and George Braque started Cubism movement.
</p>
<p>
Picasso and Braque invented a new style of paintings composed of simple bottom-up images.  Cubism movement became the most influential art movement of the 20th century and it  initiated other art movements like Abstract Art.
</p>
<p>
Why artists realized the existence of simple images in bottom-up process much earlier than scientists were able to find and prove it? Is this because complexity can be intuitively understood much earlier than logically explained?
</p>


<p><h3>Art to Science to AI</h3>
<p>
To build a bridge between Art and Artificial Intelligence we will 'connect the dots' of Mind Mapping via RDF triples - (subject, predicate, object).


<p></p>
{% highlight scala %}
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.sql.DataFrame
import org.graphframes.GraphFrame

val mindMapping=sc.parallelize(Array(
  ("Image Recognition","how?","Artificial Intelligence"),
   ("Artificial Intelligence","subclass","CNN Deep Learning"),
   ("CNN Deep Learning","what?","Simple Forms"),
   ("CNN Deep Learning","when?","2012"),
   ("Image Recognition","how?","Brain Science"),
   ("Brain Science","subclass","Perception"),
   ("Perception","what?","Simple Cells"),
   ("Perception","when?","1958"),
   ("Image Recognition","how?","Modern Art"),
   ("Modern Art","subclass","Cubism"),
   ("Cubism","what?","Simple Images"),
   ("Cubism","when?","1907")
   )).toDF("subject","predicate","object")

display(mindMapping)

subject,predicate,object
Image Recognition,how?,Artificial Intelligence
Artificial Intelligence,subclass,CNN Deep Learning
CNN Deep Learning,what?,Simple Forms
CNN Deep Learning,when?,2012
Image Recognition,how?,Brain Science
Brain Science,subclass,Perception
Perception,what?,Simple Cells
Perception,when?,1958
Image Recognition,how?,Modern Art
Modern Art,subclass,Cubism
Cubism,what?,Simple Images
Cubism,when?,1907



{% endhighlight %}
</p><p>

On Mind Mapping RDF triples we will build knowledge graph. We will use the same technique as for Word2Vec2Graph model - Spark GraphFrames.

<p></p>

<p>
</p>

{% highlight scala %}
val graphNodes=mindMapping.select("subject").
  union(mindMapping.select("object")).distinct.toDF("id")
val graphEdges=mindMapping.select("subject","object","predicate").
  distinct.toDF("src","dst","edgeId")
val graph = GraphFrame(graphNodes,graphEdges)

{% endhighlight %}


<h3>Query the Knowledge Graph</h3>
</p><p>

<p>As a graph query language we will use Spark GraphFrames 'find' function. <br>Get triples with 'how?' predicate:  </p>
{% highlight scala %}
val line=graph.
   find("(a) - [ab] -> (b)").
   filter($"ab.edgeId"==="how?").select("a.id","ab.edgeId","b.id").
      toDF("node1","edge12","node2")

display(line)
node1,edge12,node2
Image Recognition,how?,Modern Art
Image Recognition,how?,Artificial Intelligence
Image Recognition,how?,Brain Science
{% endhighlight %}


<p>Show them in 'motif' language: </p>
{% highlight scala %}
val line=graph.
   find("(a) - [ab] -> (b)").
   filter($"ab.edgeId"==="how?").select("a.id","ab.edgeId","b.id").
   map(s=>("(" +s(0).toString + ") - [" + s(1).toString +"] -> (" + s(2).toString  +  ");" ))

display(line)
value
(Image Recognition) - [how?] -> (Modern Art);
(Image Recognition) - [how?] -> (Artificial Intelligence);
(Image Recognition) - [how?] -> (Brain Science);

{% endhighlight %}


<p>Connect two triples with the second predicate equal 'when?' </p>
{% highlight scala %}
val line=graph.
   find("(a) - [ab] -> (b); (b) - [bc] -> (c) ").
   filter($"bc.edgeId"==="when?").
   select("a.id","ab.edgeId", "b.id","bc.edgeId","c.id").
   toDF("node1","edge12","node2","edge23","node3")

display(line.orderBy('node3))

node1,edge12,node2,edge23,node3
Modern Art,subclass,Cubism,when?,1907
Brain Science,subclass,Perception,when?,1958
Artificial Intelligence,subclass,CNN Deep Learning,when?,2012

{% endhighlight %}


<p>In 'motif':   </p>
{% highlight scala %}
val line=graph.
   find("(a) - [ab] -> (b); (b) - [bc] -> (c) ").
   filter($"bc.edgeId"==="when?").
   select("a.id","ab.edgeId", "b.id","bc.edgeId","c.id").
   map(s=>("(" +s(0).toString+") - ["+s(1).toString+"] -> ("+s(2).toString+") - ["+s(3).toString+"] -> ("+s(4).toString+")"))

display(line)
value
(Artificial Intelligence) - [subclass] -> (CNN Deep Learning) - [when?] -> (2012)
(Brain Science) - [subclass] -> (Perception) - [when?] -> (1958)
(Modern Art) - [subclass] -> (Cubism) - [when?] -> (1907)

{% endhighlight %}


<p>Create phrases of two triple with the last object like 'simple':  </p>
{% highlight scala %}
val line=graph.
   find("(a) - [ab] -> (b); (b) - [bc] -> (c) ").
   filter($"c.id".rlike("Simple")).select("a.id","b.id","c.id").
   map(s=>("In "+s(0).toString+" area "+s(1).toString+" discovered "+s(2).toString.toLowerCase))

display(line)

{% endhighlight %}

<font style="color:rgb(80, 96, 160);">
<small>
<br>
In Modern Art area Cubism discovered simple images
<br>
In Brain Science area Perception discovered simple cells
<br>
In Artificial Intelligence area CNN Deep Learning discovered simple forms
</small>
</font>
</p><p>
</p><p>

<p>Create phrases with all triples:   </p>
{% highlight scala %}
val line=graph.
   find("(a) - [ab] -> (b); (b) - [bc] -> (c); (c) - [cd] -> (d); (c) - [ce] -> (e) ").
   filter($"cd.edgeId"==="when?").filter($"ce.edgeId"==="what?").
   select("a.id","b.id","c.id","d.id","e.id").map(s=>("For " +s(0).toString.toLowerCase+" "+s(4).toString.toLowerCase+" were discovered by "+s(1).toString+" ("+ s(2).toString+") in "+s(3).toString))

display(line)
{% endhighlight %}

<font style="color:rgb(80, 96, 160);">
<small>
<br>
For image recognition simple images were discovered by Modern Art (Cubism) in 1907
<br>
For image recognition simple cells were discovered by Brain Science (Perception) in 1958
<br>
For image recognition simple forms were discovered by Artificial Intelligence (CNN Deep Learning) in 2012
</small>
</font>
</p><p>
</p><p>


<h3>Graph Image</h3>
</p><p>

Convert to dot language, join lines with 'digraph {}' and use Gephi tool for graph visualization.
<p></p>
{% highlight scala %}

def graph2dot(graph: GraphFrame): DataFrame = {
  graph.edges.distinct.
    map(s=>("\""+s(0).toString +"\" -> \""
      +s(1).toString +"\""+" [label=\""+(s(2).toString)+"\"];")).
      toDF("dotLine")  

{% endhighlight %}


<font style="color:rgb(60, 60, 60);">
<i><small>
<br>digraph {
<br>"Perception" -> "Simple Cells" [label="what?"];
<br>"CNN Deep Learning" -> "2012" [label="when?"];
<br>"Cubism" -> "Simple Images" [label="what?"];
<br>"Cubism" -> "1907" [label="when?"];
<br>"Artificial Intelligence" -> "CNN Deep Learning" [label="subclass"];
<br>"Brain Science" -> "Perception" [label="subclass"];
<br>"Modern Art" -> "Cubism" [label="subclass"];
<br>"Image Recognition" -> "Artificial Intelligence" [label="how?"];
<br>"Image Recognition" -> "Brain Science" [label="how?"];
<br>"Perception" -> "1958" [label="when?"];
<br>"CNN Deep Learning" -> "Simple Forms" [label="what?"];
<br>"Image Recognition" -> "Modern Art" [label="how?"];
<br>}
</small>
</i>
</font>

<a href="#">
    <img src="{{ site.baseurl }}/img/mmpic1.jpg" alt="Post Sample Image" width="777">
</a>

<p><h3>More Questions...</h3>
<p>


</p>
<ul>
<li>CNN Deep Learning was created based on Brain Science discovery of simple cells, not as magic. <i>Why are we worry about unexplainable AI but do not worry about our brain work processes?</i></li>
<li>More than 50 years before simple cells were discovered by Brain Science Cubism paintings showed us how our perception works. <i>Why some very complicated problems were demonstrated by artists much earlier than were proved by scientists? Is it because intuition works much faster than logical thinking?</i></li>
<li><i>Can some AI ideas be created based on intuition and not yet proved by science?</i></li>

</ul>



</p>

<p>


<p><h3>Next Post - Associations and Deep Learning</h3>
<p>
In the next post we will deeper look at deep learning for data associations.</p>
