---
layout:     post
title:      "Knowledge Graph for Data Mining"
subtitle:   "Building knowledge graph in Spark without SPARQL"
date:       2019-09-24 12:00:00
author:     "Melenar"
header-img: "img/pic91.jpg"
---

<p><h3>Why Knowledge Graphs?</h3>

<p>
Knowledge graph was introduced by Google in 2012 then it was adapted by many companies as Enterprise Knowledge Graph. </p>
Knowledge graph is well known for:
<ul>
<li>Google Knowledge Graph - a powerful search engine</li>
<li>Enterprise Knowledge Graph - a powerful way of integrating various data</li>
</ul>

Knowledge graph concept is much wider than search and integration. In one of our posts we used knowledge graph as Mind Mapping. In this post we will show how knowledge graph can give us a deeper view on data.
</p>
<p>Traditionally knowledge graphs are using SPARQL language for data analysis.
We will show how to build a knowledge graph in Spark and explore data using Spark DataFrame and Spark GraphFrames.
</p>

<h3>Read and Clean the Data</h3>
For data we use a Kaggle dataset
<i><a href="https://www.kaggle.com/momanyc/museum-collection">
'Museum of Modern Art Collection'</a></i> with information about titles and artists of  MoMA collection. From this dataset we've got some data about paintings of several artists.</p>
{% highlight scala %}
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.sql.DataFrame
import org.graphframes.GraphFrame
import org.apache.spark.sql.functions._

val data=sqlContext.read.format("csv").
  option("header","true").load("/FileStore/tables/daYes2.txt")

{% endhighlight %}

<p>Example:</p>

{% highlight scala %}
display(data.select("Artist","ArtistBio","BeginDate","EndDate","Nationality","Gender","Title","Date").limit(2))
Joan Miró,(Spanish, 1893–1983),(1893),(1983),(Spanish),(Male),
   Rope and People, I,Barcelona, March 27, 1935
Paul Klee,(German, born Switzerland. 1879–1940),(1879),(1940),(German),(Male),
   Fire in the Evening,1929

{% endhighlight %}


<p><h3>Artist Nationalities</h3>
<p>In this post we will play with data about artist biographies and it the following post - about their paintings. To clean the data, first we will remove parentheses from some columns: </p>

{% highlight scala %}

val aboutArtist=data.select("Artist","ArtistBio","BeginDate","EndDate","Nationality","Gender").
  distinct.
  withColumn("x1",regexp_replace(col("Nationality"),"[^A-Za-z]","")).
  withColumn("x2",regexp_replace(col("Gender"),"[^A-Za-z]","")).
  withColumn("x3",regexp_replace(col("BeginDate"),"[^0-9]","")).
  withColumn("x4",regexp_replace(col("EndDate"),"[^0-9]","")).
  select("Artist","ArtistBio","x3","x4","x1","x2").
  toDF("Artist","ArtistBio","BeginDate","EndDate","Nationality","Gender")

Vasily Kandinsky,"(French, born Russia. 1866–1944)",1866,1944,French,Male
Franz Marc,"(German, 1880–1916)",1880,1916,German,Male
Henri Matisse,"(French, 1869–1954)",1869,1954,French,Male
Joan Miró,"(Spanish, 1893–1983)",1893,1983,Spanish,Male
Egon Schiele,"(Austrian, 1890–1918)",1890,1918,Austrian,Male
Marc Chagall,"(French, born Belarus. 1887–1985)",1887,1985,French,Male
Vincent van Gogh,"(Dutch, 1853–1890)",1853,1890,Dutch,Male
Jackson Pollock,"(American, 1912–1956)",1912,1956,American,Male
Paul Signac,"(French, 1863–1935)",1863,1935,French,Male
Paul Cézanne,"(French, 1839–1906)",1839,1906,French,Male
Kazimir Malevich,"(Russian, born Ukraine. 1878–1935)",1878,1935,Russian,Male
Georges Braque,"(French, 1882–1963)",1882,1963,French,Male
Natalia Goncharova,"(Russian, 1881–1962)",1881,1962,Russian,Female
Max Beckmann,"(German, 1884–1950)",1884,1950,German,Male
Piet Mondrian,"(Dutch, 1872–1944)",1872,1944,Dutch,Male
Paul Klee,"(German, born Switzerland. 1879–1940)",1879,1940,German,Male
Claude Monet,"(French, 1840–1926)",1840,1926,French,Male
Pablo Picasso,"(Spanish, 1881–1973)",1881,1973,Spanish,Male
Paul Gauguin,"(French, 1848–1903)",1848,1903,French,Male
Oskar Kokoschka,"(Austrian, 1886–1980)",1886,1980,Austrian,Male

{% endhighlight %}

<p>For the first knowledge graph for data exploration we will use columns 'Artist' and 'Nationality'. Create graph vertices (nodes):  </p>

{% highlight scala %}
val graphNodesNationality=aboutArtist.select("Artist").
  union(aboutArtist.select("Nationality")).distinct.toDF("id")

{% endhighlight %}

<p>Create graph edges in (subject, object, predicate) form. As a predicate we will use the column name 'Nationality':  </p>

{% highlight scala %}

val graphEdgesNationality=aboutArtist.select("Artist","Nationality").
  toDF("src","dst").withColumn("edgeId",lit("Nationality")).
  distinct

{% endhighlight %}


<p>Build a knowledge graph:  </p>

{% highlight scala %}
val graphNationality = GraphFrame(graphNodesNationality,graphEdgesNationality)
{% endhighlight %}

<p>See graph vertices:  </p>
{% highlight scala %}
display(graphNationality.vertices)
Georges Braque
Pablo Picasso
Egon Schiele
Kazimir Malevich
Jackson Pollock
Marc Chagall
Natalia Goncharova
Paul Signac
Franz Marc
Piet Mondrian
Vincent van Gogh
Paul Cézanne
Vasily Kandinsky
Paul Klee
Henri Matisse
Claude Monet
Oskar Kokoschka
Paul Gauguin
Max Beckmann
Joan Miró
American
Austrian
Dutch
French
German
Russian
Spanish
{% endhighlight %}

<p>See graph edges:  </p>

{% highlight scala %}

display(graphNationality.edges)
src,dst,edgeId
Vincent van Gogh,Dutch,Nationality
Claude Monet,French,Nationality
Paul Gauguin,French,Nationality
Paul Cézanne,French,Nationality
Egon Schiele,Austrian,Nationality
Paul Klee,German,Nationality
Natalia Goncharova,Russian,Nationality
Piet Mondrian,Dutch,Nationality
Henri Matisse,French,Nationality
Franz Marc,German,Nationality
Kazimir Malevich,Russian,Nationality
Vasily Kandinsky,French,Nationality
Paul Signac,French,Nationality
Pablo Picasso,Spanish,Nationality
Oskar Kokoschka,Austrian,Nationality
Max Beckmann,German,Nationality
Joan Miró,Spanish,Nationality
Georges Braque,French,Nationality
Jackson Pollock,American,Nationality
Marc Chagall,French,Nationality
{% endhighlight %}

<p>For graph visualization we will use Gephi. Function to translate graph edges to dot language:</p>

{% highlight scala %}
def graph2dot(graph: GraphFrame): DataFrame = {
  graph.edges.distinct.
    map(s=>("\""+s(0).toString +"\" -> \""
      +s(1).toString +"\""+" [label=\""+(s(2).toString)+"\"];")).
      toDF("dotLine")  
}
{% endhighlight %}

<p>Translate graph edges to dot language:</p>

{% highlight scala %}
display(graph2dot(graphNationality))

"Vincent van Gogh" -> "Dutch" [label="Nationality"];
"Claude Monet" -> "French" [label="Nationality"];
"Paul Gauguin" -> "French" [label="Nationality"];
"Paul Cézanne" -> "French" [label="Nationality"];
"Egon Schiele" -> "Austrian" [label="Nationality"];
"Paul Klee" -> "German" [label="Nationality"];
"Natalia Goncharova" -> "Russian" [label="Nationality"];
"Piet Mondrian" -> "Dutch" [label="Nationality"];
"Henri Matisse" -> "French" [label="Nationality"];
"Franz Marc" -> "German" [label="Nationality"];
"Kazimir Malevich" -> "Russian" [label="Nationality"];
"Vasily Kandinsky" -> "French" [label="Nationality"];
"Paul Signac" -> "French" [label="Nationality"];
"Pablo Picasso" -> "Spanish" [label="Nationality"];
"Oskar Kokoschka" -> "Austrian" [laBornbel="Nationality"];
"Max Beckmann" -> "German" [label="Nationality"];
"Joan Miró" -> "Spanish" [label="Nationality"];
"Georges Braque" -> "French" [label="Nationality"];
"Jackson Pollock" -> "American" [label="Nationality"];
"Marc Chagall" -> "French" [label="Nationality"];
{% endhighlight %}

<p>Knowledge graph for Artist Nationalities:</p>


<a href="#">
    <img src="{{ site.baseurl }}/img/moma2.jpg" alt="Post Sample Image" width="560" height="500">
</a>


<p><h3>Countries of Birth </h3>
<p>
Next we will look at countries where artists were born. ArtistBio mentions born in country only if artists's nationality has changes. So we will get a country of birth from ArtistBio column:
</p>
{% highlight scala %}

val artistData1=aboutArtist.filter(col("ArtistBio").rlike("born ")).
  withColumn("bornHs",split($"ArtistBio","born ")(1)).
  withColumn("bornCountry",split($"bornHs",". 1")(0)).drop("bornHs")

Artist,ArtistBio,BeginDate,EndDate,Nationality,Gender,bornCountry
Vasily Kandinsky,"(French, born Russia. 1866–1944)",1866,1944,French,Male,Russia
Marc Chagall,"(French, born Belarus. 1887–1985)",1887,1985,French,Male,Belarus
Kazimir Malevich,"(Russian, born Ukraine. 1878–1935)",1878,1935,Russian,Male,Ukraine
Paul Klee,"(German, born Switzerland. 1879–1940)",1879,1940,German,Male,Switzerland


{% endhighlight %}

<p>When born country is not mentioned in Artist Bio column we will translate artist's nationality to country of birth based on nationality to country mapping:</p>

{% highlight scala %}
val nationality2country=sc.parallelize(Array(
  ("French","France"),
  ("Russian","Russia"),
  ("German","Germany"),
  ("American","USA"),
  ("Dutch","Netherlands"),
  ("Spanish","Spain"),
  ("Austrian","Austria"),
  ("Ukrainian","Ukraine"),
  ("Belarusian","Belarus"),
  ("Swiss","Switzerland")
  )).toDF("nation","country")
display(nationality2country)

val artistData2=aboutArtist.filter(not(col("ArtistBio").rlike("born "))).
  join(nationality2country,col("Nationality").contains(col("nation"))).
  drop("nation")

Artist,ArtistBio,BeginDate,EndDate,Nationality,Gender,country
Paul Gauguin,"(French, 1848–1903)",1848,1903,French,Male,France
Claude Monet,"(French, 1840–1926)",1840,1926,French,Male,France
Georges Braque,"(French, 1882–1963)",1882,1963,French,Male,France
Paul Cézanne,"(French, 1839–1906)",1839,1906,French,Male,France
Paul Signac,"(French, 1863–1935)",1863,1935,French,Male,France
Henri Matisse,"(French, 1869–1954)",1869,1954,French,Male,France
Natalia Goncharova,"(Russian, 1881–1962)",1881,1962,Russian,Female,Russia
Max Beckmann,"(German, 1884–1950)",1884,1950,German,Male,Germany
Franz Marc,"(German, 1880–1916)",1880,1916,German,Male,Germany
Jackson Pollock,"(American, 1912–1956)",1912,1956,American,Male,USA
Piet Mondrian,"(Dutch, 1872–1944)",1872,1944,Dutch,Male,Netherlands
Vincent van Gogh,"(Dutch, 1853–1890)",1853,1890,Dutch,Male,Netherlands
Pablo Picasso,"(Spanish, 1881–1973)",1881,1973,Spanish,Male,Spain
Joan Miró,"(Spanish, 1893–1983)",1893,1983,Spanish,Male,Spain
Oskar Kokoschka,"(Austrian, 1886–1980)",1886,1980,Austrian,Male,Austria
Egon Schiele,"(Austrian, 1890–1918)",1890,1918,Austrian,Male,Austria

{% endhighlight %}

<p>Combine the tables and build a graph.</p>

{% highlight scala %}
val artistData=artistData1.union(artistData2)  

val graphNodesBornCountry=artistData.select("Artist").
  union(artistData.select("bornCountry")).distinct.toDF("id")
val graphEdgesBornCountry=artistData.select("Artist","bornCountry").toDF("src","dst").
  withColumn("edgeId",lit("bornCountry")).
  distinct
val graphBornCountry = GraphFrame(graphNodesBornCountry,graphEdgesBornCountry)
display(graph2dot(graphBornCountry))

"Pablo Picasso" -> "Spain" [label="bornCountry"];
"Claude Monet" -> "France" [label="bornCountry"];
"Marc Chagall" -> "Belarus" [label="bornCountry"];
"Kazimir Malevich" -> "Ukraine" [label="bornCountry"];
"Paul Cézanne" -> "France" [label="bornCountry"];
"Oskar Kokoschka" -> "Austria" [label="bornCountry"];
"Piet Mondrian" -> "Netherlands" [label="bornCountry"];
"Henri Matisse" -> "France" [label="bornCountry"];
"Vincent van Gogh" -> "Netherlands" [label="bornCountry"];
"Paul Klee" -> "Switzerland" [label="bornCountry"];
"Georges Braque" -> "France" [label="bornCountry"];
"Jackson Pollock" -> "USA" [label="bornCountry"];
"Vasily Kandinsky" -> "Russia" [label="bornCountry"];
"Paul Signac" -> "France" [label="bornCountry"];
"Joan Miró" -> "Spain" [label="bornCountry"];
"Paul Gauguin" -> "France" [label="bornCountry"];
"Max Beckmann" -> "Germany" [label="bornCountry"];
"Franz Marc" -> "Germany" [label="bornCountry"];
"Natalia Goncharova" -> "Russia" [label="bornCountry"];
"Egon Schiele" -> "Austria" [label="bornCountry"];

{% endhighlight %}

<p>Where artists were born?</p>

<a href="#">
    <img src="{{ site.baseurl }}/img/moma1.jpg" alt="Post Sample Image" width="600" height="500">
</a>

<p>When you see in art museum in what country an artist was born, you rather see a country what artist's place of birth currently belongs to. Some of such countries did not exist at the time when artists were born. For example when Mark Chagall was born near Vitebsk, Russian Empire, present-day Belarus.
To change current country names to historical country names, we will change "Russia","Ukraine","Belarus" country names to "Russian Empire" and rebuild a graph:</p>

{% highlight scala %}
val artistBornData =artistData.
  withColumn("bornInCountry",when(col("bornCountry").
  isin("Russia","Ukraine","Belarus"),"Russian Empire").
  otherwise(col("bornCountry"))).drop("bornCountry")

val graphNodesBornCountry=artistBornData.select("Artist").
  union(artistBornData.select("bornInCountry")).distinct.toDF("id")
val graphEdgesBornCountry=artistBornData.
  select("Artist","bornInCountry").toDF("src","dst").
  withColumn("edgeId",lit("bornInCountry")).
  distinct
val graphBornCountry = GraphFrame(graphNodesBornCountry,graphEdgesBornCountry)
  display(graph2dot(graphBornCountry))

"Vincent van Gogh" -> "Netherlands" [label="bornInCountry"];
"Kazimir Malevich" -> "Russian Empire" [label="bornInCountry"];
"Jackson Pollock" -> "USA" [label="bornInCountry"];
"Natalia Goncharova" -> "Russian Empire" [label="bornInCountry"];
"Pablo Picasso" -> "Spain" [label="bornInCountry"];
"Paul Klee" -> "Switzerland" [label="bornInCountry"];
"Paul Signac" -> "France" [label="bornInCountry"];
"Franz Marc" -> "Germany" [label="bornInCountry"];
"Marc Chagall" -> "Russian Empire" [label="bornInCountry"];
"Piet Mondrian" -> "Netherlands" [label="bornInCountry"];
"Oskar Kokoschka" -> "Austria" [label="bornInCountry"];
"Vasily Kandinsky" -> "Russian Empire" [label="bornInCountry"];
"Max Beckmann" -> "Germany" [label="bornInCountry"];
"Henri Matisse" -> "France" [label="bornInCountry"];
"Claude Monet" -> "France" [label="bornInCountry"];
"Georges Braque" -> "France" [label="bornInCountry"];
"Egon Schiele" -> "Austria" [label="bornInCountry"];
"Joan Miró" -> "Spain" [label="bornInCountry"];
"Paul Gauguin" -> "France" [label="bornInCountry"];
"Paul Cézanne" -> "France" [label="bornInCountry"];
{% endhighlight %}

Historical birth country names:
<a href="#">
    <img src="{{ site.baseurl }}/img/moma4.jpg" alt="Post Sample Image" width="678" height="500">
</a>

<p><h3>Born Nationality</h3>
Now we can look at artists' nationalities when they were born. We'll add 'Russian Empire' to nationality to country mapping:</p>
{% highlight scala %}
val nationality2countryHistory=nationality2country.
  filter(not('country.isin("Ukraine","Belarus"))).
  withColumn("countryHistory",
    when(col("country")==="Russia","Russian Empire").otherwise(col("country"))).
  drop("country").toDF("bornNationality","countryHistory")

val artistBornNationality=artistBornData.
  join(nationality2countryHistory,
  col("bornInCountry")===col("countryHistory")).
  drop("countryHistory")
{% endhighlight %}

<p>Build a graph </p>
{% highlight scala %}
val graphNodesBornNationality=artistBornNationality.select("Artist").  
  union(artistBornNationality.select("bornNationality")).distinct.toDF("id")
val graphEdgesBornNationality=artistBornNationality.
  select("Artist","bornNationality").toDF("src","dst").
  withColumn("edgeId",lit("bornNationality")).
  distinct
val graphBornNationality = GraphFrame(graphNodesBornNationality,graphEdgesBornNationality)

"Vincent van Gogh" -> "Dutch" [label="bornNationality"];
"Paul Signac" -> "French" [label="bornNationality"];
"Pablo Picasso" -> "Spanish" [label="bornNationality"];
"Jackson Pollock" -> "American" [label="bornNationality"];
"Joan Miró" -> "Spanish" [label="bornNationality"];
"Piet Mondrian" -> "Dutch" [label="bornNationality"];
"Henri Matisse" -> "French" [label="bornNationality"];
"Kazimir Malevich" -> "Russian" [label="bornNationality"];
"Paul Gauguin" -> "French" [label="bornNationality"];
"Georges Braque" -> "French" [label="bornNationality"];
"Paul Cézanne" -> "French" [label="bornNationality"];
"Franz Marc" -> "German" [label="bornNationality"];
"Marc Chagall" -> "Russian" [label="bornNationality"];
"Claude Monet" -> "French" [label="bornNationality"];
"Max Beckmann" -> "German" [label="bornNationality"];
"Egon Schiele" -> "Austrian" [label="bornNationality"];
"Oskar Kokoschka" -> "Austrian" [label="bornNationality"];
"Paul Klee" -> "Swiss" [label="bornNationality"];
"Vasily Kandinsky" -> "Russian" [label="bornNationality"];
"Natalia Goncharova" -> "Russian" [label="bornNationality"];

{% endhighlight %}
Artists' born nationalities:
<a href="#">
    <img src="{{ site.baseurl }}/img/moma5.jpg" alt="Post Sample Image" width="500" height="700">
</a>


<p><h3>Join All Graphs</h3>
To put Artists Bio into the same graph we can calculate nodes and edges for Gender, BeginDate and EndDate, then union them with nodes and edges we created (country of birth, nationality and born nationality) and build a graph. </p>
<p>Another way to build a knowledge graph is to automate the process. Get column names for nodes and edges:</p>

{% highlight scala %}
val artistBio=artistBornNationality.drop("ArtistBio")
val columnList=artistBio.columns
Array[String] = Array(Artist, BeginDate, EndDate, Nationality, Gender, bornInCountry, bornNationality)
{% endhighlight %}

<p>Create edges for {'Artist','other column'} pairs and nodes for all columns:</p>

{% highlight scala %}
var graphNodes: DataFrame =Seq(("")).toDF("id")
var graphEdges: DataFrame =Seq(("","","")).toDF("src","dst","edgeId")
var idx=0
for (column <- columnList ) {  
  graphNodes=graphNodes.union(artistBio.select(column))
  if (idx>0) {
    graphEdges=graphEdges.union(artistBio.select(artistBio.columns(0),column).
      toDF("src","dst").withColumn("edgeId",lit(column)))
  }
  idx=idx+1
 }
{% endhighlight %}


<p>Edges:</p>
{% highlight scala %}
display(graphEdges.orderBy("src","edgeId"))
Claude Monet,1840,BeginDate
Claude Monet,1926,EndDate
Claude Monet,Male,Gender
Claude Monet,French,Nationality
Claude Monet,France,bornInCountry
Claude Monet,French,bornNationality
Egon Schiele,1890,BeginDate
Egon Schiele,1918,EndDate
Egon Schiele,Male,Gender
Egon Schiele,Austrian,Nationality
Egon Schiele,Austria,bornInCountry
Egon Schiele,Austrian,bornNationality
Franz Marc,1880,BeginDate
Franz Marc,1916,EndDate
Franz Marc,Male,Gender
Franz Marc,German,Nationality
Franz Marc,Germany,bornInCountry
Franz Marc,German,bornNationality
Georges Braque,1882,BeginDate
Georges Braque,1963,EndDate
Georges Braque,Male,Gender
Georges Braque,French,Nationality
Georges Braque,France,bornInCountry
Georges Braque,French,bornNationality
Henri Matisse,1869,BeginDate
Henri Matisse,1954,EndDate
Henri Matisse,Male,Gender
Henri Matisse,French,Nationality
Henri Matisse,France,bornInCountry
Henri Matisse,French,bornNationality
Jackson Pollock,1912,BeginDate
Jackson Pollock,1956,EndDate
Jackson Pollock,Male,Gender
Jackson Pollock,American,Nationality
Jackson Pollock,USA,bornInCountry
Jackson Pollock,American,bornNationality
Joan Miró,1893,BeginDate
Joan Miró,1983,EndDate
Joan Miró,Male,Gender
Joan Miró,Spanish,Nationality
Joan Miró,Spain,bornInCountry
Joan Miró,Spanish,bornNationality
Kazimir Malevich,1878,BeginDate
Kazimir Malevich,1935,EndDate
Kazimir Malevich,Male,Gender
Kazimir Malevich,Russian,Nationality
Kazimir Malevich,Russian Empire,bornInCountry
Kazimir Malevich,Russian,bornNationality
Marc Chagall,1887,BeginDate
Marc Chagall,1985,EndDate
Marc Chagall,Male,Gender
Marc Chagall,French,Nationality
Marc Chagall,Russian Empire,bornInCountry
Marc Chagall,Russian,bornNationality
Max Beckmann,1884,BeginDate
Max Beckmann,1950,EndDate
Max Beckmann,Male,Gender
Max Beckmann,German,Nationality
Max Beckmann,Germany,bornInCountry
Max Beckmann,German,bornNationality
Natalia Goncharova,1881,BeginDate
Natalia Goncharova,1962,EndDate
Natalia Goncharova,Female,Gender
Natalia Goncharova,Russian,Nationality
Natalia Goncharova,Russian Empire,bornInCountry
Natalia Goncharova,Russian,bornNationality
Oskar Kokoschka,1886,BeginDate
Oskar Kokoschka,1980,EndDate
Oskar Kokoschka,Male,Gender
Oskar Kokoschka,Austrian,Nationality
Oskar Kokoschka,Austria,bornInCountry
Oskar Kokoschka,Austrian,bornNationality
Pablo Picasso,1881,BeginDate
Pablo Picasso,1973,EndDate
Pablo Picasso,Male,Gender
Pablo Picasso,Spanish,Nationality
Pablo Picasso,Spain,bornInCountry
Pablo Picasso,Spanish,bornNationality
Paul Cézanne,1839,BeginDate
Paul Cézanne,1906,EndDate
Paul Cézanne,Male,Gender
Paul Cézanne,French,Nationality
Paul Cézanne,France,bornInCountry
Paul Cézanne,French,bornNationality
Paul Gauguin,1848,BeginDate
Paul Gauguin,1903,EndDate
Paul Gauguin,Male,Gender
Paul Gauguin,French,Nationality
Paul Gauguin,France,bornInCountry
Paul Gauguin,French,bornNationality
Paul Klee,1879,BeginDate
Paul Klee,1940,EndDate
Paul Klee,Male,Gender
Paul Klee,German,Nationality
Paul Klee,Switzerland,bornInCountry
Paul Klee,Swiss,bornNationality
Paul Signac,1863,BeginDate
Paul Signac,1935,EndDate
Paul Signac,Male,Gender
Paul Signac,French,Nationality
Paul Signac,France,bornInCountry
Paul Signac,French,bornNationality
Piet Mondrian,1872,BeginDate
Piet Mondrian,1944,EndDate
Piet Mondrian,Male,Gender
Piet Mondrian,Dutch,Nationality
Piet Mondrian,Netherlands,bornInCountry
Piet Mondrian,Dutch,bornNationality
Vasily Kandinsky,1866,BeginDate
Vasily Kandinsky,1944,EndDate
Vasily Kandinsky,Male,Gender
Vasily Kandinsky,French,Nationality
Vasily Kandinsky,Russian Empire,bornInCountry
Vasily Kandinsky,Russian,bornNationality
Vincent van Gogh,1853,BeginDate
Vincent van Gogh,1890,EndDate
Vincent van Gogh,Male,Gender
Vincent van Gogh,Dutch,Nationality
Vincent van Gogh,Netherlands,bornInCountry
Vincent van Gogh,Dutch,bornNationality

{% endhighlight %}

<p>Build a knowledge graph:</p>

{% highlight scala %}
val graphNodesArtistBio=graphNodes.filter('id=!="").distinct
val graphEdgesArtistBio=graphEdges.filter('src=!="").distinct
val graphArtistBio = GraphFrame(graphNodesArtistBio,graphEdgesArtistBio)

{% endhighlight %}


<p>Save a graph nodes and edges for future posts:</p>

{% highlight scala %}
graphArtistBio.vertices.write.parquet("graphNodesArtistBio")
graphArtistBio.edges.write.parquet("graphEdgesArtistBio")
{% endhighlight %}

<p>Reload graph nodes and edges and rebuild knowledge graph:</p>

{% highlight scala %}
val graphArtistNodes = sqlContext.read.parquet("graphNodesArtistBio")
val graphArtistEdges = sqlContext.read.parquet("graphEdgesArtistBio")
val graphArtist = GraphFrame(graphArtistNodes, graphArtistEdges)
{% endhighlight %}

<a href="#">
    <img src="{{ site.baseurl }}/img/moma3.jpg" alt="Post Sample Image" width="888" height="600">
</a>

<p><h3>Knowledge Graph for Data Mining</h3>
Traditionally knowledge graph data analysis is using SPARQL language. Instead of SPARQL we will use:
<ul>
<li>Spark DataFrame language</li>
<li>Spark GraphFrames we will use a graph query language - motif 'find' function.</li>
</ul>

<p><h4>Artists that Changed Nationalities</h4>
<p>To see which artists changed their nationalities we can query ArtistBio table using Spark DataFrame language:</p>

{% highlight scala %}
val diffNationalityDF=artistBio.filter('Nationality=!='bornNationality).
  select("Artist","bornNationality","Nationality")

Artist,bornNationality,Nationality
Paul Klee,Swiss,German
Vasily Kandinsky,Russian,French
Marc Chagall,Russian,French
{% endhighlight %}

<p>Or we can use graph query language - Spark GraphFrames Motif 'find' function:</p>
{% highlight scala %}

val diffNationalityGF=graphArtist.
   find("(a) - [ab] -> (b); (a) - [ac] -> (c)").
   filter($"ab.edgeId"==="bornNationality").filter($"ac.edgeId"==="Nationality").
   filter(not($"c.id".contains($"b.id"))).
   select("a.id","b.id","c.id").toDF("Artist","bornNationality","Nationality")

Artist,bornNationality,Nationality
Marc Chagall,Russian,French
Vasily Kandinsky,Russian,French
Paul Klee,Swiss,German

{% endhighlight %}

<p><h4>Artists Born in the Same Country</h4>
<p>To find pairs of artists that were born in the same country using Spark DataFrame language we have to self-join ArtistBio table:</p>

{% highlight scala %}
val sameCountryBirthDF=artistBio.select("Artist","bornInCountry").
  join(artistBio.select("Artist","bornInCountry").toDF("Artist2","bornInCountry2"),
  'bornInCountry==='bornInCountry2).
  filter('Artist<'Artist2).select("bornInCountry","Artist","Artist2").distinct

display(sameCountryBirthDF.filter('bornInCountry==="Austria"))
bornInCountry,Artist1,Artist2
Austria,Egon Schiele,Oskar Kokoschka
{% endhighlight %}

<p>Using motif 'find' function is an elegant way to find artists pairs: </p>

{% highlight scala %}
val sameCountryBirthGF=graphArtist.
   find("(a) - [ac] -> (c); (b) - [bc] -> (c)").
   filter($"ac.edgeId"===$"bc.edgeId" && $"ac.edgeId"==="bornInCountry").
   filter($"a.id"<$"b.id").
   select("c.id","a.id","b.id").toDF("bornInCountry","Artist1","Artist2").distinct

display(sameCountryBirthGF.filter('bornInCountry==="Austria"))
bornInCountry,Artist1,Artist2
Austria,Egon Schiele,Oskar Kokoschka
{% endhighlight %}

<p><h4>Biography Relationships</h4>

<p>(Subject, Object, Predicate) data structure is different that tabular data structure in particularly it  builds and explains data relationships. The graphArtist graph shows groups of artists that were born in the same country, had same nationality or same gender. Are there any other Artists relationships?</p>

{% highlight scala %}
val sharedPredicates=graphArtist.
   find("(a) - [ac] -> (c); (b) - [bc] -> (c)").
   filter($"a.id"=!=$"b.id").
   filter($"ac.edgeId" <= $"bc.edgeId").
   select("ac.edgeId","bc.edgeId").
   toDF("predicate1","predicate2").distinct

display(sharedPredicates.orderBy("predicate1","predicate2"))
predicate1,predicate2
BeginDate,BeginDate
BeginDate,EndDate
EndDate,EndDate
Gender,Gender
Nationality,Nationality
Nationality,bornNationality
bornInCountry,bornInCountry
bornNationality,bornNationality

{% endhighlight %}

<p>The same years in artist biographies:</p>

{% highlight scala %}
val sameDates=graphArtist.
   find("(a) - [ac] -> (c); (b) - [bc] -> (c)").
   filter($"a.id" < $"b.id").filter($"ac.edgeId".rlike("Date")).
   select("c.id","a.id","ac.edgeId","b.id","bc.edgeId").
   toDF("SameYear","Artist1","Year1","Artist2","Year2").distinct

display(sameDates.orderBy('SameYear))
SameYear,Artist1,Year1,Artist2,Year2
1881,Natalia Goncharova,BeginDate,Pablo Picasso,BeginDate
1890,Egon Schiele,BeginDate,Vincent van Gogh,EndDate
1935,Kazimir Malevich,EndDate,Paul Signac,EndDate
1944,Piet Mondrian,EndDate,Vasily Kandinsky,EndDate

{% endhighlight %}


<p><h3>Next Post - Paintings</h3>
In the next several posts we will deeper look at MoMA dataset.</p>
