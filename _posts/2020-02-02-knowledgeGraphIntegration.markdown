---
layout:     post
title:      "Knowledge Graph for Data Integration"
subtitle:   "Connect various datasets via Spark Knowledge Graph"
date:       2020-02-02 12:00:00
author:     "Melenar"
header-img: "img/modern225.jpg"
---

<p><h3>Data Integration </h3>

In this post we will show how to use knowledge graph to integrate data of different data types from multiple data sources.<br>
We will integrate the following datasets:

<ul>
<li>
Kaggle dataset that we used in the previous two posts: <i><a href="https://www.kaggle.com/momanyc/museum-collection">
'Museum of Modern Art Collection'</a></i>.
</li>
<li>
Data from MoMA exhibition:
<i><a href="https://www.moma.org/interactives/exhibitions/2012/inventingabstraction/?page=artists"> 'Inventing Abstraction 1910-1925'</a></i>

</li>
<li>
Data about a timeline of the
<i><a href="https://drawpaintacademy.com/modern-art-movements/"> 'Modern Art Movements'</a></i>

</li>

</ul>

<h3>Museum of Modern Art Collection</h3>
<p></p>

Data in Museum of Modern Art Collection has information about artists, there biographies and there paintings. In two previous posts based on this data we demonstrated how to use Knowledge Graph for data mining and semantics.
<br>In the <i><a href="
http://sparklingdataocean.com/2019/09/24/knowledgeGraphDataAnalysis/
">
'Knowledge Graph for Data Mining' </a></i> post we created Artist Biography knowledge graph connecting artist names with their nationalities, countries where they were born, genders and life years.
This knowledge graph allowed us to find different groups of artists, for example artists that were born in the same country or artists that changed their nationalities. <p></p>
We will start data integration from the Artist Biography knowledge graph nodes and edges:
<p></p>
{% highlight scala %}
val artistBioNodes = sqlContext.read.parquet("graphNodesArtistBio")
val artistBioEdges = sqlContext.read.parquet("graphEdgesArtistBio")
{% endhighlight %}

{% highlight scala %}
val artistBioGraph=GraphFrame(artistBioNodes, artistBioEdges)
{% endhighlight %}

<a href="#">
    <img src="{{ site.baseurl }}/img/moma53.jpg" alt="Post Sample Image" width="655" height="400">
</a>

This graph is artist name centric:
<p></p>
{% highlight scala %}
val bioArtistList=artistBioGraph.edges.select("src").distinct
display(bioArtistList.orderBy("src"))
Claude Monet
Egon Schiele
Franz Marc
Georges Braque
Henri Matisse
Jackson Pollock
Joan Miró
Kazimir Malevich
Marc Chagall
Max Beckmann
Natalia Goncharova
Oskar Kokoschka
Pablo Picasso
Paul Cézanne
Paul Gauguin
Paul Klee
Paul Signac
Piet Mondrian
Vasily Kandinsky
Vincent van Gogh
{% endhighlight %}

<p></p>
Edge types in the Artist Biography knowledge graph are the following:
<p></p>
{% highlight scala %}
display(artistBioEdges.select("edgeId").distinct.orderBy('edgeId))
BeginDate
EndDate
Gender
Nationality
bornInCountry
bornNationality
{% endhighlight %}

<p></p>
To define edge destination types we will show 3 examples of destinations for all edge types:
<p></p>
{% highlight scala %}
import org.apache.spark.sql.expressions.Window
val partitionWindow = Window.partitionBy($"edgeId").orderBy($"dst")
display(artistBioEdges.select("dst","edgeId").distinct.
withColumn("rn",row_number().over(partitionWindow)).
filter("rn<4").orderBy('edgeId,'rn))
  1839,BeginDate,1
  1840,BeginDate,2
  1848,BeginDate,3
  1890,EndDate,1
  1903,EndDate,2
  1906,EndDate,3
  Female,Gender,1
  Male,Gender,2
  American,Nationality,1
  Austrian,Nationality,2
  Dutch,Nationality,3
  Austria,bornInCountry,1
  France,bornInCountry,2
  Germany,bornInCountry,3
  American,bornNationality,1
  Austrian,bornNationality,2
  Dutch,bornNationality,3
{% endhighlight %}

<p></p>
Ontology graph edges:
<p></p>
{% highlight scala %}
val artistBioOntoEdged=sc.parallelize(Array(
  ("Year","BeginDate"),("Year","EndDate"),
  ("Gender","Gender"),("Nationality","Nationality"),
  ("Nationality","bornNationality"),("Country","bornInCountry"))).
  toDF("dst","edgeId").withColumn("src",lit("ArtistName"))
{% endhighlight %}


<p></p>
Ontology graph:
<p></p>
{% highlight scala %}
val artistBioOntoGraph=GraphFrame(
  artistBioOntoEdged.select("src").
  union(artistBioOntoEdged.select("dst")).
  distinct.toDF("id"),
  artistBioOntoEdged.distinct)
{% endhighlight %}

<p></p>
{% highlight scala %}
display(graph2dot(artistBioOntoGraph))
"Year" -> "EndDate" [label="ArtistName"];
"Nationality" -> "Nationality" [label="ArtistName"];
"Year" -> "BeginDate" [label="ArtistName"];
"Nationality" -> "bornNationality" [label="ArtistName"];
"Country" -> "bornInCountry" [label="ArtistName"];
"Gender" -> "Gender" [label="ArtistName"];
{% endhighlight %}

<p></p>
Ontology of Artist Biography knowledge graph:

<a href="#">
    <img src="{{ site.baseurl }}/img/moma36.jpg" alt="Post Sample Image" width="601" height="601">
</a>


<h3>"Inventing Abstraction 1910-1925" MoMA Exhibition Data</h3>
<p>
As the next set of data we will use data from MoMA exhibition "Inventing Abstraction 1910-1925"
<a href="https://www.moma.org/interactives/exhibitions/2012/inventingabstraction/?page=artists"> presented many abstraction artists.</a>
The following artists from our Artist Biography knowledge graph were presented on that exhibition:

<small><i>
<br>
<font style="color:rgb(99, 99, 99);">
<ul type="circle">
<li> Vasily Kandinsky
</li><li> Franz Marc
</li><li> Kazimir Malevich
</li><li> Natalia Goncharova
</li><li> Piet Mondrian
</li><li> Paul Klee
</li><li> Pablo Picasso
</li>
</ul>
</font>
</i></small>
<p></p>
MoMA website has a lot of interesting information about this exhibition. In particularly this exhibition's
<i><a href="https://www.moma.org/interactives/exhibitions/2012/inventingabstraction/?page=connections"> Artist Connections graph</a></i> illustrated productive relationships between artists. From this network we've got pair relationships between artists from the Artist Biography knowledge graph:

<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/moma33c.jpg" alt="Post Sample Image" width="888" height="600">
</a>
<p></p>
Connections between artist pairs:
<p></p>
{% highlight scala %}
val artistPairsDF=sc.parallelize(Array(
  ("Vasily Kandinsky","Franz Marc"),
  ("Vasily Kandinsky","Kazimir Malevich"),
  ("Vasily Kandinsky","Natalia Goncharova"),
  ("Vasily Kandinsky","Paul Klee"),
  ("Franz Marc","Paul Klee"),
  ("Kazimir Malevich","Natalia Goncharova"),
  ("Natalia Goncharova","Pablo Picasso"),
  ("Paul Klee","Pablo Picasso"))).
toDF("col1","col2")
{% endhighlight %}

<p></p>
Knowledge graph edges:
<p></p>
{% highlight scala %}
val artistPairEdges=artistPairsDF.
  union(artistPairsDF.select("col2","col1")).toDF("src","dst").
  withColumn("edgeId",lit("relationship"))
{% endhighlight %}

<p></p>
Artist pairs knowledge graph:
<p></p>
{% highlight scala %}
val artistPairGraph=GraphFrame(
  artistPairEdges.select("src").
  union(artistPairEdges.select("dst")).distinct.toDF("id"),
  artistPairEdges.distinct)
{% endhighlight %}

<p></p>
Artist pairs knowledge graph:
<p></p>
{% highlight scala %}
display(graph2dot(artistPairGraph))
"Franz Marc" -> "Vasily Kandinsky" [label="relationship"];
"Vasily Kandinsky" -> "Natalia Goncharova" [label="relationship"];
"Natalia Goncharova" -> "Vasily Kandinsky" [label="relationship"];
"Vasily Kandinsky" -> "Franz Marc" [label="relationship"];
"Vasily Kandinsky" -> "Kazimir Malevich" [label="relationship"];
"Paul Klee" -> "Vasily Kandinsky" [label="relationship"];
"Paul Klee" -> "Franz Marc" [label="relationship"];
"Paul Klee" -> "Pablo Picasso" [label="relationship"];
"Natalia Goncharova" -> "Pablo Picasso" [label="relationship"];
"Franz Marc" -> "Paul Klee" [label="relationship"];
"Natalia Goncharova" -> "Kazimir Malevich" [label="relationship"];
"Pablo Picasso" -> "Paul Klee" [label="relationship"];
"Kazimir Malevich" -> "Natalia Goncharova" [label="relationship"];
"Vasily Kandinsky" -> "Paul Klee" [label="relationship"];
"Kazimir Malevich" -> "Vasily Kandinsky" [label="relationship"];
"Pablo Picasso" -> "Natalia Goncharova" [label="relationship"];

{% endhighlight %}

<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/moma41.jpg" alt="Post Sample Image" width="333" height="300">
</a>
<p></p>

<p></p>
Ontology:
<p></p>
{% highlight scala %}
"ArtistName" -> "ArtistName" [label="relationship"];
{% endhighlight %}

<p></p>
Ontology of connections between artists:

<a href="#">
    <img src="{{ site.baseurl }}/img/moma35.jpg" alt="Post Sample Image" width="333" height="300">
</a>


<h3>Integrate Artist Biography knowledge graph with artists' relationships</h3>
<p>
<p></p>

<p><h4>Metadata integration</h4>

Ontology of integrated graph:

<a href="#">
    <img src="{{ site.baseurl }}/img/moma39.jpg" alt="Post Sample Image" width="616" height="616">
</a>

<p><h4>Data integration</h4>

There are two ways to integrate Artist Biography knowledge graph with Artists Pairs knowledge graph:
<ul>
<li>
Add Artists Pairs edges to Artist Biography edges
</li><li>
Overlap Artists Pairs nodes with Artist Biography nodes
</li>
</ul>

Add Artists Pairs edges to Artist Biography edges
<p></p>
{% highlight scala %}
val unionArtistPairBioEdges=artistPairGraph.edges.
  union(artistBioGraph.edges)
val unionArtistPairBioGraph=GraphFrame(
  unionArtistPairBioEdges.select("src").
  union(unionArtistPairBioEdges.select("dst")).distinct.toDF("id"),
  unionArtistPairBioEdges.distinct)
{% endhighlight %}

Overlap Artists Pairs nodes with Artist Biography nodes
<p></p>
{% highlight scala %}
val overlapArtistPairBioEdges=artistBioGraph.edges.
  join(artistPairGraph.vertices,'src==='id || 'dst==='id).
  select("src","dst","edgeId").union(artistPairGraph.edges)
val overlapArtistPairBioGraph=GraphFrame(
  overlapArtistPairBioEdges.select("src").
    union(overlapArtistPairBioEdges.select("dst")).distinct.toDF("id"),
  overlapArtistPairBioEdges.distinct)   
{% endhighlight %}

<a href="#">
    <img src="{{ site.baseurl }}/img/moma42.jpg" alt="Post Sample Image" width="471" height="434">
</a>

<p></p>
<h3>Modern Art Movements Timeline Data</h3>
<p></p>
Our next dataset is data about a timeline of the
<i><a href="https://drawpaintacademy.com/modern-art-movements/"> Modern Art Movements:</a></i>

<p></p>
{% highlight scala %}
display(sc.textFile("/FileStore/tables/aboutModerArt").toDF("charLine"))
{% endhighlight %}
<p></p>

<small>
<font style="color:rgb(33, 33, 33);">
  1872 – 1892
  <br>
  Impressionism
  <br>
  Summary: Masters of color and light. Marked a radical departure from the realistic academic painting that had dominated the eras prior.
  <br>
  "Key artists: Claude Monet, Pierre-Auguste Renoir, Camille Pissarro, Edgar Degas, Edouard Manet, Mary Cassatt"
  <br>
  """When you go out to paint try to forget what object you have before you - a tree, a house, a field or whatever. Merely think, here is a little square of blue, here an oblong of pink, here a streak of yellow, and paint it just as it looks to you, the exact color and shape, until it emerges as your own naive impression of the scene before you."" Claude Monet"Early
  <p></p>
  1880s - 1914
  <br>
  Post-Impressionism
  <br>
  "Summary: Emphasis on symbolic content and the artist's interpretation of the world. Post-impressionism shared many of the characteristics of Impressionism such as the use of vivid colors, expressive brushwork and everyday subjects. But there seemed to be a focus on distorted forms, geometric shapes and unnaturalistic colors to depict emotions and feelings. Artists often used the pointillism technique, which involved placing small dabs of distinct color."
<br>
  "Key artists: Paul Cézanne (the ""father of Post-Impressionism""), Vincent Van Gogh, Paul Gauguin, Georges-Pierre Seurat, Paul Signac"
<br>
  """I dream of painting and then I paint my dream."" Vincent van Gogh"  
<p></p>
  1907 – 1922
  <br>
  Cubism
  <br>
  "Summary: Focused on abstraction and geometric shapes, rather than space, perspective and realistic rendering."
  <br>
  "Key artists: Pablo Picasso, Georges Braque, Juan Gris, Fernand Léger"
  <br>
  """Cubism is not a reality you can take in your hand. It's more like a perfume, in front of you, behind you, to the sides, the scent is everywhere but you don't quite know where it comes from."" Pablo Picasso"
  <p></p>
  1924-1966
  <br>
  Surrealism
  <br>
  "Summary: Depicted dreams, fantasies and the unconscious state. Often incorporated the juxtaposition of incompatible elements."
  <br>
  "Key artists: Joan Miró, Salvador Dalí, René Magritte, André Breton, Yves Tanguy, Frida Kahlo, Max Ernst, Méret Oppenheim"
  <br>
  """Surrealism is destructive, but it destroys only what it considers to be shackles limiting our vision."" Salvador Dalí"
  </font>
</small>

<p></p>
To convert this data to DataFrame we will do the following:
<ul type="square">
<li> Index data by zipping text lines with range from 0 to 84
</li><li> Transform to DataFrame (index, line)
</li><li> Calculate reminder index%5 - add a column "rem"
</li><li> Calculate deviser index/5- add a column "div"
</li>
</ul>
<p></p>
{% highlight scala %}
import org.apache.spark.sql.functions._
val data=(Seq.range(0,85) zip  
  sc.textFile("/FileStore/tables/aboutModerArt").collect).
  toDF("inx","charLine").withColumn("rem",col("inx")%5).
  withColumn("div",floor(col("inx")/5))
display(data)
inx,charLine,rem,div
0,1872 – 1892,0,0
1,Impressionism,1,0
2,Summary: Masters of color and light. Marked a radical departure from the realistic academic painting that had dominated the eras prior.,2,0
3,"Key artists: Claude Monet, Pierre-Auguste Renoir, Camille Pissarro, Edgar Degas, Edouard Manet, Mary Cassatt",3,0

{% endhighlight %}
<p></p>
Next we will combine data by art movements by self-joining the table several times:
<p></p>
{% highlight scala %}
val dataModernArt=data.filter('rem===0).
   join(data.filter('rem===1),"div").
   join(data.filter('rem===2),"div").
   join(data.filter('rem===3),"div").
   join(data.filter('rem===4),"div").
   drop("rem","inx").
   toDF("rownumber","time","artMovement","summary","keyArtists","aboutArt")

display(dataModernArt)
{% endhighlight %}

<p></p>
Then we will split "keyArtists" column to Key Artist names:
<p></p>
{% highlight scala %}

val dataModernArtArtist=dataModernArt.drop("summary","aboutArt").
   withColumn("keyArtistList",
   regexp_replace(regexp_replace(col("keyArtists"),"\\(.+?\\)",""),
   "Key artists: ","")).
   withColumn("keyArtistText",explode(split(col("keyArtistList"),","))).
   withColumn("keyArtist",trim(col("keyArtistText")))

   display(dataModernArtArtist.select("rownumber","time","artMovement","keyArtist"))
   0,1872 – 1892,Impressionism,Claude Monet
   0,1872 – 1892,Impressionism,Pierre-Auguste Renoir
   0,1872 – 1892,Impressionism,Camille Pissarro
   0,1872 – 1892,Impressionism,Edgar Degas
   0,1872 – 1892,Impressionism,Edouard Manet
   0,1872 – 1892,Impressionism,Mary Cassatt
   1,Early 1880s - 1914,Post-Impressionism,Paul Cézanne
   1,Early 1880s - 1914,Post-Impressionism,Vincent Van Gogh
   1,Early 1880s - 1914,Post-Impressionism,Paul Gauguin
   1,Early 1880s - 1914,Post-Impressionism,Georges-Pierre Seurat
   1,Early 1880s - 1914,Post-Impressionism,Paul Signac
   2,1905 - 1910,Fauvism,Henri Matisse
   2,1905 - 1910,Fauvism,André Derain
   2,1905 - 1910,Fauvism,Maurice de Vlaminck
   2,1905 - 1910,Fauvism,Kees van Dongen
   3,1905 – 1933,Expressionism,Ernst Ludwig Kirchner
   3,1905 – 1933,Expressionism,Käthe Kollwitz
   3,1905 – 1933,Expressionism,Wassily Kandinsky
   3,1905 – 1933,Expressionism,Edvard Munch
   4,1907 – 1922,Cubism,Pablo Picasso
   4,1907 – 1922,Cubism,Georges Braque
   4,1907 – 1922,Cubism,Juan Gris
   4,1907 – 1922,Cubism,Fernand Léger
   5,1909 – late 1920s,Futurism,Umberto Boccioni
   5,1909 – late 1920s,Futurism,Carlo Carrà
   5,1909 – late 1920s,Futurism,Giacomo Balla
   5,1909 – late 1920s,Futurism,Natalia Goncharova
   6,1913 – late 1920s,Suprematism,Kazimir Malevich
   6,1913 – late 1920s,Suprematism,El Lissitzky
   6,1913 – late 1920s,Suprematism,Olga Rozanova
   6,1913 – late 1920s,Suprematism,Lyubov Popova
   7,1917 – 1931,De Stijl,Theo van Doesburg
   7,1917 – 1931,De Stijl,Piet Mondrian
   7,1917 – 1931,De Stijl,Vilmos Huszar
   7,1917 – 1931,De Stijl,Gerrit Rietveld
   8,1915-late 1930s,Constructivism,Vladimir Tatlin
   8,1915-late 1930s,Constructivism,Alexander Rodchenko
   8,1915-late 1930s,Constructivism,Varvara Stepanova
   8,1915-late 1930s,Constructivism,Aleksandra Ekster
   8,1915-late 1930s,Constructivism,Lyubov Popova
   9,1916-1924,Dada,Francis Picabia
   9,1916-1924,Dada,Hugo Ball
   9,1916-1924,Dada,Hans Arp
   9,1916-1924,Dada,Tristan Tzara
   10,1924-1966,Surrealism,Joan Miró
   10,1924-1966,Surrealism,Salvador Dalí
   10,1924-1966,Surrealism,René Magritte
   10,1924-1966,Surrealism,André Breton
   10,1924-1966,Surrealism,Yves Tanguy
   10,1924-1966,Surrealism,Frida Kahlo
   10,1924-1966,Surrealism,Max Ernst
   10,1924-1966,Surrealism,Méret Oppenheim
   11,1943 – 1965,Abstract Expressionism,Willem de Kooning
   11,1943 – 1965,Abstract Expressionism,Clyfford Still
   11,1943 – 1965,Abstract Expressionism,Mark Rothko
   11,1943 – 1965,Abstract Expressionism,Jackson Pollock
   12,Mid-1950s - early 1970s,Pop Art,Andy Warhol
   12,Mid-1950s - early 1970s,Pop Art,Roy Lichtenstein
   12,Mid-1950s - early 1970s,Pop Art,Claes Oldenburg
   12,Mid-1950s - early 1970s,Pop Art,James Rosenquist
   12,Mid-1950s - early 1970s,Pop Art,Richard Hamilton
   13,Early 1960s - late 1960s,Minimalism,Frank Stella
   13,Early 1960s - late 1960s,Minimalism,Tony Smith
   13,Early 1960s - late 1960s,Minimalism,Carl Andre
   13,Early 1960s - late 1960s,Minimalism,Richard Serra
   13,Early 1960s - late 1960s,Minimalism,Dan Flavin
   14,Mid-1960s onwards,Conceptual Art,Joseph Kosuth
   14,Mid-1960s onwards,Conceptual Art,Walter de Maria
   14,Mid-1960s onwards,Conceptual Art,John Baldessari
   14,Mid-1960s onwards,Conceptual Art,Sol LeWitt
   14,Mid-1960s onwards,Conceptual Art,Joseph Beuys
   15,1974-1984,Pictures Generation,Cindy Sherman
   15,1974-1984,Pictures Generation,Barbara Kruger
   15,1974-1984,Pictures Generation,Robert Longo
   15,1974-1984,Pictures Generation,Richard Prince
   16,Late 1970s - early 1990s,Neo-Expressionism,Georg Baselitz
   16,Late 1970s - early 1990s,Neo-Expressionism,Julian Schnabel
   16,Late 1970s - early 1990s,Neo-Expressionism,Francesco Clemente
   16,Late 1970s - early 1990s,Neo-Expressionism,Jean-Michel Basquiat
{% endhighlight %}

<p></p>
From Modern Art Movements Key Artists we will take the list of artists from our Artist Biography knowledge graph:
<p></p>
{% highlight scala %}
val modernArtData = dataModernArtArtist.
  select("time","artMovement","keyArtist").
  join(bioArtistList,'keyArtist==='src).drop("src").distinct

display(modernArtArtist)
time,artMovement,keyArtist
1905 - 1910,Fauvism,Henri Matisse
1943 – 1965,Abstract Expressionism,Jackson Pollock
1924-1966,Surrealism,Joan Miró
Early 1880s - 1914,Post-Impressionism,Paul Gauguin
1907 – 1922,Cubism,Georges Braque
Early 1880s - 1914,Post-Impressionism,Paul Cézanne
1907 – 1922,Cubism,Pablo Picasso
Early 1880s - 1914,Post-Impressionism,Paul Signac
1917 – 1931,De Stijl,Piet Mondrian
1872 – 1892,Impressionism,Claude Monet
1909 – late 1920s,Futurism,Natalia Goncharova
1913 – late 1920s,Suprematism,Kazimir Malevich
{% endhighlight %}

<p></p>
We will build Modern Art Movement knowledge graph based on ontology:

<a href="#">
    <img src="{{ site.baseurl }}/img/moma38b.jpg" alt="Post Sample Image" width="333" height="300">
</a>

<p></p>
Modern Art Movement knowledge graph edges:
<p></p>
{% highlight scala %}
val modernArtEdges=
  modernArtData.select("keyArtist","artMovement").toDF("src","dst").
  withColumn("edgeId",lit("artMovement")).union(
  modernArtData.select("artMovement","time").toDF("src","dst").
  withColumn("edgeId",lit("time"))).distinct
{% endhighlight %}

<p></p>
Modern Art Movement knowledge graph:
<p></p>
{% highlight scala %}
val modernArtGraph=GraphFrame(
  modernArtEdges.select("src").
  union(modernArtEdges.select("dst")).distinct.toDF("id"),
  modernArtEdges.distinct)
{% endhighlight %}

<p></p>
{% highlight scala %}
display(graph2dot(modernArtGraph))
"Jackson Pollock" -> "Abstract Expressionism" [label="artMovement"];
"Fauvism" -> "1905 - 1910" [label="time"];
"Paul Cézanne" -> "Post-Impressionism" [label="artMovement"];
"Cubism" -> "1907 – 1922" [label="time"];
"Kazimir Malevich" -> "Suprematism" [label="artMovement"];
"Impressionism" -> "1872 – 1892" [label="time"];
"Abstract Expressionism" -> "1943 – 1965" [label="time"];
"Futurism" -> "1909 – late 1920s" [label="time"];
"Paul Signac" -> "Post-Impressionism" [label="artMovement"];
"Piet Mondrian" -> "De Stijl" [label="artMovement"];
"Henri Matisse" -> "Fauvism" [label="artMovement"];
"Pablo Picasso" -> "Cubism" [label="artMovement"];
"Suprematism" -> "1913 – late 1920s" [label="time"];
"Natalia Goncharova" -> "Futurism" [label="artMovement"];
"Georges Braque" -> "Cubism" [label="artMovement"];
"Joan Miró" -> "Surrealism" [label="artMovement"];
"Post-Impressionism" -> "Early 1880s - 1914" [label="time"];
"De Stijl" -> "1917 – 1931" [label="time"];
"Claude Monet" -> "Impressionism" [label="artMovement"];
"Surrealism" -> "1924-1966" [label="time"];
"Paul Gauguin" -> "Post-Impressionism" [label="artMovement"];
{% endhighlight %}

<a href="#">
    <img src="{{ site.baseurl }}/img/moma44b.jpg" alt="Post Sample Image" width="579" height="500">
</a>
<p><h3>Artist Pairs within Modern Art Movement Key Artists list</h3>

<p> </p>
Ontology of integrated graph:
<a href="#">
    <img src="{{ site.baseurl }}/img/moma50.jpg" alt="Post Sample Image" width="371" height="453">
</a>
<p></p>

<p><h4>Data integration</h4>
<p></p>

<p>To add artist pair relationship info to modern art movement info we will combine edges of Modern Art Movement and Artist Pairs knowledge graphs:

{% highlight scala %}
val modernArtPairGraph=GraphFrame(
  artistPairGraph.edges.union(modernArtGraph.edges).select("src").
  union(artistPairGraph.edges.union(modernArtGraph.edges).select("dst")).distinct.toDF("id"),
  artistPairGraph.edges.union(modernArtGraph.edges)
)
{% endhighlight %}

{% highlight scala %}
display(graph2dot(modernArtPairGraph))
"Vasily Kandinsky" -> "Franz Marc" [label="relationship"];
"Vasily Kandinsky" -> "Kazimir Malevich" [label="relationship"];
"Paul Klee" -> "Vasily Kandinsky" [label="relationship"];
"Paul Klee" -> "Franz Marc" [label="relationship"];
"Paul Klee" -> "Pablo Picasso" [label="relationship"];
"Abstract Expressionism" -> "1943 – 1965" [label="time"];
"Futurism" -> "1909 – late 1920s" [label="time"];
"Natalia Goncharova" -> "Pablo Picasso" [label="relationship"];
"Paul Signac" -> "Post-Impressionism" [label="artMovement"];
"Piet Mondrian" -> "De Stijl" [label="artMovement"];
"Franz Marc" -> "Paul Klee" [label="relationship"];
"Natalia Goncharova" -> "Kazimir Malevich" [label="relationship"];
"Pablo Picasso" -> "Paul Klee" [label="relationship"];
"Henri Matisse" -> "Fauvism" [label="artMovement"];
"Kazimir Malevich" -> "Natalia Goncharova" [label="relationship"];
"Vasily Kandinsky" -> "Paul Klee" [label="relationship"];
"Pablo Picasso" -> "Cubism" [label="artMovement"];
"Suprematism" -> "1913 – late 1920s" [label="time"];
"Natalia Goncharova" -> "Futurism" [label="artMovement"];
"Kazimir Malevich" -> "Vasily Kandinsky" [label="relationship"];
"Georges Braque" -> "Cubism" [label="artMovement"];
"Joan Miró" -> "Surrealism" [label="artMovement"];
"Post-Impressionism" -> "Early 1880s - 1914" [label="time"];
"De Stijl" -> "1917 – 1931" [label="time"];
"Claude Monet" -> "Impressionism" [label="artMovement"];
"Surrealism" -> "1924-1966" [label="time"];
"Pablo Picasso" -> "Natalia Goncharova" [label="relationship"];
"Paul Gauguin" -> "Post-Impressionism" [label="artMovement"];
{% endhighlight %}
</p><p>
<a href="#">
    <img src="{{ site.baseurl }}/img/moma49c.jpg" alt="Post Sample Image" width="679" height="500">
</a>
<p><h3>Add Artist Biography info to Modern Art Movement key artists</h3>

<p></p>
Ontology of integrated graph:
<a href="#">
    <img src="{{ site.baseurl }}/img/moma46.jpg" alt="Post Sample Image" width="611" height="471">
</a>

<p><h4>Data integration</h4>
<p></p>
<p>From Artist Biography knowledge graph we will take information about nationalities and born countries of modern art key artists.
</p>
{% highlight scala %}
val modernArtKeyArtists = modernArtData.select("keyArtist")
display(modernArtKeyArtists.orderBy("keyArtist"))
Claude Monet
Georges Braque
Henri Matisse
Jackson Pollock
Joan Miró
Kazimir Malevich
Natalia Goncharova
Pablo Picasso
Paul Cézanne
Paul Gauguin
Paul Signac
Piet Mondrian
{% endhighlight %}


<p>Combine edges of Artist Biography knowledge graph and edges of Modern Art Movement knowledge graph and exclude edges related to genders and dates:
</p>
{% highlight scala %}
val modernArtBioEdges=artistBioGraph.edges.
  join(modernArtKeyArtists,'src==='keyArtist).drop("keyArtist").
  union(modernArtGraph.edges).
  filter(not('edgeId.isin("EndDate","BeginDate","Gender")))
{% endhighlight %}


<p>Biographies of Modern Art Movement key artists:
</p>
{% highlight scala %}
val modernArtBioGraph=GraphFrame(
  modernArtBioEdges.select("src").
  union(modernArtBioEdges.select("dst")).distinct.toDF("id"),
  modernArtBioEdges)
{% endhighlight %}


<a href="#">
    <img src="{{ site.baseurl }}/img/moma45c.jpg" alt="Post Sample Image" width="700" height="471">
</a>
<p></p>


<p></p>
<p><h3>Integrate All Three Knowledge Graphs</h3>
<p><h4>Metadata integration</h4>

</p><p>
<p></p>
Ontology of integrated graph:

<a href="#">
    <img src="{{ site.baseurl }}/img/moma40b.jpg" alt="Post Sample Image" width="665" height="471">
</a>
<p></p>

<h4>Data Integration</h4>
<p>
First from each graph we will get artist name list then calculate overlap of these artist lists.
<br>Artist names from Artist Biography knowledge graph: </p>

{% highlight scala %}
val artistBioNodes=artistBioGraph.vertices.distinct.toDF("id1")
{% endhighlight %}
</p><p>

Artist names from Artist Pairs knowledge graph: </p>

{% highlight scala %}
val artistPairNodes=artistPairGraph.vertices.distinct.toDF("id2")
{% endhighlight %}
</p><p>

Artist names from Modern Art Movement knowledge graph: </p>

{% highlight scala %}
val modernArtNodes=modernArtGraph.vertices.distinct.toDF("id3")
{% endhighlight %}
</p><p>

There are only three artist names that are in each of three knowledge graphs: </p>

{% highlight scala %}
val artistNodes=artistBioNodes.join(artistPairNodes,'id1==='id2).
  join(modernArtNodes,'id2==='id3).drop("id2","id3").toDF("id")
display(artistNodes)
Pablo Picasso
Kazimir Malevich
Natalia Goncharova
{% endhighlight %}

</p><p>

<p>
Next, we need to get edges from knowledge graphs related to overlapping artists.  

<br>As Artist Biography knowledge graph is artist name centric, to get combined graph edges we will filter them by edge 'src': </p>
{% highlight scala %}
val artistBioEdges=artistBioGraph.edges.
  join(artistNodes,'src==='id).drop("id")
{% endhighlight %}
</p><p>

<p>
To get edges from Artist Pairs knowledge graph we need to filter them by both edge parameters - 'src' and edge 'dst': </p>
{% highlight scala %}
val artistPairEdges=artistPairGraph.edges.join(artistNodes,'src==='id1).
  join(artistNodes.toDF("id2"),'dst==='id2).drop("id1","id2").distinct
{% endhighlight %}
</p><p>

<p>
Modern Art Movement knowledge graph has two types of edges: Artist -> Modern Art Movement and Modern Art Movement -> time period. {Artist, Modern Art Movement} edges :</p>
{% highlight scala %}
val modernArtEdges1=modernArtGraph.edges.
  join(artistNodes,'src==='id1).drop("id1")
{% endhighlight %}
</p><p>

<p>
{Modern Art Movement, time period} edges: </p>
{% highlight scala %}
val modernArtEdges2=modernArtGraph.edges.join(modernArtEdges1.
  select("dst").toDF("dst1"),'src==='dst1).drop("dst1")
{% endhighlight %}
</p><p>

<p></p>
Edges of combined graph:
<p></p>
{% highlight scala %}
val artistEdges=artistBioEdges.
  union(artistPairEdges).
  union(modernArtEdges1).
  union(modernArtEdges2)
{% endhighlight %}


<p></p>
Build a combine graph:
<p></p>
{% highlight scala %}
val artistGraph=GraphFrame(artistNodes,artistEdges)
display(graph2dot(artistGraph).orderBy("dotLine"))
"Cubism" -> "1907 – 1922" [label="time"];
"Futurism" -> "1909 – late 1920s" [label="time"];
"Kazimir Malevich" -> "1878" [label="BeginDate"];
"Kazimir Malevich" -> "1935" [label="EndDate"];
"Kazimir Malevich" -> "Male" [label="Gender"];
"Kazimir Malevich" -> "Natalia Goncharova" [label="relationship"];
"Kazimir Malevich" -> "Russian Empire" [label="bornInCountry"];
"Kazimir Malevich" -> "Russian" [label="Nationality"];
"Kazimir Malevich" -> "Russian" [label="bornNationality"];
"Kazimir Malevich" -> "Suprematism" [label="artMovement"];
"Natalia Goncharova" -> "1881" [label="BeginDate"];
"Natalia Goncharova" -> "1962" [label="EndDate"];
"Natalia Goncharova" -> "Female" [label="Gender"];
"Natalia Goncharova" -> "Futurism" [label="artMovement"];
"Natalia Goncharova" -> "Kazimir Malevich" [label="relationship"];
"Natalia Goncharova" -> "Pablo Picasso" [label="relationship"];
"Natalia Goncharova" -> "Russian Empire" [label="bornInCountry"];
"Natalia Goncharova" -> "Russian" [label="Nationality"];
"Natalia Goncharova" -> "Russian" [label="bornNationality"];
"Pablo Picasso" -> "1881" [label="BeginDate"];
"Pablo Picasso" -> "1973" [label="EndDate"];
"Pablo Picasso" -> "Cubism" [label="artMovement"];
"Pablo Picasso" -> "Male" [label="Gender"];
"Pablo Picasso" -> "Natalia Goncharova" [label="relationship"];
"Pablo Picasso" -> "Spain" [label="bornInCountry"];
"Pablo Picasso" -> "Spanish" [label="Nationality"];
"Pablo Picasso" -> "Spanish" [label="bornNationality"];
"Suprematism" -> "1913 – late 1920s" [label="time"];
{% endhighlight %}

<p></p>

<a href="#">
    <img src="{{ site.baseurl }}/img/moma52.jpg" alt="Post Sample Image" width="665" height="471">
</a>

<p><h3>Next Post - Paintings</h3>
In the next several posts we will continue looking at Knowledge Graphs as more natural way to represent data.</p>
