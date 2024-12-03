---
layout:     post
title:      "GNN Link Prediction for Country Population, Life Expectancy, Happiness "
subtitle:   "Knowledge Graph of Country Populations: how to deal with variety of data domains"
date:       2024-12-26 12:00:00
author:     "Melenar"
header-img: "img/page125b.jpg"
---
<p><h3> Introduction</h3>
<p></p>
Graphs are everywhere in our lives. They represent molecules in chemistry, roads in navigation, and even our social networks like Facebook, a molecule graph, a city map, a social network graph.
<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/gkgSlide1.jpg" alt="Post Sample Image" width="765" >
</a>
<p></p>
<p></p>

So why graphs?

<p></p>

People don’t think sequentially, especially when solving complex problems, often engaging in mind wandering.
This may explain why the smartest people don’t have more neurons but instead have more connections, enabling dynamic and non-linear thinking.
GNNs replicate this by modeling data as graphs, helping us uncover complex relationships and patterns in neuroscience.
<p></p>
What is graph thinking for data analysis: if humans think this way, why wouldn't we force machine to think this way?

What is machine graph thinking? Data domains that can be represented as graphs can use graph thinking to get insights in data. GNN models were created for this reason.
<p></p>
To represent data as graph, start with thinking about relationships. Why relationships are important?
<p></p>
Using GNN for knowledge graphs that are built on variety of date domains is a challenge.
<p></p>
For spatial data, graph structure can be built based on coordinates. In EEG data we introduced it as
<p></p>
For countries neighbors can be defined by countries that they share borders with.

<p></p>
The next step will be to link these subgraphs through common connections. For example, when we have different domains for the same entities (countries), we connect the same countries across subgraphs.
<p></p>
Many data types can be converted to vectors -- everything -to- vector.
<p></p>
In this study we will put together information about:
long time data for country populations
a few years information about population by age;
median, age 0-5, age 15-64,  65-older
several years of information about country life expectancy
income
country territory
density
agriculture

<p></p>

<p></p>

To define neighbors for countries we can use information about country borders -
Getting information about pairs of countries that share borders will be interesting, especially if we have information about types and quality of borders -- how can people cross the borders.
<p></p>
Two types of country borders: land borders and sea borders.
<p></p>

Land Borders:
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/gkgMap1.jpg" alt="Post Sample Image" width="700" >
</a>
</p><p>
Sea Borders:
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/gkgMap2.jpg" alt="Post Sample Image" width="700" >
</a>
</p><p>
How locations can affect lives? For bigger countries it's more difficult to figure out, for example territory of Russia is very big with a variety of climates.

<p></p>

Borders between the countries are like synapses between neurons. The quality (edge weight) is changing with time: the open the borders are, the better are connections between the countries. The more closed borders are the stronger the conflict between countries.


<p></p>

We will analyze how the border qualities affect how similar are life quality and longertivity.
<p></p>


<p></p>

<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/gkgSlide2.jpg" alt="Post Sample Image" width="479" >
</a>
<p></p>


<p></p>

<h3>Methodologies</h3>
<h4>Prepare Input Data for the Model</h4>
As input data for GNN Link Prediction model we need to create small labeled graphs:
<p></p>
<ul>
<li>
Edges
</li><li>
Nodes with features.
</li><li>

</li></ul>


  <p></p>

<h4>Train the GNN Graph Classification Model</h4>

  In this study we uses a GCNConv model from PyTorch Geometric Library as a GNN graph classification model. The GCNConv model is a type of graph convolutional network that applies convolutional operations to extract meaningful features from the input graph data (edges, node features, and the graph-level labels). The code for the model is taken from a PyG tutorial.
  </p><p>
  <p></p>

<p></p>
<h4>XXXd</h4>
To calculate cosine similarities between pairs of vectors, we will use Cosine Similarities function:  
    <p></p>
    {% highlight python %}
    **
    {% endhighlight %}
    <p></p>

    <p></p>


<p></p>      


<p></p>


<h4>Node Embedding: Catching Pre-Final Vectors in GNN Link Prediction</h4>
<p>GNN Graph Classification with pre-final stop, .</p>

<p></p>
<h4>Interpreting Model Results: Cosine Similarity </h4>
    <p>With node embeddings in place, .</p>

    Select pairs of vectors with cosine similarities higher than 0.95.
        <p></p>  
    <p>This integrated approach allows us to delve deeper into .</p>
<p></p>
<h4>Interpreting Model Results: Symmetry Metrics </h4>


<p></p>
<p></p>
<p></p>
<p></p>
<h3>Experiments Overview </h3>
<p></p>
<p></p>
<p></p>
<p>This section outlines the experimental framework used to </p>
<p></p>
<p></p>
<p></p>     
<h4>Data Sources</h4>
<p></p>
<p></p>
<p></p>
<h5>Data Sources for Country Land Borders</h5>
<p></p>
<p></p>
<p></p>
<p></p>
Country land boundaries and associated metadata are sourced from Natural Earth’s Admin 0 - Countries dataset, version 5.0.1 (2023). This dataset is publicly available under the CC0 1.0 Public Domain Dedication and includes information on GDP, population, and administrative classifications.
<p></p>

Path to your zip file in Google Drive
Path to extract the files
Unzip the file
<p></p>
<p></p>
<p></p>               
{% highlight python %}
import zipfile
import os
zip_file_path = "/content/drive/My Drive/Geo/ne_10m_admin_0_countries.zip"
extract_to_path = "/content/drive/My Drive/Geo/"
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_path)
{% endhighlight %}
<p></p>
<p></p>
<h5>Data Sources for Country Sea Borders</h5>
<p></p>
<p></p>
<p></p>
<p></p>     
Sea boundaries and Exclusive Economic Zones (EEZs) were obtained from Marineregions.org's World EEZ v12 dataset, version 12 (2023). The dataset is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
<p></p>
<p></p>
Path to the ZIP file
Directory to extract files
Unzip the file
List the extracted files
<p></p>
Next,
<p></p>               
{% highlight python %}
import zipfile
import os
zip_file_path = "/content/drive/My Drive/Geo/World_EEZ_v12_20231025_gpkg.zip"
extract_to_path = "/content/drive/My Drive/Geo/World_EEZ_v12_20231025_gpkg/"
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_path)
extracted_files = os.listdir(extract_to_path)
{% endhighlight %}
<p></p>
<p></p>
<h5>Data Sources for Country Life Expectancy</h5>
<p></p>
<p></p>
<p></p>
Next, Load the dataset with additional options
Skip the first 4 rows (based on World Bank CSV format)
Ensure the delimiter is a comma
Use the Python engine for parsing flexibility
<p></p>               
{% highlight python %}
import pandas as pd
file_path ='/content/drive/My Drive/Geo/API_SP.DYN.LE00.IN_DS2_en_csv_v2_99.csv'
life_expectancy = pd.read_csv(
    file_path,
    skiprows=4,          
    delimiter=",",       
    engine="python"      
)
{% endhighlight %}
<p></p>
<p></p>
Next, Extract unique country names
Print the total number of countries and their names
<p></p>               

<p></p>
<p></p>
Next,
<p></p>               
{% highlight python %}
xxxx
{% endhighlight %}
<p></p>
<p></p>
Next,
<p></p>               
{% highlight python %}
xxxx
{% endhighlight %}
<p></p>

<p></p>
<p></p>
<h4>Input Data Preparation</h4>
<p></p>
<p></p>
<p></p>
{% highlight python %}
xxxx
{% endhighlight %}
<p></p>
<p></p>
When integrating multiple datasets with potentially different country naming conventions, it's essential to standardize country names systematically. Below are best practices and tools for handling this efficiently. As country indicators, instead of country names we used ISO_A3 code.
<p></p>
<p></p>
<h5>Step by Step</h5>
<p></p>
<p></p>
<p></p>
{% highlight python %}
xxxx
{% endhighlight %}
<p></p>
<p></p>
<h5>Land Neighbors Graph</h5>
<p></p>
To read borders between countries from the shapefile and extract them as connections (edges) in a graph, follow these steps:
Load the Shapefile
Ensure you've loaded the dataset correctly:
<p></p>

Load the shapefile
Inspect the dataset
<p></p>
<p></p>
<p></p>
{% highlight python %}
import geopandas as gpd
shapefile_path = "/content/drive/My Drive/natural_earth/ne_110m_admin_0_countries.shp"
world = gpd.read_file(shapefile_path)
{% endhighlight %}
<p></p>
<p></p>
<p></p>
<p></p>
Simplify the Dataset
Focus on the necessary columns:

Country Name: Typically NAME, SOVEREIGNT, or similar.
Geometry: Contains polygon data for borders.
Simplify the dataset:
<p></p>
<p></p>
<p></p>
{% highlight python %}
world = world[['NAME', 'geometry']]
world = world.rename(columns={'NAME': 'country'})
{% endhighlight %}<p></p>
<p></p>
<p></p>
<p></p>
Identify Neighboring Countries
Use geopandas spatial operations to find countries that share borders:
 Create a dictionary to store country neighbors
Find all countries that touch the current country's borders
<p></p>
<p></p>
<p></p>
{% highlight python %}
world['geometry'] = world['geometry'].buffer(0)
neighbors = {}
for index, country in world.iterrows():
    touching = world[world.geometry.touches(country.geometry)]['country'].tolist()
    neighbors[country['country']] = touching
{% endhighlight %}
<p></p>
<p></p>
<p></p>
Create the Graph
Convert the neighbors dictionary into a graph using networkx:
<p></p>

Initialize a graph

Add edges (connections between countries)

Inspect the graph
<p></p>
<p></p>
<p></p>
{% highlight python %}
import networkx as nx
G = nx.Graph()
for country, neighbor_list in neighbors.items():
    for neighbor in neighbor_list:
        G.add_edge(country, neighbor)
{% endhighlight %}<p></p>
<p></p>
<p></p>
<h5>Sea Neighbors Graph</h5>
<p></p>
<p></p>

Path to the ZIP file
Directory to extract files
Unzip the file
List the extracted files
<p></p>
<p></p>
<p></p>
{% highlight python %}
import zipfile
import os
zip_file_path = "/content/drive/My Drive/Geo/World_EEZ_v12_20231025_gpkg.zip"
extract_to_path = "/content/drive/My Drive/Geo/World_EEZ_v12_20231025_gpkg/"
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_path)
extracted_files = os.listdir(extract_to_path)
{% endhighlight %}

<p></p>
<p></p>
Ensure Valid Geometry
Some polygons might have invalid geometries, which can cause issues during spatial operations. Fix any invalid geometries using buffer(0):
Confirm the geometries are valid -- # Should return True
<p></p>
<p></p>
<p></p>
{% highlight python %}
eez['geometry'] = eez['geometry'].buffer(0)
{% endhighlight %}<p></p>
<p></p>
<p></p>
<p></p>
To find overlapping EEZs (Sea Neighbors), find which countries share overlapping EEZ boundaries using spatial operations:
Loop through each EEZ polygon to find overlaps
<p></p>
<p></p>
<p></p>
{% highlight python %}
country_pairs = []
for index, country in eez_sea.iterrows():
    overlapping_countries = eez_sea[eez_sea.geometry.intersects(country.geometry)]['country'].tolist()
    overlapping_countries = [c for c in overlapping_countries if c != country['country']]
    for neighbor in overlapping_countries:
        pair = tuple(sorted([country['country'], neighbor]))
        if pair not in country_pairs:
            country_pairs.append(pair)
{% endhighlight %}
<p></p>
To create the Sea Neighbor Graph, convert the sea_neighbors dictionary into a graph using networkx:
<p></p>

Create the graph
Initialize the graph

 Add edges to the graph
 Print basic graph info
<p></p>
<p></p>
<p></p>
{% highlight python %}
import networkx as nx
import matplotlib.pyplot as plt
sea_graph = nx.Graph()
for pair in country_pairs:
    sea_graph.add_edge(pair[0], pair[1])  
isolated_nodes = list(nx.isolates(sea_graph))
sea_graph.remove_nodes_from(isolated_nodes)
{% endhighlight %}
<p></p>
<p></p>
<p></p>
<h5>Combine Land and Sea Graphs </h5>
<p></p>
<p></p>
<p></p>
Add 'land' attribute to all land borders
<p></p>
{% highlight python %}
for u, v in land_graph.edges:
    land_graph[u][v]['type'] = 'land'
{% endhighlight %}
<p></p>

<p></p>
Add 'sea' attribute to all sea borders
<p></p>
{% highlight python %}
for u, v in sea_graph.edges:
    sea_graph[u][v]['type'] = 'sea'
{% endhighlight %}
<p></p>

<p></p>
Combine land and sea graphs
If an edge exists in both graphs, mark it as both land and sea
<p></p>
<p></p>
<p></p>
{% highlight python %}
combined_graph = nx.compose(land_graph, sea_graph)
for u, v in combined_graph.edges:
    if land_graph.has_edge(u, v) and sea_graph.has_edge(u, v):
        combined_graph[u][v]['type'] = 'both'  # Mark as both land and sea
{% endhighlight %}
<p></p>
<p></p>
<p></p>

<h5>World Bank life expectancy dataset </h5>

join it with the both_graph (representing countries as nodes), you can follow these steps

Step 1: Load and Inspect the Data
Load the CSV file to inspect its structure and contents.

<p></p>
<p></p>
Filter and Reshape the Data
Extract the relevant columns and reshape the data into a more usable format.
<p></p>
Keep relevant columns

Melt the dataframe to long format (year as a variable)

Convert year to integer
Drop rows with NaN values
 Preview the reshaped data
<p></p>
{% highlight python %}
life_expectancy_filtered = life_expectancy.drop(columns=["Indicator Name", "Indicator Code"])
life_expectancy_long = life_expectancy_filtered.melt(
    id_vars=["Country Name", "Country Code"],
    var_name="Year",
    value_name="Life Expectancy"
)
life_expectancy_long["Year"] = pd.to_numeric(life_expectancy_long["Year"], errors="coerce")
life_expectancy_long = life_expectancy_long.dropna()
{% endhighlight %}

<p></p>
To join the life expectancy data with the graph, map the Country Code or Country Name from the dataset to the nodes in your graph.
Get a list of nodes in the graph
Check if the graph uses country names or codes
Example: If the graph uses country names
<p></p>
<p></p>
{% highlight python %}
import networkx as nx
mapped_data = life_expectancy_long[life_expectancy_long["Country Name"].isin(nodes)]
{% endhighlight %}
<p></p>
<p></p>
<p></p>
<p></p>
Group life expectancy data by country for easy access

Add life expectancy as node features

Check a sample node's data
For missing data, assign an empty dictionary
<p></p>
<p></p>
<p></p>
{% highlight python %}
life_expectancy_dict = life_expectancy_long.groupby("Country Name") \
    .apply(lambda df: df.set_index("Year")["Life Expectancy"].to_dict()) \
    .to_dict()
for node in both_graph.nodes:
    if node in life_expectancy_dict:
        nx.set_node_attributes(both_graph, {node: life_expectancy_dict[node]}, name="life_expectancy")
    else:
        nx.set_node_attributes(both_graph, {node: {}}, name="life_expectancy")
{% endhighlight %}

<p></p>
<p></p>

<h4>Model Training</h4>
    <p>The model training phase
<p></p>


    <p></p>               



<p></p>

    <p></p>
<h4>Rewiring Knowledge Graph</h4>



    <p></p>

<p></p>  

<p></p>               


    <p></p>
Next,
    <p></p>               
{% highlight python %}
xxxx
{% endhighlight %}
    <p></p>


<p></p>

<p></p>






<p></p>








    <h3>In Conclusion</h3>
<p></p>
    In this study,
<p></p>
    Our work
<p></p>
    This research marks
<p></p>    

<p></p>
One of the challenges with using GNN for knowledge graph is that knowledge graphs consist of variety of different data domains and different formats of data values. In this study we introduce a method to deal with these challenges.  
<p></p>
We start with domain specific subgraphs and run GNN link prediction models on each subgraphs. The output of these models will be vectors of the same size.



<p></p>
We also can combine across domains different entities if connections exist in initial knowledge graph. For example, in one domain has artists as nodes, another domain paintings,  and another domain art museums than we can connect artists with paintings and pinting with museums that have that paintings.
<p></p>
<p></p>

<p></p>
<p></p>
