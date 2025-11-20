---
layout:     post
title:      "Temporal–Spatial GNN Fusion for Climate Analytics"
subtitle:   "Voronoi-Based City Networks Reveal Hidden Temperature Dynamics"
date:       2025-06-25 12:00:00
author:     "Melenar"
header-img: "img/pageVec19.jpg"
---

<h2>Spatio-Temporal Graphs for Climate Patterns</h2>
<p>
  Here we combine <strong>time series graphs</strong> with a global <strong>spatial graph</strong>
  of cities to create a spatio-temporal view of climate. Decades of daily temperatures are turned
  into graph-based climate fingerprints for each city, then injected into a Voronoi-based world
  graph where neighboring regions share borders. A GNN refines this fused graph so we can see
  which distant cities behave almost identically, which nearby cities quietly diverge, and which
  “bridge cities” connect climate regions. The same method can be applied to any setting where
  signals evolve over time and are anchored in space—sensors, assets, patients, or networks.
</p>


<h2>Conference</h2>
<p>
  The work <em>“GNN Fusion of Voronoi Spatial Graphs and City–Year Temporal Graphs for Climate Analysis”</em>
  was presented at <strong>SCAI 2025</strong> in Ljubljana, Slovenia, on <strong>8 October 2025</strong>.
  The corresponding paper is currently not yet published in archival proceedings.
</p>



<h2>Fusing Space &amp; Time with Graph Neural Networks</h2>
<p>
  Climate isn’t only about <em>where</em> you are; it’s about how conditions <em>evolve together</em> across the map.
  Here we use <strong>Graph Neural Networks (GNNs)</strong> to connect <strong>1,000 cities</strong> across
  <strong>40 years</strong> of daily temperatures, revealing hidden climate “neighborhoods” and long-range “bridges”
  that geography alone can’t explain. Modeling both space and time as graphs shows when distant regions behave like
  close neighbors—and when nearby cities drift apart.
</p>

<figure>
  <img src="{{ site.baseurl }}/img/voronoi34.jpg" alt="Four perspectives: Average, Spatial, Temporal, Spatial+Temporal">
  <figcaption>Figure 1. Four perspectives combining spatial and temporal graph inputs for climate analysis.</figcaption>
</figure>

<p>This diagram shows the four different ways we explore climate similarity, based on combinations of input data and graph structures. Here is color guide:
<ul>
  <li><strong>Yellow area</strong> — represents the use of <strong>raw data</strong>, specifically the <em>average temperature vectors</em> for each city.</li>
  <li><strong>Light green area</strong> — represents the use of <strong>temporal data</strong>, meaning the <em>city-level graphs</em> built from year-to-year climate patterns.</li>
  <li><strong>Purple area</strong> — represents the use of <strong>spatial data</strong>, specifically the <em>Voronoi graph</em> that defines proximity between cities.</li>
</ul>


<h3>Four perspectives we analyze</h3>
<ul>
  <li><strong>Average:</strong> a city’s typical year-round temperature profile</li>
  <li><strong>Spatial:</strong> proximity connections via a Voronoi graph</li>
  <li><strong>Temporal:</strong> how climate evolves across years within one city</li>
  <li><strong>Spatial + Temporal:</strong> fusion of both for the richest signal</li>
</ul>

<h3>Why this matters</h3>
<p>
  Most climate work emphasizes either geography or time series. This study fuses both:
  <strong>space</strong> (how cities influence one another through adjacency) and
  <strong>time</strong> (how each city’s climate changes year to year). The joint view exposes patterns a single axis
  misses—regional belts that shift together, regime matches across oceans, and nearby microclimates that diverge.
</p>

<h3>How the model learns similarity</h3>
<p>We build two complementary graph families and train GNNs with link prediction to learn embeddings:</p>
<ul>
  <li><strong>Spatial graphs:</strong> nodes are cities; edges come from Voronoi adjacency.</li>
  <li><strong>Temporal city-year graphs:</strong> each city becomes a chain of yearly nodes; edges link years with high cosine similarity of their daily temperature vectors.</li>
</ul>

<h3>Feature spaces produced</h3>
<ol>
  <li><strong>Average temperature vectors</strong> (365-dim per city)</li>
  <li><strong>City graph embeddings</strong> from GNN graph classification over city-year structures</li>
  <li><strong>Link-prediction embeddings (average vectors):</strong> predict spatial edges from average profiles</li>
  <li><strong>Link-prediction embeddings (city-graph vectors):</strong> same task using city-level GNN vectors, fusing spatial and temporal signals</li>
</ol>

<h3>Reusable blueprint</h3>
<p>
  Although the focus is temperature, this two-stream graph fusion is a reusable approach for evolving systems—mobility,
  pollution, public health, and more.
</p>



<p></p>




<p></p>
<h2>Transforming Data to Graphs</h2>
<p></p>
<p></p>
<h3>Raw Data</h3>
<p></p>

This study is based on a large and detailed dataset: 


<ol>
  <li>1,000 the most populous cities worldwide </li>
  <li>Geo-coordinates</li>
  <li>40 years of daily temperatures (1980–2019)</li>
</ol>

<p></p>
This data is giving us long-term, high-resolution climate information across the globe. It allows us to fuse geography (space) with decades of daily temperatures (time) and
reveal hidden climate neighborhoods & bridges.

<p></p>
<h3>Spatial Graph -- Voronoi</h3>
<p></p>
To model geography we use a Voronoi diagram. Think of several coffee shops in a town: each shop “owns” the area closer to it than to any other shop. We apply the same idea to cities. Each city becomes a node; two cities are connected if their Voronoi regions share a boundary. Some neighbors are nearby; others can be surprisingly far apart (see the Québec–Porto example).
<p></p>

<p></p>

<a href="#">
    <img src="{{ site.baseurl }}/img/voronoi41.jpg" alt="Post Sample Image" width="333" >
</a>
<p></p>

<ul>
  <li><strong>Coffee shops → areas:</strong> each site covers its closest region.</li>
  <li><strong>Neighbors →  edges:</strong> regions that share a border become connected.</li>
  <li><strong>Cities → nodes:</strong> cities sit at Voronoi centers; shared borders define graph edges.</li>

</ul>


<p></p>
We use the Voronoi diagram to define natural neighbors between cities: space is partitioned so every location belongs to its closest city. Cities are connected if their regions share a border. For example, Québec, Canada and Porto, Portugal are neighbors because no more-populated cities lie between them at that scale; the shared Voronoi boundary falls near the midpoint of the great-circle path.
   <a href="#">
       <img src="{{ site.baseurl }}/img/voronoi17.jpg" alt="Post Sample Image" width="777" >
   </a>

<p></p>
Because cities are on the curved Earth, we first project latitude/longitude into a planar coordinate system (EPSG:3857) so the Voronoi algorithm—which expects Cartesian coordinates—can run. Any equal, consistent projection is acceptable here; the goal is topological adjacency, not precise distances.
<p></p>
This produces a spatial graph over 1,000 cities in which each node carries a 365-value average-temperature vector (mean daily temperatures across 40 years). The result captures true proximity determined by spatial partitioning—not arbitrary distance thresholds—and highlights dense regional clusters as well as isolated cities.
<p></p>
A spatial graph that connects all 1,000 cities built using a Voronoi diagram, which links cities that are geographically close
Each city is a node, described by a 365-value vector (its average daily temperatures across 40 years)
This gives us a global view of how nearby climates relate

To understand the <strong>spatial structure</strong>, we used <strong>Voronoi diagrams</strong> to define natural neighborhoods. Cities are connected if their regions share a border, creating a network that reflects true proximity — not based on arbitrary distance cutoffs, but shaped by how space is divided. This helps capture how some cities are part of dense regional clusters, while others are more isolated.

<p></p>
<h3>Temporal City Graphs</h3>
<p></p>

<p></p>
We looked at the <strong>temporal behavior</strong> of climate in each city — how daily temperatures have changed (or remained consistent) across decades. Some locations show highly stable seasonal cycles, while others exhibit more variation year to year. We constructed temporal graphs for each city, with nodes representing city - year, with a 365-value vector of daily temperatures. Graph edges we defined by pairs of nodes with cosine similarities higher than threshold. Temporal graphs represent relationships between daily temperature vectors for different years.
For each graph we added a virtual node to transform disconnected graphs into single connected components.
<p></p>
<ul>
  <li><strong>Nodes</strong> = city-year; features = daily temperature</li>
  <li><strong>Edges</strong> = high cosine similarity (year↔year)</li>
  <li><strong>Virtual nodes</strong> to ensure connectivity</li>

</ul>

<p></p>



<a href="#">
    <img src="{{ site.baseurl }}/img/voronoi42.jpg" alt="Post Sample Image" width="555" >
</a>
<p></p>
You can see as examples city graph for Malaga, Spain with stable weather and highly similar yearly weather and Orenburg, Russia where daily temperature vectors are very different year by year.
<p></p>
<p></p>
<h2>GNN Models</h2>
<p></p>
We will use Graph Neural Networks (GNNs) to learn more from these graphs by turning each city (or each city-year) into a meaningful vector. In both cases, the GNN models help us represent each city as a learned vector, shaped by its spatial context or its climate history over time. These new vectors can then be used to compare cities, group them, or detect unusual patterns. From each model, we extract the pre-final output vectors — these are 128-dimensional embeddings that capture the learned information.

<p></p>
<h3>Temporal GNN - Graph Classification</h3>
<p></p>
To capture how each city’s climate has changed over time, we applied a GNN Graph Classification model to 1,000 city-level temporal graphs. The model learns from the structure and connections within each graph, summarizing a city’s long-term climate behavior — whether it’s stable, variable, or somewhere in between.
<p></p>
The model requires small graphs with labels. As a simple proxy, we labeled cities based on their distance from the equator, using latitude as an indicator of climate variability. The model outputs a 128-dimensional vector that represents each city’s climate history. Future work could refine these labels with more detailed, data-driven methods.
<p></p>

<p></p>

<p></p>
For classifying city graphs, we used the Graph Convolutional Network (GCNConv) model from the PyTorch Geometric Library (PyG). The GCNConv model allowed us to extract feature vectors from the graph data, enabling us to perform a binary classification to determine whether the climate for each city was 'stable' or 'unstable'.
<p></p>

<p></p>

<p></p>
<p></p>
<h3>Spatial GNN - Link Prediction</h3>
<p></p>
For the spatial analysis, we build a Voronoi-based graph that links all 1 000 cities worldwide and train a GNN link-prediction model on it. Each node is initialized with a 365-dimensional “average-temperature” vector, enabling the network to learn how climate patterns propagate among neighboring cities. The model then produces a fresh embedding for every city—one that blends its own climate signature with the influence of its Voronoi neighbors.
<p></p>

<p></p>
For spatial graphs we used GNN Link Prediction model based on GraphSAGE algorithm, which generates node embeddings based on attributes and neighbors without retraining. Our study employs a GNN Link Prediction model from the Deep Graph Library (DGL) library.

<p></p>
<h3>Joint Spatial and Temporal Modeling</h3>
<p></p>
To capture both temporal and spatial influences on climate, we apply a GNN Link Prediction model. The graph structure comes from Voronoi-based city connections, while the node features are the learned embeddings from the city-level temporal graphs produced by the GNN Classification model. This allows us to explore how geography and long-term climate behavior interact together.

<p></p>
<h2>Methods</h2>

<p></p>
The diagram below shows how we built a climate similarity model using graph neural networks. First, we connected the 1000 most populated cities using Voronoi-based geography — cities are linked if their zones share a border. Then, we used 40 years of temperature data to describe each city in two ways: one based on raw daily averages, and one using advanced GNN models that learn from how each city's climate changed over time. These feature vectors help us compare cities and uncover deep climate patterns around the world.




   <a href="#">
       <img src="{{ site.baseurl }}/img/voronoi16b.jpg" alt="Post Sample Image" width="404" >
   </a>
Fig. 1. Overview of the proposed method combining Voronoi-based spatial graphs with GNN pipelines for climate similarity and classification.
<p></p>


<p></p>
<p></p>






<h2>Coding and Observations</h2>


<p></p>
<h3>Data Source: Climate Data</h3>
<p></p>
Our primary dataset, sourced from Kaggle, is titled:
<i><a href="
https://www.kaggle.com/hansukyang/temperature-history-of-1000-cities-1980-to-2020">"Temperature History of 1000 cities 1980 to 2020"</a></i> - daily temperature from 1980 to 2020 years for 1000 most populous cities in the world. This dataset provides a comprehensive record of average daily temperatures in Celsius for the 1000 most populous cities worldwide, spanning from 1980 to 2019.
<p></p>



<p></p>

   <a href="#">
       <img src="{{ site.baseurl }}/img/preFinFig1.jpg" alt="Post Sample Image" width="678" >
   </a>
Fig. 1. Latitude Distribution of the 1000 Most Populous Cities.
<p></p>  

To begin our climate analysis, we created a simple but effective climate profile for each city. The dataset includes daily temperature readings for 1000 cities across multiple years. By averaging the temperatures for each day of the year across all available years, we produced a single 365-dimensional vector per city.

This average vector captures the city’s typical annual temperature pattern and serves as a foundational node feature for later graph-based models.
</p>

<p></p>  

  <p></p>

  <p></p>
  {% highlight python %}
  df=rawData
  daily_cols = list(map(str, range(365)))
  city_avg_vectors = df.groupby('cityInd')[daily_cols].mean().reset_index()
  city_avg_vectors.shape
  (1000, 366)
  {% endhighlight %}
  <p></p>
<p></p>

<p></p>

<h3>Voronoi Graph Construction</h3>

<p>
To capture spatial relationships between cities, we built a <strong>Voronoi diagram</strong>, which naturally defines neighboring regions based on geographic proximity.
</p>

<p>
First, we projected the latitude and longitude of each city into a flat 2D coordinate system using the <strong>EPSG:3857 map projection</strong>, as required by the Voronoi algorithm.
</p>

<pre><code>
from pyproj import Transformer
from scipy.spatial import Voronoi
import numpy as np

# Project geographic coordinates to 2D plane
transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
projected = np.array([
    transformer.transform(lon, lat)
    for lon, lat in zip(cityData['lng'], cityData['lat'])
])
</code></pre>

<p>
We then computed the Voronoi diagram using SciPy:
</p>

<pre><code>
# Compute Voronoi diagram
vor = Voronoi(projected)
</code></pre>

<p>
The Voronoi output contains pairs of neighboring points via <code>vor.ridge_points</code>, which lists index pairs for cities whose regions share a border. We converted these indices into our unique city identifiers (<code>cityInd</code>) and created a DataFrame representing the edge list of our spatial graph:
</p>

<pre><code>
# Extract unique neighbor pairs
neighbors = set(tuple(sorted((p1, p2))) for p1, p2 in vor.ridge_points)

# Build edge list DataFrame
rows = []
for i, j in neighbors:
    rows.append({
        'city1': cityData.iloc[i]['cityInd'],
        'city2': cityData.iloc[j]['cityInd'],
    })
voronoi_df = pd.DataFrame(rows)
</code></pre>

<p><strong>Example of Voronoi-based neighboring city pairs:</strong></p>




<table class="compact" style="width:auto;">
  <colgroup>
    <col style="width:60px;">
    <col style="width:60px;">
  </colgroup>
  <thead>
    <tr><th>City_1</th><th>City_2</th></tr>
  </thead>
  <tbody>
    <tr><td>155</td><td>810</td></tr>
    <tr><td>60</td><td>801</td></tr>
    <tr><td>40</td><td>185</td></tr>
    <tr><td>874</td><td>905</td></tr>
    <tr><td>686</td><td>705</td></tr>
  </tbody>
</table>




<p>
This edge list defines the connections in our spatial graph, showing which cities are considered neighbors based on shared Voronoi borders.
</p>

<h4>Calculating Distances Between Neighboring Cities</h4>

<p>
To measure distances between neighboring cities, we used the same 2D projected coordinates:
</p>

<pre><code>
rows = []
for i, j in neighbors:
    dist_km = round(np.linalg.norm(projected[i] - projected[j]) / 1000, 5)
    rows.append({
        'city1': cityData.iloc[i]['cityInd'],
        'city2': cityData.iloc[j]['cityInd'],
        'distance_km': dist_km
    })
voronoi_distances_df = pd.DataFrame(rows)
</code></pre>

<p>
To make the results more readable, we combined city names and countries:
</p>

<pre><code>
# Create readable city labels
cityData['city_country'] = cityData['city_ascii'] + ', ' + cityData['country']
city_lookup = cityData.set_index('cityInd')['city_country']

# Add city names to distances DataFrame
voronoi_distances_df['city1_name'] = voronoi_distances_df['city1'].map(city_lookup)
voronoi_distances_df['city2_name'] = voronoi_distances_df['city2'].map(city_lookup)
</code></pre>



<p></p>
Statistics on distance Between Neighboring Cities (in kilometers)
<pre>
count     2983.00  
mean       638.67  
std       1170.90  
min          2.27  
25%        176.22  
50%        340.92  
75%        658.97  
max      25870.97
</pre>
<p></p>


<p>
This structure provides both the graph topology (neighbors) and distance information, forming the basis for spatial climate analysis using graph models.
</p>


<p></p>

<p></p>


<p></p>

<p></p>

<p></p>
<h3>Temporal GNN: Graph Classification</h3>
<p></p>
For our analysis, each city was modeled as a graph, with nodes representing specific \{city, year\} pairs. These nodes encapsulate a full year of daily temperature values, allowing us to examine long-term temporal trends across time. To enable classification, each city graph was labeled as either stable or unstable, based on its geographic latitude. The assumption here is that cities located closer to the equator tend to have more stable climate patterns, with less seasonal fluctuation, while those farther from the equator generally experience greater variability.
<p></p>
We divided the cities into two groups using their latitude values---one closer to the equator, the other at higher latitudes---creating a binary classification task for our Graph Neural Network (GNN) model. The bar chart below shows the latitude distribution of all 1000 cities, highlighting a dense cluster between 20\textdegree{} and 60\textdegree{} in the Northern Hemisphere and a sparser spread in the Southern Hemisphere. The equator is marked by a dashed line for reference.

<p></p>
<h4>Input Graph Data Preparation</h4>
<p></p>

Before training our GNN for classification, we need to label each city graph as either <strong>stable</strong> or <strong>unstable</strong> in terms of climate. To do this, we sort all 1000 cities by their absolute latitude — under the assumption that cities closer to the equator (low latitude) tend to have more stable temperature patterns over time.
<p></p>
We assign a label of <code>0</code> to the 500 cities nearest the equator and a label of <code>1</code> to the 500 cities farther away. These labels serve as ground truth for training the graph classification model.
<p></p>
Here’s the code used to sort the data and assign the classification labels:

<p></p>
<p></p>
{% highlight python %}
df_sorted = df.loc[df['lat'].abs().sort_values().index]
df_sorted['label'] = [0 if i < 500 else 1 for i in range(1000)]
df_sorted.reset_index(drop=True, inplace=True)
df_sorted['labelIndex'] = df_sorted.index
cityLabels = df_sorted.sort_values('cityInd', ascending=True)
cityLabels.reset_index(drop=True, inplace=True)
min_lat = cityLabels['lat'].min()
max_lat = cityLabels['lat'].max()
print("Minimum Latitude:", min_lat)
print("Maximum Latitude:", max_lat)
Minimum Latitude: -41.3
Maximum Latitude: 64.15
{% endhighlight %}

<p></p>

After assigning stability labels to cities, we merge this information with the original temperature dataset. Each city-year pair includes a daily temperature vector (365 values), and we focus on the year 1980 as the representative graph structure for every city.
<p></p>
We also extract important metadata — including city name, coordinates, and region — to keep track of each graph's identity during analysis.

<p></p>
{% highlight python %}
subData=rawData.merge(lineScore,on='cityInd',how='inner')
subData.reset_index(drop=True, inplace=True)
values1=subData.iloc[:, 9:374]
metaGroups=subData[(subData['nextYear']==1980)].iloc[:,[1,2,3,4,7,374]]
metaGroups.reset_index(drop=True, inplace=True)
metaGroups['index']=metaGroups.index
{% endhighlight %}
<p></p>

<p>
To build graphs for each city, we first define a cosine similarity threshold. We will use this threshold to determine which years within a city are connected based on the similarity of their temperature profiles.
</p>

<p>
For example, if two years have a cosine similarity greater than <code>0.925</code>, we connect them with an edge in that city’s graph. This approach helps us capture internal climate consistency and variability over time.
</p>


<p></p>
<p></p>
{% highlight python %}
from torch_geometric.loader import DataLoader
import random
datasetTest=list()
cosList = [0.925]
{% endhighlight %}
<p></p>



<p></p>
<p></p>
{% highlight python %}
from torch_geometric.loader import DataLoader
datasetTest=list()
datasetModel=list()
import random
for cos in cosList:
  for g in range(1000):
    cityName=metaGroups.iloc[g]['city_ascii']
    country=metaGroups.iloc[g]['country']
    label=metaGroups.iloc[g]['label']
    cityInd=metaGroups.iloc[g]['cityInd']
    data1=subData[(subData['cityInd']==cityInd)]
    values1=data1.iloc[:, 9:374]
    fXValues1= values1.fillna(0).values.astype(float)
    fXValuesPT1=torch.from_numpy(fXValues1)
    fXValuesPT1avg=torch.mean(fXValuesPT1,dim=0).view(1,-1)
    fXValuesPT1union=torch.cat((fXValuesPT1,fXValuesPT1avg),dim=0)
    cosine_scores1 = pytorch_cos_sim(fXValuesPT1, fXValuesPT1)
    cosPairs1=[]
    score0=cosine_scores1[0][0].detach().numpy()
    for i in range(40):
      year1=data1.iloc[i]['nextYear']
      cosPairs1.append({'cos':score0, 'cityName':cityName, 'country':country,'label':label,
          'k1':i, 'k2':40, 'year1':year1, 'year2':'XXX',
          'score': score0})
      for j in range(40):
        if i<j:
          score=cosine_scores1[i][j].detach().numpy()
          # print(cos)
          if score>cos:
            year2=data1.iloc[j]['nextYear']
            cosPairs1.append({'cos':cos, 'cityName':cityName, 'country':country,'label':label,
                'k1':i, 'k2':j, 'year1':year1, 'year2':year2,
                'score': score})
    dfCosPairs1=pd.DataFrame(cosPairs1)
    edge1=torch.tensor(dfCosPairs1[['k1',	'k2']].T.values)
    dataset1 = Data(edge_index=edge1)
    dataset1.y=torch.tensor([label])
    dataset1.x=fXValuesPT1union
    datasetTest.append(dataset1)
    if label>=0:
      datasetModel.append(dataset1)
    loader = DataLoader(datasetTest, batch_size=32)
    loader = DataLoader(datasetModel, batch_size=32)
{% endhighlight %}
<p></p>

<p></p>
<h4>GNN Graph Classification Model Training</h4>
<p></p>
<p>
Once our graph dataset is ready, we divide it into training and testing splits — using 888 city graphs for training and the remaining 112 for testing. Each graph represents one city, with nodes representing years and features capturing daily temperature patterns.
</p>

<p>
We use PyTorch Geometric’s <code>DataLoader</code> to batch graphs efficiently and iterate through them during training. Below, we also define a 3-layer Graph Convolutional Network (GCN) with a global pooling layer that summarizes each graph into a single embedding.
</p>

<p>
The final layer outputs a prediction for each graph: whether it represents a <strong>stable</strong> or <strong>unstable</strong> climate pattern.
</p>
<p></p>
{% highlight python %}
torch.manual_seed(12345)
train_dataset =  dataset[:888]
test_dataset = dataset[888:]
rom torch_geometric.loader import DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(365, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)
    def forward(self, x, edge_index, batch, return_graph_embedding=False):
        # Node Embedding Steps
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        graph_embedding = global_mean_pool(x, batch)  
        if return_graph_embedding:
            return graph_embedding  
        x = F.dropout(graph_embedding, p=0.5, training=self.training)
        x = self.lin(x)
        return x
model = GCN(hidden_channels=128)
{% endhighlight %}
<p></p>

With the model and data loaders set up, we now train our Graph Neural Network (GCN) using a standard cross-entropy loss function. We optimize using Adam and evaluate the model's accuracy on both training and test sets after each epoch.
<p></p>
The training loop runs for 76 epochs, showing how well the model is learning to classify cities based on their climate patterns over time.

<p></p>
{% highlight python %}
from IPython.display import Javascript
model = GCN(hidden_channels=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
def train():
    model.train()
    for data in train_loader:
         out = model(data.x.float(), data.edge_index, data.batch)  
         loss = criterion(out, data.y)  
         loss.backward()  
         optimizer.step()
         optimizer.zero_grad()  
def test(loader):
     model.eval()
     correct = 0
     for data in loader:
         out = model(data.x.float(), data.edge_index, data.batch)
         pred = out.argmax(dim=1)
         correct += int((pred == data.y).sum())
     return correct / len(loader.dataset)  
for epoch in range(1, 77):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
{% endhighlight %}
<p></p>

<pre>
Epoch: 061, Train Acc: 0.9110, Test Acc: 0.8750
Epoch: 062, Train Acc: 0.9392, Test Acc: 0.9107
Epoch: 063, Train Acc: 0.9291, Test Acc: 0.8839
Epoch: 064, Train Acc: 0.9110, Test Acc: 0.9196
Epoch: 065, Train Acc: 0.9471, Test Acc: 0.9107
Epoch: 066, Train Acc: 0.9448, Test Acc: 0.9286
Epoch: 067, Train Acc: 0.9279, Test Acc: 0.8929
Epoch: 068, Train Acc: 0.9392, Test Acc: 0.9107
Epoch: 069, Train Acc: 0.9032, Test Acc: 0.8661
Epoch: 070, Train Acc: 0.9414, Test Acc: 0.9286
Epoch: 071, Train Acc: 0.9426, Test Acc: 0.9018
Epoch: 072, Train Acc: 0.9448, Test Acc: 0.9196
Epoch: 073, Train Acc: 0.9448, Test Acc: 0.9107
Epoch: 074, Train Acc: 0.9448, Test Acc: 0.9107
Epoch: 075, Train Acc: 0.9448, Test Acc: 0.9196
Epoch: 076, Train Acc: 0.9414, Test Acc: 0.9196
</pre>

<p></p>
{% highlight python %}
Epoch: 076, Train Acc: 0.9414, Test Acc: 0.9196
{% endhighlight %}

<p></p>
<h4>GNN Graph Classification Model Results</h4>
<p></p>

Once the model is trained, we can use it to extract vector representations (embeddings) for each city graph. These embeddings capture structural and feature-based patterns learned during training — essentially summarizing each city’s climate behavior over time.
<p></p>
Below, we retrieve the graph embedding for the first city in our dataset. The model outputs a 128-dimensional vector that can later be used for clustering, similarity analysis, or further graph-based tasks.
<p></p>
{% highlight python %}
g=0
out = model(dataset[g].x.float(), dataset[g].edge_index, dataset[g].batch, return_graph_embedding=True)
out.shape
torch.Size([1, 128])
{% endhighlight %}
<p></p>
<p>
After training our GNN classifier, we loop through all 1000 city graphs and extract their 128-dimensional embeddings using the model’s <code>return_graph_embedding=True</code> mode. These embeddings capture the climate structure of each city graph and can be used for downstream tasks such as clustering, similarity analysis, or building meta-graphs.
</p>

<p>
We collect these vectors into a unified DataFrame called <code>city_graph_vectors</code>, where each row corresponds to a single city (indexed by <code>cityInd</code>) and each column holds part of its graph embedding.
</p>
{% highlight python %}
softmax = torch.nn.Softmax(dim = 1)
graphUnion=[]
for g in range(graphCount):
  label=dataset[g].y[0].detach().numpy()
  out = model(dataset[g].x.float(), dataset[g].edge_index, dataset[g].batch, return_graph_embedding=True)
  output = softmax(out)[0].detach().numpy()
  pred = out.argmax(dim=1).detach().numpy()
  graphUnion.append({'index':g,'vector': out.detach().numpy()})
df=graphUnion_df
df['vector'] = df['vector'].apply(lambda x: np.array(x).flatten())
city_graph_vectors = pd.DataFrame(df['vector'].to_list())
city_graph_vectors.insert(0, 'cityInd', df['index'])
city_graph_vectors.columns = ['cityInd'] + list(range(city_graph_vectors.shape[1] - 1))
{% endhighlight %}
<p></p>


<p></p>

<h3>Spatial GNN: Link Prediction</h3>
<p></p>
We used the GNN Link Prediction model from DGL library for Voronoi graph with average vectors and Voronoi graph with city-graph embedding vectors.  
<p></p>
We build a DGL graph where each node represents a city and edges connect Voronoi neighbors.
Node features are provided in two forms:
<p></p>

<ul>
  <li><strong>Raw vectors:</strong> 365-dimensional averages of daily temperature over 40 years.</li>
  <li><strong>Embedded vectors:</strong> Learned representations from city-level temporal graphs via a GNN classification model.</li>
</ul>

<p>Graph construction steps (same for both feature types):</p>

<ol>
  <li>Load vector data and remove non-feature columns (e.g., <code>cityInd</code>).</li>
  <li>Convert Voronoi-based city pairs to PyTorch edge tensors.</li>
  <li>Create the DGL graph with nodes and Voronoi edges.</li>
  <li>Assign raw or embedded vectors as node features.</li>
</ol>


<p></p>
{% highlight python %}
avg_vectors=pd.read_csv(theAvgPath)
import torch
import dgl
features = torch.tensor(avg_vectors.drop(columns='cityInd').values, dtype=torch.float32)
src_nodes = torch.tensor(edges['city1'].values, dtype=torch.int64)
dst_nodes = torch.tensor(edges['city2'].values, dtype=torch.int64)
g = dgl.graph((src_nodes, dst_nodes), num_nodes=features.shape[0])
g.ndata['feat'] = features
g
Graph(num_nodes=1000, num_edges=2983,
      ndata_schemes={'feat': Scheme(shape=(365,), dtype=torch.float32)}
      edata_schemes={})
{% endhighlight %}

<p></p>
<p></p>

<h4>Splitting Edges for Training and Testing</h4>

<p>
For link prediction, we need both real (positive) and fake (negative) edges.
We split Voronoi edges into training and test sets, then sample an equal number
of non-connected city pairs as negatives.
</p>

<ol>
  <li>Extract and shuffle city-to-city edges; convert to PyTorch tensors.</li>
  <li>Split 90% for training, 10% for testing.</li>
  <li>Build a sparse adjacency matrix to detect missing (negative) edges.</li>
  <li>Sample negative edges with no connection or self-loop, matching positive count.</li>
</ol>


<p></p>
{% highlight python %}
import scipy.sparse as sp
import numpy as np
import torch
eids = np.random.permutation(g.number_of_edges())
eids = torch.tensor(eids, dtype=torch.int64)  
test_size = int(len(eids) * 0.1)
train_size = len(eids) - test_size
test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]
adj = sp.coo_matrix((np.ones(len(u)), (u.tolist(), v.tolist())),
      shape=(g.number_of_nodes(), g.number_of_nodes()))
adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
neg_u, neg_v = np.where(adj_neg != 0)
neg_eids = np.random.choice(len(neg_u), g.number_of_edges(), replace=False)
test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
train_g = dgl.remove_edges(g, eids[:test_size])
{% endhighlight %}


<p></p>
<p></p>
<h4>Building a GraphSAGE Model</h4>

<p>
To learn meaningful representations of each city in our graph, we use a two-layer GraphSAGE model. GraphSAGE (Graph Sample and Aggregate) is a popular Graph Neural Network architecture that generates node embeddings by aggregating information from a node’s neighbors.
</p>

<p>
In our model, each layer applies a <code>mean</code> aggregator to combine neighbor features and passes the result through a ReLU activation. The second layer refines the hidden representation.
</p>

<p>
Here’s the code that defines the GraphSAGE model using DGL’s built-in <code>SAGEConv</code> layer:
</p>
<p></p>
{% highlight python %}
from dgl.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
{% endhighlight %}

<p></p>

<p></p>

<p>


<p>
<h4>Link Prediction: Dot Product and MLP</h4>

<p>
To predict whether two cities should be connected, we use link prediction methods that score the similarity between node embeddings.
</p>

<p>
<code>DotPredictor</code> uses a simple dot product to measure alignment between nodes, while <code>MLPPredictor</code> applies a small neural network to learn more flexible scoring patterns.
</p>

</p>
</p>

<p>

</p>
<p></p>
{% highlight python %}
import dgl.function as fn
class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return g.edata['score'][:, 0]
class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)
    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']
{% endhighlight %}
<p></p>


<p></p>
<h4>Model Setup and Evaluation</h4>

<p>
We define the full training pipeline using our GraphSAGE model and a predictor. You can easily switch between <code>DotPredictor</code> and <code>MLPPredictor</code> by updating one line.
</p>

<p>
The <code>compute_loss</code> function uses binary cross-entropy to learn from both positive and negative edges. We also use the AUC (Area Under the Curve) score to evaluate how well the model distinguishes between real and false edges.
</p>


<p></p>

<p></p>
{% highlight python %}
model = GraphSAGE(train_g.ndata['feat'].shape[1], 128)
pred = DotPredictor()
def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)
def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)
{% endhighlight %}
<p></p>

<p></p>
<h4>Training the Model</h4>

<p>
We optimize both the GraphSAGE encoder and the link predictor using the Adam optimizer. During each training epoch, the model generates node embeddings, computes scores for positive and negative edges, and updates its weights using binary cross-entropy loss.
</p>

<p>
The model is trained for 4000 epochs, with periodic logging of the training loss.
</p>
<p></p>
<p></p>
{% highlight python %}
optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=1e-4 )
all_logits = []
for e in range(4000):
    h = model(train_g, train_g.ndata['feat'])
    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    loss = compute_loss(pos_score, neg_score)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if e % 200 == 0:
        print('In epoch {}, loss: {}'.format(e, loss))
{% endhighlight %}

<p></p>
Average raw vectors:
<p></p>
<pre>
In epoch 0, loss: 204028.6875
In epoch 200, loss: 1625.9397
In epoch 400, loss: 558.7700
In epoch 600, loss: 291.7234
In epoch 800, loss: 179.5440
In epoch 1000, loss: 120.3145
In epoch 1200, loss: 84.1568
In epoch 1400, loss: 60.9932
In epoch 1600, loss: 44.9907
In epoch 1800, loss: 34.1410
In epoch 2000, loss: 26.5716
In epoch 2200, loss: 21.0573
In epoch 2400, loss: 16.8369
In epoch 2600, loss: 13.7170
In epoch 2800, loss: 11.2253
In epoch 3000, loss: 9.2902
In epoch 3200, loss: 7.8638
In epoch 3400, loss: 6.7238
In epoch 3600, loss: 5.7895
In epoch 3800, loss: 4.9161
</pre>
<p></p>
<p></p>
City Graph embedded vectors:
<p></p>
<pre>
In epoch   0, loss: 9616.3594
In epoch 200, loss:  306.7737
In epoch 400, loss:  121.4521
In epoch 600, loss:   62.3646
In epoch 800, loss:   35.9576
In epoch 1000, loss:  23.0823
In epoch 1200, loss:  16.0522
In epoch 1400, loss:  11.7496
In epoch 1600, loss:   8.9312
In epoch 1800, loss:   7.0417
In epoch 2000, loss:   5.7560
In epoch 2200, loss:   4.7962
In epoch 2400, loss:   4.0738
In epoch 2600, loss:   3.5095
In epoch 2800, loss:   3.0330
In epoch 3000, loss:   2.6442
In epoch 3200, loss:   2.3126
In epoch 3400, loss:   2.0293
In epoch 3600, loss:   1.7847
In epoch 3800, loss:   1.6135
</pre>

<p></p>
<p></p>

<h4>Evaluating the Model</h4>

<p>
To assess performance, we evaluated the trained link prediction model on a held-out test set using the
AUC (Area Under the ROC Curve) metric, which measures the model’s ability to distinguish between actual
and non-existent links — higher values indicate better predictive accuracy.
</p>

<p>The evaluation was performed using two types of node features:</p>

<ul>
  <li>Average daily temperature vectors: AUC 0.823015</li>
  <li>City graph embedded vectors: AUC 0.808995</li>
</ul>

<p>The AUC was computed using the following code:</p>
<p></p>
<p></p>
{% highlight python %}
from sklearn.metrics import roc_auc_score
with torch.no_grad():
    pos_score = pred(test_pos_g, h)
    neg_score = pred(test_neg_g, h)
    print('AUC', compute_auc(pos_score, neg_score))

{% endhighlight %}
<p></p>
<p></p>
<h4>Extracting Node Embeddings from GNN Link Prediction</h4>

<p>
Once the GNN Link Prediction model has been trained, we can extract the learned node embeddings — 128-dimensional vectors that capture both climate and geographic context. These embeddings represent how each city relates to its Voronoi-based neighbors in terms of temperature trends and spatial structure.
</p>

<p>
To make the embeddings easier to analyze and merge with other datasets, we convert them into a DataFrame format and assign each row its corresponding cityInd.
</p>
<p></p>

{% highlight python %}
import pandas as pd
import torch
h_numpy = h.detach().numpy()  
embedding_table = pd.DataFrame(h_numpy)
embedding_table['cityInd'] = embedding_table.index  
{% endhighlight %}
<p></p>

<p></p>

<h2>Findings</h2>

<p></p>

<h3>Voronoi Graph Observations</h3>

<p></p>

<h4>The Biggest Triangle in the Voronoi Graph</h4>

<p>
This map shows the largest "triangle" formed by neighboring cities in our Voronoi graph.
</p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/voronoi35.jpg" alt="Post Sample Image" width="876" >
</a>
<p></p>
<p>
<strong>Wellington, New Zealand</strong> is one corner of this triangle. Its two nearest neighbors aren’t even nearby — they’re across the ocean in <strong>Port Elizabeth, South Africa</strong> and <strong>Mar del Plata, Argentina</strong>.
</p>

<p>
This triangle highlights how the Voronoi graph connects cities based on available neighbors, not strict distance. In isolated parts of the world, cities can be far apart yet still linked — an effect that’s especially important when exploring global climate patterns.
</p>

<h4>How Cities Are Distributed in the Voronoi Graph</h4>

<p>
Through areas of Voronoi diagrams we can see how the world’s 1,000 most populated cities are distributed — and how that shapes our Voronoi graph. The map shows each city as a colored dot, where the color reflects how much space that city covers in the graph. This gives a picture of where cities are packed close together and where they’re more isolated.
<p></p>

</p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/voronoi45.jpg" alt="Post Sample Image" width="876" >
</a>
<p></p>

<p></p>


<ul>
  <li><strong>Green = </strong>small cells (dense).</li>
  <li><strong>Yellow = </strong>medium.</li>
  <li><strong>Red = </strong>large (sparse).</li>
</ul>

<p>
</p>
You can see on the map that countries with the highest population —India and China —have large green areas. In Europe it’s interesting to compare Germany and France: similar title population and similar territories, but population distribution by cities are very different.
<p></p>


Populated-city distribution may reflect geography and climate.
<p></p>

<h3>Nearby vs. Distant Similarities</h3>


<p></p>
The spatial Voronoi graph and the four types of vectors create many opportunities to explore climate patterns across cities. We can look for similarities, differences, clusters, or unexpected relationships. But including all these analyses in one blog post would make it too long and hard to follow.
<p></p>
In this post, we’ll share just a few simple examples to show what’s possible. More detailed results and additional patterns will be covered in separate posts.
<p></p>

The combination of the <strong>spatial Voronoi graph</strong> and the four types of vectors opens up many ways to explore global climate patterns. There’s an almost endless list of questions we can ask — from obvious ones like which cities have similar climate profiles, to more complex ideas like how geographic structure interacts with long-term climate change.
<p></p>
<p></p>

<h4>Distant but similar: Wellington ↔ Mar del Plata (regime match) </h4>

<p></p>
The tables below highlight two interesting groups of city pairs based on both their physical distance and climate similarity across different types of vectors.

<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/voronoi43.jpg" alt="Post Sample Image" width="800" >
</a>

<p></p>
Despite being thousands of kilometers apart, these cities show high climate similarity — especially when considering temporal patterns and spatial relationships together. This suggests that geographic distance alone doesn't tell the full story when it comes to climate behavior.
<p></p>
<h4>Nearby but different: New York ↔ Brooklyn (microclimate)</h4>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/voronoi44.jpg" alt="Post Sample Image" width="800" >
</a>

<p></p>
In most cases, neighboring cities naturally share highly similar climates. But there are exceptions — like New York and Brooklyn, where spatial and combined similarity scores are unexpectedly low. This reflects how factors like microclimates, urban environments, or modeling limitations can produce unexpected results, even at small distances.
<p></p>


<p></p>  

<h3>New Graph Metrics: Closeness and Betweenness</h3>

Throughout, climate similarity means cosine similarity between the indicated vectors; for path-based metrics we use edge weights 𝑤 = 1 − cosine. Each set of maps uses the same spatial backbone: edges come from the Voronoi graph, where two cities are adjacent if their cells share a border.

<p></p>
What changes across panels is the edge weight, derived from cosine similarity computed from one of four representations (Average, Temporal, Spatial, Spatial+Temporal), with vectors normalized prior to cosine. The topology stays fixed; the weights—and therefore any shortest- path–based measures—change with the chosen vectors. Smaller weights mean higher climate similarity.
<p></p>

<p></p>
<h4>Closeness Centrality</h4>
<p></p>



<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/centrality_all.jpg" alt="Post Sample Image" width="800" >
</a>
<p></p>
In the closeness centrality maps, cities with high closeness are, on average, at short weighted distance from many others—i.e., they are similar to many cities. Dense climate regions such as Europe and East Asia typically stand out. Differences between panels reveal how each representation defines “similar,” shifting which cities appear most central.

<p></p>
<p></p>
<h4>Betweenness Centrality</h4>
<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/Betweenness_all.jpg" alt="Post Sample Image" width="800" >
</a>

<p></p>

In the betweenness maps, different weightings emphasize different connectors: high-betweenness cities lie on many shortest routes. The Spatial+Temporal view surfaces more long- range intermediaries than Average (notably in Africa, South America, and the Pacific). We also observe slight polarization in Spatial and Spatial+Temporal; the reason for this requires further research.


<p></p>


<h2>To Be Continued...</h2>

<p>
In future posts, we’ll dig deeper into the questions this approach opens up, like:
</p>

<ul>
  <li>How are cities clustered or isolated in the Voronoi graph — and how does that look across different parts of the world?</li>
  <li>How might those spatial patterns relate to climate stability or change?</li>
  <li>How do different types of vectors — from simple temperature averages to learned embeddings — reveal new climate connections?</li>
  <li>Where do geography and climate patterns align — and where do they tell different stories?</li>
</ul>

<p>
There’s a lot more to explore. These first examples are just a glimpse of how spatial graphs and climate data together can uncover hidden patterns.
</p>


<p></p>

<h3>Broader Impact</h3>


<p></p>
<ul>
  <li><strong>New perspective:</strong> Graph-based AI reveals hidden relationships across space and time — beyond what traditional text or image analyses capture.</li>
  <li><strong>Transferable method:</strong> The same fusion pipeline applies to cities, mobility, environmental systems, and public health.</li>
  <li><strong>Next steps — vectors:</strong> We showed only a few examples; many more applications and vector analyses remain to explore.</li>
  <li><strong>Next steps — healthcare:</strong> Extend the approach to disease-spread modeling and hospital-capacity planning.</li>
</ul>

<p></p>    






<p></p>

<p></p>








<p></p>



<p></p>

<p></p>

<p></p>
<p></p>
