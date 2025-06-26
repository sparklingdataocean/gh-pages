---
layout:     post
title:      "Voronoi-Based Spatial Graph Construction for GNN Models"
subtitle:   "Enhancing Climate Data Analysis through Spatial and Temporal Graphs"
date:       2025-10-04 12:00:00
author:     "Melenar"
header-img: "img/pageVec19.jpg"
---

<p></p>
<p><h2> Introduction</h2>
<p></p>
What if we could see the world’s cities not just as dots on a map, but as regions shaped by how close they are to one another? Using a Voronoi diagram, we’ve done just that—turning global population hubs into a network of neighboring zones. Each city becomes a node, connected to others that share a geographical boundary. Then, we add another layer: temperature. By combining 40 years of daily temperature data with this spatial network, we can begin to uncover how climate behaves not just in isolation, but in relation to neighboring regions. This approach opens a new lens into understanding how geography and climate interact—and what those patterns might tell us about the changing world.
<p></p>
"img/pageVec19.jpg"
"img/page1c.jpg"
"img/pageGnnClimate6.jpg"
<p></p>
"img/pageVec53.jpg"
<p></p>

<p></p>

<p></p>



In this post, we
<p></p>

<p></p>


<p></p>


<p></p>

<p></p>








<p></p>
<h2>Introduction</h2>

<p>
This project explores how <strong>daily temperature behaves across 1,000 of the world’s most populated cities</strong>, using <strong>40 years of historical data</strong>. Rather than looking at each city in isolation, we focus on how climate behaves across geography and over time — and how those patterns can be studied together.
</p>

<p>
To understand the <strong>spatial structure</strong>, we used <strong>Voronoi diagrams</strong> to define natural neighborhoods. Cities are connected if their regions share a border, creating a network that reflects true proximity — not based on arbitrary distance cutoffs, but shaped by how space is divided. This helps capture how some cities are part of dense regional clusters, while others are more isolated.
</p>

<p>
We also looked at the <strong>temporal behavior</strong> of climate in each city — how daily temperatures have changed (or remained consistent) across decades. Some locations show highly stable seasonal cycles, while others exhibit more variation year to year.
</p>

<p>
To explore these patterns in more depth, we applied <strong>Graph Neural Networks (GNNs)</strong> across four main scenarios:
</p>

<ul>
  <li><strong>Average:</strong> based on a city's typical year-round temperature profile</li>
  <li><strong>Spatial:</strong> based on proximity to other cities using the Voronoi graph</li>
  <li><strong>Temporal:</strong> based on how climate evolves over time within a single location</li>
  <li><strong>Spatial + Temporal:</strong> combining both location and long-term behavior for a richer perspective</li>
</ul>

<p>
By layering these perspectives, we aim to uncover how climate behaves across connected regions — where it stays consistent, where it shifts, and where unexpected patterns emerge.
</p>

<p></p>


<p></p>





<p></p>

<p></p>

<p></p>

<h2>Methods</h2>
<p></p>
The diagram below shows how we built a climate similarity model using graph neural networks. First, we connected the 1000 most populated cities using Voronoi-based geography — cities are linked if their zones share a border. Then, we used 40 years of temperature data to describe each city in two ways: one based on raw daily averages, and one using advanced GNN models that learn from how each city's climate changed over time. These feature vectors help us compare cities and uncover deep climate patterns around the world.

<h3>Method Overview</h3>


   <a href="#">
       <img src="{{ site.baseurl }}/img/voronoi16b.jpg" alt="Post Sample Image" width="678" >
   </a>
Fig. 1. Overview of the proposed method combining Voronoi-based spatial graphs with GNN pipelines for climate similarity and classification.
<p></p>
On the top left, we started by building a Voronoi graph — a map where cities are connected if they share a boundary in the global Voronoi diagram. Then, we calculated each city’s average daily temperature vector by combining 40 years of data into one 365-day snapshot.
<p></p>
Together, these formed the input for our first GNN Link Prediction model, which learned which cities were most similar — not just by distance, but by their average climate patterns.
<p></p>
But we didn’t stop there. We also looked at how each city’s climate has changed over time. We created a graph for every city, connecting years with similar temperature patterns. Then, we trained a GNN Graph Classification model to label cities as having stable or unstable climates — and from that, we got a new vector that captured each city’s climate behavior over time.
<p></p>
Finally, we ran a second GNN Link Prediction model using these new city vectors, still connected by the Voronoi graph. This let us compare cities based on how their climates evolve, not just their averages.
<h3>Graph Construction and Climate Labeling</h3>
<p></p>

In this study, we utilized GNN Graph Classification models to analyze small labeled graphs created from nodes and edges. We constructed graphs for each city, with nodes representing specific city-year pairs and edges defined by pairs of nodes with cosine similarities higher than threshold values. Each graph was labeled as either 'stable' or 'unstable' based on the city's geographical latitude.
<p></p>
<p></p>




<h4>Implementation of GCNConv for Graph Classification</h4>

<p></p>
For classifying these graphs, we used the Graph Convolutional Network (GCNConv) model from the PyTorch Geometric Library (PyG). The GCNConv model allowed us to extract feature vectors from the graph data, enabling us to perform a binary classification to determine whether the climate for each city was 'stable' or 'unstable'.
<p></p>

<h4>Python Code for Extracting Pre-Final Vectors: Graph Embedding</h4>
<p></p>
This function defines a custom Graph Convolutional Network (GCN) model using the PyTorch Geometric (PyG) library. The model is designed for classifying graphs, such as determining the climate stability of cities based on their temperature data. Here's a detailed breakdown of the function:
<p></p>
<p></p>  

<p></p>

<p></p>


<p></p>  

<p></p>
{% highlight python %}
xxxxx
{% endhighlight %}


<p></p>

After training the Graph Convolutional Network (GCN) model, this code snippet extracts the graph embedding for a specific graph in the dataset:
The graph embedding is stored in <code><span style="color: blue;">out</span></code>, capturing the structural and feature information of the entire graph.

<p></p>
{% highlight python %}
xxx
{% endhighlight %}
<p></p>


<p></p>

<p></p>
{% highlight python %}
xxxxx
{% endhighlight %}
<p></p>
<p></p>
Cosine similarity function:
<p></p>
{% highlight python %}
xxx
{% endhighlight %}
<p></p>


<h2>Experiments Overview</h2>
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

<p></p>



<p></p>

<p></p>



<h3>Voronoi Graph</h3>
<p></p>
To build a meaningful spatial graph between cities, we used a Voronoi diagram — a way of dividing space where each region contains all points closest to a particular city. This gives us a natural way to define neighbors: two cities are connected if their Voronoi regions share a border.
<p></p>
For example, Quebec, Canada and Porto, Portugal are neighbors because there are no highly populated cities betwenn them. Geometrically, the shares Voronoi diagram border line will cross the midpoint between them.
   <a href="#">
       <img src="{{ site.baseurl }}/img/voronoi17.jpg" alt="Post Sample Image" width="567" >
   </a>
Distance Distribution by Regions
<p></p>
Because cities are located on the curved surface of the Earth, we first converted their latitude and longitude into a flat 2D coordinate system using a map projection (EPSG:3857). This step is crucial because the Voronoi algorithm expects Cartesian coordinates in a Euclidean space.
<p></p>
Once projected, we used SciPy’s Voronoi class to compute the diagram. From that, we extracted the pairs of cities that share borders using vor.ridge_points, which gives us the indices of city pairs with adjacent Voronoi regions.
<p></p>
We then converted those indices into our unique cityInd identifiers and built a simple DataFrame of all Voronoi-based neighbor connections — forming the edge list of our spatial graph.
<p></p>

<p></p>
{% highlight python %}
from pyproj import Transformer
from scipy.spatial import Voronoi
import numpy as np
transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
projected = np.array([transformer.transform(lon, lat)
    for lon, lat in zip(cityData['lng'], cityData['lat'])])
vor = Voronoi(projected)
vor_neighbors = vor.ridge_points
neighbors = set(tuple(sorted((p1, p2))) for p1, p2 in vor.ridge_points)
rows = []
for i, j in neighbors:
    rows.append({
        'city1': cityData.iloc[i]['cityInd'],
        'city2': cityData.iloc[j]['cityInd'],
    })
voronoi_df = pd.DataFrame(rows)
{% endhighlight %}

<p>Example: each row below represents a pair of neighboring cities based on shared Voronoi borders:</p>

<pre>
city1   city2
 155     810  
  60     801  
  40     185  
 874     905  
 686     705  
</pre>

<p>This edge list defines the connections in our spatial graph for further distance and climate similarity analysis.</p>

<h4>Voronoi Neighbor Pair (Example)</h4>

<p></p>
<p></p>
<h4>Calculating Distances Between Neighboring Cities Using Voronoi Borders</h4>
<p></p>
To measure how far apart neighboring cities are, we first used a Voronoi diagram to identify which cities share a border. Each city’s geographic coordinates (latitude and longitude) were then projected into a flat 2D space using the EPSG:3857 map projection. This let us calculate accurate Euclidean distances in meters.
<p></p>
For every city pair connected by a Voronoi edge, we computed the distance between them and converted it into kilometers. We also kept track of each city’s unique identifier (cityInd) so that we could easily combine this data with other parts of our analysis.
<p></p>


<p></p>
{% highlight python %}
rows = []
for i, j in neighbors:
    dist_km = round(np.linalg.norm(projected[i] - projected[j]) / 1000, 5)
    rows.append({
        'city1': cityData.iloc[i]['cityInd'],
        'city2': cityData.iloc[j]['cityInd'],
        'distance_km': dist_km
    })
voronoi_distances_df = pd.DataFrame(rows)
{% endhighlight %}

<h4>Adding City and Country Labels</h4>

<p>To make our results easier to understand, we created a column that combines each city's name and country. This allows us to clearly display city pairs instead of using just numerical IDs.</p>

<p>First, we created a new column called <code>city_country</code> by combining the city name and country. Then we used each city’s <code>cityInd</code> as a key to map these names into the merged dataset.</p>

{% highlight python %}
cityData['city_country'] = cityData['city_ascii'] + ', ' + cityData['country']
city_lookup = cityData.set_index('cityInd')['city_country']
merged_df['city1_name'] = merged_df['city1'].map(city_lookup)
merged_df['city2_name'] = merged_df['city2'].map(city_lookup)
{% endhighlight %}

<p></p>
<h4>Distance Between Neighboring Cities (in kilometers)</h4>
<pre>
count     2983.000000  
mean       638.67  
std       1170.90  
min          2.27  
25%        176.22  
50%        340.92  
75%        658.97  
max      25870.97
</pre>
<p></p>

<h4>Adding Region Information to City Pairs</h4>
<p></p>
To further enrich our dataset, we linked each city pair with their respective geographical regions (zone). First, we created a mapping from each city's unique identifier (cityInd) to its corresponding region. Then, we added region details directly into our pairs dataset.
<p></p>
{% highlight python %}
region_map = cityData.set_index('cityInd')['zone'].to_dict()
voronoi_distances_df['region1'] = voronoi_distances_df['city1'].map(region_map)
voronoi_distances_df['region2'] = voronoi_distances_df['city2'].map(region_map)
{% endhighlight %}
  <p></p>
<p></p>

<p></p>

   <a href="#">
       <img src="{{ site.baseurl }}/img/voronoi10.jpg" alt="Post Sample Image" width="567" >
   </a>
Distance Distribution by Regions
<p></p>
<h4>Multi-View Node Representations for Climate Graphs</h4>
<p>
With the Voronoi-based spatial graph complete, we now shift our focus to calculating node features — the data that powers our graph-based models. For each city (node), we generate different types of vector representations that capture various dimensions of climate behavior.
</p>

<p>
First, we compute <strong>average daily temperature vectors</strong> by averaging the temperature across all available years, giving us a 365-dimensional profile for each city. These serve as the most direct representation of climate patterns over a typical year.
</p>

<p>
Next, we introduce <strong>graph-level embeddings</strong> extracted from a GNN Graph Classification model. These 128-dimensional vectors represent each city’s temporal temperature graph, encoded through a model trained to distinguish between stable and unstable climates. This gives us a more abstract, learned representation of each city's climate dynamics.
</p>

<p>
Then, we use these temperature-based vectors as input features for a <strong>GNN Link Prediction</strong> model. This approach learns embeddings for each city (node) based on its Voronoi neighbors and temperature trends, helping us understand how temperature profiles relate to geographic proximity.
</p>

<p>
Finally, we experiment with <strong>GNN Link Prediction using the city graph embeddings</strong> (from the previous step) as node features. This closes the loop, combining spatial relationships and abstract temporal behavior into one unified graph model — enabling richer predictions and insights into climate similarity between cities.
</p>







<p></p>

<h3>Average Temperature Vectors</h3>
<p>
To begin our climate analysis, we created a simple but effective climate profile for each city. The dataset includes daily temperature readings for 1000 cities across multiple years. By averaging the temperatures for each day of the year across all available years, we produced a single 365-dimensional vector per city.
</p>

<p>
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



<p></p>
<p></p>
<h3>GNN Graph Classification Model</h3>
<p></p>
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
<h3>GNN Link Prediction Model</h3>
<p></p>
<h4>DGL Graph Based on Voronoi Edges with Raw and Embedded City Vectors</h4>

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

<h3>🌐 Comparing Climate-Based City Embeddings and Graph Connectivity</h3>

<p>
With our spatial graph constructed from Voronoi neighbors and several types of node embeddings in hand, we now turn to comparing these city representations. Each type of vector reflects a unique perspective on urban climate data — some rooted in raw temperature values, others learned from GNN models.
</p>

<p>
We examine four types of node features:
</p>
<ul>
  <li><strong>Average Temperature Vectors:</strong> Basic yearly climate profiles averaged over time (365 values per city).</li>
  <li><strong>City Graph Embeddings:</strong> Learned vectors from a GNN Graph Classification model, based on structural changes across years.</li>
  <li><strong>Link Prediction Embeddings (on average vectors):</strong> Node embeddings learned to predict spatial connectivity using average climate patterns.</li>
  <li><strong>Link Prediction Embeddings (on city graph vectors):</strong> Embeddings trained on Voronoi edges but using city-level GNN vectors as input features — a fusion of spatial and temporal signals.</li>
</ul>

<p>
For each embedding type, we:
</p>
<ul>
  <li>Calculate <strong>cosine similarity</strong> between connected city pairs in the Voronoi graph</li>
  <li>Analyze the <strong>distribution of similarities</strong> to understand alignment between spatial and climate patterns</li>
  <li>Identify edge cases — e.g., cities close together but climatically dissimilar</li>
</ul>

<p>
Beyond pairwise similarity, we also use the Voronoi-based spatial graph (with edge weights derived from cosine similarity or distance) to compute <strong>graph centrality metrics</strong>. This includes:
</p>
<ul>
  <li><strong>Betweenness Centrality:</strong> Identifies cities that lie on many shortest paths — potential climate "bridges" between regions.</li>
  <li><strong>Degree Centrality:</strong> Highlights cities with many close climate or spatial neighbors.</li>
</ul>

<p>
This dual approach — comparing embeddings and analyzing spatial connectivity — helps us understand not just how similar cities are, but which ones are most structurally important in the global climate network.
</p>




{% highlight python %}
xxx
{% endhighlight %}



<p></p>

<h4>Calculating the Distribution of Cosine Similarities Between Cities</h4>
<p></p>

To understand how alike neighboring cities are in terms of climate, we calculated the cosine similarity between their average daily temperature vectors. A higher similarity score means the two cities share more similar yearly temperature patterns.
<p></p>
We did this for every pair of cities connected in the Voronoi graph:
<p></p>
{% highlight python %}
cosine_rows = []
for _, row in voronoi_distances_df.iterrows():
    i = row['city1']
    j = row['city2']
    vec_i = city_avg_vectors.loc[i].values
    vec_j = city_avg_vectors.loc[j].values
    cos_sim = cosine_similarity(vec_i, vec_j)
    cosine_rows.append({
        'city1': i,
        'city2': j,
        'cosine_similarity': cos_sim
    })
voronoi_cosines_df = pd.DataFrame(cosine_rows)
merged_df = pd.merge(voronoi_distances_df, voronoi_cosines_df, on=['city1', 'city2'])
{% endhighlight %}
<p></p>   
This final merged_df combines spatial distance and climate similarity—giving us a powerful dataset to explore relationships between how close cities are geographically and how similar they are climatologically.
<p></p>


<h4>Cosine Similarity Statistics</h4>
<p>
We compared cosine similarities between neighboring cities across four types of vectors:
raw average temperatures, GNN graph embeddings, GNN pre-final classification vectors, and GNN link prediction embeddings.
</p>

<p>
Average temperature vectors showed the strongest similarity (mean ≈ 0.99) with very low variability, confirming that neighboring cities often share similar climates.
</p>

<p>
GNN graph embeddings had a wider range, capturing structural differences and deeper spatial relationships.
Pre-final classification vectors provided a balanced view—more abstract than raw data but still climate-driven.
Link prediction embeddings also showed strong alignment, though with slightly more variability.
</p>

<p>
Together, these distributions reveal how different vector types capture varying degrees of geographic and climatic similarity.
</p>

<p></p>

<h4>Voronoi-Based Spatial Graph</h4>
<p>The diagram below illustrates how cities are connected based on shared Voronoi borders.</p>



<p></p>
<figure>
    <img src="{{ site.baseurl }}/img/voronoi11.jpg" alt="Traditional EEG Graph Example" style="width:90%; margin:auto;">
    <figcaption>This figure from our previous study represents construction of a Unified Knowledge Graph.</figcaption>
</figure>
<p></p>

<p></p>
<figure>
    <img src="{{ site.baseurl }}/img/voronoi12.jpg" alt="Traditional EEG Graph Example" style="width:90%; margin:auto;">
    <figcaption>This figure from our previous study represents construction of a Unified Knowledge Graph.</figcaption>
</figure>
<p></p>

<p></p>
<figure>
    <img src="{{ site.baseurl }}/img/voronoi13.jpg" alt="Traditional EEG Graph Example" style="width:90%; margin:auto;">
    <figcaption>This figure from our previous study represents construction of a Unified Knowledge Graph.</figcaption>
</figure>
<p></p>

Table~\ref{tab:lowest_avg_cosine} highlights the five city pairs with the lowest cosine similarity based on their average daily temperature vectors. Despite moderate or long geographic distances in some cases, such as between Porto and Quebec, the temperature patterns differ significantly. This indicates diverse climate behaviors even among cities with some spatial proximity.

<p></p>
<figure>
    <img src="{{ site.baseurl }}/img/voronoi15.jpg" alt="Traditional EEG Graph Example" style="width:98%; margin:auto;">
    <figcaption>Lowest Cosine Similarity Scores Between Voronoi Neighboring Cities Based on Average Temperature Vectors.</figcaption>
</figure>
<p></p>

<p></p>
<figure>
    <img src="{{ site.baseurl }}/img/voronoi14.jpg" alt="Traditional EEG Graph Example" style="width:98%; margin:auto;">
    <figcaption>This figure from our previous study represents construction of a Unified Knowledge Graph.</figcaption>
</figure>
<p></p>




<p>While neighboring cities tend to be relatively close geographically, the climate similarity between them is often extremely high — with most pairs scoring above 0.99 in cosine similarity. However, there are some interesting outliers where cities are nearby but climatically quite different.</p>
<p></p>  
<p></p>    
The table below compares two extremes in our Voronoi-based spatial graph: the five closest and five farthest city pairs that share a Voronoi border.
<p></p>

<p></p>

   <a href="#">
       <img src="{{ site.baseurl }}/img/voronoi8.jpg" alt="Post Sample Image" width="678" >
   </a>
Fig. 1. Latitude Distribution of the 1000 Most Populous Cities.
<p></p>  
Unsurprisingly, nearby cities like Jerusalem and Al Quds, or New York and Brooklyn, are virtually identical in climate (cosine similarity = 1.0) — they’re essentially part of the same metro areas.
<p></p>
What’s more surprising is that some extremely distant pairs—like Wellington (New Zealand) and Mar del Plata (Argentina)—also have very similar climate patterns, even though they're on opposite sides of the globe.
<p></p>
This shows that our spatial graph doesn't just capture proximity—it also reveals deeper patterns in how cities experience temperature. One exception is Reykjavik and Krasnoyarsk, which are both cold, but with very different seasonal patterns, explaining their lower similarity score.
<p></p>

<p></p>

   <a href="#">
       <img src="{{ site.baseurl }}/img/voronoi9.jpg" alt="Post Sample Image" width="567" >
   </a>
Fig. 1. Latitude Distribution of the 1000 Most Populous Cities.


<p></p>


<p></p>

<p></p>
{% highlight python %}
xxx
{% endhighlight %}
<p></p>


    <p></p>


<p></p>

<p></p>




<p></p>
{% highlight python %}
xxx
{% endhighlight %}


<p></p>

<h3>In Conclusion</h3>
<p></p>


In this study,
<p></p>



<p></p>

<p></p>    





<p></p>

<p></p>

<p></p>
<p></p>
