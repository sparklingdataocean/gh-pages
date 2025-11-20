---
layout:     post
title:      "Exploring Document Comparison with GNN Graph Classification"
subtitle:   "Extracting Subgraphs from Semantic Graphs and Applying GNN Graph Classification"
date:       2023-07-07 12:00:00
author:     "Melenar"
header-img: "img/pageNlpGc1e.jpg"
---
<h2>Semantic Graph for Text Understanding</h2>
<p>
  In this project, we treat text as a semantic graph rather than a flat sequence of words.
  From Wikipedia biographies of modern artists, we build a graph where nodes are meaningful
  word pairs and edges connect them when they appear together or in related contexts.
  Each node carries a transformer embedding, and a Graph Neural Network learns to “rewire”
  this graph—strengthening important links, downplaying weak ones, and revealing which
  artists and concepts truly sit close together or far apart in meaning.
</p>
<p>
  This semantic graph becomes a foundation for NLP tasks: it supports richer recommendations
  (finding both similar and contrastive artists or documents), cleans and enriches the
  underlying knowledge graph, and provides structure-aware representations that go beyond
  simple embedding cosine similarity. Instead of just asking “how similar are these texts?”,
  we can ask “how is their semantic neighborhood wired?” and let Graph AI answer from the
  topology of the semantic graph itself.
</p>

<h2>Conference &amp; Publication</h2>
<p>
  This work was presented at <strong>FRUCT35</strong>, the 35th Conference of the Open Innovations
  Association, held in Tampere, Finland, from <strong>24–26 April 2024</strong>, as the paper
  <em>“Enhancing NLP through GNN-Driven Knowledge Graph Rewiring and Document Classification”</em>.
  It was published in the conference proceedings with the doi:
  <a href="https://doi.org/10.23919/FRUCT61870.2024.10516410" target="_blank" rel="noopener">
    10.23919/FRUCT61870.2024.10516410
  </a>.
</p>




<p><h2>GNN Graph Classification for Semantic Graphs</h2>
<p></p>
In our previous studies, we focused on the exploration of knowledge graph rewiring to uncover unknown relationships between modern art artists. In one study
<u><a href="https://www.springerprofessional.de/en/building-knowledge-graph-in-spark-without-sparql/18375090">'Building Knowledge Graph in Spark Without SPARQL'</a></u>, we utilized artist biographies, known artist relationships, and data on modern art movements to employ graph-specific techniques, revealing hidden patterns within the knowledge graph.
</p><p>
In more recent study <u><a href="https://www.scitepress.org/Link.aspx?doi=10.5220/0011664400003393">'Rewiring Knowledge Graphs by Link Predictions'</a></u> our approach involved the application of GNN link prediction models. We trained these models on Wikipedia articles, specifically focusing on biographies of modern art artists. By leveraging GNN, we successfully identified previously unknown relationships between artists.


</p><p>

This study aims to extend earlier research by applying GNN graph classification models for document comparison, specifically using Wikipedia articles on modern art artists. Our methodology will involve transforming the text into semantic graphs based on co-located word pairs, then generating subsets of these semantic subgraphs as input data for GNN graph classification models. Finally, we will employ GNN graph classification models for a comparative analysis of the articles.

</p><p>
<p><h3>Introduction</h3>

The year 2012 marked a significant breakthrough in the fields of deep learning and knowledge graphs. It was during this year that Convolutional Neural Networks (CNN) gained prominence in image classification with the introduction of AlexNet. At the same time, Google introduced knowledge graphs, which revolutionized data integration and management. This breakthrough highlighted the superiority of CNN techniques over traditional machine learning approaches across various domains. Knowledge graphs enriched data products with intelligent and magical capabilities, transforming the way information is organized, connected, and understood.
</p><p>
For several years, deep learning and knowledge graphs progressed in parallel paths. CNN deep learning excelled at processing grid-structured data but faced challenges when dealing with graph-structured data. Graph techniques effectively represented and reasoned about graph structured data but lacked the powerful capabilities of deep learning. In the late 2010s, the emergence of Graph Neural Networks (GNN) bridged this gap and combined the strengths of deep learning and graphs. GNN became a powerful tool for processing graph- structured data through deep learning techniques.


<p></p>

</p><p>
GNN models allow to use deep learning algorithms for graph structured data by modeling entity relationships and capturing structures and dynamics of graphs. GNN models are being used for the following tasks to analyze graph-structured data: node classification, link prediction, and graph classification. Node classification models predict label or category of a node in a graph based on its local and global neighborhood structures. Link prediction models predict whether a link should exist between two nodes based on node attributes and graph topology. Graph classification models classify entire graphs into different categories based on their graph structure and attributes: edges, nodes with features, and labels on graph level.

</p><p>

</p><p>

GNN graph classification models are developed to classify small graphs and in practice they are commonly used in the fields of chemistry and medicine. For example, chemical molecular structures can be represented as graphs, with atoms as nodes, chemical bonds as edges, and graphs labeled by categories.

</p><p>
One of the challenges in GNN graph classification models lies in their sensitivity, where detecting differences between classes is often easier than identifying outliers or incorrectly predicted results. Currently, we are actively engaged in two studies that focus on the application of GNN graph classification models to time series classification tasks:

<u><a href="http://sparklingdataocean.com/2023/02/11/cityTempGNNgraphs/"> 'GNN Graph Classification for Climate Change Patterns'</a></u> and <u><a href="http://sparklingdataocean.com/2023/05/08/classGraphEeg/"> 'GNN Graph Classification for EEG Pattern Analysis'</a></u>.

</p><p>
In this post, we address the challenges of GNN graph classification on semantic graphs for document comparison. We demonstrate effective techniques to harness graph topology and node features in order to enhance document analysis and comparison. Our approach leverages the power of GNN models in handling semantic graph data, contributing to improved document understanding and similarity assessment.
</p><p>


</p><p>
To create semantic graph from documents we will use method that we introduced in our post
<u><a href="http://sparklingdataocean.com/2022/11/09/knowledgeGraph4NlpGnn/"> 'Find Semantic Similarities by GNN Link Predictions'</a></u>. In that post we demonstrated how to use GNN link prediction models to revire knowledge graphs.
For experiments of that study we looked at semantic similarities and dissimilarities between biographies of 20 modern art artists based on corresponding Wikipedia articles. One experiment was based on traditional method implemented on full test of articles and cosine similarities between reembedded nodes. In another scenario, GNN link prediction model ran on top of articles represented as semantic graphs with nodes as pairs of co-located words and edges as pairs of nodes with common words.
</p><p>
In this study, we expand on our previous research by leveraging the same data source and employing similar graph representation techniques. However, we introduce a new approach by constructing separate semantic graphs dedicated to each individual artist. This departure from considering the entire set of articles as a single knowledge graph enables us to focus on the specific relationships and patterns related to each artist. By adopting this approach, we aim to capture more targeted insights into the connections and dynamics within the knowledge graph, allowing for a deeper exploration of the relationships encoded within the biographies of these artists.
</p><p>



<p><h3>Methods</h3>
<p></p>

The input data for GNN graph classification models consists of a collection of labeled small graphs composed of edges and nodes with associated features. In this section we will describe data processing and model training in the following order:

<ul>
<li>Text preprocessing to transform raw data to semantic graphs. </li>
<li>Node embedding process.</li>
<li>The process of semantic subgraph extraction.</li>
<li>Training GNN graph classification model.</li>
</ul>



<p><h4>From Raw Data to Semantic Graph</h4>
<p></p>
To transform text data to semantic graph with nodes as co-located word pairs we will do the following:



</p>
<ul>
<li>Tokenize Wikipedia text and exclude stop words.</li>
<li>Get nodes as co-located word pairs.</li>
<li>Get edges between nodes.</li>
<li>Build semantic graph.</li>
</ul>
<p></p>
To generate edges we will find pair to pair neighbors following text sequences within articles and joint pairs that have common words.
<p></p>
{% highlight python %}
if pair1=[leftWord1, rightWord1],
   pair2=[leftWord2, rightWord2]
   and rightWord1=leftWord2,
then there is edge12={pair1, pair2}

{% endhighlight %}
<p></p>

Graph edges built based of these rules will cover word to word sequences and word to word chains within articles. On nodes and edges described above we will built an semantic graphs.
</p><p>
</p><p>
<h4>Node Embedding</h4>
</p><p>
To translate text of pairs of co-located to vectors we will use transformer model from Hugging Face: <u><a href="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"> 'all-MiniLM-L6-v2'</a></u>. This is a sentence-transformers model that maps text to a 384 dimensional vector space.


</p><p>
<p><h4>Extract Semantic Subgraphs</h4>
</p><p>


As input data for GNN graph classification model we need a set of labeled small graphs. In this study from each document of interest we will extract a set of subgraphs. By extracting relevant subgraphs from both documents, GNN graph classification models can compare the structural relationships and contextual information within the subgraphs to assess their similarity or dissimilarity. One of the ways to extract is getting subgraphs as neighbors and neighbors of neighbors of nodes with high centralities. In this study we will use betweenness centrality metrics.

</p><p>

</p><p>
<h4>Train the Model</h4>
</p><p>
The GNN graph classification model is designed to process input graph data, including both the edges and node features, and is trained on graph-level labels. In this case, the input data structure consists of the following components:
</p><p>
<ul>
<li>
Edges in a graph capture the relationships between nodes.
</li><li>
Nodes with embedded features would be embedded into the nodes to provide additional information to the GNN graph classification model.
</li><li>
Graph-level labels are assigned to the entire graph, and the GNN graph classification model leverages these labels to identify and predict patterns specific to each label category.
</li></ul>
<p></p>

As GNN graph classification model we will use a GCNConv (Graph Convolutional Network Convolution) activation model. The model code is taken from tutorial of the <u><a href="https://pytorch-geometric.readthedocs.io/en/latest/"> 'PyTorch Geometric Library (PyG)'</a></u>. The GCNConv graph classification model is a type of graph convolutional network that uses convolution operations to aggregate information from neighboring nodes in a graph. It takes as input graph data (edges, node features, and the graph-level labels) and applies graph convolutional operations to extract meaningful features from the graph structure.
<p></p>
The Python code for the GCNConv model is provided by the PyG library. The code for converting data to the PyG data format, model training and interpretation techniques are described below.
<p></p>


<h3>Experiments</h3>
<p></p>
<h4>Data Source</h4>
<p></p>

As the data source for this study we used text data from Wikipedia articles about 20 modern art artists. Here is the list of artists and Wikipedia text size distribution:
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/artStats.jpg" alt="Post Sample Image" width="1000">
</a>
<p></p>

<p>Based on Wikipedia text size distribution, the most well known artist in our artist list is Vincent van Gogh and the most unknown artist is Franz Marc:</p>

<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/artImg1.jpg" alt="Post Sample Image" width="345">
</a>
<p></p>

<p></p>
More detail information is available in our post <u><a href="http://sparklingdataocean.com/2022/07/23/knowledgeGraph4GNN/">'Rewiring Knowledge Graphs by Link Predictions'</a></u>.

<p></p>

To estimate document similarities based on GNN graph classification model, we experimented with pairs of highly connected artists and highly disconnected artists.

Pairs of artists were selected based on our study <u><a href="https://www.springerprofessional.de/en/building-knowledge-graph-in-spark-without-sparql/18375090">"Building Knowledge Graph in Spark without SPARQL"</a></u>.
This picture illustrates relationships between modern art artists based on their biographies and art movements:

<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/artStats2b.jpg" alt="Post Sample Image" width="1000">
</a>
<p></p>

<p>As highly connected artists, we selected Pablo Picasso and Georges Braque, artists with well known strong relationships between them: both Pablo Picasso and Georges Braque were pioneers of cubism art movement.
<p></p>
As highly disconnected artists, we selected Claude Monet and Kazimir Male- vich who were notably distant from each other: they lived in different time peri- ods, resided in separate countries, and belonged to contrasting art movements: Claude Monet was a key artist of impressionism and Kazimir Malevich a key artist of Suprematism.</p>

<p></p>

For a more detailed exploration of the relationships between modern art artists discovered through knowledge graph techniques, you can refer to our post:

<u><a href="http://sparklingdataocean.com/2020/02/02/knowledgeGraphIntegration/">"Knowledge Graph for Data Integration"</a></u>.


<p></p>


<p></p>

<p></p>

<p></p>
<h4>Transform Text Document to Semantic Graph</h4>
<p></p>
For each selected Wikipedia article we transformed text to semantic graphs by the following steps:

<ul>
<li>Tokenize Wikipedia text and excluded stop words.</li>
<li>Generate nodes as co-located word pairs.</li>
<li>Calculate edges as joint pairs that have common words. These edges represente word sequences and word chains within articles.</li>
</ul>


<p></p>
<h5>Tokenize Wikipedia text</h5>
<p></p>
{% highlight python %}
from nltk.tokenize import RegexpTokenizer
tokenizer =RegexpTokenizer(r'[A-Za-z]+')

wikiArtWords=wikiArtists['Wiki']
  .apply(lambda x: RegexpTokenizer(r'[A-Za-z]+').tokenize(x)).reset_index()

wikiArtWords=wikiArtWords.explode(['words'])
wordStats=pd.merge(wikiArtWords,listArtists)

artistWordStats=wordStats.groupby('Artist').count()
  .sort_values('words',ascending=False)
{% endhighlight %}



<p></p>
<h5>Exclude stop words</h5>
<p></p>

Exclude stop words and short words woth length<4:
<p></p>
{% highlight python %}
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
dfStopWords=pd.DataFrame (STOPWORDS, columns = ['words'])
dfStopWords['stop']="stopWord"
stopWords=pd.merge(wikiArtWords,dfStopWords,on=['words'], how='left')

nonStopWords=stopWords[stopWords['stop'].isna()]
nonStopWords['stop'] = nonStopWords['words'].str.len()
nonStopWords=nonStopWords[nonStopWords['stop']>3]
nonStopWords['words']= nonStopWords['words'].str.lower()
nonStopWords.reset_index(drop=True, inplace=True)
nonStopWords['idxWord'] = nonStopWords.index
{% endhighlight %}

<p></p>

<p></p>


<p></p>

<p></p>

<p></p>
<h5>Generated nodes as co-located word pairs</h5>
<p></p>
Get pairs of co-located words:
<p></p>
{% highlight python %}
bagOfWords=pd.DataFrame(nonStopWordsSubset['words'])
bagOfWords.reset_index(drop=True, inplace=True)
bagOfWords['idxWord'] = bagOfWords.index

indexWords = pd.merge(nonStopWordsSubset,bagOfWords, on=['words','idxWord'])
idxWord1=indexWords
  .rename({'words':'word1','idxArtist':'idxArtist1','idxWord':'idxWord1'}, axis=1)
idxWord2=indexWords
  .rename({'words':'word2','idxArtist':'idxArtist2','idxWord':'idxWord2'}, axis=1)

leftWord=idxWord1.iloc[:-1,:]
leftWord.reset_index(inplace=True, drop=True)
rightWord = idxWord2.iloc[1: , :].reset_index(drop=True)  

pairWords=pd.concat([leftWord,rightWord],axis=1)
pairWords = pairWords.drop(pairWords[pairWords['idxArtist1']!=pairWords['idxArtist2']].index)
pairWords.reset_index(drop=True, inplace=True)

{% endhighlight %}
<p></p>

Drop duplicates {artist, word1, word2}

<p></p>

{% highlight python %}
cleanPairWords = pairWords
cleanPairWords = cleanPairWords.drop_duplicates(
  subset = ['idxArtist1', 'word1', 'word2'], keep = 'last').reset_index(drop = True)
cleanPairWords['wordpair'] =
  cleanPairWords["word1"].astype(str) + " " + cleanPairWords["word2"].astype(str)
cleanPairWords['nodeIdx']=cleanPairWords.index

{% endhighlight %}
<p></p>



<p></p>


<p></p>
<h5>Calculated edges as joint pairs that have common words.</h5>
<p></p>
Index data:
<p></p>

{% highlight python %}
nodeList1=nodeList
  .rename({'word2':'theWord','wordpair':'wordpair1','nodeIdx':'nodeIdx1'}, axis=1)
nodeList2=nodeList
  .rename({'word1':'theWord','idxArtist1':'idxArtist2','wordpair':'wordpair2',
  'nodeIdx':'nodeIdx2'}, axis=1)
allNodes=pd.merge(nodeList1,nodeList2,on=['theWord'], how='inner')

{% endhighlight %}

<p></p>

<p></p>
<h4>Input Data Preparation</h4>
<p></p>
<h5>Transform Text to Vectors</h5>
As mentioned above, for text to vector translation we used ’all- MiniLM-L6- v2’ transformer model from Hugging Face.
<p></p>
Get unique word pairs for embedding
<p></p>
{% highlight python %}
bagOfPairWords=nodeList
bagOfPairWords = bagOfPairWords.drop_duplicates(subset='wordpair')
bagOfPairWords.reset_index(inplace=True, drop=True)
bagOfPairWords['bagPairWordsIdx']=bagOfPairWords.index
{% endhighlight %}
<p></p>
<p></p>
Transform node features to vectors:
<p></p>
{% highlight python %}
model = SentenceTransformer('all-MiniLM-L6-v2')
wordpair_embeddings = model.encode(cleanPairWords["wordpair"],convert_to_tensor=True)

{% endhighlight %}

<p></p>


<p></p>
<h5>Prepare Input Data for GNN Graph Classification Model</h5>
<p></p>

In GNN graph classification, the input to the model is typically a set of small graphs that represent entities in the dataset. These graphs are composed of nodes and edges, where nodes represent entities, and edges represent the relationships between them. Both nodes and edges may have associated features that describe the attributes of the entity or relationship, respectively. These features can be used by the GNN model to learn the patterns and relationships in the data, and classify or predict labels for the graphs. By considering the structure of the data as a graph, GNNs can be particularly effective in capturing the complex relationships and dependencies between entities, making them a useful tool for a wide range of applications.

<p></p>
To prepare the input data for the GNN graph classification model, we generated labeled semantic subgraphs from each document of interest. These subgraphs were constructed by selecting neighbors and neighbors of neighbors around specific ”central” nodes. The central nodes were determined by identifying the top 500 nodes with the highest betweenness centrality within each document.
<p></p>
By focusing on these central nodes and their neighboring nodes, we aimed to capture the relevant information and relationships within the document. This approach allowed us to create labeled subgraphs that served as the input data for the GNN graph classification model, enabling us to classify and analyze the documents effectively.

<p></p>
{% highlight python %}
import networkx as nx
list1 = [3]  
list2 = [6]  
radius=2
datasetTest=list()
datasetModel=list()
dfUnion=pd.DataFrame()
seeds=[]
for artist in list1 + list2:
  if artist in list1:
    label=0
  if artist in list2:
    label=1
    edgeInfo0=edgeInfo[edgeInfo['idxArtist']==artist]
    G=nx.from_pandas_edgelist(edgeInfo0,  "wordpair1", "wordpair2")
    betweenness = nx.betweenness_centrality(G)
    sorted_nodes = sorted(betweenness, key=betweenness.get, reverse=True)
    top_nodes = sorted_nodes[:500]
    dfTopNodes=pd.DataFrame(top_nodes,columns=['seedPair'])
    dfTopNodes.reset_index(inplace=True, drop=True)
    for seed in dfTopNodes.index:
      seed_node=dfTopNodes['seedPair'].iloc[seed]
      seeds.append({'label':label,'artist':artist, 'seed_node':seed_node, 'seed':seed})
      foaf_nodes = nx.ego_graph(G, seed_node, radius=radius).nodes
      dfFoaf=pd.DataFrame(foaf_nodes,columns=['wordpair'])
      dfFoaf.reset_index(inplace=True, drop=True)
      dfFoaf['foafIdx']=dfFoaf.index
      words_embed = words_embeddings.merge(dfFoaf, on='wordpair')
      values1=words_embed.iloc[:, 0:384]
      fXValues1= values1.fillna(0).values.astype(float)
      fXValuesPT1=torch.from_numpy(fXValues1)
      graphSize=dfFoaf.shape[0]
      # dfFoaf.tail()
      oneGraph=[]
      for i in range(graphSize):
        pairi=dfFoaf['wordpair'].iloc[i].split(" ", 1)
        pairi1 = pairi[0]
        pairi2 = pairi[1]
        # for j in range(i+1,graphSize):
        for j in range(graphSize):
          pairj=dfFoaf['wordpair'].iloc[j].split(" ", 1)
          pairj1 = pairj[0]
          pairj2 = pairj[1]
          if ( pairi2==pairj1):
            oneGraph.append({'label':label, 'artist':artist,'seed':seed,
                             'centralNode':seed_node, 'k1':i, 'k2':j, 'pairi':pairi, 'pairj':pairj})
      dfGraph=pd.DataFrame(oneGraph)
      dfUnion = pd.concat([dfUnion, dfGraph], ignore_index=True)
      edge1=torch.tensor(dfGraph[['k1','k2']].T.values)
      dataset1 = Data(edge_index=edge1)
      dataset1.y=torch.tensor([label])
      dataset1.x=fXValuesPT1
      datasetModel.append(dataset1)
      loader = DataLoader(datasetModel, batch_size=32)
{% endhighlight %}
<p></p>

Model size
<p></p>
{% highlight python %}
modelSize=len(dataset)
modelSize
1000
{% endhighlight %}
<p></p>
<p></p>

<p><h4>Training the Model</h4>
<p></p>
For this study we used the code provided by PyTorch Geometric as tutorial on GCNConv graph classification models - we just slightly tuned it for our data:

<p></p>
<p><h5>Randomly split data to training and tesing</h5>
<p></p>

<p></p>

<p></p>
{% highlight python %}
import random
torch.manual_seed(12345)
percent = 0.13
sample_size = int(modelSize * percent)
train_size=int(modelSize-sample_size)
test_dataset = random.sample(dataset, sample_size)
train_dataset = random.sample(dataset, train_size)
{% endhighlight %}

<p></p>

<p></p>
{% highlight python %}
print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')
Number of training graphs: 870
Number of test graphs: 130
{% endhighlight %}

<p></p>

<p></p>
<p></p>
{% highlight python %}
from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()
{% endhighlight %}
<p></p>
<h5>Prepare the model:</h5>

<p></p>
{% highlight python %}
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(384, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)        
        return x
model = GCN(hidden_channels=64)
print(model)
{% endhighlight %}
<p></p>

<p><h5>Train the Model:</h5>
<p></p>

<p></p>

<p></p>
{% highlight python %}
from IPython.display import Javascript
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

model = GCN(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
   model.train()
   for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x.float(), data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader):
    model.eval()
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x.float(), data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.

for epoch in range(1, 10):
   train()
   train_acc = test(train_loader)
   test_acc = test(test_loader)
   print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
{% endhighlight %}

<p></p>
<p></p>
To estimate the model results we used the same model accuracy metrics as in the PyG tutorial.
<p></p>
<p></p>
<p></p>

<p><h5>Accuracy Metrics of the Model:</h5>
<p></p>

As we mentioned above, the GNN graph classification model exhibits higher sensitivity for classification compared to the GNN link prediction model. In both scenarios, we trained the models for 9 epochs.
<p></p>
Given the distinct differences between Monet and Malevich as artists, we anticipated achieving high accuracy metrics. However, the surprising outcome was obtaining perfect metrics as 1.0000 for training data and 1.0000 for testing right from the initial training step.

<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/modelStats1.jpg" alt="Post Sample Image" width="345">
</a>
<p></p>

In the classification of Wikipedia articles about Pablo Picasso and Georges Braque, we were not anticipating the significant differentiation between these two documents: these artists had very strong relationships in biography and art movements. Also GNN link prediction models classified these artists as highly similar.
<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/modelStats2.jpg" alt="Post Sample Image" width="345">
</a>
<p></p>
This observation highlights the high sensitivity of the GNN graph classifica- tion model and emphasizes the ability of the GNN graph classification model to capture nuanced differences and provide a more refined classification approach compared to the GNN Link Prediction models.

<p><h4>Model Results</h4>
<p></p>

<p></p>
To interpret model results we calculated the softmax probabilities for each class output by the model. The softmax probabilities represent the model's confidence in its prediction for each class.
<p></p>

{% highlight python %}
modelSize
1000
{% endhighlight %}

<p></p>
<p></p>

<p></p>
{% highlight python %}
softmax = torch.nn.Softmax(dim = 1)
graph1=[]
for g in range(modelSize):
  label=dataset[g].y[0].detach().numpy()
  out = model(dataset[g].x.float(), dataset[g].edge_index, dataset[g].batch)
  output = softmax(out)[0].detach().numpy()
  pred = out.argmax(dim=1).detach().numpy()
  graph1.append({'index':g,
                 'label':label,'pred':pred[0],
                 'prob0':round(output[0], 4),'prob1':round(output[1], 4)})
{% endhighlight %}
<p></p>



<p></p>
One of the challenges encountered when utilizing the GNN graph classification model for text classification is the identification of outliers. In the scenario of classifying Wikipedia articles about the biographies of Claude Monet and Kazimir Malevich, the trained model did not detect any outliers.
<p></p>
In the case of Pablo Picasso and Georges Braque, we found that despite their shared biographies and involvement in the same art movements, there were no- table differences between their respective Wikipedia articles. The GNN graph classification model identified these articles as highly dissimilar, suggesting dis- tinct characteristics and content within their biographies. During our analysis of 1000 subgraphs, we encountered only 8 outliers.
<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/modelStats3.jpg" alt="Post Sample Image" width="444">
</a>
<p></p>



<p></p>
<p></p>
<p><h3>Conclusion</h3>


<p></p>
GNN graph classification is a powerful machine learning technique designed for object classification when the objects can be represented as graphs. By mapping object elements to nodes and their relationships to edges, GNN graph classification models capture complex interdependencies among the elements. This approach is particularly valuable when traditional machine learning methods struggle to capture complex relationships.
<p></p>
GNN graph classification has been successfully applied in various domains, including molecule classification, image recognition, protein classification, social networks, brain connectivity networks, road networks, and climate data analysis.
<p></p>
In this paper, we investigate the application of GNN graph classification models in NLP for document comparison, aiming to uncover document similarities and dissimilarities using graph topology and node features. We focus on comparing Wikipedia articles about modern art artists and demonstrate the potential of these models in extracting relevant information and identifying patterns. Additionally, we address challenges related to model sensitivity and outlier detection in GNN graph classification.
<p></p>
We investigated the effectiveness of GNN graph classification in capturing different types of relationships among artists. We specifically selected two pairs of artists, one representing highly connected relationships (Pablo Picasso and Georges Braque) and the other representing highly disconnected relationships (Claude Monet and Kazimir Malevich). As expected, in the case of classifying Wikipedia articles on the biographies of Claude Monet and Kazimir Malevich, no outliers were detected.
<p></p>
In the case of Pablo Picasso and Georges Braque, despite their shared biographies and association with the cubism art movement, we identified substantial differences in their respective articles. The GNN graph classification model categorized these articles as highly dissimilar and out of 1000 subgraphs, we encountered only 8 outliers, further emphasizing the model’s sensitivity in capturing the nuanced differences between the documents.
<p></p>
Future research can further explore the applications of GNN graph classification models in NLP, with a focus on addressing sensitivity and outlier detection challenges. Additionally, efforts can be made to deeper understanding of semantic graph relationships.
<p></p>
In conclusion, our study advances document comparison using GNN graph classification, offering valuable insights for text analysis and knowledge discovery. It contributes to the growing field of GNN-based methods in NLP, opening avenues for future research and practical applications.
<p></p>
<p><h3>Next Post - Graph Connectors</h3>

In the next spost we will start a new topic related to knowledge graphs and GNN.
<p></p>
