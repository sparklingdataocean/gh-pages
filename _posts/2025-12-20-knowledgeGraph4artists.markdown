---
layout:     post
title:      "Connecting Artists through Unified Knowledge Graph"
subtitle:   "Integrating Text and Images with Graph Neural Networks to Explore Artistic Relationships"
date:       2025-12-20 12:00:00
author:     "Melenar"
header-img: "img/artPage2d.jpg"
---

<p><h2>IntroductionXXX: Knowledge Graphs Exploration</h2>

Art and artists are deeply connected, reflecting creativity, culture, and history. Graphs provide a powerful way to explore these relationships by representing artists, artworks, and movements as nodes, and their connections as edges. This helps us uncover patterns, find hidden links, and better understand the evolution of art. With advanced tools like Graph Neural Networks (GNNs), we can predict new connections, group related ideas, and gain even deeper insights into the art world.
<p></p>

<p><h3>Building on Our Previous Research</h3>
<p></p>

Our previous study focused on the <strong>Construction of a Unified Knowledge Graph</strong> to analyze the relationships between artists and their works. Starting with raw data comprising artist biographies and painting images, two separate graphs are created: an <strong>Artist Graph</strong> and a <strong>Painting Graph</strong>.
<p></p>


This is the link to corresponding post <i><a href="http://127.0.0.1:4000/2025/01/01/knowledgeGraph4artPaintings/">'Connecting Art and Data: Building a Unified Knowledge Graph'</a></i>.

<p></p>
Textual data from artist biographies is transformed into vectors using Large Language Models (LLMs), while visual data from painting images is processed into vectors using Convolutional Neural Networks (CNNs). These initial embeddings are further refined through Graph Neural Networks (GNNs) to produce consistent, high-quality representations for the nodes in both graphs.
<p></p>
The final step involves combining the Artist Graph and Painting Graph into a single <strong>Unified Knowledge Graph</strong>, where links represent the connections between artists and their respective works. This integrated graph provides a robust framework for exploring the intricate relationships within the art world, enabling tasks such as classification, clustering, and link prediction.
<p></p>


<p></p>
<figure>
    <img src="{{ site.baseurl }}/img/artPaint15.jpg" alt="Traditional EEG Graph Example" style="width:61%; margin:auto;">
    <figcaption>This figure from our previous study represents construction of a Unified Knowledge Graph.</figcaption>
</figure>
<p></p>

<p></p>


<p></p>

<p></p>



<p></p>
<p><h2>This Study: Introduction</h2>
<p></p>



<p></p>

In this study we use the same dataset, <i><a href="https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time">
'Best Artworks of All Time'</a></i>. Data was taken from Kaggle and it features information and artwork images from 50 of the most influential artists in history. The dataset includes detailed artist metadata sourced from Wikipedia, as well as a collection of high-resolution images scraped from artchallenge.ru.
<p></p>
As text data analysis for our previous study we used short bio information provided in raw data. In this study we will be  exploring text data from Wikipedia articles about the same 50 artists. For deeper semantic analysis, we will use a different approach: instead of building it on artist short bios, we will build initial semantic graph on co-located word pairs.

<p></p>
Semantic graph built of co-located word pairs represents a deeper view on text documents: not only high key words (or key word pairs) but as graphs, a variety of other semantic connections. We will look at nodes with high centralities (like nodes with high betweenness) and will build local graphs around them. As nodes with high betweenness represent 'hubs' with high connectivity around them, local graphs will represent tightly connected text topics. We will generace several highest topics for each of 50 artists and run GNN Graph Classification model. From that model we'll graph pre-final vectors that would represent embedded graphs.    
<p></p>

For GNN Graph Classification, to use this model from PyTorch geometric library, we need graph labeling. For this study, we can select like 20 tor graphs from each Artist wiki docs, then index them in the betweenness order, then classify by even indexes for even number artists and odd indexes for odd number artists.
<p></p>
Then we'll get a cosine similarity matrix for ~1000 graph embedded vectors and find more linear algebra scenarios -- closest artists, far away within the same artists, or within similar artists. Closets neighbors. Build another graph on top of them and look at that graph for clustering, outliers...
<p></p>

<p></p>
This semantic graph method was introduced in our previous study. It was posted in <i><a href="http://127.0.0.1:4000/2022/11/09/knowledgeGraph4NlpGnn/">'Find Semantic Similarities by GNN Link Predictions'</a></i>, presented in a conference and published in paper
</p><p>


<p><h2>Methods</h2>



</p><p>

<p></p>


<p><h3>Building Initial Semantic Graph</h3>
<p></p>
To build initial semantic knowledge graph we will use the following steps:

</p>
<ul>
<li>Tokenize Wikipedia text and exclude stop words.</li>
<li>Get nodes as word pairs that are co-located within articles.</li>
<li>Get edges as pair to pair neighbors following text sequences within articles.</li>
<li>Get edges as joint pairs that have common words. These edges will represent word chains within articles and across them.</li>
</ul>


<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/artWiki1.jpg" alt="Post Sample Image" width="404">
</a>

<p></p>

Graph edges built based of these rules will cover word to word sequences and word to word chains within articles. More important, they will connect different articles by covering word to word chains across articles.
</p><p>
On nodes and edges described above we will built an initial knowledge graph.

</p><p>
<p><h3>Transform Text to Vectors</h3>
</p><p>
As a method of text to vector translation we will use a transformer model from the Hugging Face <i><a href="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2">'all-MiniLM-L6-v2'</a></i>. In this model, the text is encoded into a 384-dimensional vector space, producing a tensor of size <code>[50, 384]</code>, where each row represents an artist.
</p>

<p>
These embeddings are assigned as node features in the graph using the <code>ndata['feat']</code> attribute. This enables:
</p>

<ul>
  <li>Using vectors as input for GNN link prediction models.</li>
  <li>Generating additional graph edges by calculating a cosine similarity matrix and selecting highly connected vector pairs.</li>
</ul>

<p>
This approach provides a robust foundation for graph-based analysis, capturing both semantic relationships and potential hidden connections between artists.
</p>

</p><p>
<p><h3>Run GNN Link Prediction Model</h3>
</p><p>

As Graph Neural Networks link prediction model we will use a GraphSAGE link prediction model from Deep Graph Library (DGL). The model is built on two GrapgSAGE layers  and computes node representations by averaging neighbor information.

The code for this model is provided by DGL tutorial <i><a href="https://docs.dgl.ai/en/0.8.x/tutorials/blitz/4_link_predict.html">DGL Link Prediction using Graph Neural Networks</a></i>.



</p><p>

The results of this code are embedded nodes that can be used for further analysis such as node classification, k-means clustering, link prediction and so on. In this study we used it for link prediction by estimating cosine similarities between embedded nodes.
<p></p>



<h3>Unified Knowledge Graph</h3>


<p>
To analyze relationships between artists and their works, we created a Unified Knowledge Graph by combining the Artist graph and Painting graph. This integration enables consistent multi-modal data embedding and analysis.
</p>

<p>
Artist biographies were transformed into 384-dimensional vectors using a transformer model, then reduced to 128-dimensional embeddings through GNN link prediction. Paintings were similarly represented as 2048-dimensional vectors using a pre-trained CNN model, which were also reduced to 128-dimensional embeddings.
</p>

<p>
To unify the graphs, unique indices ensured no overlap between artist and painting nodes, with artist nodes offset by 1000. Painting file names were processed to map each painting to its corresponding artist, and edges were added to represent these relationships.
</p>

<p>
The resulting Unified Knowledge Graph integrates artist and painting data into a single structure, providing a foundation for advanced graph-based analysis.
</p>

<h4>GNN Link Prediction Model for the Unified Graph</h4>
<p>
<p>We applied the GNN Link Prediction model three times across different stages of the study:</p>

<p><strong>1. Artist Graph:</strong> The model was used to process artist biographies, initially embedded as 384-dimensional vectors derived from text data. The GNN Link Prediction model reduced these embeddings to a consistent size of 128 dimensions, capturing semantic relationships between artists effectively.</p>

<p><strong>2. Painting Graph:</strong> For paintings, visual features extracted via a CNN model were initially represented as 2048-dimensional vectors. The GNN Link Prediction model was applied to reduce these high-dimensional embeddings to 128 dimensions, emphasizing meaningful stylistic or thematic connections between paintings.</p>

<p><strong>3. Unified Knowledge Graph:</strong> After merging the Artist and Painting graphs into a single structure, the GNN Link Prediction model was run again on the unified graph. This step refined the node embeddings, ensuring consistent 128-dimensional representations across all nodes while integrating multi-modal relationships between artists and their works.</p>

<p>By running the model at each stage, we achieved consistent embeddings that facilitate advanced graph-based analysis, uncovering complex patterns and relationships within the unified framework.</p>

</p>




<h2>Experiments</h2>
<p></p>
<h3>Data Source Analysis</h3>
<p></p>
As data source we used text data from kaggle.com: <i><a href="https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time">
'Best Artworks of All Time'</a></i>.



<p></p>
<h4>Metadata Overview</h4>
<p>This dataset provides the following key files:</p>
<ul>
  <li>
    <strong><code>artists.csv</code></strong>:
    <ul>
      <li>Contains metadata for 50 influential artists, including:</li>
      <li><strong>Name</strong>, <strong>Genre</strong>, <strong>Nationality</strong>, <strong>Biography</strong>, and <strong>Years</strong> (lifespan or active period).</li>
    </ul>
  </li>
  <li>
    <strong><code>resized.zip</code></strong>:
    <ul>
      <li>A collection of resized artwork images, optimized for faster processing and reduced storage.</li>
      <li>Ideal for machine learning workflows requiring efficient model training and testing.</li>
    </ul>
  </li>
</ul>
<p>These files provide a compact yet comprehensive resource for analyzing and classifying artworks.</p>


<p></p>
<h4>Raw Data Processing</h4>
<p></p>

The code loads the artist metadata from a CSV file, selects relevant columns (name, genre, nationality, bio, years), sorts the data alphabetically by name, resets the index, and assigns a sequential artistIndex based on the new order.
<p></p>
{% highlight python %}
artists = pd.read_csv("/content/drive/My Drive/Art/artists.csv")
artists_dff = artists[['name', 'genre', 'nationality','bio', 'years']]
artists_df= artists_dff.sort_values('name').reset_index(drop=True)
artists_df['artistIndex'] = artists_df.index
{% endhighlight %}



<p>Split the standardized 'years' column into 'start_year' and 'end_year': </p>
<p></p>
<ul>
  <li>
    The <code>years</code> column is cleaned to replace special characters (e.g., em-dash) with a hyphen and split into <code>start_year</code> and <code>end_year</code>.
  </li>
  <li>
    The <code>genre</code> column is exploded, creating separate rows for each genre associated with an artist, and the DataFrame is reset to include <code>artistIndex</code>, <code>name</code>, <code>start_year</code>, <code>end_year</code>, and <code>genre</code>.
  </li>
  <li>
    A new DataFrame, <code>artist2nationality</code>, is created by splitting the <code>nationality</code> column into multiple entries, exploding it into separate rows, and resetting the index to include <code>artistIndex</code>, <code>name</code>, <code>start_year</code>, <code>end_year</code>, and <code>nationality</code>.
  </li>
</ul>

<p></p>
{% highlight python %}
artists_df['years'] = artists_df['years'].apply(lambda x: re.sub(r'[–—]', '-', x))
artists_df[['start_year', 'end_year']] = artists_df['years'].str.split(' - ', expand=True)
  .explode('genre')
  .reset_index(drop=True)[['artistIndex', 'name', 'start_year', 'end_year', 'genre']]
artist2nationality = artists_df.assign(nationality=artists_df['nationality'].str.split(',')) \
.explode('nationality')
.reset_index(drop=True)[['artistIndex', 'name', 'start_year', 'end_year', 'nationality']]
{% endhighlight %}

<p></p>

<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/artPaint1.jpg" alt="Post Sample Image" width="777">
</a>
<p></p>
<h3>Artist Graph</h3>
<p></p>
<h4>Building Graph on Artists</h4>
<p>To construct a graph where nodes represent artists and edges are created based on shared attributes, follow these steps:</p>
<ul>
  <li>
    <strong>Create nodes:</strong> Add a node for each artist using their <code>artistIndex</code> and <code>name</code>.
  </li>
  <li>
    <strong>Create genre-based edges:</strong> For each genre, add edges between all pairs of artists who share that genre.
  </li>
  <li>
    <strong>Create nationality-based edges:</strong> Similarly, for each nationality, add edges between all pairs of artists who share that nationality.
  </li>
</ul>
<p></p>


<p></p>
Below is the Python code used to implement this graph:

<p></p>

{% highlight python %}
import networkx as nx
G = nx.Graph()
for _, row in artist2genre.iterrows():
    G.add_node(row['artistIndex'], name=row['name'])# !pip install torch
for genre, group in artist2genre.groupby('genre'):
    artist_indices = group['artistIndex'].tolist()
    for i in range(len(artist_indices)):
        for j in range(i + 1, len(artist_indices)):
            G.add_edge(artist_indices[i], artist_indices[j], genre=genre)
for nationality, group in artist2nationality.groupby('nationality'):
    artist_indices = group['artistIndex'].tolist()
    for i in range(len(artist_indices)):
        for j in range(i + 1, len(artist_indices)):
            G.add_edge(artist_indices[i], artist_indices[j], nationality=nationality)
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Average degree: {nx.density(G) * (G.number_of_nodes() - 1)}")
Number of nodes: 50
Number of edges: 218
Average degree: 8.72
{% endhighlight %}

<p></p>

<p></p>
<p>The following steps describe how to convert graph edges into a DataFrame for analysis:</p>
<ul>
  <li><strong>Convert edges to a DataFrame:</strong> Extract the edges from the graph <code>G</code> and organize them into a DataFrame with columns for source nodes, target nodes, and edge attributes.</li>
  <li><strong>Expand edge attributes:</strong> If the edge attributes are stored as dictionaries, expand these into separate columns to make them more accessible.</li>
  <li><strong>Display the edges DataFrame:</strong> View the resulting DataFrame to verify its structure and content.</li>
</ul>
<p>The Python code snippet below demonstrates these steps:</p>
<p></p>

{% highlight python %}
import pandas as pd
edges_df = pd.DataFrame(list(G.edges(data=True)), columns=['Source', 'Target', 'Attributes'])
if not edges_df.empty and isinstance(edges_df['Attributes'].iloc[0], dict):
    attributes_df = edges_df['Attributes'].apply(pd.Series)
    edges_df = pd.concat([edges_df.drop(columns=['Attributes']), attributes_df], axis=1)
{% endhighlight %}

<p></p>
Convert edges to dgl format:
<p></p>

{% highlight python %}
import dgl
import dgl.nn as dglnn
import dgl.data
from dgl.data import DGLDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
unpickEdges=edges_df
edge_index=torch.tensor(unpickEdges[['Source','Target']].T.values)
u,v=edge_index[0],edge_index[1]
g=dgl.graph((u,v))
g=dgl.add_self_loop(g)
g
Graph(num_nodes=50, num_edges=268,
      ndata_schemes={}
      edata_schemes={})
{% endhighlight %}
<p></p>

<h4>Transform Text to Vectors</h4>
<p></p>

<p>This section describes how text data, specifically the artist biographies, is transformed into vector embeddings to be used as node features in a graph. We utilized the <code>'all-MiniLM-L6-v2'</code> model from Hugging Face for this purpose, which generates high-quality sentence embeddings.</p>
<ul>
  <li>
    <strong>Model Selection:</strong> The <code>'all-MiniLM-L6-v2'</code> model, a lightweight yet powerful transformer, was used for efficient text-to-vector translation.
  </li>
  <li>
    <strong>Embedding Creation:</strong> Artist biographies from the DataFrame are encoded into a tensor of size <code>[50, 384]</code>, where each row represents a 384-dimensional vector for an artist.
  </li>
  <li>
    <strong>Assigning Features:</strong> These embeddings are assigned as node features to the graph using the <code>ndata['feat']</code> attribute.
  </li>
</ul>

<p></p>
The Python code below demonstrates the embedding process and assignment to the graph:
<p></p>

{% highlight python %}
model = SentenceTransformer('all-MiniLM-L6-v2')
node_embeddings = modelST.encode(artists_df['bio'],convert_to_tensor=True)
node_embeddings = node_embeddings.to(torch.device('cpu'))
gNew.ndata['feat'] = node_embeddings
node_embeddings.shape
torch.Size([50, 384])
{% endhighlight %}
<p></p>

Input graph
<p></p>
{% highlight python %}
g
Graph(num_nodes=50, num_edges=268,
      ndata_schemes={'feat': Scheme(shape=(384,), dtype=torch.float32)}
      edata_schemes={})
{% endhighlight %}
<p></p>

<p></p>
<h4>GNN Link Prediction Model Training</h4>
<p>This subsection describes the process of training a GNN link prediction model, using code adapted from the DGL library. The focus is on leveraging the built-in functionalities for implementing and training GNN-based link prediction tasks.</p>
<p>The training pipeline includes constructing positive and negative edge graphs for the training and testing phases, defining the GNN architecture, and training the model using standard techniques. The training concludes with an evaluation of the model's performance using the AUC (Area Under the Curve) metric.</p>

<p></p>


{% highlight python %}
u, v = g.edges()
eids = np.arange(g.number_of_edges())
eids = np.random.permutation(eids)
test_size = int(len(eids) * 0.1)
train_size = g.number_of_edges() - test_size
test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]
adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
neg_u, neg_v = np.where(adj_neg != 0)
neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
train_g = dgl.remove_edges(g, eids[:test_size])
{% endhighlight %}
<p></p>


<p></p>

<p></p>

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
{% highlight python %}
import dgl.function as fn
train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())
test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())
class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return g.edata['score'][:, 0]
{% endhighlight %}

<p></p>
<p></p>
{% highlight python %}
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


{% highlight python %}
model = GraphSAGE(train_g.ndata['feat'].shape[1], 128)
pred = MLPPredictor(128)
def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)
def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)
optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)
{% endhighlight %}

<p></p>
<p></p>
{% highlight python %}
all_logits = []
for e in range(100):
    h = model(train_g, train_g.ndata['feat'].float())
    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    loss = compute_loss(pos_score, neg_score)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if e % 5 == 0:
        print('In epoch {}, loss: {}'.format(e, loss))
{% endhighlight %}

<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/artPaint2.jpg" alt="Post Sample Image" width="333" >
</a>

<p></p>
<p></p>
Final evaluation:
<p></p>
{% highlight python %}
from sklearn.metrics import roc_auc_score
with torch.no_grad():
    pos_score = pred(test_pos_g, h)
    neg_score = pred(test_neg_g, h)
    print('AUC', compute_auc(pos_score, neg_score))
AUC 0.9171597633136095
{% endhighlight %}
<p></p>

<p>To ensure the embedded vectors are preserved and accessible across sessions, they are saved to Google Drive as PyTorch tensors. This allows seamless reuse without recomputing the embeddings, saving time and computational resources.</p>
<p></p>
{% highlight python %}
import torch
torch.save(h, '/content/drive/My Drive/Art/artists_h.pt')
load_h = torch.load('/content/drive/My Drive/Art/artists_h.pt')
{% endhighlight %}
<p></p>





<p></p>



<p></p>



<p></p>




<h3>Unified Knowledge Graph</h3>
<p></p>
<h4>Combining Artist and Painting Graphs into a Unified Knowledge Graph</h4>
<p>We started by constructing two separate graphs:</p>
<ul>
  <li>
    <strong>Artist Graph:</strong> Artist biographies were converted into vectors of size <code>384</code> using a transformer model, followed by GNN link prediction, which transformed these embeddings into vectors of size <code>128</code>.
  </li>
  <li>
    <strong>Painting Graph:</strong> Paintings were represented as vectors of size <code>2048</code> using a pre-trained CNN model, followed by GNN link prediction, which transformed these embeddings into vectors of size <code>128</code>.
  </li>
</ul>
<p><strong>Next Step:</strong> We will combine the Artist graph and Painting graph into a unified Knowledge Graph. This Knowledge Graph (suggested name: <code>ArtKnowledgeGraph</code>) will map painting nodes to their corresponding artist nodes. The node features for this unified graph will be represented by vectors of size <code>128</code>, ensuring a consistent embedding space across all node types.</p>

<p></p>

<p></p>
<p></p>

<p>To create a unified Knowledge Graph, we need to ensure that nodes from both the Artist graph and the Painting graph have unique indices. This adjustment is achieved by offsetting the indices of the Artist nodes:</p>
<ul>
  <li>
    <strong>Unique indexing:</strong> Since the Painting graph contains 1000 nodes (indexed from <code>0</code> to <code>999</code>), we add an offset of <code>1000</code> to the indices of the Artist nodes. This ensures that Artist node indices start from <code>1000</code> and do not overlap with Painting node indices.
  </li>
  <li>
    <strong>Prepare adjusted Artist nodes:</strong> After offsetting the indices, the Artist nodes are prepared with the updated <code>nodeIdx</code> and their corresponding names. These are then included in the unified Knowledge Graph.
  </li>
</ul>
<p></p>
{% highlight python %}
artist_graph_nodes = pd.read_csv("/content/drive/My Drive/Art/artist_graph_nodes.csv")
artist_graph_edges = pd.read_csv("/content/drive/My Drive/Art/artist_graph_edges.csv")
artist_graph_nodes['Node'] = artist_graph_nodes['nodeIdx'] + 1000
artist_graph_nodes = artist_graph_nodes[['Node', 'name']]
artist_graph_edges['Source'] = artist_graph_edges['Source'] + 1000
artist_graph_edges['Target'] = artist_graph_edges['Target'] + 1000
{% endhighlight %}
<p></p>

<p></p>
Knowledge Graph nodes: head and tail
<a href="#">
    <img src="{{ site.baseurl }}/img/artPaint8.jpg" alt="Post Sample Image" width="499">
</a>
<p></p>


Painting nodes are prepared for the unified Knowledge Graph by mapping each painting to its corresponding artist and assigning unique artist indices.
<p></p>
<ul>
  <li>
    Load painting nodes:The painting graph nodes are loaded from a CSV file, and the column <code>file_name</code> is renamed to <code>name</code> for consistency.
  </li>
  <li>
    Map paintings to artists: A new column, <code>artist_name</code>, is created by extracting the artist's name from the painting's file name. This assumes the artist's name is the first two words in the file name, separated by underscores.
  </li>
  <li>
    Assign unique artist indices: An <code>artist_index</code> column is created by dividing the painting node index (<code>Node</code>) by 20 (integer division) and adding an offset of <code>1000</code>. This ensures unique indexing for artists across the graph.
  </li>
</ul>
<p></p>

{% highlight python %}
painting_graph_nodes = pd.read_csv("/content/drive/My Drive/Art/painting_graph_nodes.csv")
painting_graph_nodes = painting_graph_nodes.rename(columns={'file_name': 'name'})
painting_artist_graph_nodes=painting_graph_nodes
painting_artist_graph_nodes['artist_name'] = painting_graph_nodes['name']
   .apply(lambda x: x.split('_')[0]+' '+x.split('_')[1] )
painting_artist_graph_nodes['artist_index'] = painting_graph_nodes['Node']
   .apply(lambda x: (x // 20)+1000)

{% endhighlight %}

<p></p>


<p></p>
<p></p>
{% highlight python %}
xxx
{% endhighlight %}
<p></p>
<p></p>

In this post we demonstrated how to use transformers and GNN link predictions to rewire knowledge graphs.
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/artPaint11.jpg" alt="Post Sample Image" width="234" height="314">
</a>
<p></p>
{% highlight python %}
artist_graph_nodes['nodeIdx'] = artist_graph_nodes['nodeIdx'] + 1000
artist_graph_nodes = artist_graph_nodes[['nodeIdx', 'name']]
{% endhighlight %}
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/artPaint12.jpg" alt="Post Sample Image" width="234" height="314">
</a>
<p></p>
{% highlight python %}
artist_graph_edges['Source'] = artist_graph_edges['Source'] + 1000
artist_graph_edges['Target'] = artist_graph_edges['Target'] + 1000
artist_graph_edges = artist_graph_edges[['Source', 'Target']]
{% endhighlight %}
<p></p>
<p></p>
{% highlight python %}
painting_artist_graph_nodes=painting_graph_nodes
painting_artist_graph_nodes['artist_name'] =
    painting_graph_nodes['name'].apply(lambda x: x.split('_')[0]+' '+x.split('_')[1] )
painting_artist_graph_nodes['artist_index'] =
    painting_graph_nodes['Node'].apply(lambda x: (x // 20)+1000)

painting_graph_nodes.tail()
{% endhighlight %}
<p></p>

<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/artPaint13.jpg" alt="Post Sample Image" width="444" height="500">
</a>

<p>To establish connections between painting nodes and their corresponding artist nodes, edges are created by matching paintings to artists based on their indices:</p>
<ul>
  <li><strong>Find matching artist nodes:</strong> For each painting node, the <code>artist_index</code> is used to identify the matching artist node in <code>artist_graph_nodes</code>.</li>
  <li><strong>Create edges:</strong> For each match, an edge is added between the painting node and the artist node. These edges represent the relationship between a painting and its creator.</li>
  <li><strong>Convert edges to DataFrame:</strong> The resulting edges are stored in a DataFrame with <code>Source</code> and <code>Target</code> columns, representing the painting and artist nodes, respectively.</li>
</ul>
<p>The following code implements this process:</p>
<p></p>
{% highlight python %}
edges = []
for _, painting_row in painting_artist_graph_nodes.iterrows():
    artist_matches = artist_graph_nodes[artist_graph_nodes['nodeIdx'] ==
        painting_row['artist_index']]
for _, artist_row in artist_matches.iterrows():
    edges.append((painting_row['Node'], artist_row['nodeIdx']))
painting_artist_graph_edges = pd.DataFrame(edges, columns=['Source', 'Target'])
{% endhighlight %}

<p></p>
The unified Knowledge Graph is created by combining nodes and edges from the Painting and Artist graphs. Painting nodes and Artist nodes are concatenated into a single DataFrame, ensuring all nodes are represented. Similarly, edges between paintings and their artists (painting_artist_graph_edges) and edges within the Artist graph (artist_graph_edges) are concatenated into a single DataFrame of edges. The resulting Knowledge Graph contains 1218 edges, ready for further graph-based processing and analysis.
<p></p>
{% highlight python %}
knowledge_graph_nodes = pd.concat([painting_nodes, artist_nodes], ignore_index=True)
knowledge_graph_edges = pd.concat([painting_artist_graph_edges, artist_graph_edges],
    ignore_index=True)
knowledge_graph_edges.shape
(1218, 2)
{% endhighlight %}
<p></p>


<h4>Transforming the Unified Graph to DGL Format</h4>
<p></p>
<p></p>
<p></p>
The Knowledge Graph is constructed as a DGL graph object. The edges from knowledge_graph_edges are transformed into a PyTorch tensor to define the connections between nodes. These edges are used to create a DGL graph, and self-loops are added to ensure that each node is connected to itself, which can improve GNN performance. The node features are represented by knowledge_graph_embeddings, with each node having a 128-dimensional feature vector. The resulting graph contains 1050 nodes and 2268 edges, making it ready for graph-based learning tasks.

<p></p>
{% highlight python %}
unpickEdges=knowledge_graph_edges
edge_index=torch.tensor(unpickEdges[['Source','Target']].T.values)
u,v=edge_index[0],edge_index[1]
gKG=dgl.graph((u,v))
gKG=dgl.add_self_loop(gKG)
gKG.ndata['feat'] = knowledge_graph_embeddings
gKG
Graph(num_nodes=1050, num_edges=2268,
      ndata_schemes={'feat': Scheme(shape=(128,), dtype=torch.float32)}
      edata_schemes={})
{% endhighlight %}
<p></p>

<p></p>
<h4>GNN Link Prediction Model for Unified Knowledge Graph</h4>

<p></p>
<p>The final phase of training involves running the GNN link prediction model for 2000 epochs. During each epoch, the model performs a forward pass to compute node embeddings, calculates scores for positive and negative edges, and computes the loss. Gradients are backpropagated to update model parameters, optimizing the model for link prediction. Progress is logged every 100 epochs, showing the loss to monitor convergence.</p>
<p>After training, the model's performance is evaluated on the test set. Positive and negative edge scores are computed, and the AUC (Area Under the Curve) metric is used to assess the model's ability to distinguish between true and false links. The final AUC score achieved is <code>0.9007</code>, indicating strong predictive performance.</p>

<p></p>
{% highlight python %}
all_logits = []
for e in range(2000):
    # forward
    h = model(train_g, train_g.ndata['feat'].float())
    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    loss = compute_loss(pos_score, neg_score)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if e % 100 == 0:
        print('In epoch {}, loss: {}'.format(e, loss))
{% endhighlight %}
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/artPaint10.jpg" alt="Post Sample Image" width="374">
</a>
<p></p>
{% highlight python %}
from sklearn.metrics import roc_auc_score
with torch.no_grad():
    pos_score = pred(test_pos_g, h)
    neg_score = pred(test_neg_g, h)
    print('AUC', compute_auc(pos_score, neg_score))
AUC 0.9007130791642577
{% endhighlight %}
<p></p>
<p></p>
<p>The node embeddings for the unified Knowledge Graph are saved and reloaded to enable persistent storage and reuse:</p>
<ul>
  <li><strong>Save embeddings:</strong> The node embeddings, represented by the variable <code>h</code>, are saved to Google Drive as a PyTorch tensor using <code>torch.save</code>. This ensures that the embeddings can be preserved for later use.</li>
  <li><strong>Load embeddings:</strong> The saved embeddings are reloaded from Google Drive using <code>torch.load</code>. This allows for seamless reuse of the computed embeddings without requiring recomputation.</li>
</ul>
<p></p>
<p></p>
{% highlight python %}
torch.save(h, '/content/drive/My Drive/Art/artist_knowledge_graph_h.pt')
load_h = torch.load('/content/drive/My Drive/Art/artist_knowledge_graph_h.pt')
{% endhighlight %}
<p></p>
<h2>Interpreting GNN Link Prediction Model Results</h2>

<p>The results of the GNN Link Prediction (LP) model go beyond merely predicting links; they generate <em>embedded vectors</em> for each node in the Unified Knowledge Graph. These vectors, consistent in size, provide high-dimensional representations of the nodes, capturing both contextual and relational information. Such embeddings enable advanced analysis using techniques like <em>cosine similarity</em>, <em>clustering</em>, and <em>building new graph layers</em>, offering deeper insights into the structure and relationships within the graph.</p>

<p>The Unified Knowledge Graph in this study consists of <strong>50 artist nodes</strong> and <strong>1000 painting nodes</strong>, all encoded as vectors of size 128. The GNN LP model ensures a consistent embedding space across all nodes, enabling seamless analysis and comparisons between different node types.</p>

<p>In this section, we explore two main areas of interpretation:</p>
<ul>
    <li><strong>Artist-artist relationships:</strong> Examining similarities in the embedded vectors to identify shared themes, influences, or overlapping artistic philosophies.</li>
    <li><strong>Painting-painting connections:</strong> Analyzing stylistic or contextual similarities between paintings through their embeddings.</li>
</ul>

By focusing on embeddings rather than direct link predictions, this study highlights the ability of GNN models to uncover nuanced patterns and provide a deeper understanding of the complex relationships within the art world.

<p></p>
<h3>Model Results Analysis</h3>
<p></p>
In this step, we calculate the cosine similarity between all pairs of nodes in the Unified Knowledge Graph using their embedded vectors. The similarity scores, generated by a PyTorch function, quantify the relationships between nodes based on their high-dimensional embeddings.
<p></p>
The result is stored in a DataFrame containing 1,102,500 rows, representing all possible pairs of 1,050 nodes (50 artists and 1,000 paintings). Each row includes the indices of the two nodes and their corresponding similarity score, which serves as the foundation for analyzing relationships between nodes in the graph.
<p></p>
{% highlight python %}
cosine_scores_gnn = pytorch_cos_sim(load_h, load_h)
pairs_scores = []
for i in range( len(cosine_scores_gnn)):
    for j in range(len(cosine_scores_gnn)):  
        pairs_scores.append({
            'idx1': i,
            'idx2': j,
            'score': cosine_scores_gnn[i][j].item()  # Use `.item()` for scalar tensors
        })

df=pd.DataFrame(pairs_scores)
df.shape
(1102500, 3)
{% endhighlight %}
<p></p>
<p></p>
<p>This step enhances the similarity DataFrame by joining node information from the Unified Knowledge Graph. For each pair of indices (<code>idx1</code> and <code>idx2</code>), node metadata such as names and types are added, providing contextual information about the nodes involved in each similarity computation.</p>

<p>Key steps include:</p>
<ul>
    <li><strong>Joining node information:</strong> The <code>idx1</code> and <code>idx2</code> columns are matched with the <code>nodeIdx</code> column in the Knowledge Graph nodes DataFrame, adding relevant details about each node.</li>
    <li><strong>Renaming columns:</strong> After the join, node metadata (<code>nodeName</code> and <code>nodeType</code>) is renamed to <code>node1_name</code>, <code>node1_type</code>, <code>node2_name</code>, and <code>node2_type</code> for clarity.</li>
    <li><strong>Dropping redundant columns:</strong> Extra columns from the join (<code>nodeIdx_x</code> and <code>nodeIdx_y</code>) are removed, keeping the DataFrame clean and focused on relevant information.</li>
</ul>

<p>The resulting DataFrame includes enriched details about each pair of nodes, laying the groundwork for deeper analysis of relationships within the graph.</p>


<p></p>
{% highlight python %}
df = df.merge(knowledge_graph_nodes, left_on='idx1', right_on='nodeIdx', how='left')
df = df.rename(columns={'nodeName': 'node1_name', 'nodeType': 'node1_type'})
df = df.merge(knowledge_graph_nodes, left_on='idx2', right_on='nodeIdx', how='left')
df = df.rename(columns={'nodeName': 'node2_name', 'nodeType': 'node2_type'})
df = df.drop(columns=['nodeIdx_x', 'nodeIdx_y'])
graph_pairs = df
{% endhighlight %}
<p></p>

<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/artPaint20.jpg" alt="Post Sample Image" width="555">
</a>
<p></p>

<h4>Cosine Similarity Score Distributions</h4>

<p>The GNN Link Prediction model generates similarity scores for each pair of nodes in the Unified Knowledge Graph. These scores represent the relationships between nodes and provide insights into the structure and connections within the graph.</p>


<p>For all node pairs, excluding self-loops, scores range from -0.515 to 1.000, with a mean of approximately 0.460. This indicates that many pairs exhibit moderate similarity, while higher scores highlight stronger relationships.</p>
<p></p>
{% highlight python %}
filtered_graph_pairs = graph_pairs[graph_pairs['idx1'] != graph_pairs['idx2']]
overall_scores = filtered_graph_pairs['score']
overall_stats = overall_scores.describe()
print("Overall Score Distribution:\n", overall_stats)
Overall Score Distribution:
 count   1101450.000000
mean          0.460064
std           0.337415
min          -0.515314
25%           0.217390
50%           0.471504
75%           0.744774
max           1.000000
{% endhighlight %}
<p></p>
<p>Artist-artist pairs show a higher mean score of 0.621, reflecting strong semantic relationships such as shared influences or overlapping styles.
<p></p>

<p></p>
{% highlight python %}
artist_artist_pairs = filtered_graph_pairs[
    (filtered_graph_pairs['node1_type'] == 'artist') & (filtered_graph_pairs['node2_type'] == 'artist')
]
artist_artist_scores = artist_artist_pairs['score']
artist_stats = artist_artist_scores.describe()
print("Artist-Artist Score Distribution:\n", artist_stats)
Artist-Artist Score Distribution:
 count    2450.000000
mean        0.621267
std         0.291015
min        -0.281765
25%         0.455145
50%         0.688079
75%         0.860471
max         0.992316
{% endhighlight %}
<p></p>



In contrast, painting-painting pairs have a mean score of 0.477, demonstrating diverse visual and contextual connections between artworks.</p>
<p></p>
{% highlight python %}
painting_painting_pairs = filtered_graph_pairs[
    (filtered_graph_pairs['node1_type'] == 'painting') & (filtered_graph_pairs['node2_type'] == 'painting')
]
painting_scores = painting_painting_pairs['score']
painting_stats = painting_scores.describe()
print("Painting-Painting Score Distribution:\n", painting_stats)
Painting-Painting Score Distribution:
 count    999000.000000
mean          0.477173
std           0.338798
min          -0.515314
25%           0.232907
50%           0.492818
75%           0.769807
max           1.000000
{% endhighlight %}
<p></p>


<p>These distributions reveal the model's ability to capture meaningful relationships across different types of nodes, setting the stage for deeper analysis of highly connected and less connected nodes. Such insights are valuable for exploring relationships in art and uncovering hidden patterns.</p>
<p></p>
<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/artPaint21.jpg" alt="Post Sample Image" width="711">
</a>
<p></p>
<p>This analysis explores the distribution of cosine similarity scores for painting-painting pairs. The scores are grouped into intervals ranging from -1.0 to 1.0 in steps of 0.1. Data is filtered to include only painting-painting connections, and a histogram visualizes the frequency of pairs within each interval. The results highlight a concentration of pairs in higher intervals (e.g., <code>0.9-1.0</code>), suggesting strong stylistic or thematic similarities between many paintings. In contrast, lower intervals reveal fewer connections, emphasizing weaker or negative correlations.</p>

<p></p>
{% highlight python %}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
bins = np.arange(-1.0, 1.1, 0.1)
bins[-1] = bins[-1] + 0.001
painting_painting_pairs = filtered_graph_pairs[
    (filtered_graph_pairs['node1_type'] == 'painting') &
    (filtered_graph_pairs['node2_type'] == 'painting')
]
frequency, bin_edges = np.histogram(painting_painting_pairs['score'], bins=bins)
intervals = [f"{round(bin_edges[i], 1)}-{round(bin_edges[i + 1], 1)}" for i in range(len(bin_edges) - 1)]
freq_df = pd.DataFrame({'Interval': intervals, 'Frequency': frequency})
freq_df = freq_df.iloc[::-1]
print(freq_df)
plt.bar(freq_df['Interval'], freq_df['Frequency'], color='skyblue', alpha=0.7)
plt.xticks(rotation=45)
plt.title("Cosine Similarity Distribution for Painting-Painting Pairs")
plt.xlabel("Cosine Similarity Intervals")
plt.ylabel("Frequency")
plt.gca().invert_xaxis()  # Invert x-axis for descending order
plt.grid(axis='y')
plt.tight_layout()
plt.show()
{% endhighlight %}
<p></p>
<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/artPaint24.jpg" alt="Post Sample Image" width="500">
</a>
<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/artPaint22.jpg" alt="Post Sample Image" width="654">
</a>
<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/artPaint23.jpg" alt="Post Sample Image" width="654">
</a>
<p></p>

<p></p><p></p>


<p></p><p></p>


<p></p>

<p></p>
<h3>Painting-Painting Connections</h3>
<p>This section focuses on analyzing connections between paintings based on their similarity scores, emphasizing relationships that span across different artists. By filtering and enriching the DataFrame, we can better understand how paintings from various artists relate to one another.</p>

<p>Key steps include:</p>
<ul>
    <li><strong>Filtering painting-painting pairs:</strong> Rows where both nodes are of type <code>painting</code> are extracted, creating a subset of the data that exclusively examines relationships between paintings.</li>
    <li><strong>Extracting artist names:</strong> For each painting, the artist’s name is derived from the file name by splitting the string and capturing the relevant segment. The artist names are stored in new columns <code>artist1</code> and <code>artist2</code>.</li>
    <li><strong>Cross-artist connections:</strong> Connections between paintings created by different artists are identified by filtering for pairs where <code>artist1</code> is not equal to <code>artist2</code>. Additionally, only unique pairs are retained by ensuring <code>idx1 &lt; idx2</code>.</li>
</ul>
<p></p>
The resulting DataFrame highlights cross-artist painting relationships, providing insights into stylistic or thematic similarities between works by different creators.</p>

<p></p>
{% highlight python %}
painting_pairs=df[(df['node1_type']=='painting') & (df['node2_type']=='painting')]
painting_pairs['artist1']=painting_pairs['node1_name'].apply(lambda x: x.split('_')[-2])
painting_pairs['artist2']=painting_pairs['node2_name'].apply(lambda x: x.split('_')[-2])
dfp=painting_pairs

{% endhighlight %}
<p></p>
<p></p>
<h4>Cross-Artist Painting Connections Analysis</h4>
<p>
The analysis focuses on identifying connections between paintings created by different artists, highlighting stylistic or thematic similarities. By filtering pairs of paintings where the creators are distinct, and ensuring unique combinations, we analyzed a total of <strong>490,000 cross-artist connections</strong>. The top-scoring pairs based on cosine similarity are presented below, offering insights into overlapping influences and shared artistic elements.
</p>
<p></p>
{% highlight python %}
diffArtistPaintings= dfp[(dfp['artist1']!=dfp['artist2']) & (dfp['idx1'] < dfp['idx2'])]
diffArtistPaintings.shape : (490000, 9)
diffArtistPaintings.sort_values(by='score').tail(9)
{% endhighlight %}
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/artPaint16.jpg" alt="Post Sample Image" width="555">
</a>
<p></p>
Observations:
<ul>
  <li><strong>Stylistic Overlaps:</strong> High similarity scores suggest shared stylistic elements or thematic connections between paintings, even from different movements.</li>
  <li><strong>Artistic Influence:</strong> These links may indicate underlying inspirations or techniques shared across eras or artistic philosophies.</li>
  <li><strong>Applications:</strong> The insights can support art recommendations, historical analyses, or new narratives in understanding artistic relationships.</li>
</ul>
<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/artPaint19.jpg" alt="Post Sample Image" width="777">
</a>
<p></p>
<h4>Exploring Painting-Painting Connections Within the Same Artist</h4>
<p>This table highlights painting-painting connections within the works of the same artist, showcasing the least similar pairs based on cosine similarity scores derived from the GNN embeddings.</p>

<p></p>
<p></p>
<p></p>
<p></p>
{% highlight python %}
artistPaintings= dfp[(dfp['artist1']==dfp['artist2']) & (dfp['idx1'] < dfp['idx2'])]
artistPaintings.sort_values(by='score').head(11)
{% endhighlight %}
<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/artPaint17b.jpg" alt="Post Sample Image" width="555">
</a>
<p></p>
Key Observations
<ul>
  <li><strong>Low Cosine Similarity Within Same Artist's Works:</strong>
    <ul>
      <li><em>Hieronymus Bosch</em> has the lowest similarity score (0.2722), indicating significant variation between the chosen paintings, likely due to differences in themes or styles.</li>
      <li><em>Pierre-Auguste Renoir</em> and <em>Edouard Manet</em> also have scores near 0.29, reflecting diversity in artistic elements.</li>
    </ul>
  </li>
  <p></p>
  <a href="#">
      <img src="{{ site.baseurl }}/img/artPaint18.jpg" alt="Post Sample Image" width="777">
  </a>
  <p></p>
  <li><strong>Mikhail Vrubel Dominates:</strong> The table features multiple pairs from <em>Mikhail Vrubel</em>, with scores ranging from 0.2952 to 0.3195, highlighting his stylistic versatility.</li>
  <p></p>
  <a href="#">
      <img src="{{ site.baseurl }}/img/artPaint31.jpg" alt="Post Sample Image" width="579">
  </a>
  This visualization highlights the painting "Lilac" by Mikhail Vrubel, which exhibits a low cosine similarity (<0.32) to his other works. This deviation offers insight into how specific pieces can stand apart from an artist's typical style, enriching our understanding of their creative range and versatility.
  <p></p>
</ul>




<p></p>
<p>The low similarity scores between paintings of the same artist emphasize <strong>stylistic versatility</strong> or exploration within their body of work. These results provide valuable insights into the evolution and experimentation of individual artists, offering a deeper understanding of their creative journeys.</p>



<p></p>
<h3>Artist-Artist Connections</h3>

<p></p>
This analysis explores the relationships between artists by categorizing their connections based on cosine similarity scores and metadata such as shared genres or nationalities.

<p></p>
First, pairs of artist nodes were isolated and merged with existing edge data to identify relationships such as shared genres or nationalities. Missing values in the <code>genre</code> and <code>nationality</code> columns were replaced with "None" to standardize the dataset for further processing.</p>


<p></p>
{% highlight python %}
artist_pairs=df[(df['node1_type']=='artist') & (df['node2_type']=='artist')]
combined_df['genre'] = combined_df['genre'].fillna('None')
combined_df['nationality'] = combined_df['nationality'].fillna('None')
combined_df = pd.merge(
    artist_pairs,
    artist_edges,
    on=['idx1', 'idx2'],
    how='left'
)
{% endhighlight %}
<p></p>
<p></p>
<p></p>

<p>To better understand the nature of artist connections, a new column, <code>edge_type</code>, was introduced to classify relationships into the following categories:</p>
<ul>
  <li><strong>Genre:</strong> When artists share a common artistic style or movement.</li>
  <li><strong>Nationality:</strong> When artists belong to the same country.</li>
  <li><strong>Both:</strong> When artists share both genre and nationality.</li>
  <li><strong>None:</strong> When no direct connection exists in the provided metadata.</li>
</ul>

<p>This categorization offers valuable insights into the nature of artist-artist relationships, helping uncover patterns and anomalies within the graph.</p>
<p></p>
{% highlight python %}
def determine_edge_type(row):
    if row['genre'] != 'None' and row['nationality'] != 'None':
        return 'both'
    elif row['genre'] != 'None':
        return 'genre'
    elif row['nationality'] != 'None':
        return 'nationality'
    else:
        return 'none'
combined_df['edge_type'] = combined_df.apply(determine_edge_type, axis=1)
combined_df[combined_df['edge_type']!='none'].tail(11)
{% endhighlight %}
<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/artPaint25.jpg" alt="Post Sample Image" width="734">
</a>
<p></p>
<p></p>
{% highlight python %}
stats = combined_df.groupby('edge_type')['score']
  .agg(['mean', 'median', 'min', 'max', 'std']).reset_index()
print("Statistics by Edge Type:")
print(stats)
Statistics by Edge Type:
     edge_type      mean    median       min       max       std
0         both  0.751972  0.790473  0.345487  0.989279  0.210772
1        genre  0.817520  0.864702  0.260232  0.978745  0.149252
2  nationality  0.719641  0.791969  0.052798  0.992316  0.245162
3         none  0.617343  0.685730 -0.281765  1.000000  0.296322
{% endhighlight %}
<p></p>
To better understand the relationship between edge types and their corresponding similarity scores, we use a boxplot. The plot is arranged in the following order: both, genre, nationality, and none, ensuring consistency and clarity. A uniform light gray color is applied to all boxes for a clean and professional appearance.

<p></p>
{% highlight python %}
plt.figure(figsize=(8, 5))
order = ['both', 'genre', 'nationality', 'none']  # Specify the desired order
sns.boxplot(x='edge_type', y='score', data=combined_df, order=order, color='lightgray')  # Use single color
plt.title("Score Distribution by Edge Type", fontsize=14)
plt.xlabel("Edge Type", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.show()
{% endhighlight %}
<p></p>

This visualization provides insights into how different edge types (e.g., genre-based or nationality-based connections) influence the similarity scores among artists in the graph.

<a href="#">
    <img src="{{ site.baseurl }}/img/artPaint26.jpg" alt="Post Sample Image" width="474">
</a>
<p></p>

<p>In this analysis, we focus on understanding the relationships between artist nodes by exploring similarity scores for different edge types. By examining both high and low scores, we gain insights into hidden connections and variations among artists.</p>


<p></p>

<h4>Low Scores with 'Both'</h4>
<p>Even when artists share both genre and nationality, low similarity scores reveal nuanced differences. These pairs demonstrate the diversity that can exist even within closely related artistic and cultural contexts.</p>
<p></p>
{% highlight python %}
df=combined_df
low_scores_both = df[(df['edge_type'] == 'both') & (df['idx2'] > df['idx1'])]
print("Low scores with 'genre' edge type:")
low_scores_both.sort_values(by='score').head(7)
{% endhighlight %}
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/artPaint30.jpg" alt="Post Sample Image" width="579">
</a>
<p></p>
With the exception of Titian and Leonardo da Vinci, all pairs are French artists from Impressionism or Post-Impressionism, yet their low cosine similarity scores highlight nuanced stylistic differences within these closely related movements.

<p></p>


<p></p>
<h4>Low Scores with 'Nationality'</h4>
<p>Artists sharing the same nationality but exhibiting weak similarities highlight diversity within a shared cultural framework. These cases may represent unique interpretations or individualistic approaches despite shared backgrounds.</p>

<p></p>
{% highlight python %}
low_scores_nationality = df[(df['edge_type'] == 'nationality') & (df['idx2'] > df['idx1'])]
print("Low scores with 'nationality' edge type:")
low_scores_nationality.sort_values(by='score').head(11)
{% endhighlight %}

<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/artPaint29.jpg" alt="Post Sample Image" width="579">
</a>
<p></p>
The low cosine similarity scores between artists of the same nationality often reflect differences in time periods and artistic movements. For example, Rembrandt and Piet Mondrian represent vastly different eras—17th-century Baroque and 20th-century Abstract art—while Camille Pissarro and Eugene Delacroix belong to distinct movements like Impressionism and Romanticism. These temporal and stylistic disparities highlight the diversity within shared cultural backgrounds.
<p></p>


<p></p>
<h4>Low Scores with 'Genre'</h4>
<p>For artist pairs connected by the same genre but showing low similarity scores, we explore how artists within a single genre can differ significantly in their styles, techniques, or approaches to art.</p>

<p></p>
{% highlight python %}
low_scores_genre = df[(df['edge_type'] == 'genre') &  (df['idx2'] > df['idx1'])]
print("Low scores with 'genre' edge type:")
low_scores_genre.sort_values(by='score').head()
{% endhighlight %}
<p></p>
<p></p>

<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/artPaint28.jpg" alt="Post Sample Image" width="579">
</a>
<p></p>
For artists connected by the same genre, even the lowest cosine similarity scores are relatively high, indicating a baseline level of shared characteristics. However, these pairs still highlight stylistic diversity within the same genre, such as the contrasting approaches of Eugene Delacroix and William Turner in Romanticism or the differences between Albrecht Dürer and Hieronymus Bosch in the Northern Renaissance.


<p></p>

<h4>High Scores with 'None'</h4>
<p>Artists with no shared genre or nationality but high similarity scores reveal unexpected connections. These relationships may indicate shared influences, overlapping themes, or stylistic similarities that are not captured by explicit attributes.</p>


<p></p>
{% highlight python %}
high_scores_none = df[(df['edge_type'] == 'none') & (df['idx2'] > df['idx1'])]
print("High scores with 'none' edge type:")
high_scores_none.sort_values(by='score',ascending=False).head(7)
{% endhighlight %}
<p></p>

<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/artPaint27.jpg" alt="Post Sample Image" width="579">
</a>
<p></p>
These high scores, despite no overlaps in genre or nationality, suggest deeper, unrecognized influences or thematic and stylistic parallels that transcend traditional classifications, warranting further exploration. One such connection is between Pablo Picasso (1881–1973), the Spanish pioneer of Cubism, and Mikhail Vrubel (1856–1910), a Russian Symbolist painter. Although they lived in different periods and represented contrasting artistic movements, a historical link bridges their work. In 1906, Sergey Diaghilev brought Vrubel’s works to Paris, where they captured the admiration of a young Pablo Picasso.
<p></p>

<p></p>



<p></p>
{% highlight python %}
if pair1=[leftWord1, rightWord1],
   pair2=[leftWord2, rightWord2]
   and rightWord1=leftWord2,
then there is edge12={pair1, pair2}

{% endhighlight %}




<h2>Conclusion</h2>

<p>
This study presented a Unified Knowledge Graph framework that combines text and image data to explore artistic relationships. By utilizing Graph Neural Networks (GNNs) for embedding and link prediction, we uncovered meaningful connections within and across artist and painting nodes, revealing both expected patterns and surprising insights.
</p>

<p>
<strong>Key findings include:</strong>
</p>

<ul>
  <li><strong>Cross-Artist Painting Connections:</strong> Identified thematic overlaps and stylistic diversity between works created by different artists.</li>
  <li><strong>Artist-Specific Variations:</strong> Highlighted the evolution and experimentation within individual artists' portfolios, showcasing their creative range.</li>
  <li><strong>Unexpected Influences:</strong> Discovered surprising connections, such as historical links between Pablo Picasso and Mikhail Vrubel, demonstrating the broader potential of this approach.</li>
</ul>

<p>
The Unified Knowledge Graph provides a powerful tool for analyzing complex datasets, offering fresh perspectives on art history. It also paves the way for applications in recommendation systems, classification tasks, and cultural analysis. Future work will focus on expanding the graph with additional data types and improving model interpretability.
</p>
