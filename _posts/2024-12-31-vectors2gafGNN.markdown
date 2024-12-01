---
layout:     post
title:      "Pre-final222 Vector Caching in Graph Classification"
subtitle:   "Enhancing Graph Neural Networks: Introducing Pre-final Vector Caching in Graph Classification"
date:       2024-12-31 12:00:00
author:     "Melenar"
header-img: "img/vectorPage6b.jpg"
---
<p><h3> Introduction222</h3>
<p></p>
Linear algebra is indispensable in the world of machine learning and artificial intelligence, primarily because it provides a way to efficiently represent and manipulate data. Whether we're dealing with matrices or vectors, these mathematical structures help us model complex problems in a computationally manageable form. In recent years, the surge in popularity of deep learning models has highlighted the versatility of linear algebra across various domains. These models often require data to be transformed into vectors.
<p></p>
Whether it's images, audio, text, or data from social networks, converting different types of information into a consistent vector format allows deep learning algorithms to process and analyze it effectively. This transformation is not just a technical necessity but a gateway to developing AI applications that are both innovative and capable of cross-domain functionality. Applying linear algebra techniques is a prerequisite for pushing the boundaries of what machines can learn and how they learn it, making it a cornerstone of modern AI development.
<p></p>
Linear algebra is being used for a variety of aggragtes and patterns like clustering, classification, regression and so on. Step by step data processing within deep learning neural network pipelines are represented as vectors.  Linear algebra is foundational to machine learning, serving as the backbone for a multitude of data aggregation and pattern recognition techniques, including clustering, classification, and regression. Within deep learning, particularly in neural network pipelines, data processing is meticulously structured where each step typically involves operations on vectors.
<p></p>
This study delves into the specialized realm of Graph Neural Networks (GNNs), a dynamic area of study where input data of node features must be represented as vectors. GNNs uniquely capture the nuances of graph structures, not just through the individual nodes but also via the overall graph topology. This allows GNNs to harness both the attribute and relational information encoded in graphs, offering a powerful tool for domain-specific challenges.
<p></p>
Graph Neural Networks are increasingly applied across various fields where data can be naturally represented as graphs. This includes social networks, biological networks, knowledge graphs, and more, where entities and their interconnections can be modeled to uncover deep insights. In this poster, we will explore how GNNs process these graph-based data representations through advanced linear algebraic operations to perform key tasks of GNNs: node classification, link prediction, and graph classification.
<p></p>

Node Classification and Link Prediction data processings are based on node embeddings and Graph Classifications are based on the whole graph embedding. When catching pre-final vectors from Node Classification and Link Prediction model processing, we are getting vectors that represent edbedded node features. These vectors can be used for node classification or regression; for link prediction these vectors are typically used to decide if edges between node pairs exist based on threshold cosine similarities. Vectors that are catched from model pipelines can be applied for many other techniques, like clustering, finding the closest members, triangle analysis, and so on. Explorations of getting pre-final embedded vectors for node classification and link prediction  techniques are used in some studies.
<p></p>
Input data for GNN Graph Classifications models consist of many labeled 'small graphs'. Such models that are commonly used in chemistry and biology, can also be applied to small graphs taken from other data domains. In social network graphs, these techniques can be used for analysis of surroundings of points of interests. Points of interest can be selected as nodes with high centrality metrics and subgraphs around points of interest can be taken as their 'friends' and 'friends of friends'.
<p></p>
Time series data can be segmented into small graphs using sliding window techniques that capture local temporal patterns and consider each time segment as a distinct 'small graph'. This method, segmenting time series into overlapping small graphs, excels in capturing short-term time series data fluctuations and it's a good method to detect short-term variability and rapid temporal data changes.
<p></p>


<p></p>
While there are studies on pre-final vector capturing from GNN Link Prediction and Node Classification models, based on our reseach, currently there are no studies that investigate capturing embedded whole graphs from GNN Graph Classification models. In this study we will show how to catch the whole 'small graphs' embedded vectors from such models. By catching these embbeded vectors and applying to these vectors similarity measures, we can do much more than graph classification: we can apply methods like clustering, or for any graph, we can find closest neigbors, or we can use small graphs as nodes and create graphs on top of small graphs.
<p></p>

Vector similarity measures are essential in solving many data mining problems. Methods that are commonly used to compare vectors are cosine similarities or dot products. In this study, we will also examine Symmetry Metrics, metrics based on unsupervised machine learning models that transform pairwise vectors to Gramian Angular Fields (GAF) images, classify them to symmetric and asymmetric classes, and calculate probabilities for pairwise vectors to get into the symmetric classes. We introduced Symmetry Metrics in our previous studies [] where we showed that they can be applied to many data domains where subjects can be mapped to vectors. In those studies we also showed than symmetry metrics are more sensitive than cosine similarity metrics and therefore they are useful for anomaly detection.
<p></p>
<p></p>

To distinguish between similar and dissimilar vector pairs we will generate training data as 'same' and 'different' classes. For the 'same' class we will generate self-reflected, mirror vectors

<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/nldl_img6.jpg" alt="Post Sample Image" width="500" >
</a>
</p><p>

and for 'different' class we will generate joint vectors as a random combination of non-equal pairs. Mirror vectors of the 'same' class transformed to GAF images represent symmetric images and GAF images of the 'different' class  - asymmetric images.  

<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/vectorSymmet1c.jpg" alt="Post Sample Image" width="500" >
</a>
</p><p>

CNN image classification model is trained to distinguish between symmetric and asymmetric images.  Similarity metric of this model is a probability for joint vectors converted to GAF image to get into the 'same' class.



<p></p>

Also, in our previous studies[], we discovered GNN Graph Classification model high sencitivities and explored the potential of using these models for anomaly detections. For anomaly detection, merging of GNN Graph Classifications with Symmetry Metrics becomes very interesting.
<p></p>
<p></p>



<p><h3> Related Work</h3>
<p></p>
Start with DL history CNN and KG, then getting together as GNN. Tell that these techniques can be manipulated from one to another. CNN symmetry metrics -- getting pairs fo wectors and comparing their similarities through GAF. GNN Graph classification - getting embedded vectors that represent the whole graphs and then tranforming pairs of embedded graphs to GAF images and compare them through symmetry metrics.  
<p></p>
In 2012, a significant breakthrough occurred in the fields of deep learning and knowledge graphs. Convolutional Neural Network (CNN) image classification was introduced through AlexNet[], showcasing its superiority over previous machine learning techniques in various domains []. Concurrently, Google introduced knowledge graphs, enabling machines to understand relationships between entities and revolutionizing data integration and management, enhancing products with intelligent and ’magical’ capabilities[].
<p></p>
The growth of deep learning and knowledge graphs occurred simultaneously for years, with CNN excelling at grid-structured data tasks but struggling with graph-structured ones. Conversely, graph techniques thrived on graph structured data but lacked deep learning’s capability. In the late 2010s, Graph Neural Networks (GNN) emerged, combining deep learning and graph processing, and revolutionizing how we handle graph-structured data. In the late 2010s, Graph Neural Networks emerged as a fusion of deep learning and knowledge graphs. GNN enable complex data analysis and predictions by effectively capturing relationships between graph nodes.
<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/climateGnnGc1.jpg" alt="Post Sample Image" width="479" >
</a>
<p></p>
(Picture from a book: Bronstein, M., Bruna, J., Cohen, T., and Velickovic ́, P.
“Geometric deep learning: Grids, groups, graphs, geodesics, and gauges”, 2021)
</p><p>
The “Geometric deep learning" paper was written in 2021 when the biggest animal in Deep Learning zoo was CNN. If that paper was written in 2023-2024, with no questions, the biggest animal would be LLM (Large Language Models). Who knows, what will be the biggest Deep Learning animal in 2-3 years from now...
<p></p>
In this study we will explore GNN models, in particular, GNN Graph Classifications, and CNN models that we will use as non-traditional method for vector similarity calculations, Symmetry Metrics
<p></p>

<p></p>

<h3>Methodologies</h3>
<p></p>
<h4>Cosine Similarity Method</h4>
To calculate cosine similarities between pairs of vectors, we will use Cosine Similarities function:  
    <p></p>
    {% highlight python %}
    import torch
    def pytorch_cos_sim(a: Tensor, b: Tensor):
        return cos_sim(a, b)
    def cos_sim(a: Tensor, b: Tensor):
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)
        if len(a.shape) == 1:
            a = a.unsqueeze(0)
        if len(b.shape) == 1:
            b = b.unsqueeze(0)
        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))
    {% endhighlight %}
    <p></p>

    <p></p>
<h4>Symmetry Metrics Method</h4>
In out previous study we demonstrated that for Symmetry Metrics, a model trained on one data domain, can be used to calculate how similar are vectors taken from another data domain. For this study we will use the model that was trained on all climate data.

For trained model, data processing and model training used the following steps:
<ul>
<li>Created pairs of pairwise vectors: self-reflected, mirror vectors for ’same’ class and concatenated different vectors for ’different’ class</li>
<li>Transformed joint vectors to GAF images for image classification</li>
<li>Trained CNN image classification on transfer learning to distinguish between symmetric and asymmetric images</li>
</ul>

<p></p>
To calculate how similar to each other are vectors for another data domain we will

<ul>
<li>Combine them as joint vectors</li>
<li>Transformed joint vectors to GAF images</li>
<li>Run GAF images through trained image classification model</li>
<li>Predict vector similarities based on the trained model by fast.ai function 'learn.predict'.</li>
</ul>

<p></p>  
Detail information and coding for data preparation, training and interpretation techniques for Symmetry Metrics are described in details in our previous posts [].
<p></p>      
<h4>Prepare Input Data for the Model</h4>

As input data for GNN Graph Classification model we need to create small labeled graphs:
<p></p>
<ul>
<li>
Edges
</li><li>
Nodes with features.
</li><li>
Labels on graph level.
</li></ul>

To get graph labels for input data, we classified cities into stable and unstable climate patterns by utilizing average cosines between consecutive years.
To generate small graph, for each city we calculated cosine similarity matrices and transformed them into graphs by taking only vector pairs with cosine similarities higher than a threshold. Also, for each graph we added a virtual node to transform disconnected graphs into single connected components.
  <p></p>


<h4>Train the GNN Graph Classification Model</h4>

In this study we uses a GCNConv model from PyTorch Geometric Library as a GNN graph classification model. The GCNConv model is a type of graph convolutional network that applies convolutional operations to extract meaningful features from the input graph data (edges, node features, and the graph-level labels). The code for the model is taken from a PyG tutorial.
</p><p>
<p></p>




<p></p>


<h4>Graph Embedding: Catching Pre-Final Vectors in GNN Graph Classification</h4>
<p>GNN Graph Classification with pre-final stop, .</p>

<p></p>
<h4>Interpreting Model Results: Cosine Similarity </h4>
    <p>With node embeddings in place, .</p>

    Select pairs of vectors with cosine similarities higher than 0.95.
        <p></p>  
    <p>This integrated approach allows us to delve deeper into .</p>
<p></p>
<h4>Interpreting Model Results: Symmetry Metrics </h4>


 Symmetry Metrics based on interpretation of the trained model results
To calculate how similar are vectors to each other we will combine them as joint vectors and transform to GAF images. Then we will run GAF images through trained image classification model and use probabilities of getting to the ’same’ class as symmetry metrics. To predict vector similarities based on the trained model, we will use fast.ai function 'learn.predict'.


<p></p>
<h3>Experiments Overview</h3>
    <p>This section outlines the experimental framework used to </p>

    <h4>Data Source: Climate Data</h4>

    As a data source we will use climate data from Kaggle data sets:
    <i><a href="
    https://www.kaggle.com/hansukyang/temperature-history-of-1000-cities-1980-to-2020">"Temperature History of 1000 cities 1980 to 2020"</a></i> - daily temperature from 1980 to 2020 years for 1000 most populous cities in the world.

    <p></p>



    <h4>Input Data Preparation</h4>




    <h4>Model Training</h4>
    <p>The model training phase
<p></p>


    <p></p>               



<p></p>

    <p></p>
<h4>Rewiring Knowledge Graph</h4>



    <p></p>
To pinpoint the main influencers in the knowledge graph, we employed betweenness centrality metrics. This approach enabled us to identify nodes that serve as crucial connectors facilitating information flow throughout the network. The initial step involves creating a graph using the NetworkX library:
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
The results of our previous climate time series study showed that cities located near the Mediterranean Sea had high similarity to a smooth line, indicating stable and consistent temperature patterns.  In one of climate analysis scenarios we found that most of cities with high similarities to a smooth line are located on Mediterranean Sea not far from each other. Here is a clockwise city list: Marseille (France), Nice (France), Monaco (Monaco), Genoa (Italy), Rome (Italy), Naples (Italy), and Salerno (Italy):
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/nldl_img9.jpg" alt="Post Sample Image" width="333" >
</a>
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

<p></p>

<p></p>
<p></p>
