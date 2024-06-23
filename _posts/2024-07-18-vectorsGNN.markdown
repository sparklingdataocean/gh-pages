---
layout:     post
title:      "Intermediate Vector Caching in GNN Graph Classification"
subtitle:   "Enhancing Climate Data Analysis through Intermediate Vector Caching in GNN Graph Classification Models"
date:       2024-07-18 12:00:00
author:     "Melenar"
header-img: "img/pageVec23.jpg"
---
In our latest research, we explored the crucial role of linear algebra in deep learning for efficient data representation and manipulation. We focused on extracting pre-final vectors from GNN Graph Classification models and analyzing embedded 'small graphs' using climate time series data. By combining linear algebra with these models, we significantly improved climate data analysis results. Additionally, we created meta-graphs on top of the embedded small graphs, extending our analytical capabilities and enhancing overall insights.

<p><h3> Introduction</h3>
<p></p>
Linear algebra is essential in machine learning and AI because it helps represent and manipulate data efficiently. Using matrices and vectors, we can model complex problems in a way that's easy for computers to handle. Recently, the rise of deep learning models has shown just how versatile and useful linear algebra is across different fields.

<p></p>

In deep learning, converting different types of data—like images, audio, text, and social network information—into a uniform vector format is crucial. This standardization makes it easier for algorithms to process and analyze the data, leading to innovative AI applications across multiple domains. Linear algebra plays a key role in this process, supporting methods like clustering, classification, and regression by enabling the manipulation and analysis of data in neural network pipelines. Every step in these pipelines involves vector operations, underscoring the critical importance of linear algebra in advancing deep learning technology.

<p></p>
In this study, we explore how to capture pre-final vectors from GNN processes and apply these intermediate vectors to a range of tasks beyond their primary functions. GNNs are particularly good at tasks like node classification, link prediction, and graph classification. While node classification and link prediction rely on node embeddings, graph classification focuses on whole graph embeddings. These pre-final vectors, representing embedded node features, have a wide array of applications. They can be used for node classification, regression, clustering, identifying nearest neighbors, and triangle analysis. This versatility highlights their potential to extend the capabilities of GNN models significantly.

<p></p>

For example, the GraphSAGE link prediction model in the Deep Graph Library (DGL) generates pre-final vectors, or embeddings, for each node instead of making direct link predictions. These embeddings capture the nodes' features and their relationships within the graph. Researchers have explored using these pre-final vectors for various tasks such as node classification, clustering, regression, and triangle analysis within the graph.

<p></p>

While the potential of pre-final vectors from link prediction models has been explored, our research reveals a gap in studying embedded whole graphs from GNN Graph Classification models. These models effectively capture graph structures by considering individual nodes and overall topology, leveraging both attribute and relational information within small graphs. This capability makes GNN Graph Classification models particularly powerful for domain-specific challenges in areas like social networks, biological networks, and knowledge graphs. In our study, we demonstrate how to capture embedded vectors of entire 'small graphs' from these models and use them for further graph data analysis.

<p></p>

GNN Graph Classification models use many labeled small graphs as input data. While traditionally used in chemistry and biology, these models can also be applied to small graphs from other domains. For instance, in social networks, they analyze points of interest identified by high centrality metrics, including their friends and friends of friends. Additionally, time series data can be segmented into small graphs using sliding window techniques, capturing short-term variability and rapid changes for dynamic data analysis. This versatility makes GNN Graph Classification models valuable for a wide range of applications.

<p></p>

For our experiments, we're using a climate time series dataset from Kaggle, which includes daily temperature data over 40 years for the 1000 most populous cities in the world. For each city, we'll create a graph where nodes represent combinations of cities and years, with node features being daily temperature vectors for each city-year pair. To define the edges of the graph, we'll select pairs of vectors with cosine similarities higher than a specified threshold.

<p></p>
We will validate our methods for capturing pre-final vectors and demonstrate their effectiveness in managing and analyzing dynamic datasets. By capturing these embedded vectors and applying similarity measures, we can go beyond simple graph classification. We'll use these techniques for clustering, finding the closest neighbors within any graph, and even creating meta-graphs by using small graphs as nodes. This approach opens up exciting possibilities for deeper data analysis and insights.

<p></p>


<p></p>

<p></p>


<p></p>
<p></p>






<p></p>


<p></p>
<p></p>



<p><h3> Related Work</h3>
<p></p>
In 2012, a significant breakthrough occurred in the fields of deep learning and knowledge graphs. Convolutional Neural Network (CNN) image classification was introduced through AlexNet [5], showcasing its superiority over previous machine learning techniques in various domains [6]. Concurrently, Google intro- duced knowledge graphs, enabling machines to understand relationships between
Climate Data Analysis through Intermediate Vectors from GNN Models 3
entities and revolutionizing data integration and management, enhancing prod- ucts with intelligent and ’magical’ capabilities [7].
<p></p>
<p></p>
The growth of deep learning and knowledge graphs occurred simultaneously for years, with CNNs excelling at grid-structured data tasks but struggling with graph-structured ones. Conversely, graph techniques thrived on graph-structured data but lacked deep learning’s capability. In the late 2010s, Graph Neural Net- works (GNNs) emerged, combining deep learning and graph processing, and revolutionizing how we handle graph-structured data by enabling complex data analysis and predictions through effective capture of relationships between graph nodes [8]. Starting in 2022, Large Language Models (LLMs) became prominent in the deep learning landscape and currently most of deep learning research attention is in LLM area. We still hope that GNN will continue to be emerging.
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

<p></p>

<p></p>

<h3>Methods</h3>
<p></p>


    <p></p>




<p></p>  

<p></p>      
<h4>Graph Construction and Climate Labeling</h4>

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

In this study, we utilized GNN Graph Classification models to analyze small labeled graphs structured from nodes and edges. Graphs were calculated for each city, with nodes corresponding to specific city-year pairs and edges defined as node pairs with cosine similarities higher than threshold values. Graphs were labeled as ’stable’ or ’unstable’ based on the city’s geographical latitude.
<p></p>


<h4>Implementation of GCNConv for Graph Classification</h4>

For the classification of these graphs, we deployed the Graph Convolutional Network (GCNConv) model from the PyTorch Geometric Library (PyG) [17]. The GCNConv model allows us to extract feature vectors from graph data, enabling a final binary classification to determine ’stable’ or ’unstable’ climates.

</p><p>
<p></p>




<p></p>






<p></p>
<h3>Experiments Overview</h3>

    <h4>Data Source: Climate Data</h4>

    As a data source we will use climate data from Kaggle data sets:
    <i><a href="
    https://www.kaggle.com/hansukyang/temperature-history-of-1000-cities-1980-to-2020">"Temperature History of 1000 cities 1980 to 2020"</a></i> - daily temperature from 1980 to 2020 years for 1000 most populous cities in the world.

    <p>
   This dataset provides a comprehensive record of average daily temperatures in Celsius for the 1000 most populous cities world- wide, spanning from 1980 to 2019. Utilizing this extensive dataset, we developed a Graph Neural Network (GNN) Graph Classification model aimed at analyzing and interpreting the climatic behaviors of these urban centers.
   <p></p>
   For our analysis, each city was represented as an individual graph, with nodes corresponding to specific city-year pairs. These nodes encapsulate the temperature data for their respective years, facilitating a detailed examination of temporal climatic patterns within each city.
   <p></p>
   The graphs were labeled as ’stable’ or ’unstable’ based on the latitude of the cities. We assumed that cities closer to the equator exhibit less tempera- ture variability and hence more stability. This assumption aligns with observed climatic trends, where equatorial regions generally experience less seasonal vari- ation compared to higher latitudes. To categorize the cities, we divided the 1000 cities into two groups based on their latitude, with one group consisting of cities nearer to the equator and the other group comprising cities at higher latitudes.
   <p></p>
  <p></p>

   <a href="#">
       <img src="{{ site.baseurl }}/img/preFinFig1.jpg" alt="Post Sample Image" width="678" >
   </a>
Fig. 1. Latitude Distribution of the 1000 Most Populous Cities.
   <p></p>
   Picture on Fig. 1 displays the latitude distribution of the 1000 most popu- lous cities covered in our study. It highlights the geographical diversity of these cities and provides a clear visual representation of their spread across different latitudes. Understanding this distribution is crucial for analyzing the climatic variations that may affect each city, reinforcing our assessment of temperature stability relative to geographic location.

    <p></p>
  <p></p>   <p></p>


    <h4>Data Preparation and Model Training</h4>

    <p></p>

    In the development of the Graph Neural Network (GNN) Graph Classification model for analyzing climate data, we constructed individual graphs for each city, labeled as ’stable’ or ’unstable’ based on latitude. Edges within these graphs were defined based on high cosine similarities between node pairs, reflecting similar temperature trends. To ensure uniform structure across all graphs, virtual nodes were introduced, improving graph connectivity and aiding in model generaliza- tion across various urban climates.
    <p></p>
    For our analysis, we employed the GCNConv model from the PyTorch Geo- metric (PyG) library [22]. This model is specifically used for extracting pre-final feature vectors from graphs before finalizing classification decisions. These vec- tors are crucial for subsequent detailed analyses of climate patterns.
    <p></p>
    The GCNConv model’s performance was quantitatively evaluated, showing high accuracy rates: approximately 94% on training data and 92% on test data. This performance highlights the model’s effectiveness in detecting and classify- ing anomalous climate trends using daily temperature data represented through graphs.
    <p></p>


    <h4>Application of Graph Embedded Vectors: Cosine Similarity Analysis</h4>
    <p>
    After training the GNN Graph Classification model, each city graph was trans- formed into a graph embedded vector. These vectors served as a foundational element for subsequent data analyses.

    <h5>Analysis of Cosine Similarity Matrix of Graph-Embedded Vectors:</h5>
    We constructed a cosine similarity matrix for 1000 cities to identify closely re- lated climate profiles, facilitating nuanced comparisons and clustering based on embedded vector data.
    </p><p>

    <p></p>    
To illustrate this, we examined the closest neighbors of the graph vectors for Tokyo, Japan (the largest city in our dataset), and Gothenburg, Sweden (the smallest city in our dataset). As shown in Table 1, Tokyo’s closest neighbors are primarily major Asian cities, indicating strong climatic and geographical simi- larities. Similarly, Table 2 shows that Gothenburg’s nearest neighbors are pre- dominantly European cities, reflecting similar weather patterns across Northern and Central Europe.
<p></p>    
Table 1. Closest Neighbors of Tokyo, Japan (Lat 35.69, Long 139.69). Based on Cosine
Similarity
     <a href="#">
         <img src="{{ site.baseurl }}/img/preFinTab1.jpg" alt="Post Sample Image" width="404" >
     </a>

     <p></p>

     <p></p>
Table 2. Closest Neighbors of Gothenburg, Sweden (Lat 57.71, Long 12.00). Based on Cosine Similarity
      <a href="#">
          <img src="{{ site.baseurl }}/img/preFinTab2.jpg" alt="Post Sample Image" width="383" >
      </a>




    <p></p>   
    We also identified vector pairs with the lowest cosine similarity, specifically -0.543011, between Ulaanbaatar, Mongolia, and Shaoguan, China. This negative similarity suggests stark climatic differences. Additionally, the pair with cosine similarity closest to 0.0 (-0.000047), indicating orthogonality, is between Nan- chang, China, and N’Djamena, Chad. This near-zero similarity underscores the lack of significant relationship between these cities’ climatic attributes.            



<p></p>

<h5>The cosine similarity matrix distribution</h5> from the embedded city graphs shows notable peaks for values over 0.9 and between -0.4 to -0.2, indicating dis- tinct clustering based on climatic profiles. Some clusters show high similarity, reflecting nearly identical climates, while others display moderate similarities, highlighting shared but less pronounced features.

<p></p>

 <a href="#">
     <img src="{{ site.baseurl }}/img/preFinFig2.jpg" alt="Post Sample Image" width="678" >
 </a>
Fig. 2. Cosine Similarity Distribution.
<p></p>

These patterns are graphically represented in Fig. 2 and detailed in Table 4. The figure clearly shows the skewed distribution, emphasizing areas with the highest concentration of values. This visualization is essential for interpreting the relational dynamics of the cities, providing insights into the clustering patterns from their climate data.
<p></p>
Table 3. Distribution of Cosine Similarities.
      <a href="#">
          <img src="{{ site.baseurl }}/img/preFinTab3.jpg" alt="Post Sample Image" width="256" >
      </a>

<p></p>

<h4>Application of Graph Embedded Vectors: Graphs Derived from Cosine Similarity Thresholds</h4>
<p></p>
Based on the observed distribution of cosine similarities, we generated three distinct graphs for further analysis. Each graph was constructed using different cosine similarity thresholds to explore the impact of these thresholds on city pair distances.
<p></p>  
<b>For the first graph</b>, we used a high similarity threshold (cosine similarity > 0.9). The statistics for the distances between city pairs in this graph are as follows:
– Mean distance: 7942.658 km
– Median distance: 7741.326 km
– Standard deviation: 5129.801 km – Minimum distance: 1.932 km
– Maximum distance: 19975.287 km
<p></p>               
The shortest distance pair is between Jerusalem, Israel (latitude 31.7784, lon- gitude 35.2066) and Al Quds, West Bank (latitude 31.7764, longitude 35.2269). These cities are geographically very close, underscoring their proximity with nearly identical latitude and longitude coordinates. The longest distance pair is between Quito, Ecuador (latitude -0.2150, longitude -78.5001) and Pekanbaru, Indonesia (latitude 0.5650, longitude 101.4250). These cities are on opposite sides of the world, as reflected by their dramatically contrasting geographical coordinates, spanning a vast distance across the globe.
<p></p>
<i>For the second graph,</i> defined by a cosine similarity threshold ranging from -0.4 to -0.2, we observed a moderate level of climatic similarity among city pairs. The key statistics for this graph are as follows:
– Mean distance: 8648.245 km
– Median distance: 8409.507 km
– Standard deviation: 4221.592 km – Minimum distance: 115.137 km
– Maximum distance: 19963.729 km
    <p></p>   
    For this graph, the shortest distance pair is between Kabul, Afghanistan (lat- itude 34.5167, longitude 69.1833) and Jalalabad, Afghanistan (latitude 34.4415, longitude 70.4361). The longest distance pair is between Mendoza, Argentina (latitude -32.8833, longitude -68.8166) and Shiyan, China (latitude 32.5700, lon- gitude 110.7800).
    <p></p>
For the third graph, we used a high similarity threshold (cosine similarity > 0.99), resulting in connected components of sizes [514, 468, 7, 5]. The largest connected component, with 514 nodes, predominantly includes cities with stable climates (475 nodes labeled as 0) and a smaller portion with unstable climates (39 nodes labeled as 1). The second-largest component, containing 468 nodes, primarily consists of cities with unstable climates (451 nodes labeled as 1) and a few with stable climates (17 nodes labeled as 0). These findings indicate that cities within the same climate category (stable or unstable) exhibit higher sim- ilarity, leading to larger connected components, whereas the similarities across different climate categories are less pronounced.
    <p></p>
Table 4. Cities in the Third Connected Component (7 Nodes)
      <a href="#">
          <img src="{{ site.baseurl }}/img/preFinTab4.jpg" alt="Post Sample Image" width="383" >
      </a>
    <p></p>
Table 5. Cities in the Fourth Connected Component (5 Nodes)
          <a href="#">
              <img src="{{ site.baseurl }}/img/preFinTab5.jpg" alt="Post Sample Image" width="383" >
          </a>
              <p></p>
In the smaller connected components, city graphs represent areas on the border between stable and unstable climates. Tables 4 and 5 provide details of these cities, highlighting their transitional nature. Table 4 lists the cities from the third component, consisting of 7 nodes, while Table 5 lists the cities from the fourth component, comprising 5 nodes. These cities illustrate the variability and complexity of climatic relationships, showing a blend of stable and unstable climatic conditions. This underscores the nuanced and intricate climatic patterns that exist at the boundaries between different climate categories.
<p></p>

{% highlight python %}
xxxx
{% endhighlight %}
    <p></p>


<p></p>

<p></p>

<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/nldl_img9.jpg" alt="Post Sample Image" width="333" >
</a>
<p></p>




<p></p>








    <h3>In Conclusion</h3>
<p></p>
    In this study, we explored how to capture pre-final vectors from GNN models, focusing on their application in GNN Graph Classification. Linear algebra helps represent and manipulate diverse data types, turning them into uniform vector formats that deep learning models can process.
<p></p>
    We highlighted how GNN Graph Classification models capture complex graph structures using advanced linear algebra operations. A key part of our research was embedding entire ’small graphs’ from these models. This method opens new possibilities for analyzing and clustering small graphs, identifying closest neigh- bors, and building meta-graphs.
<p></p>

Our findings show that integrating linear algebra with GNNs improves the efficiency and scalability of these models, broadening their use across various domains. By capturing and analyzing embedded graphs from GNN Graph Clas- sifications, we can enhance data analysis and predictive capabilities, advancing artificial intelligence and its applications.
<p></p>    





<p></p>

<p></p>

<p></p>
<p></p>
