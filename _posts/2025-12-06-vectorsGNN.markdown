---
layout:     post
title:      "Unlocking the Power of Pre-Final Vectors in GNN Graph Classification"
subtitle:   "Utilizing Intermediate Vectors from GNN Graph Classification to Enhance Climate Analysis"
date:       2026-01-06 12:00:00
author:     "Melenar"
header-img: "img/gnnge5b.jpg"
---



<p></p>
<p><h2> Introduction</h2>
<p></p>
Graphs are everywhere—molecules that show how atoms connect, or social networks that map out friendships. They capture relationships that ordinary tables or lists can’t show. But because graphs come in so many shapes and sizes, they can be difficult to analyze directly.
<p></p>
That’s where GNN graph classification comes in. Instead of one giant network, it works with many small graphs—each representing its own structure. By turning these graphs into vectors, we get a common format that makes them easy to compare, classify, and analyze.

<p></p>

<p></p>
Imagine you have a bunch of different molecules. Each one can be seen as a little graph: the atoms are the nodes, and the bonds between them are the edges. This graph view captures the structure of the molecule—how its parts connect and interact.
Using GNN graph classification, we can turn each of these molecular graphs into a vector, a set of numbers that encodes the structure in a consistent way. Once molecules are represented like this, we can compare them more easily, predict their properties, group similar ones together, or even spot patterns that might lead to new discoveries.
<p></p>
When most people think of graphs, they picture social networks—each person as a node and each friendship as an edge. Now imagine zooming in on smaller pieces of that network, like one person’s friend circle or a close-knit community. Each of these can be treated as its own little graph.
<p></p>
We can also think about small graphs in terms of dimension. A molecule is a zero-dimensional graph—it’s self-contained, like a single point with no larger context. A friend circle inside a social network is a multi-dimensional graph, because it sits within a bigger network and carries more context from the larger structure.
<p></p>
And there’s also a middle ground: the one-dimensional case, which sits between fully isolated graphs and multi-dimensional subgraphs. Here we use sliding graphs, where each snapshot is a one-dimensional slice of the larger timeline. For example, stock prices can be broken into overlapping segments to capture market fluctuations, while climate records such as daily temperatures can be segmented to reveal short-term shifts. Each segment becomes its own small graph, making it possible to analyze patterns and changes step by step.
<p></p>


<p></p>



<p></p>
<h2>How Raw Data Becomes Graphs</h2>
<p></p>

Different types of raw data—like time series, text, or images—can all be transformed into graph form. Depending on the goal, this might mean creating a single comprehensive graph, generating sliding graphs that capture sequential slices, or extracting local subgraphs that highlight specific regions of interest.
<p></p>

 <a href="#">
     <img src="{{ site.baseurl }}/img/gnnge2b.jpg" alt="Post Sample Image" width="888" >
 </a>
Data Flow: From Raw Data to Graph Classification.
 <p></p>

<p></p>
No matter the construction method, the outcome is a set of small graphs made up of nodes and edges. These give us a consistent way to represent very different data sources, so they can all be explored and compared using graph classification.

<p></p>
We will validate the methods for capturing pre-final vectors and demonstrate their effectiveness in managing and analyzing dynamic datasets. By capturing these embedded vectors and applying similarity measures to them, we will extend beyond graph classification to apply methods like clustering, finding the closest neighbors for any graph, or even using small graphs as nodes to create meta-graphs on top of small graphs.
<p></p>


<p></p>

<h3>Using Climate Data to Build Graphs</h3>

In this post, we’ll use climate data as a hands-on example to show how different kinds of graphs can be built and analyzed. We’ll look at Single Graphs, Sliding Graphs, and Local Subgraphs, each offering a different perspective but sharing the same building blocks—nodes and edges.
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/gnnge3b.jpg" alt="Post Sample Image" width="999" >
</a>
Climate Data Flow: From Raw Data to Pre-final Vectors.
<p></p>
<p></p>
From there, we’ll walk through how these graphs are processed with PyTorch Geometric for GNN graph classification, producing pre-final vectors that we can use for deeper analysis. This way, abstract graph concepts become more concrete, with real code examples grounded in climate data.
<p></p>


<h3>Data Source: Climate Data</h3>
<p></p>
For our study, we use a Kaggle dataset <i><a href="
https://www.kaggle.com/hansukyang/temperature-history-of-1000-cities-1980-to-2020">"Temperature History of 1000 cities 1980 to 2020"</a></i> , which provides 40 years of daily average temperature records (in Celsius) for the world’s 1000 most populous cities. Alongside the long time series, it also includes geographical coordinates for each city, making it especially valuable for graph-based analysis.
<p></p>
This mix of temporal and spatial information lets us represent the data in several forms—Single Graphs, Sliding Graphs, and Local Subgraphs. That flexibility makes it an excellent foundation for both building graph models and walking through coding examples in this blog series.
<p></p>

<p></p>
<h3>Data Pre-processing</h3>

<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/gnnge3c.jpg" alt="Post Sample Image" width="489" >
</a>
<p></p>
<p></p>
The raw dataset is organized as a table where each row corresponds to a city. Alongside identifiers like city name, coordinates, and country, the table includes sequences of daily temperature values for each year. This tabular form is the starting point for transforming climate records into graph structures.
<p></p>
Climate: Raw Data:
 <p></p>
 <a href="#">
     <img src="{{ site.baseurl }}/img/gnnge1a.jpg" alt="Post Sample Image" width="777" >
 </a>



 <p></p>




 <h4>Function: <code>climate_matrix_to_long</code></h4>
<p>
  Reshapes a wide <em>city–year temperature table</em> (daily columns <code>0..364/365</code>)
  into a tidy, day-level format with one row per city and date.
</p>
<ul>
  <li>Builds a unique city key (<code>cityName = "city, country"</code>).</li>
  <li>Adds a simple latitude-based label (0 = closer to equator, 1 = farther).</li>
  <li>Converts day indices into real calendar dates.</li>
</ul>
<p>The result is a clean dataset ready for graph construction and analysis.</p>


 <p></p>


 <p></p>
 <p></p>
 <p></p>

{% highlight python %}
import pandas as pd
import numpy as np
def climate_matrix_to_long(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cityName"] = df["city_ascii"].astype(str) + ", " + df["country"].astype(str)
    city_lat = (
        df[["city_ascii", "country", "lat"]]
        .drop_duplicates()
        .dropna(subset=["lat"])
        .copy()
    )
    city_lat["abs_lat"] = city_lat["lat"].abs()
    city_lat = city_lat.sort_values(["abs_lat", "city_ascii", "country"])
       .reset_index(drop=True)
    cut = len(city_lat) // 2
    city_lat["city_label"] = (city_lat.index >= cut).astype(int)  
    def _is_intlike(c):
        try:
            int(c); return True
        except:
            return False
    daily_cols =
       sorted([c for c in df.columns if _is_intlike(c)], key=lambda x: int(x))
    id_vars =
       ["cityName","city_ascii","country","lat","lng","zone","cityInd","nextYear"]
    long = df.melt(id_vars=id_vars, value_vars=daily_cols,
        var_name="doy", value_name="value")
    long["doy"] = long["doy"].astype(int)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    base = pd.to_datetime(long["nextYear"].astype(int).astype(str)
       + "-01-01", errors="coerce")
    long["date"] = base + pd.to_timedelta(long["doy"], unit="D")
    long = long.rename(columns={"nextYear": "year"})
    long = long.merge(city_lat[["city_ascii","country","city_label"]],
                      on=["city_ascii","country"], how="left")
    return long[["cityName","lat","lng","cityInd","year","date","doy","value","city_label"]]
{% endhighlight %}


 </p><p>


<p></p>
<p></p>

<p></p>



<p></p>


<p></p>  

<p></p>
{% highlight python %}
climate_long = climate_matrix_to_long(rawData)
climate_long.head()
{% endhighlight %}
<p></p>
Climate Data transformed to long matrix:
<p></p>

 <a href="#">
     <img src="{{ site.baseurl }}/img/gnnge1b.jpg" alt="Post Sample Image" width="654" >
 </a>

 <p></p>



 <p></p>

 <p></p>
 <p></p>

 <p></p>
 <h3>Single Graphs</h3>
 <p></p>
 <a href="#">
     <img src="{{ site.baseurl }}/img/gnnge3d.jpg" alt="Post Sample Image" width="489" >
 </a>
 <p></p>

 <p>
   A <strong>city-graph</strong> is a single, self-contained graph for one city—independent of all others.
   Each node is a <em>city–year</em> with features equal to that year’s daily temperatures.
   Edges link pairs of city–year nodes when their temperature vectors exceed a cosine-similarity threshold.
   The graph label (e.g., <code>stable</code> vs <code>unstable</code>) comes from the city’s latitude group.
 </p>
 <a href="#">
     <img src="{{ site.baseurl }}/img/voronoi42.jpg" alt="Post Sample Image" width="489" >
 </a>

 <p></p>
 <p><strong>Node table for single-graph construction</strong></p>
 <ul>
   <li><strong>graph_name</strong>: <code>str(cityInd)</code> (one graph per city)</li>
   <li><strong>graph_label</strong>: <code>city_label</code> (latitude-based 0/1)</li>
   <li><strong>node_name</strong>: <code>{cityInd}~{year}</code> (unique city–year key)</li>
   <li><strong>features</strong>: pivot days → columns to a fixed 365-length vector <code>f0..f364</code> (DOY 0..364);
       if a <code>365</code> column exists (leap year), drop it for consistency.</li>
 </ul>

 <p>
   In short: pivot daily values by (city, year), attach labels/IDs, and produce a tidy node table ready for per-city single-graph construction.
 </p>



 <p></p>
 {% highlight python %}
 import pandas as pd
 def build_city_graph_nodes_table(df: pd.DataFrame) -> pd.DataFrame:
     df2 = climate_long.sort_values(["cityInd", "year", "date"], kind="stable")
     wide = df2.pivot_table(
         index=["cityInd", "city_label", "year"],
         columns="doy",
         values="value",
         aggfunc="first"
     )
     wide = wide.reindex(columns=range(365))  
     wide.columns = [f"f{i}" for i in range(365)]
     wide = wide.reset_index()
     wide["graph_name"] = wide["cityInd"].astype(int).astype(str)
     wide["graph_label"] = wide["city_label"].astype(int)
     wide["node_name"] = wide["cityInd"].astype(int).astype(str) + "~"
        + wide["year"].astype(int).astype(str)
     cols = ["graph_name", "graph_label", "node_name"]
        + [f"f{i}" for i in range(365)]
  return wide[cols]
 {% endhighlight %}
 <p></p>
 <p></p>
 {% highlight python %}
 single_nodes_table = build_city_graph_nodes_table(climate_long)
 single_nodes_table.tail()
 {% endhighlight %}
 <p></p>
 <a href="#">
     <img src="{{ site.baseurl }}/img/gnnge1c.jpg" alt="Post Sample Image" width="777" >
 </a>


 <p></p>
 <h3>Sliding Graphs</h3>
 <p></p>
 <a href="#">
     <img src="{{ site.baseurl }}/img/gnnge3e.jpg" alt="Post Sample Image" width="489" >
 </a>
 <p></p>
 For one-dimensional small graphs, we’ll use the sliding-graph technique introduced in our earlier post and paper. We take a long time series and cut it into overlapping segments; each segment becomes a node with the segment’s values as features. Nodes are connected when their segments exceed a cosine-similarity threshold, and consecutive nodes are grouped into small graph snapshots that move along the series. This yields a stream of compact graphs that capture short-term patterns while preserving temporal order.


 <p></p>
 <p></p>
 <img src="{{ site.baseurl }}/img/eegSlide3.jpg" alt="Sliding graphs" width="478"
     style="border:2px solid #ccc; border-radius:6px;">

<p></p>
To build the long 1-D lines, we pick a single calendar day and pull its temperature for every city across all years, then concatenate the values in a fixed, reproducible order (e.g., sort by year, then by city). In this post we create two such lines—January 1 and April 1—and use them as baseline sequences for later steps. On January 1, we expect a stronger hemispheric split (Northern winter vs. Southern summer), wider spread from polar cold to subtropical heat, and big within-hemisphere differences driven by altitude and distance to the ocean. On April 1, that contrast softens: the Northern Hemisphere is warming out of winter while the Southern is cooling into autumn, so distributions partially converge—still distinct, but with more overlap and a slightly tighter overall spread. In short: the two lines should be neither very different nor very similar—tropics stay stable, temperate regions shift, and the hemispheric gradient remains but is less stark in April.
     <p></p>
<p></p>
{% highlight python %}
def extract_cityInd_year_doy_values_sorted(df, month, day):
    sub = df.loc[
        (df["date"].dt.month == month) & (df["date"].dt.day == day),
        ["cityInd", "year", "doy", "value"]
    ].copy()
    sub = sub.sort_values(["year", "cityInd"], kind="stable")
    sub["cityInd~year~doy"] = (
        sub["cityInd"].astype(str) + "~"
           + sub["year"].astype(str) + "~" + sub["doy"].astype(str)
    )
    return sub[["cityInd~year~doy", "value"]].reset_index(drop=True)
{% endhighlight %}
     <p></p>

     <p></p>
     <p></p>

 <p></p>
 {% highlight python %}
jan1_vals = extract_cityInd_year_doy_values_sorted(climate_long, 1, 1)
apr1_vals = extract_cityInd_year_doy_values_sorted(climate_long, 4, 1)
 {% endhighlight %}
 <p></p>
 <p></p>
 <a href="#">
     <img src="{{ site.baseurl }}/img/gnnge4a.jpg" alt="Post Sample Image" width="222" >
 </a>
<p></p>
The function: <strong>build_sliding_graph_nodes</strong> turns one long time series (two columns: <code>indicator</code>, <code>value</code>) into a <em>node table</em> for sliding graphs.
</p>
 <ul>
   <li><strong>Overlapping nodes:</strong> window the values with length <code>W</code> and stride <code>S</code>.
       Features → <code>f0..f{W-1}</code>; <code>node_name</code> is the indicator at the window start.</li>
   <li><strong>Graph grouping:</strong> group <code>G</code> consecutive nodes, moving by <code>Sg</code>.
       <code>graph_name</code> is the indicator at the group start; <code>graph_label</code> is the provided integer (e.g., 0/1).</li>
   <li><strong>Output:</strong> DataFrame with columns
       <code>graph_name</code>, <code>graph_label</code>, <code>node_name</code>, <code>f0..f{W-1}</code>.
       If too few nodes exist to form one graph, returns an empty frame with these columns.</li>
 </ul>

 <p>This function uses the following parameters:</p>
 <ul>
   <li><code>indicator_col</code> (default <code>"indicator"</code>), <code>value_col</code> (default <code>"value"</code>)</li>
   <li><code>W</code>, <code>S</code>: node window size &amp; stride; <code>G</code>, <code>Sg</code>: nodes per graph &amp; graph

 <p></p>

<p></p>
{% highlight python %}
import pandas as pd
import numpy as np
def build_sliding_graph_nodes(
    df: pd.DataFrame,
    W: int = 30, S: int = 7,      
    G: int = 30, Sg: int = 7,     
    indicator_col: str = "indicator",
    value_col: str = "value",
    graph_label: int = 0          
) -> pd.DataFrame:
    ind = df[indicator_col].astype(str).to_numpy()
    vals = df[value_col].to_numpy(dtype=float)
    N = len(df)
    fcols = [f"f{i}" for i in range(W)]
    empty_cols = ["graph_name", "graph_label", "node_name"] + fcols
    node_starts = np.arange(0, N - W + 1, S, dtype=int)
    N_nodes = len(node_starts)
    node_names = ind[node_starts]
    X = np.stack([vals[i:i+W] for i in node_starts], axis=0)  
    if N_nodes < G:
        return pd.DataFrame(columns=empty_cols)
    graph_starts = np.arange(0, N_nodes - G + 1, Sg, dtype=int)
    rows = []
    for gs in graph_starts:
        gname = node_names[gs]           
        block_names = node_names[gs:gs+G]
        block_feats = X[gs:gs+G]
        for nm, feat in zip(block_names, block_feats):
            row = {"graph_name": gname, "graph_label": int(graph_label),
               "node_name": nm}
            row.update({f"f{i}": float(feat[i]) for i in range(W)})
            rows.append(row)
    return pd.DataFrame(rows, columns=empty_cols)
{% endhighlight %}
<p></p>
<p>We built node tables for the Jan&nbsp;1 and Apr&nbsp;1 baseline lines using function <code>build_sliding_graph_nodes</code> with parameters: <code>W=32</code>, <code>S=8</code>, <code>G=32</code>, and <code>Sg=10</code>; set <code>indicator_col="cityInd~year~doy"</code> and <code>value_col="value"</code>; assigned <code>graph_label=0</code> to Jan&nbsp;1 and <code>graph_label=1</code> to Apr&nbsp;1; then concatenated the results into a single table: <code>sliding_nodes_table = pd.concat([nodes_table_0, nodes_table_1], ignore_index=True)</code>.</p>

<p></p>


<p></p>  

<p></p>
{% highlight python %}
nodes_table_0 = build_sliding_graph_nodes
   (jan1_vals, W=32, S=8, G=32, Sg=10,
   indicator_col="cityInd~year~doy", value_col="value",graph_label=0)
nodes_table_1 = build_sliding_graph_nodes
   (apr1_vals, W=32, S=8, G=32, Sg=10,
   indicator_col="cityInd~year~doy", value_col="value",graph_label=1)
sliding_nodes_table = pd.concat([nodes_table_0, nodes_table_1],
   ignore_index=True)
{% endhighlight %}
<p></p>

<a href="#">
    <img src="{{ site.baseurl }}/img/gnnge6c.jpg" alt="Post Sample Image" width="789" >
</a>


<p></p>
<h3>Nodes of Single and Sliding Graphs</h3>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/gnnge3f.jpg" alt="Post Sample Image" width="489" >
</a>
<p></p>
Each small graph uses the same node schema:
<ul>
  <li><strong>graph name:</strong> ID of the graph (e.g., city for single graphs; anchor for sliding).</li>
  <li><strong>graph label:</strong> class tag for the graph (e.g., 0/1).</li>
  <li><strong>node name:</strong> unique node identifier.</li>
  <li><strong>node features:</strong> fixed-length numeric vector (daily temps for single graphs; W-length window for sliding).</li>
</ul>
<p></p>
For single graphs and sliding graphs, we first build the nodes with their feature vectors, then add edges by linking pairs of nodes whose feature vectors exceed a cosine-similarity threshold. For a local subgraph, we do it differently: we start by selecting a local neighborhood (e.g., geographically nearby cities or a focused region/time slice), define edges using spatial/structural proximity (e.g., radius/k-NN/adjacency), and then attach features to the nodes within that localized structure.
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/gnnge4b.jpg" alt="Post Sample Image" width="777" >
</a>

<a href="#">
    <img src="{{ site.baseurl }}/img/gnnge4c.jpg" alt="Post Sample Image" width="777" >
</a>
<p></p>
<p></p>
<h3>Edges of Single and Sliding Graphs</h3>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/gnnge3g.jpg" alt="Post Sample Image" width="489" >
</a>
<p></p>
This step partitions the node table into separate small graphs, compares node feature vectors within each partition using cosine similarity, and adds undirected edges between pairs that exceed a chosen threshold. It then appends a single hub node to each graph—its features are the mean of that graph’s nodes—and connects the hub to all nodes to guarantee connectivity and provide a global anchor. The result is an augmented node set (original nodes plus hubs) and an edge list; partitions remain isolated from one another, and the threshold controls how dense the similarity edges are.
<p></p>
<ul>
  <li>Partitions <code>nodes_table</code> by <code>graph_name</code>; each partition is one small graph.</li>
  <li>Within each graph, connects node pairs whose feature vectors meet a <strong>cosine similarity ≥ threshold</strong> (undirected).</li>
  <li>Adds a per-graph <em>virtual hub</em> node with mean features and connects it to all nodes for connectivity/context.</li>
  <li>Outputs two tables: updated nodes (original + virtual) and an edges table; no links across different <code>graph_name</code> values.</li>
  <li>Edge density controlled by the cosine threshold (higher ⇒ fewer similarity edges).</li>
</ul>

<p></p>
{% highlight python %}
import pandas as pd
import numpy as np
def build_cosine_edges_with_virtual(
    nodes_table: pd.DataFrame,
    cosine_threshold: float = 0.9,
    virtual_node_name: str = "__VIRTUAL__"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fcols = [c for c in nodes_table.columns if isinstance(c, str)
       and c.startswith("f")]
    fcols.sort(key=lambda x: int(x[1:]))
    if not fcols:
        raise ValueError("No feature columns found (expected f0..fK).")
    edges = []
    virtual_rows = []
    for gname, g in nodes_table.groupby("graph_name", sort=False):
        glabel = int(g["graph_label"].iloc[0])
        X_full = g[fcols].to_numpy(dtype=float)
        valid_mask = ~np.isnan(X_full).any(axis=1)
        if valid_mask.sum() >= 2:
            X = X_full[valid_mask]
            names_valid = g.loc[valid_mask, "node_name"].astype(str).to_numpy()
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            Xn = X / norms
            S = Xn @ Xn.T
            n = S.shape[0]
            iu, ju = np.triu_indices(n, k=1)
            keep = S[iu, ju] >= cosine_threshold
            for i, j in zip(iu[keep], ju[keep]):
                left, right = names_valid[i], names_valid[j]
                if left > right:
                    left, right = right, left
                edges.append((gname, left, right))
        vfeat = np.nanmean(X_full, axis=0)
        vrow = {"graph_name": gname, "graph_label": glabel,
           "node_name": virtual_node_name}
        vrow.update({c: float(v) for c, v in zip(fcols, vfeat)})
        virtual_rows.append(vrow)
        all_names = g["node_name"].astype(str).to_numpy()
        for nm in all_names:
            left, right = virtual_node_name, nm
            if left > right:
                left, right = right, left
            edges.append((gname, left, right))
    nodes_out =
       pd.concat([nodes_table, pd.DataFrame(virtual_rows)],
       ignore_index=True)
    edges_df =
       pd.DataFrame(edges, columns=["graph_name", "left", "right"])
       .drop_duplicates()
    return nodes_out, edges_df{% endhighlight %}
<p></p>

<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/gnnge6a.jpg" alt="Post Sample Image" width="521" >
</a>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/gnnge6b.jpg" alt="Post Sample Image" width="521" >
</a>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/gnnge6d.jpg" alt="Post Sample Image" width="999" >
</a>
<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/gnnge6e.jpg" alt="Post Sample Image" width="999" >
</a>
<p></p>
<p></p>
<h3>Local Subgraphs</h3>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/gnnge3h.jpg" alt="Post Sample Image" width="489" >
</a>
<p></p>
To build multi-dimensional small subgraphs on climate data, first we will build one globe-spanning backbone graph. To build this spatial graph we will use Voronoi diagram. To figure out Voronoi graph, imagine several Starbucks in town. Each of them covers the territory that is closer to it then to other Starbucks. We will consider two Starbucks as neighbors if they share the Voronoi borders.   
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/voronoi41.jpg" alt="Post Sample Image" width="314" >
</a>
<p></p>
For climate based Voronoi graph, each city becomes a node. They are connected to others that share a geographical boundary. Some neighbors will be close to each other, some neighbors far away, like Porto and Québec.
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/voronoi17.jpg" alt="Post Sample Image" width="628" >
</a>
<p></p>
<p></p>

In practice, we project latitude/longitude to a planar map (e.g., EPSG:3857), compute the Voronoi diagram, and turn every shared border into an undirected edge (optionally cross-checking with the Delaunay triangulation).
<p></p>

<p></p>



Onto this backbone we attach city-level features—starting with 365-day climatology vectors (average day-of-year temperatures), and later swapping in richer temporal or spatio-temporal embeddings. When helpful, we turn cosine similarity between city vectors into an edge weight. This “big graph” then acts like a map: to study any place or region, we simply crop a local subgraph (k-hop or radius neighborhood) and analyze that focused piece without rebuilding the world each time.
<p></p>

<p><code>build_city_avg_vectors_long(...)</code> creates one row per city with its average day-of-year temperature profile across all years.</p>
<ul>
  <li><strong>Output:</strong> <code>cityInd</code> (+ optional metadata) plus features <code>f0..f364</code> (or <code>f365</code> if kept).</li>
  <li><strong>Steps:</strong> clean <code>doy</code>/<code>value</code> → optionally drop day 365 → average by (<code>cityInd</code>, <code>doy</code>) using <code>agg</code> → pivot wide and rename to <code>f*</code>.</li>
  <li><strong>Params:</strong> <code>drop_day_365</code>, <code>agg</code> (e.g., "mean"/"median"/callable), <code>include_meta</code> (add <code>cityName</code>, <code>lat</code>, <code>lng</code>, <code>city_label</code>).</li>
  <li><strong>Use cases:</strong> node features for spatial graphs, similarity searches, or clustering.</li>
</ul>
<p></p>
{% highlight python %}
import pandas as pd
import numpy as np
def build_city_avg_vectors_long(
    climate_long: pd.DataFrame,
    drop_day_365: bool = True,
    agg="mean",                 
    include_meta: bool = True   
) -> pd.DataFrame:
    df = climate_long.copy()
    df["doy"] = pd.to_numeric(df["doy"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["doy", "value"])
    df["doy"] = df["doy"].astype(int)
    if drop_day_365:
       df = df[df["doy"] <= 364]
    gd = (df.groupby(["cityInd", "doy"], as_index=False)["value"]
       .aggregate(agg))
    wide = gd.pivot(index="cityInd", columns="doy", values="value")
    max_doy = 364
       if drop_day_365 else (365
       if (df["doy"].max() >= 365)
       else df["doy"].max())
    wide = wide.reindex(columns=range(0, max_doy + 1), fill_value=np.nan)
    wide.columns = [f"f{int(c)}" for c in wide.columns]
    out = wide.reset_index()
    if include_meta:
        meta = (df.groupby("cityInd", as_index=False)
                  .agg(cityName=("cityName", "first"),
                       lat=("lat", "first"),
                       lng=("lng", "first"),
                       city_label=("city_label", "first")))
        out = meta.merge(out, on="cityInd", how="left")
        # Order columns
        fcols = [c for c in out.columns if c.startswith("f")]
        out = out[["cityInd", "cityName", "lat", "lng", "city_label"] + fcols]
    else:
        fcols = [c for c in out.columns if c.startswith("f")]
        out = out[["cityInd"] + fcols]
    return out
{% endhighlight %}
<p></p>
{% highlight python %}
city_avg_vectors = build_city_avg_vectors_long(climate_long)
city_avg_vectors.head(2)
{% endhighlight %}
<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/gnnge4f.jpg" alt="Post Sample Image" width="888" >
</a>
<p></p>
This function builds a spatial neighbor graph from city coordinates: it treats each city as a node, tessellates the map with a Voronoi diagram, and returns an undirected edge list linking city pairs whose regions touch—after light preprocessing to keep the geometry stable.
<p></p>
<ul>
  <li><strong>Input:</strong> DataFrame with city IDs and WGS84 coordinates (defaults: <code>cityInd</code>, <code>lng</code>, <code>lat</code>); rows with missing coords are dropped.</li>
  <li><strong>Steps:</strong> sort by ID for reproducible indexing &rarr; project lon/lat to planar CRS (EPSG:3857, meters) &rarr; add tiny (~10 m) deterministic jitter to exact duplicate points &rarr; compute the Voronoi diagram &rarr; take neighbor pairs from <code>ridge_points</code> &rarr; map indices back to city IDs &rarr; enforce <code>city1 &lt; city2</code> and de-duplicate.</li>
  <li><strong>Output:</strong> undirected edge list <code>city1, city2</code> where each pair denotes cities whose Voronoi regions share a border.</li>
</ul>

<p></p>
<p></p>
{% highlight python %}
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import Voronoi
def build_voronoi_edges(city_avg_vectors: pd.DataFrame,
    lon_col="lng", lat_col="lat",
    id_col="cityInd",
    jitter_m=10.0) -> pd.DataFrame:
    df = city_avg_vectors[[id_col, lon_col, lat_col]].dropna().copy()
    df = df.sort_values(id_col).reset_index(drop=True)
    transformer =
       Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    XY = np.array([transformer.transform(lon, lat)
       for lon, lat in zip(df[lon_col], df[lat_col])])
    rounded = np.round(XY, 6)
    dup_mask =
       pd.Series(map(tuple, rounded)).duplicated(keep=False).to_numpy()
    if dup_mask.any():
        rng = np.random.default_rng(42)
        XY[dup_mask] += rng.normal(scale=jitter_m, size=(dup_mask.sum(), 2))
    vor = Voronoi(XY)
    idx_pairs =
       set(tuple(sorted((int(i), int(j)))) for i, j in vor.ridge_points)
    edges = pd.DataFrame(
        [(df.loc[i, id_col], df.loc[j, id_col]) for i, j in idx_pairs],
        columns=["city1", "city2"]
    ).drop_duplicates().reset_index(drop=True)
    return edges
{% endhighlight %}
<p></p>

<p></p>
<p></p>
{% highlight python %}
voronoi_edges = build_voronoi_edges(city_avg_vectors)
voronoi_edges.tail()
{% endhighlight %}
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/gnnge4g.jpg" alt="Post Sample Image" width="160" >
</a>
<p></p>
<p></p>

<p></p>

<p></p>

<p></p>

<p></p>
<h3>Edges of Local Subgraphs</h3>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/gnnge3i.jpg" alt="Post Sample Image" width="489" >
</a>
<p></p>
To create many small subgraphs, we take each city as a center and collect all cities within 2 hops on the global Voronoi graph (its neighbors and their neighbors). We then keep only the edges inside this 2-hop set (the induced subgraph), assign a graph name like local_<cityInd>_k2, and use each city’s existing feature vector (f0..f364) as node features. Repeating this for every city yields a consistent set of small, comparable subgraphs centered on each location.

<p></p>
<p></p>
{% highlight python %}
import pandas as pd
import networkx as nx
def local_edges_for_nodes(voronoi_edges: pd.DataFrame,
   node_ids: list, graph_name: str):
   sub = voronoi_edges[
      voronoi_edges["city1"].isin(node_ids) &
      voronoi_edges["city2"].isin(node_ids)
   ].copy()
   left  = sub[["city1","city2"]].min(axis=1)
   right = sub[["city1","city2"]].max(axis=1)
   out = pd.DataFrame({"graph_name": graph_name, "left": left, "right": right})
return out.drop_duplicates()
def build_local_voronoi_edges(city_avg_vectors: pd.DataFrame,
   voronoi_edges: pd.DataFrame,graph_prefix: str = "local"):
   G = nx.from_pandas_edgelist(voronoi_edges, "city1", "city2",
      create_using=nx.Graph())
   k = 2  
   all_edges = []
   for cid in city_avg_vectors["cityInd"].astype(int):
      node_ids = list(nx.single_source_shortest_path_length(G, cid,
      cutoff=k).keys())
   gname = f"{graph_prefix}_{cid}_k{k}"
   all_edges.append(local_edges_for_nodes(voronoi_edges, node_ids, gname))
   edges = pd.concat(all_edges, ignore_index=True).drop_duplicates()
return edges
{% endhighlight %}
<p></p>

<p></p>
<p></p>
{% highlight python %}
local_graph_edges.tail()
{% endhighlight %}
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/gnnge4i.jpg" alt="Post Sample Image" width="222" >
</a>
<p></p>
<p></p>
{% highlight python %}
**
{% endhighlight %}
<p></p>
<p></p>
<h3>Nodes of Local Subgraphs</h3>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/gnnge3j.jpg" alt="Post Sample Image" width="489" >
</a>
<p></p>
This step partitions

<p></p>
<p></p>
{% highlight python %}
**
{% endhighlight %}
<p></p>

<p></p>
<p></p>
{% highlight python %}
**
{% endhighlight %}
<p></p>
<p></p>
<h3>Transform to Pytorch Geometric Format</h3>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/gnnge3k.jpg" alt="Post Sample Image" width="489" >
</a>
<p></p>

This function converts tabular nodes/edges into a list of PyTorch Geometric <code>Data</code> objects—one per <code>graph_name</code>.
<ul>
  <li>Detects feature columns (<code>f0..fK</code>) and builds the node feature tensor <code>x</code> (NaNs → 0).</li>
  <li>Reads the per-graph label from <code>graph_label</code> and stores it as <code>y</code>.</li>
  <li>Maps <code>node_name</code> to indices, filters edges for that graph, and constructs an undirected <code>edge_index</code> by adding both <code>(u,v)</code> and <code>(v,u)</code>; uses an empty tensor if no edges.</li>
  <li>Attaches metadata: <code>data.graph_name</code> and <code>data.node_names</code>.</li>
  <li>Returns <code>[Data(...), Data(...), …]</code> ready for loaders and GNN models.</li>
</ul>

<p></p>
{% highlight python %}
import numpy as np
import torch
from torch_geometric.data import Data
def tables_to_pyg(nodes_table, edges_table):
   fcols = sorted([c for c in nodes_table.columns
      if str(c).startswith("f")],key=lambda s: int(str(s)[1:]))
   data_list = []
   for gname, nd in nodes_table.groupby("graph_name", sort=False):
      X = nd[fcols].to_numpy(dtype=np.float32)
      x = torch.from_numpy(np.nan_to_num(X))
      y = torch.tensor([int(nd["graph_label"].iloc[0])], dtype=torch.long)
      name_to_idx = {n: i for i, n in enumerate(nd["node_name"].astype(str))}
      ed = edges_table[edges_table["graph_name"] == gname]
      src, dst = [], []
      for a, b in zip(ed["left"].astype(str), ed["right"].astype(str)):
         if a in name_to_idx and b in name_to_idx:
            ia, ib = name_to_idx[a], name_to_idx[b]
         src += [ia, ib]
         dst += [ib, ia]
      edge_index = (torch.tensor([src, dst], dtype=torch.long)
      if len(src) > 0 else torch.empty((2, 0), dtype=torch.long))
         data = Data(x=x, edge_index=edge_index, y=y)
      data.graph_name = str(gname)               
      data.node_names = nd["node_name"].tolist()
      data_list.append(data)
return data_list
{% endhighlight %}
<p></p>

<p></p>
<h3>GNN Graph Classification</h3>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/gnnge3l.jpg" alt="Post Sample Image" width="489" >
</a>
<p></p>
This step partitions

<p></p>
<h3>Use Pre-final Vectors</h3>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/gnnge3m.jpg" alt="Post Sample Image" width="489" >
</a>
<p></p>
This step partitions

<p></p>
<h3>Edges of Single and Sliding Graphs</h3>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/gnnge3g.jpg" alt="Post Sample Image" width="489" >
</a>
<p></p>
This step partitions


 <h1>Training</h1>

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
        # Graph Embedding Step
        graph_embedding = global_mean_pool(x, batch)  # [num_graphs, hidden_channels]
        if return_graph_embedding:
            return graph_embedding  # Return graph-level embedding here
        # Classification Step
        x = F.dropout(graph_embedding, p=0.5, training=self.training)
        x = self.lin(x)
        return x
model = GCN(hidden_channels=128)
print(model)
{% endhighlight %}


<p></p>

After training the Graph Convolutional Network (GCN) model, this code snippet extracts the graph embedding for a specific graph in the dataset:
The graph embedding is stored in <code><span style="color: blue;">out</span></code>, capturing the structural and feature information of the entire graph.

<p></p>
{% highlight python %}
g = 0
out = model(dataset[g].x.float(), dataset[g].edge_index, dataset[g].batch, return_graph_embedding=True)
out.shape
torch.Size([1, 128])
{% endhighlight %}
<p></p>
<ul>
    <li><em>dataset[g].x.float()</em>: Node features as floating-point tensor.</li>
    <li><em>dataset[g].edge_index</em>: Edge list of the graph.</li>
    <li><em>dataset[g].batch</em>: Batch assignment for nodes.</li>
    <li><em>return_graph_embedding=True</em>: Requests the graph-level embedding instead of classification.</li>
</ul>

<p></p>
The following code processes a series of graphs using a GCN model, applies a softmax function to the outputs, extracts predictions and graph embeddings, and stores the embeddings along with graph indices in a list for further analysis.
<p></p>
<p></p>
{% highlight python %}
softmax = torch.nn.Softmax(dim = 1)
graphUnion=[]
for g in range(graphCount):
  label=dataset[g].y[0].detach().numpy()
  out = model(dataset[g].x.float(), dataset[g].edge_index, dataset[g].batch, return_graph_embedding=True)
  output = softmax(out)[0].detach().numpy()
  pred = out.argmax(dim=1).detach().numpy()
  graphUnion.append({'index':g,'vector': out.detach().numpy()})
{% endhighlight %}
<p></p>
<p></p>
Cosine similarity function:
<p></p>
{% highlight python %}
import pandas as pd
import torch
from torch.nn.functional import normalize
def cos_sim(a: torch.Tensor, b: torch.Tensor):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    a_norm = normalize(a, p=2, dim=1)
    b_norm = normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))
{% endhighlight %}
<p></p>
This code calculates the cosine similarity between pairs of graph embeddings stored in <code><span style="color: blue;">graphUnion</span></code> and appends the results, along with their corresponding graph indices, to the <code><span style="color: blue;">cosine_similarities</span></code> list.
<p></p>
<p></p>
<p></p>
{% highlight python %}
cosine_similarities = []
for i in range(len(graphUnion)):
    for j in range(i+1, len(graphUnion)):  
        vector_i = torch.tensor(graphUnion[i]['vector'])
        vector_j = torch.tensor(graphUnion[j]['vector'])
        cos_sim_value = cos_sim(vector_i, vector_j).numpy().flatten()[0]  
        cosine_similarities.append({
            'left': graphUnion[i]['index'],
            'right': graphUnion[j]['index'],
            'cos': cos_sim_value
        })
{% endhighlight %}
<p></p>
<p></p>

<p></p>
<p></p>
{% highlight python %}
**
{% endhighlight %}
<p></p>

<p></p>
{% highlight python %}
{% endhighlight %}
<p></p>





    <p></p>

        <h4>Data Preparation and Model Training</h4>
  <p></p>
In our project, we developed a Graph Neural Network (GNN) Graph Classification model to analyze climate data. We created individual graphs for each city, labeling them as 'stable' or 'unstable' based on their latitude. Edges in these graphs were defined by high cosine similarities between node pairs, indicating similar temperature trends. To ensure consistency across all graphs, we introduced virtual nodes, which improved connectivity and helped the model generalize across different urban climates.
  <p></p>
For our analysis, we used the GCNConv model from the PyTorch Geometric (PyG) library. This model is excellent for extracting important feature vectors from graphs before making final classification decisions, which are essential for a detailed analysis of climate patterns.


<p></p>

 <a href="#">
     <img src="{{ site.baseurl }}/img/preFinalVector1.jpg" alt="Post Sample Image" width="678" >
 </a>
 <p></p>
  <p></p>
The GCNConv model performed very well, with accuracy rates of around 94% on training data and 92% on test data. These results highlight the model’s ability to effectively detect and classify unusual climate trends using daily temperature data represented in graph form.
  <p></p>   

<h4>Application of Graph Embedded Vectors: Cosine Similarity Analysis</h4>
<p></p>

  After training the GNN Graph Classification model, we transformed each city graph into an embedded vector. These vectors became the foundation for our subsequent data analyses.
<p></p>
<h5>Cosine Similarity Matrix Analysis of Graph-Embedded Vectors</h5>
<p></p>
  We constructed a cosine similarity matrix for 1000 cities to identify closely related climate profiles. This matrix allows for detailed comparisons and clustering based on the embedded vector data.
<p></p>
  To illustrate, we examined the closest neighbors of the graph vectors for Tokyo, Japan (the largest city in our dataset), and Gothenburg, Sweden (the smallest city in our dataset). Tokyo’s closest neighbors are primarily major Asian cities, indicating strong climatic and geographical similarities. Similarly, Gothenburg’s nearest neighbors are predominantly European cities, reflecting similar weather patterns across Northern and Central Europe.
<p></p>
  We also identified vector pairs with the lowest cosine similarity, specifically -0.543011, between Ulaanbaatar, Mongolia, and Shaoguan, China. This negative similarity suggests stark climatic differences. Additionally, the pair with a cosine similarity closest to 0.0 (-0.000047), indicating orthogonality, is between Nanchang, China, and N’Djamena, Chad. This near-zero similarity underscores the lack of a significant relationship between these cities’ climatic attributes.
<p></p>




    <p></p>    

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
Code to identify the top 5 closest neighbors to a specific node (node 0) based on cosine similarity values:
<p></p>
<ul>
    <li>Select neighbors where node 0 is either the 'left' or 'right' node from the DataFrame <em>dfCosSim</em>.</li>
    <li>Concatenate these rows into a single DataFrame <em>neighbors</em>.</li>
    <li>Sort the combined DataFrame by cosine similarity in descending order to prioritize the closest neighbors.</li>
    <li>Add a 'neighbor' column to identify the neighboring node, adjusting between 'left' and 'right' as needed.</li>
    <li>Select the top 5 rows with the highest cosine similarity and keep only the 'neighbor' and 'cos' columns.</li>
</ul>

    <p></p>

    {% highlight python %}
    neighbors_left = dfCosSim[dfCosSim['left'] == 0]
    neighbors_right = dfCosSim[dfCosSim['right'] == 0]
    neighbors = pd.concat([neighbors_left, neighbors_right])
    neighbors = neighbors.sort_values(by='cos', ascending=False)
    neighbors['neighbor'] = neighbors.apply(lambda row: row['right'] if row['left'] == 0 else row['left'], axis=1)
    top_5_neighbors = neighbors.head(5)
    top_5_neighbors = top_5_neighbors[['neighbor', 'cos']]
    {% endhighlight %}


    </p><p>  

<h5>Analyzing Climate Profiles with Cosine Similarity Matrix</h5>
<p></p>


The cosine similarity matrix distribution from the embedded city graphs reveals distinct clustering patterns, with notable peaks for values over 0.9 and between -0.4 to -0.2. These peaks indicate clusters of cities with nearly identical climates and those with shared but less pronounced features. This skewed distribution highlights areas with the highest concentration of values, providing essential insights into the relational dynamics and clustering patterns of the cities based on their climate data. The bar chart clearly illustrates how cities with similar climate profiles group together.

<p></p>
Table 3. Distribution of Cosine Similarities.
      <a href="#">
          <img src="{{ site.baseurl }}/img/preFinTab3.jpg" alt="Post Sample Image" width="256" >
      </a>

<p></p>
<p></p>

 <a href="#">
     <img src="{{ site.baseurl }}/img/preFinFig2.jpg" alt="Post Sample Image" width="678" >
 </a>
<p></p>
Code for distribution of cosine similarities
<p></p>
{% highlight python %}
import matplotlib.pyplot as plt
plt.figure(figsize=(7, 5))  # Adjust the size of the figure, swapped dimensions for vertical orientation
plt.hist(dfCosSim['cos'], bins=25, alpha=0.75,
         color='CornflowerBlue',
         orientation='horizontal')  # Set orientation to horizontal
plt.title('Distribution of Cosine Similarities')
plt.ylabel('Cosine Similarity')  # Now y-axis is cosine similarity
plt.xlabel('Frequency')  # And x-axis is frequency
plt.grid(True)
plt.show()
{% endhighlight %}

<p></p>
<h4>Application of Graph Embedded Vectors: Graphs Derived from Cosine Similarity Thresholds</h4>
<p></p>
Based on the observed distribution of cosine similarities, we generated three distinct graphs for further analysis, each using different cosine similarity thresholds to explore their impact on city pair distances.


<p></p>
To calculate distances between cities we used the following code:
<p></p>
{% highlight python %}
from math import sin, cos, sqrt, atan2, radians
def dist(lat1,lon1,lat2,lon2):
  rlat1 = radians(float(lat1))
  rlon1 = radians(float(lon1))
  rlat2 = radians(float(lat2))
  rlon2 = radians(float(lon2))
  dlon = rlon2 - rlon1
  dlat = rlat2 - rlat1
  a = sin(dlat / 2)**2 + cos(rlat1) * cos(rlat2) * sin(dlon / 2)**2
  c = 2 * atan2(sqrt(a), sqrt(1 - a))
  R=6371.0
  return R * c

  def cityDist(city1,country1,city2,country2):
    lat1=cityMetadata[(cityMetadata['city_ascii']==city1)
      & (cityMetadata['country']==country1)]['lat']
    lat2=cityMetadata[(cityMetadata['city_ascii']==city2)
      & (cityMetadata['country']==country2)]['lat']
    lon1=cityMetadata[(cityMetadata['city_ascii']==city1)
      & (cityMetadata['country']==country1)]['lng']
    lon2=cityMetadata[(cityMetadata['city_ascii']==city2)
      & (cityMetadata['country']==country2)]['lng']
    return dist(lat1,lon1,lat2,lon2)  

{% endhighlight %}
<p></p>


The following function filters a DataFrame for high cosine similarity values, creates a graph, and adds edges between nodes with high similarities, ready for further analysis or visualization.
<p></p>
{% highlight python %}
import networkx as nx
import matplotlib.pyplot as plt
df=dfCosSim
high_cos_df = df[df['cos'] > 0.9]
G = nx.Graph()
if not high_cos_df.empty:
    for index, row in high_cos_df.iterrows():
        G.add_edge(row['left'], row['right'], weight=row['cos'])
{% endhighlight %}

<p></p>

<p></p>


The following code enriches the edges of the graph <em>G</em> with distance information and then collects all the distance values into a list for further analysis:
<p></p>
{% highlight python %}
for _, row in distData.iterrows():
  if G.has_edge(row['left'], row['right']):
    G[row['left']][row['right']]['distance'] = row['distance']

distances = [attr['distance'] for u, v, attr in G.edges(data=True)]
mean_distance = np.mean(distances)
median_distance = np.median(distances)
std_deviation = np.std(distances)
min_distance = np.min(distances)
max_distance = np.max(distances)
{% endhighlight %}
<p></p>
This code iterates through the <em>distData</em> DataFrame, checks for existing edges in the graph <em>G</em>, and adds distance attributes to these edges. It then calculates the mean, median, standard deviation, minimum, and maximum of the distance values.
<p></p>
<b>For the first graph</b>, we used a high similarity threshold (cosine similarity > 0.9).

The statistics for the distances between city pairs in the first graph are as follows:
<ul>
    <li><strong>Mean distance</strong>: 7942.658 km</li>
    <li><strong>Median distance</strong>: 7741.326 km</li>
    <li><strong>Standard deviation</strong>: 5129.801 km</li>
    <li><strong>Minimum distance</strong>: 1.932 km</li>
    <li><strong>Maximum distance</strong>: 19975.287 km</li>
</ul>


<p></p>               
The shortest distance pair is between Jerusalem, Israel, and Al Quds, West Bank, with nearly identical latitude and longitude coordinates (31.7784, 35.2066 for Jerusalem and 31.7764, 35.2269 for Al Quds), highlighting their close proximity. In contrast, the longest distance pair is between Quito, Ecuador, and Pekanbaru, Indonesia. These cities, located on opposite sides of the world, have dramatically different geographical coordinates (-0.2150, -78.5001 for Quito and 0.5650, 101.4250 for Pekanbaru), spanning a vast distance across the globe.

<p></p>


<b>For the second graph</b>, defined by a cosine similarity threshold ranging from -0.4 to -0.2, we observed a moderate level of climatic similarity among city pairs. The key statistics for this graph are as follows:

<p></p>

<p></p>
<ul>
    <li><strong>Mean distance</strong>: 8648.245 km</li>
    <li><strong>Median distance</strong>: 8409.507 km</li>
    <li><strong>Standard deviation</strong>: 4221.592 km</li>
    <li><strong>Minimum distance</strong>: 115.137 km</li>
    <li><strong>Maximum distance</strong>: 19963.729 km</li>
</ul>

<p></p>

For this graph, the shortest distance pair is between Kabul, Afghanistan (latitude 34.5167, longitude 69.1833) and Jalalabad, Afghanistan (latitude 34.4415, longitude 70.4361). The longest distance pair is between Mendoza, Argentina (latitude -32.8833, longitude -68.8166) and Shiyan, China (latitude 32.5700, longitude 110.7800).

<p></p>

Both the first and second graphs had just one connected component. To generate a graph with several connected components, we examined graphs with very high thresholds.
<p></p>
<b>For the third graph</b>, we used a high similarity threshold (cosine similarity > 0.99), resulting in connected components of sizes [514, 468, 7, 5]. The largest connected component, with 514 nodes, predominantly includes cities with stable climates (475 nodes labeled as stable) and a smaller portion with unstable climates (39 nodes labeled as unstable). The second-largest component, containing 468 nodes, primarily consists of cities with unstable climates (451 nodes labeled as unstable) and a few with stable climates (17 nodes labeled as stable). These findings indicate that cities within the same climate category (stable or unstable) exhibit higher similarity, leading to larger connected components, whereas similarities across different climate categories are less pronounced.
<p></p>
Table 4. Cities in the Third Connected Component (7 Nodes)
      <a href="#">
          <img src="{{ site.baseurl }}/img/preFinTab4.jpg" alt="Post Sample Image" width="383" >
      </a>
<p></p>
In the smaller connected components, city graphs represent areas on the border between stable and unstable climates. The cities in these smaller components illustrate the variability and complexity of climatic relationships, showing a blend of stable and unstable climatic conditions. This underscores the nuanced and intricate climatic patterns that exist at the boundaries between different climate categories.
<p></p>




Table 5. Cities in the Fourth Connected Component (5 Nodes)
          <a href="#">
              <img src="{{ site.baseurl }}/img/preFinTab5.jpg" alt="Post Sample Image" width="383" >
          </a>
              <p></p>

<p></p>
<p></p>



    <p></p>


<p></p>

<p></p>

<p></p>

<p></p>




<p></p>

<h3>In Conclusion</h3>
<p></p>


In this study, we explored how pre-final vectors from GNN models can be applied in GNN Graph Classification. We showed that linear algebra is vital in transforming various data types into uniform vector formats that deep learning models can effectively use.
<p></p>
Our research demonstrated how GNN Graph Classification models capture complex graph structures through advanced linear algebra techniques. By embedding entire 'small graphs' from these models, we opened up new possibilities for analyzing and clustering small graphs, finding nearest neighbors, and creating meta-graphs.
<p></p>
The results suggest that combining linear algebra with GNNs enhances the models' efficiency and scalability, making them useful in many fields. By capturing and analyzing embedded graphs from GNN Graph Classification models, we can significantly improve data analysis and predictive abilities, advancing artificial intelligence and its many applications.
<p></p>



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
