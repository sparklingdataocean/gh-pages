---
layout:     post
title:      "Monitoring Market Topology Shifts"
subtitle:   "Sliding-graph embeddings for SPY vs sector ETFs"
date:       2026-01-26 12:00:00
author:     "Melenar"
header-img: "img/finsg15e.jpg"
---

LLMs excel at language, but they are natively sequential—making them a poor default tool when the signal is structural and relational. Financial time series have a similar blind spot: standard rolling comparisons summarize local movement, but can miss early shifts in market topology. When relationships matter, Graph AI offers an alternative: we represent each series as a sequence of sliding graphs, use a GNN to embed each graph snapshot into a vector, and align embedding streams to produce local similarity timelines for SPY vs sector ETFs. We compare this to a rolling-window baseline and flag short episodes where the two views disagree most—reviewable moments when relationships may be reorganizing even if sequential similarity looks stable.

<p></p>
<h2>Introduction</h2>


<section class="post-section" id="why-sliding-graphs-for-finance">
  <p>
    Large language models (LLMs) have become remarkably good at generating and summarizing text, but their limits show up when we ask
    them to act like general reasoning engines. As Yann LeCun and others have noted, LLMs are fundamentally <em>sequential</em>:
    their native representation is a one-dimensional ordering of symbols. That sequential topology is great for fluent continuation,
    but it is a poor fit for problems where the signal is structural, relational, or shaped by global constraints. In those settings,
    outputs can sound plausible while still being hard to verify or reconcile at the system level.
  </p>

  <p>
    This points to a broader issue of representation. Human reasoning does not rely on natural language alone; we use formal
    languages—mathematics, diagrams, programs—to make structure explicit. Mathematics is not the world itself, but a language for
    representing quantities and manipulating them under well-defined rules, which makes constraint checking and transformation
    possible. Graphs play a similar role for relationships. Nodes encode persistent entities, edges encode connections, and topology
    defines neighborhoods by connectivity rather than positional proximity. Many real-world signals depend on multi-hop dependencies,
    shared intermediaries, and global relational consistency—properties that are difficult to express faithfully in purely sequential
    form.
  </p>

  <p>
    Time series analysis inherits similar constraints. Standard financial time series models represent data as sequences of values
    indexed by time. This works well for trends, volatility, and local temporal dependence, but it often treats relationships between
    entities as secondary. Correlations are typically computed pairwise and regime changes inferred indirectly from aggregate metrics,
    even though relational reorganization across assets can happen before obvious changes appear in individual series.
  </p>

  <p>
    To address this gap for financial time series, we use <strong>Sliding Graphs</strong> and adapt them to market data. We construct
    overlapping temporal windows, build a graph representation for each window, and compute graph embeddings that capture how recent
    behavior is organized across assets. This produces structure-aware similarity timelines that help detect and localize relationship
    change (for example, sector reconfiguration or regime shifts) that value-based rolling comparisons can miss. Practically, we
    flag periods where standard rolling-window similarity and sliding-graph similarity disagree, treating that disagreement as a
    measurable indicator that market relationships are reorganizing.
  </p>
</section>

<p></p>  




<p></p>

<p></p>



<p></p>





<p></p>

<h2>From Prior Work to This Study</h2>


<p>
  This study builds directly on our prior work; the four items below are the key building blocks
  that made the current workflow possible.
</p>


<p>
  <strong>1) GNN Graph Classification for time series</strong><br>
  We first adapted <strong>GNN graph classification</strong> (commonly used for molecular graph labeling) to
  <strong>time series</strong> by turning time-local segments into graph snapshots and training a graph classifier
  on those graphs. A key practical detail in this line of work is the <strong>virtual node</strong>, added to stabilize
  message passing and produce a consistent graph-level representation.<br>
  Blog: <a href="https://sparklingdataocean.com/2023/02/11/cityTempGNNgraphs/" target="_blank" rel="noopener">https://sparklingdataocean.com/2023/02/11/cityTempGNNgraphs/</a><br>
  Presented at: <strong>ICANN 2023</strong> — Crete, Greece (Sep 2023) and <strong>COMPLEX NETWORKS 2023</strong> — Menton, France (Nov 2023).
</p>

<p>
  <strong>2) Sliding graphs (introduced on climate time series)</strong><br>
  We then formalized <strong>sliding graphs</strong>: instead of treating a long sequence as one object, we represent it
  as a <strong>sequence of overlapping graph snapshots</strong>, each capturing recent local behavior and structure.<br>
  Blog: <a href="http://sparklingdataocean.com/2024/05/25/slidingWindowGraph/" target="_blank" rel="noopener">http://sparklingdataocean.com/2024/05/25/slidingWindowGraph/</a><br>
  Presented at: <strong>ICMLT 2024</strong> — Oslo, Norway (May 2024).
</p>

<p>
  <strong>3) Pre-final vectors from GNN graph classification</strong><br>
  Next, we showed how to reuse the model’s <strong>pre-final vectors</strong> (graph embeddings right before the last layer)
  as a stable representation you can track, compare, and analyze beyond classification.<br>
  Blog: <a href="https://sparklingdataocean.com/2024/07/04/vectorsGNN/" target="_blank" rel="noopener">https://sparklingdataocean.com/2024/07/04/vectorsGNN/</a><br>
  Presented at: <strong>MLG 2024 (ECML-PKDD 2024 workshop)</strong> — Vilnius, Lithuania (Sep 2024).
</p>

<p>
  <strong>4) Time-aligned sliding-graph embeddings for new time series (EEG)</strong><br>
  Finally, we applied the same <strong>sliding graphs + embeddings</strong> workflow to EEG and introduced
  <strong>time-aligned embedding streams</strong> for comparing structure over time.<br>
  Blog: <a href="https://sparklingdataocean.com/2025/01/25/slidingWindowGraph-EEG/" target="_blank" rel="noopener">https://sparklingdataocean.com/2025/01/25/slidingWindowGraph-EEG/</a><br>
  Presented at: <strong>Brain Informatics 2025</strong> — Bari, Italy (Nov 2025).
</p>




<h2>Methods</h2>
<p></p>


<section>
  <h3>Pipeline overview: from time series to a time-aligned similarity signal</h3>

  <p>
    This pipeline converts a pair of time series into a single, time-aligned similarity timeline.
    The key idea is to compare two <em>streams of learned graph embeddings</em> at matching time points,
    rather than comparing raw values directly.
  </p>
Our pipeline for this study consists of several stages.
<p></p>
  <ol>
    <li>
      <strong>Start with two time series (A and B).</strong><br />
      The goal is to track how their relationship evolves over time, not just compute one global similarity.
    </li>

    <li>
      <strong>Create overlapping sliding windows.</strong><br />
      Each window captures a short, recent segment of behavior. Overlap ensures a dense sequence of
      time-local snapshots.
    </li>

    <li>
      <strong>Construct sliding graphs.</strong><br />
      For every window, build a small graph snapshot that encodes local structure (e.g., which
      window-elements behave similarly and how they connect). This turns time into a <em>sequence of graphs</em>.
    </li>

    <li>
      <strong>Run a GNN and keep the pre-final embeddings.</strong><br />
      A graph neural network processes each graph snapshot and produces a compact embedding vector
      that summarizes the structure of that window. Doing this for both series yields two embedding
      streams over time: <em>Vectors A</em> and <em>Vectors B</em>.
    </li>

    <li>
      <strong>Align embeddings by timestamp and compute similarity at each time point.</strong><br />
      At each aligned window time, compute a similarity score (e.g., cosine similarity) between the
      embedding from A and the embedding from B.
    </li>

    <li>
      <strong>Obtain a similarity timeline.</strong><br />
      The output is a time series of similarity scores that reflects how the two series compare
      window-by-window in the learned embedding space—capturing changes in the organization/structure
      of recent patterns.
    </li>
  </ol>

  <p>
    In practice, this similarity timeline can be scanned for intervals where relationships strengthen,
    weaken, or reorganize—providing time-local “periods of interest” for deeper interpretation.
  </p>
</section>

<p></p>
<a href="#">
      <img src="{{ site.baseurl }}/img/fings17b.jpg" alt="Post Sample Image" width="808" >
</a>
<p></p>

<p></p>
<p></p>

<p></p>
<p></p>

<p></p>


<h3>Sliding graph definition</h3>
<p></p>

In our previous study, <a href="https://dl.acm.org/doi/10.1145/3674029.3674059">GNN Graph Classification for Time Series: A New Perspective on Climate Change Analysis</a>, we introduced an approach to constructing graphs using the
<em>Sliding Window Method</em>.

<p></p>
<a href="#">
      <img src="{{ site.baseurl }}/img/slide3.jpg" alt="Post Sample Image" width="404" >
</a>
  <p></p>
  <p></p>


<p></p>
<h3>Data to Graph Transformation</h3>
<p></p>
Time series data is segmented into overlapping windows using the sliding
window technique. Each segment forms a unique graph, allowing for the analysis of local temporal dynamics.
<p></p>
In these graphs:

  <ul>
    <li>
      <strong>Nodes</strong>: Represent data points within the window, with features derived from their values.
    </li>
    <li>
      <strong>Edges</strong>: Connect pairs of nodes to maintain temporal relationships.
    </li>
  </ul>
<p></p>
Key Parameters:
  <ul>
    <li>
      <strong>Window Size (W)</strong>: Determines the size of each segment.
    </li>
    <li>
      <strong>Shift Size (S)</strong>: Defines the degree of overlap between windows.
    </li>
    <li>
      <strong>Edge Definitions</strong>: Tailored to the specific characteristics of the time series, helping detect meaningful
      patterns.
    </li>
  </ul>



<p></p>
<p></p>
<h3>Node calculation</h3>
<p></p>
For a dataset with N data points, we apply a sliding window of size W with a shift of S to create nodes. The number of nodes, N<sub>nodes</sub>, is calculated as:
    <math xmlns="http://www.w3.org/1998/Math/MathML">
        <mrow>
            <msub>
                <mi>N</mi>
                <mi>nodes</mi>
            </msub>
            <mo>=</mo>
            <mrow>
                <mo>&lfloor;</mo>
                <mfrac>
                    <mrow>
                        <mi>N</mi>
                        <mo>-</mo>
                        <mi>W</mi>
                    </mrow>
                    <mi>S</mi>
                </mfrac>
                <mo>&rfloor;</mo>
            </mrow>
            <mo>+</mo>
            <mn>1</mn>
        </mrow>
    </math>
<p></p>
<p></p>
<p></p>


<p></p>
<h3>Graph calculation</h3>
<p></p>
With the nodes determined, we construct graphs, each comprising G nodes, with a shift of S<sub>g</sub> between successive graphs. The number of graphs, N<sub>graphs</sub>, is calculated by:
    <math xmlns="http://www.w3.org/1998/Math/MathML">
        <mrow>
            <msub>
                <mi>N</mi>
                <mi>graphs</mi>
            </msub>
            <mo>=</mo>
            <mrow>
                <mo>&lfloor;</mo>
                <mfrac>
                    <mrow>
                        <msub>
                            <mi>N</mi>
                            <mi>nodes</mi>
                        </msub>
                        <mo>-</mo>
                        <mi>G</mi>
                    </mrow>
                    <msub>
                        <mi>S</mi>
                        <mi>g</mi>
                    </msub>
                </mfrac>
                <mo>&rfloor;</mo>
            </mrow>
            <mo>+</mo>
            <mn>1</mn>
        </mrow>
    </math>
<p></p>
<p></p>
<h3>Graph construction</h3>
<p></p>
Cosine similarity matrices are generated from the time series data and transformed into graph adjacency matrices.
<p></p>
  <ul>
    <li>
      <strong>Edge Creation</strong>: Edges are established for vector pairs with cosine values above a defined threshold.
    </li>
    <li>
      <strong>Virtual Nodes</strong>: Added to ensure network connectivity, enhancing graph representation.
    </li>
  </ul>
<p></p>

<p></p>
<p></p>
This framework effectively captures both local and global patterns within the time series, yielding valuable insights into temporal dynamics.
<p></p>

<p></p>
<h3>GNN graph classification</h3>
<p></p>
We employ the <em>GCNConv</em> model from the PyTorch Geometric Library for GNN Graph Classification tasks. This model performs convolutional operations, leveraging edges, node attributes, and graph labels to extract features and analyze graph structures comprehensively.
<p></p>
By combining the sliding window technique with Graph Neural Networks, our approach offers a robust framework for analyzing time series data. It captures intricate temporal dynamics and provides actionable insights into both local and global patterns, making it particularly well-suited for applications such as EEG data analysis. This method allows us to analyze time series data effectively by capturing both local and global patterns, providing valuable insights into temporal dynamics.
<p></p>

  <p>
  </p>


  <p></p>

<p></p>
<h2>Experiments</h2>
<p></p>

<p></p>
<section>
  <h3>Data</h3>

  <section class="post-section" id="raw-data">
    <p>
      For this project, we keep the dataset intentionally small: <strong>five liquid U.S. ETFs</strong> that tend to behave differently
      across market regimes—so shifts in “relationships” and sector rotation have a chance to show up:
    </p>

    <ul>
      <li><strong>SPY</strong> (S&amp;P 500)</li>
      <li><strong>XLK</strong> (Technology)</li>
      <li><strong>XLE</strong> (Energy)</li>
      <li><strong>XLF</strong> (Financials)</li>
      <li><strong>XLU</strong> (Utilities)</li>
    </ul>

    <p>
      We pull <strong>daily prices from Yahoo Finance</strong> using <code>yfinance</code>, and we use <strong>Adjusted Close</strong> so
      splits and dividends don’t create artificial jumps in the series. In our run, we request data starting <strong>1998-12-01</strong>,
      then align all tickers to the <strong>intersection of trading dates</strong> (shared window
      <strong>1998-12-22 to 2026-01-21</strong>, <strong>6811</strong> rows per ETF). The aligned series are saved to
      <strong>Parquet</strong> (one file per ticker) so downstream steps load instantly without re-downloading.
    </p>

    <p>
      Because these ETFs trade at very different price levels, we don’t plot raw prices. Instead, we normalize each series so it starts
      at <strong>1.0</strong> on the first shared trading day:
    </p>
  </section>

  </p>
  <p></p>
  <a href="#">
        <img src="{{ site.baseurl }}/img/finsg3.jpg" alt="Post Sample Image" width="808" >
  </a>
  <p></p>

</section>


<p></p>
<p>
<section class="post-section" id="spy-ratios-illustration">
  <p>
    We’re going to analyze how relationships evolve between <strong>SPY</strong> and each sector ETF. As a quick illustration, the chart
    below shows <strong>price ratios</strong> of each ETF versus SPY, which makes it easier to see when a sector is leading or lagging
    the broad market.
  </p>
  <p></p>
  <a href="#">
        <img src="{{ site.baseurl }}/img/finsg6.jpg" alt="Post Sample Image" width="808" >
  </a>
  <p></p>
  <p>
    A few major periods stand out. <strong>Tech (XLK)</strong> shows two clear eras of relative leadership: a surge during the
    <strong>dot-com period</strong> (late 1990s into the early 2000s) followed by a sharp reversal, and then a renewed, sustained push
    in the recent <strong>AI-driven boom</strong> (2023–present). <strong>Financials (XLF)</strong> is dominated by the
    <strong>2008 crisis</strong>, where the ratio collapses and the recovery afterward follows a noticeably different trajectory.
    Finally, <strong>COVID (2020)</strong> appears as a sharp relationship shock, with a rapid drawdown, quick rebound, and a clear
    reshuffling of sector leadership.
  </p>
</section>

</p>


  <section class="post-section" id="creating-graph-nodes">
    <h3>Creating graph nodes</h3>

    <p>
      For each ETF, we start with its <strong>Adjusted Close</strong> time series and turn it into a set of
      <strong>graph nodes</strong>, where each node represents a short, fixed-length “recent history” of price movement.
      Concretely, we slide a window of length <strong>W</strong> over the series (step <strong>S</strong>). Each window becomes one node.
    </p>

    <p>
      Before we do anything graph-like, we normalize each window so that it starts at the same baseline. The simplest view is price
      normalization:
    </p>

    <p>
      <strong>Normalized price(t) = AdjClose(t) / AdjClose(t₀)</strong>
    </p>

    <p>
      This forces every window to start at <strong>1.0</strong> on day one, so we focus on <em>shape</em> rather than absolute price level.
    </p>

    <p>
      In the implementation, we use the log version of the same idea because it behaves better numerically and aligns with how finance
      often models returns. For a window starting at time <span style="white-space:nowrap;">t₀</span>, we compute a
      <strong>relative log-price path</strong>:
    </p>

    <p>
      <strong>features = log(price_window) − log(price_window[0])</strong>
    </p>

    <p>
      So each node feature vector encodes the <em>trajectory</em> of that ETF over the next <strong>W</strong> points, expressed relative
      to the first value in the window. Intuitively: every node says, “starting from here, what did the local path look like?”
      This gives us a clean, comparable node representation across different ETFs and time periods—exactly what we need before
      constructing sliding graphs over these nodes.
    </p>

  </section>

  <section class="post-section" id="fin-graph-nodes-io">


    <p><strong>Inputs</strong></p>
    <ul>
      <li>
        <code>df</code> (DataFrame) with at least:
        <ul>
          <li><code>timestamp_col</code> (default <code>"timestamp"</code>)</li>
          <li><code>price_col</code> (default <code>"adj_close"</code>, must be &gt; 0; non-positive rows are dropped)</li>
        </ul>
      </li>
      <li><code>symbol</code> (string) used to build unique node names</li>
      <li>
        Windowing parameters:
        <ul>
          <li><code>W = 32</code>: node window length (feature vector size)</li>
          <li><code>S = 1</code>: node stride (step between windows)</li>
          <li><code>G = 32</code>: nodes per graph snapshot</li>
          <li><code>Sg = 6</code>: snapshot stride</li>
        </ul>
      </li>
      <li><code>graph_label</code> (int) assigned to all snapshots from this call</li>
      <li><code>eps</code> protects the log transform</li>
    </ul>



    <p><strong>Output</strong></p>
    <p>
      A DataFrame where each row is a node-in-snapshot, with columns:
      <code>graph_name</code>, <code>graph_label</code>, <code>node_name</code>, and
      <code>f0</code> … <code>f{W-1}</code>.
      If there isn’t enough data to form windows (<code>N &lt; W</code>) or snapshots (<code>n_nodes &lt; G</code>), the function returns
      an empty DataFrame with the expected columns.
    </p>
  </section>



<h4>Function: fin_graph_nodes.py</h4>
  <p></p>
{% highlight python %}
from __future__ import annotations
import numpy as np
import pandas as pd
def fin_graph_nodes(
    df: pd.DataFrame,
    symbol: str,
    timestamp_col: str = "timestamp",
    price_col: str = "adj_close",
    W: int = 32,
    S: int = 1,
    G: int = 32,
    Sg: int = 6,
    graph_label: int = 0,
    eps: float = 1e-12,   # protects log(0)
) -> pd.DataFrame:
    if W <= 0 or G <= 0 or S <= 0 or Sg <= 0:
        raise ValueError("W, G, S, Sg must be positive integers.")
    d = df[[timestamp_col, price_col]].copy()
    d[timestamp_col] =
       pd.to_datetime(d[timestamp_col], utc=True, errors="coerce")
    d[price_col] = pd.to_numeric(d[price_col], errors="coerce")
    d = d.dropna(subset=[timestamp_col, price_col])
       .sort_values(timestamp_col).reset_index(drop=True)
    d = d[d[price_col] > 0].reset_index(drop=True)
    prices = d[price_col].to_numpy(dtype=float)
    ts = d[timestamp_col].to_list()
    N = len(prices)
    out_cols = ["graph_name", "graph_label", "node_name"] +
      [f"f{i}" for i in range(W)]
    if N < W:
        return pd.DataFrame(columns=out_cols)
    logp = np.log(np.maximum(prices, eps))
    node_starts = np.arange(0, N - W + 1, S, dtype=int)
    node_names = np.array([f"{symbol}_{ts[i].isoformat()}"
    for i in node_starts], dtype=object)
    X = np.stack(
        [(logp[i:i + W] - logp[i]) for i in node_starts],  
        axis=0)  
    n_nodes = X.shape[0]
    if n_nodes < G:
        return pd.DataFrame(columns=out_cols)
    graph_starts = np.arange(0, n_nodes - G + 1, Sg, dtype=int)
    rows = []
    for gs in graph_starts:
        gname = node_names[gs]
        for j in range(G):
            nm = node_names[gs + j]
            feat = X[gs + j]
            row = {"graph_name": gname, "graph_label": int(graph_label),
            "node_name": nm}
            row.update({f"f{k}": float(feat[k]) for k in range(W)})
            rows.append(row)
   return pd.DataFrame(rows, columns=out_cols)
{% endhighlight %}
  <p></p>

<section class="post-section" id="graph-edges">
<h3>Graph edges </h3>
    <p>
      Once we have the <strong>node windows table</strong>, we build edges for each <strong>sliding graph snapshot</strong>. Edges define
      which nodes are “related” within that snapshot and control how information flows in the GNN.
    </p>

    <p><strong>Virtual-node edges (always on)</strong><br>
      For every snapshot, we add one special <strong>virtual node</strong> and connect it to every real node (a star pattern). This makes
      each input graph a <strong>single connected component</strong>, which helps <strong>GNN graph classification models</strong> learn a
      stable graph-level representation. The virtual node acts as a <strong>global anchor</strong>, aggregating signals from all nodes.
    </p>

    <p><strong>Cosine-similarity edges (optional)</strong><br>
      Within each snapshot, we optionally connect node pairs whose feature vectors are highly similar (above a cosine threshold). This
      links windows with similar local shape, capturing repeating patterns.
    </p>




    <p>
  <strong>Inputs:</strong> the previously created nodes table, a
  <code style="background:#fdecef;color:#b42318;padding:0.1em 0.35em;border-radius:0.25em;">cosine_threshold = 0.95</code>.<br>
  <strong>Outputs:</strong> a new nodes table with one virtual node added per graph, and an edges table with
  <code>graph_name</code>, <code>left</code>, <code>right</code>, and <code>edge_type</code>.
</p>

  </section>



  <p></p>
<h4>Function: fin_graph_edges.py</h4>
  <p></p>
{% highlight python %}
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
PathLike = Union[str, Path]

def _feature_cols(df: pd.DataFrame) -> list[str]:
    fcols = [c for c in df.columns
    if isinstance(c, str) and c.startswith("f") and c[1:].isdigit()]
    fcols.sort(key=lambda s: int(s[1:]))
    if not fcols:
        raise ValueError("No feature columns found (expected f0..fK).")
    return fcols

def _normalize_windows(X: np.ndarray, mode: str = "zscore", eps: float = 1e-12)
   -> np.ndarray:
    if mode == "none":
        return X
    mu = np.nanmean(X, axis=1, keepdims=True)
    Xc = X - mu
    if mode == "demean":
        return Xc
    if mode == "zscore":
        sd = np.nanstd(Xc, axis=1, keepdims=True)
        sd = np.where(sd > 0, sd, 1.0)
        return Xc / (sd + eps)
    raise ValueError("window_norm must be one of: 'zscore', 'demean', 'none'")

def build_cosine_edges_with_virtual(
    nodes_table: pd.DataFrame,
    *,
    cosine_threshold: float = 0.92,
    virtual_node_name: str = "__VIRTUAL__",
    normalize_windows: bool = True,
    window_norm: str = "zscore",
    eps: float = 1e-12,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(nodes_table, pd.DataFrame):
        raise TypeError("nodes_table must be a pandas DataFrame")
    if nodes_table.empty:
        return nodes_table.copy(),
          pd.DataFrame(columns=["graph_name", "left", "right", "edge_type"])
    required = {"graph_name", "graph_label", "node_name"}
    missing = required - set(nodes_table.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    fcols = _feature_cols(nodes_table)
    edges = []
    virtual_rows = []
    for gname, g in nodes_table.groupby("graph_name", sort=False):
        glabel = int(g["graph_label"].iloc[0])
        X_full = g[fcols].to_numpy(dtype=float)
        names_all = g["node_name"].astype(str).to_numpy()
        valid = ~np.isnan(X_full).any(axis=1)
        if int(valid.sum()) >= 2:
            X = X_full[valid]
            names_valid = g.loc[valid, "node_name"].astype(str).to_numpy()
            if normalize_windows:
                X = _normalize_windows(X, mode=window_norm, eps=eps)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            Xn = X / (norms + eps)
            sim = Xn @ Xn.T
            n = sim.shape[0]
            iu, ju = np.triu_indices(n, k=1)
            keep = sim[iu, ju] >= float(cosine_threshold)
            for i, j in zip(iu[keep], ju[keep]):
                left, right = names_valid[i], names_valid[j]
                if left > right:
                    left, right = right, left
                edges.append((gname, left, right, "cosine"))
        vfeat = np.nanmean(X_full, axis=0)
        vrow = {"graph_name": gname, "graph_label": glabel,
           "node_name": virtual_node_name}
        vrow.update({c: float(v) for c, v in zip(fcols, vfeat)})
        virtual_rows.append(vrow)
        for nm in names_all:
            left, right = virtual_node_name, nm
            if left > right:
                left, right = right, left
            edges.append((gname, left, right, "virtual"))
    nodes_out = pd.concat([nodes_table,
      pd.DataFrame(virtual_rows)], ignore_index=True)
    edges_df = pd.DataFrame(edges, columns=["graph_name",
      "left", "right", "edge_type"]).drop_duplicates()
    return nodes_out, edges_df

def fin_graph_edges(
    nodes_input: Union[pd.DataFrame, PathLike],
    *,
    cosine_threshold: float = 0.92,
    virtual_node_name: str = "__VIRTUAL__",
    normalize_windows: bool = True,
    window_norm: str = "zscore",
    save_dir: Optional[PathLike] = None,
    out_prefix: str = "etf",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(nodes_input, pd.DataFrame):
        nodes_table = nodes_input
    else:
        p = Path(nodes_input)
        if not p.exists():
            raise FileNotFoundError(p)
        if p.suffix.lower() in [".parquet", ".pq"]:
            nodes_table = pd.read_parquet(p)
        elif p.suffix.lower() == ".csv":
            nodes_table = pd.read_csv(p)
        else:
            raise ValueError("Unsupported file type. Use .parquet or .csv.")
    nodes_out, edges_df = build_cosine_edges_with_virtual(
        nodes_table,
        cosine_threshold=cosine_threshold,
        virtual_node_name=virtual_node_name,
        normalize_windows=normalize_windows,
        window_norm=window_norm,)
    if save_dir is not None:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        nodes_out.to_parquet(out_dir / f"{out_prefix}_nodes_virtual.parquet",
          index=False)
        edges_df.to_parquet(out_dir / f"{out_prefix}_edges.parquet", index=False)
    return nodes_out, edges_df
{% endhighlight %}

  <p></p>

  <section class="post-section" id="create-graph-list">
    <h3>Create graph list</h3>

    <p>
      At this stage, we convert each ticker’s saved tables into a list of small graph snapshots that a GNN can consume. We run
      <code>fin_graph_list</code> <strong>once per ticker</strong> (SPY, XLK, XLE, XLF, XLU), load that ticker’s
      <code>nodes_with_virtual</code> and <code>edges</code>, and export a PyTorch Geometric <code>Data</code> object for each
      <code>graph_name</code> window. The result is a <strong>graph list per ticker</strong>, saved to disk so we can reuse it without
      rebuilding features and edges.
    </p>

    <p>
      Later, in the GNN graph classification step, we reuse the <strong>SPY</strong> graph list four times—paired with each sector ETF
      list—so the comparison is always “SPY vs ETF,” but the graph lists themselves are created and stored independently.
    </p>

    <p>
      <strong>Inputs:</strong> one ticker’s <code>nodes_with_virtual</code> table and its <code>edges</code> table (paths to
      <code>.parquet</code>/<code>.csv</code>).<br>
      <strong>Outputs:</strong> a <code>data_list</code> of PyG <code>Data</code> graphs (one per <code>graph_name</code>), saved as a
      <code>.pkl</code> file for that ticker.
    </p>
  </section>


  <p></p>
<h4>Function: fin_graph_list.py</h4>
  <p></p>
{% highlight python %}
from __future__ import annotations
from pathlib import Path
from typing import List, Union, Optional
import numpy as np
import pandas as pd
PathLike = Union[str, Path]

def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}. Use .parquet or .csv")

def fin_graph_list(
    nodes0_path: PathLike,
    edges0_path: PathLike,
    nodes1_path: PathLike,
    edges1_path: PathLike,
    *,
    save: bool = True,
    out_name: Optional[str] = None,
) -> List["Data"]:
    import pickle
    import torch
    from torch_geometric.data import Data
    n0p, e0p, n1p,
    e1p = map(lambda p: Path(p), [nodes0_path, edges0_path, nodes1_path, edges1_path])
    for p in [n0p, e0p, n1p, e1p]:
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
    nodes0 = _load_table(n0p)
    edges0 = _load_table(e0p)
    nodes1 = _load_table(n1p)
    edges1 = _load_table(e1p)
    nodes_table = pd.concat([nodes0, nodes1], ignore_index=True)
    edges_table = pd.concat([edges0, edges1], ignore_index=True)
    fcols = [c for c in nodes_table.columns if isinstance(c, str)
       and c.startswith("f") and c[1:].isdigit()]
    fcols.sort(key=lambda s: int(s[1:]))
    if not fcols:
        raise ValueError("No feature columns found (expected f0..fK).")
    required_edges = {"graph_name", "left", "right"}
    miss_e = required_edges - set(edges_table.columns)
    if miss_e:
        raise ValueError(f"Edges table missing columns: {sorted(miss_e)}")
    data_list: List[Data] = []
    for gname, nd in nodes_table.groupby("graph_name", sort=False):
        X = nd[fcols].to_numpy(dtype=np.float32)
        x = torch.from_numpy(np.nan_to_num(X))  
        y = torch.tensor([int(nd["graph_label"].iloc[0])], dtype=torch.long)
        name_to_idx =
           {n: i for i, n in enumerate(nd["node_name"].astype(str).tolist())}
        ed = edges_table[edges_table["graph_name"] == gname]
        src, dst = [], []
        for a, b in zip(ed["left"].astype(str), ed["right"].astype(str)):
            if a in name_to_idx and b in name_to_idx:
                ia, ib = name_to_idx[a], name_to_idx[b]
                src += [ia, ib]
                dst += [ib, ia]
        edge_index = (
            torch.tensor([src, dst], dtype=torch.long)
            if len(src) > 0
            else torch.empty((2, 0), dtype=torch.long)
        )
        data = Data(x=x, edge_index=edge_index, y=y)
        data.graph_name = str(gname)
        data.node_names = nd["node_name"].astype(str).tolist()
        data_list.append(data)
    if save:
        out_dir = n0p.parent
        if out_name is None:
            sym0 = n0p.stem.replace("_with_virtual", "")
            sym1 = n1p.stem.replace("_with_virtual", "")
            out_name = f"fin_graphs_{sym0}__{sym1}.pkl"
        out_path = out_dir / out_name
        with open(out_path, "wb") as f:
            pickle.dump(data_list, f)
    return data_list
{% endhighlight %}
  <p></p>
  <p></p>


  <section class="post-section" id="gnn-graph-classification">
    <h3>GNN graph classification</h3>

    <p>
      Next, we train a PyG GNN graph classification model on pairs of graph lists. Each run combines
      SPY with one sector ETF: <strong>SPY–XLK</strong>, <strong>SPY–XLE</strong>, <strong>SPY–XLF</strong>,
      <strong>SPY–XLU</strong>. The graph lists were created earlier and reused here; this step is purely about learning from those
      prebuilt sliding-graph snapshots.
    </p>

    <p>
      The model predicts a binary graph label, but the main artifact we keep is the pre-final vectors:
      the sliding-graph embeddings (one embedding per snapshot).
    </p>

    <p>
      <strong>Inputs:</strong> two precomputed graph lists (one for SPY, one for the sector ETF).<br>
      <strong>Outputs:</strong> a time-aligned table of <strong>graph embeddings</strong> (one row per snapshot) plus
      <strong>accuracy metrics</strong>:
    </p>

    <ul>
      <li><strong>SPY–XLE:</strong> train 0.791, test 0.713</li>
      <li><strong>SPY–XLF:</strong> train 0.668, test 0.607</li>
      <li><strong>SPY–XLK:</strong> train 0.713, test 0.633</li>
      <li><strong>SPY–XLU:</strong> train 0.708, test 0.589</li>
    </ul>
  </section>

  <h3>From GNN outputs to similarity timelines</h3>

  <p>
    After training the GNN graph-classification model, we keep the <strong>pre-final embedding vectors</strong>
    produced for each sliding-graph snapshot. Because each snapshot corresponds to a specific time window,
    stacking these vectors in order gives a <strong>time-aligned embedding stream</strong> for each series.
  </p>

  <p>
    To convert embeddings into a similarity timeline, we: (1) attach each embedding to its window timestamp,
    (2) align the two embedding streams on the same timestamps, and (3) compute a single
    <strong>cosine similarity</strong> score between the two embedding vectors at each aligned time point.
  </p>

  <p>
    In parallel, we compute a <strong>sliding-window similarity</strong> baseline directly from the original
    rolling window (node) vectors. Plotting the <strong>sliding-graph</strong> and <strong>sliding-window</strong>
    similarity timelines together makes it easy to see when the two views track each other—and when they diverge.
  </p>



<h3>Interpreting the similarity timelines</h3>

<p></p>
<a href="#">
      <img src="{{ site.baseurl }}/img/finsd14.jpg" alt="Post Sample Image" width="808" >
</a>
<p></p>


<p>
We plot two similarity timelines for each pair (SPY, sector ETF) on the same dates. The goal is to compare two different “views” of how closely a sector is behaving relative to the broad market over time.
</p>

<p>
The first line is a sliding-window baseline (shown in grey). It answers a simple question: <em>do the recent price paths look similar right now?</em> In other words, it’s a local, short-horizon comparison of how the two series have been moving recently.
</p>

<p>
The second line is a sliding-graph similarity (shown in color). It uses learned graph embeddings from our sliding-graph representation. This view is meant to capture similarity in the <em>organization</em> of recent patterns, not just whether two recent paths look alike point-by-point.
</p>

<p>
Across pairs, the two timelines are often broadly consistent, but they are not identical. There are clear intervals where the window baseline stays relatively high while the graph-based similarity drops (and sometimes the reverse). That divergence is the key signal we use next — it suggests the two representations are emphasizing different aspects of the same market behavior.
</p>


<h3>Finding time periods of interest from disagreement</h3>

<p>
To identify time periods worth investigating, we focus on where the two similarity timelines disagree the most. For each pair (SPY, sector ETF), we align the two signals on the same timestamps and compute a simple “difference” timeline: when the graph-based similarity is much higher or much lower than the sliding-window baseline.
</p>

<p>
We then scan this difference signal for unusually large deviations, and group nearby deviations into short contiguous episodes. This turns scattered spikes into more interpretable time intervals.
</p>

<p>
Finally, we keep the strongest 1–2 episodes per sector as candidate periods of interest. In the current run, this produced clear episodes for Energy and Financials, only a single episode for Technology, and none for Utilities at the chosen sensitivity threshold. These episodes give us concrete, time-localized targets for deeper interpretation — moments when the graph view and the window view tell meaningfully different stories about similarity.
</p>
<p></p>
<a href="#">
      <img src="{{ site.baseurl }}/img/finsg10.jpg" alt="Post Sample Image" width="707" >
</a>
<p></p>
<p></p>
<a href="#">
      <img src="{{ site.baseurl }}/img/finsg11.jpg" alt="Post Sample Image" width="707" >
</a>
<p></p>
<p></p>
<a href="#">
      <img src="{{ site.baseurl }}/img/finsg12.jpg" alt="Post Sample Image" width="707" >
</a>
<p></p>

<p></p>

<p></p>
<p>

</p>

<h2>Conclusion</h2>

<p>
  In this post, we showed how to treat time series as a sequence of small graph snapshots, then use a GNN to produce time-aligned graph embeddings. By comparing embedding similarity to a simple rolling-window similarity baseline, we get two complementary views of how relationships evolve over time.
</p>

<p>
  The main takeaway is that the most interesting signal often appears where these two views disagree. When the window baseline and the graph-based similarity diverge, it suggests that “recent values” may look similar while the organization of patterns has changed (or vice versa). We use that disagreement as a practical way to flag candidate periods of relationship reconfiguration.
</p>

<p>
  In this run, the disagreement detector produced clear periods of interest for some sectors (notably Energy and Financials), only one for Technology, and none for Utilities at the chosen sensitivity. That’s exactly what we want at this stage: a small set of time-localized intervals worth interpreting, rather than a noisy stream of alerts.
</p>

<p>
  This is an intentionally small, public-data experiment on daily prices. The goal is not to claim predictive power from a single chart, but to demonstrate a representation and a workflow that turns “relationship change” into something measurable and reviewable.
</p>

<p>
  Looking ahead, we currently compute similarity at aligned timestamps, but we can generalize this to lagged similarity to produce lead–lag timelines—measuring whether structural changes in one series tend to precede another. While lagged similarity is not a causal proof, it can be a practical screening tool for identifying directed relationship episodes. With richer, non-public market data, the same approach can go further—for example: (1) minute-by-minute trading and pricing data can show when the market’s “health” changes and when connections between assets suddenly tighten or snap, (2) short-horizon buy/sell activity can reveal who tends to move first and when large, mechanical repositioning dominates everyday trading, and (3) internal trade records can uncover repeatable trading patterns and flag unusual periods when the usual relationships between instruments abruptly change.
</p>










<p></p>

<p></p>

<p></p>
<p></p>
