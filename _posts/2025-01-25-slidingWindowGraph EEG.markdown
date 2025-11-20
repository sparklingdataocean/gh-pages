---
layout:     post
title:      "Sliding Graph Neural Networks for EEG Analysis"
subtitle:   "Capturing Temporal Dynamics in Brain Activity"
date:       2025-01-25 12:00:00
author:     "Melenar"
header-img: "img/page115y.jpg"
---



<h2>Sliding Graphs: Watching Signals Change Like a Movie</h2>
<p>
  Sliding graphs are a way to let AI watch how a signal behaves over time, not just look at a
  summary. Instead of treating a long recording as one big block, we break it into many small,
  overlapping moments and let AI see which moments look alike and which don’t. Put together, these
  moments form an evolving map of the signal, showing when things are calm, when they shift, and
  when something unusual starts to happen. We illustrate this with EEG sleep vs rest, but the same
  idea works for machines, sensors, markets, climate, or any long signal—anywhere you want AI to
  say not only what is happening, but when things start to change.
</p>

<p></p>
<h2>Conference</h2>
<p>
  The work <em>“Time Aligned Sliding Graph Embeddings for Dynamic Time Series Analysis”</em>
  was presented at <strong>Brain Informatics 2025</strong> in Bari, Italy, on
  <strong>11 November 2025</strong>. The corresponding paper is not yet published
  in archival proceedings.
</p>

<p></p>



<p></p>
<h2>Exploring EEG Through Graph-Based Methods</h2>
<p></p>
Over the years, we are looking to uncover the secrets of brain connectivity using EEG data.
Our work has evolved from traditional graph analysis techniques to cutting-edge Graph Neural Networks (GNNs),
each step uncovering deeper insights into neural dynamics. Let’s take a closer look at these studies.

<p></p>
<p><h4>Study 1: Traditional Graph Analysis</h4></p>
<p></p>
Our journey began with a traditional graph analysis approach. In this study, we constructed connectivity graphs from EEG trials using
<strong>cosine similarity</strong> between channels. Each graph’s nodes represented EEG electrodes, and the edges reflected their functional connectivity.
<p></p>
Key Findings:
<ul>
    <li>Differences in connectivity patterns emerged between the <strong>Alcoholic</strong> and <strong>Control</strong> groups, providing insights into altered neural activity.</li>
    <li>Graph features like clustering coefficients and edge density helped highlight these differences.</li>
    <li>However, traditional methods struggled to distinguish subtle variations, particularly in <strong>single-stimulus conditions</strong>, prompting the need for more advanced techniques.</li>
</ul>
<figure>
    <img src="{{ site.baseurl }}/img/dataSource5.jpg" alt="Traditional EEG Graph Example" style="width:75%; margin:auto;">
    <figcaption>Figure 1: A sample connectivity graph constructed from EEG data using cosine similarity.</figcaption>
</figure>
<p></p>
For a deeper dive into this work, check out our post <a href="http://sparklingdataocean.com/2020/08/19/brainGraphEeg/">"EEG Patterns by Deep Learning and Graph Mining"</a> or refer to the paper <a href="https://link.springer.com/chapter/10.1007/978-3-030-87101-7_19">Time Series Pattern Discovery by Deep Learning and Graph Mining</a>.

<p></p>

<p></p>
<h4>Study 2: Graph Neural Networks for Trial Classification</h4>
<p></p>
On the second study, we introduced <strong>Graph Neural Networks (GNNs)</strong> to analyze EEG data at the trial level. Each graph represented an entire EEG trial, encapsulating the connectivity across all channels.
<p></p>
Why GNNs? GNNs brought a new level of sophistication by enabling the model to learn spatial relationships and connectivity dynamics within the graph.
<p></p>
Key Findings:
<ul>
    <li><strong>Improved Classification Accuracy:</strong> GNN Graph Classification models significantly outperformed traditional methods in differentiating between Alcoholic and Control groups.</li>
    <li><strong>Enhanced Connectivity Insights:</strong> Subtle variations in connectivity, previously missed, were captured.</li>
    <li><strong>Challenges:</strong> Misclassifications within the Control group highlighted the complexity of EEG connectivity patterns.</li>
</ul>

<p></p>

<p></p>
This approach is detailed further in our post <a href="http://sparklingdataocean.com/2023/05/08/classGraphEeg/">"GNN Graph Classification for EEG Pattern Analysis"</a> or refer to the paper <a href="https://www.springerprofessional.de/en/enhancing-time-series-analysis-with-gnn-graph-classification-mod/26751028">Enhancing Time Series Analysis with GNN Graph Classification Models</a>.

<p></p>

<h4>Study 3: Graph Neural Networks for Link Prediction</h4>
<p></p>
In our third study, the focus shifted to <strong>link prediction</strong>, using GNNs to analyze node- and edge-level connectivity. A unified graph constructed from EEG electrode distances was used to predict connectivity dynamics.
<p></p>
Key Findings:
<ul>
    <li><strong>Revealing Hidden Connectivity:</strong> GNN Link Prediction models highlighted relationships between electrodes that were previously unobserved.</li>
    <li><strong>Node Importance:</strong> Certain electrodes emerged as more central to connectivity patterns.</li>
    <li><strong>Limitations:</strong> This method focused primarily on short-term EEG segments, leaving the dynamics of long-term recordings unexplored.</li>
</ul>

<p></p>
For more on this work, check out our <a href="http://sparklingdataocean.com/2024/11/09/GNN_timeSeries_EEG/">"Graph Neural Networks for EEG Connectivity Analysis"<a href="#"></a> or refer to the paper <a href="https://iwain.lucentia.es/proceedings/">Graph Neural Networks in Action: Uncovering Patterns in EEG Time Series Data.  1st International Workshop on Artificial Intelligence for Neuroscience (IWAIN’24), pp. 4–15</a>.
<p></p>
<p></p>
<figure>
    <img src="{{ site.baseurl }}/img/brain4.jpg" alt="Traditional EEG Graph Example" style="width:75%; margin:auto;">
    <figcaption>Figure 2: A sample connectivity graph constructed from EEG data using cosine similarity.</figcaption>
</figure>


<p></p>
<h2>Looking Ahead: Current Study</h2>
<p></p>  


This study applies GNN Sliding Graph Classification to long-time EEG series, capturing evolving neural activity during sleep and rest. This approach reveals extended brain states, uncovering transitions and sustained neural processes, offering deeper insights into EEG dynamics over time.
<p></p>  
Imagine moving a sliding window through EEG data—like watching a movie, scene by scene. Each window captures a brief moment in time. Now imagine building a graph that doesn’t just follow the timeline, but connects moments that belong to the same theme—even if they’re far apart. Like linking “research” and “presentation” as part of the same goal. That graph becomes a story—showing how different moments are connected by meaning, not just time.
<p></p>  
<figure>
    <img src="{{ site.baseurl }}/img/brain41.jpg" alt="Traditional EEG Graph Example" style="width:99%; margin:auto;">
    <figcaption>Slide from demo on conference Brain Informatics (BI 2025) in Bari, Italy.</figcaption>
</figure>
<p></p>

<p></p>



<p></p>


<h2>GNN Sliding Graph Classification: Introduction</h2>

<p></p>

In our previous work, we introduced two key methods for time series analysis. The methodology consists of three key steps:


<ul>
  <li>
    <strong>Sliding Graph Construction:</strong> Transform time series data into graph structures by segmenting it into overlapping windows. Each graph captures localized temporal and spatial relationships, representing distinct patterns over the chosen time frame.
  </li>
  <li>
    <strong>GNN Graph Classification:</strong> Utilize GNNs to classify these graphs, extracting high-level features from their topology while preserving the structural and temporal dependencies in the data.
  </li>
  <li>
    <strong>Pre-final Vectors:</strong> Obtain graph embeddings (pre-final vectors) from the GNN Graph Classification model during classification. These embeddings represent the learned topological features and are further analyzed to reveal temporal and structural patterns in the time series.
  </li>
</ul>


<p></p>

Both methods were successfully applied to <em>climate time series data</em>, revealing complex patterns in large-scale datasets. However, these techniques have never been combined in a single study.

<p></p>

In this study, we integrate these approaches and apply them to <strong>EEG time series data</strong>, specifically in the context of <em>sleep studies</em>. EEG analysis presents unique challenges, requiring methods that can detect both <em>long-term trends</em> and <em>local brain connectivity changes</em>. By leveraging <strong>sliding graph construction</strong> and <strong>pre-final vector extraction</strong>, we aim to uncover <em>hidden EEG patterns</em> that traditional signal processing techniques might miss.
<p></p>
Objectives of This Study
<p></p>
<ul>
    <li>Demonstrate the effectiveness of <strong>graph-based models</strong> for long-duration biomedical signal analysis.</li>
    <li>Validate the generalizability of <strong>GNN Sliding Graph Classification</strong> and <strong>Pre-Final Vectors</strong> beyond climate data, applying them to neuroscience.</li>
</ul>

This approach bridges sliding window techniques and graph-based modeling, providing a powerful framework for analyzing complex temporal EEG data. By capturing both localized and global topological patterns, it enhances our understanding of brain activity dynamics during sleep.

<p></p>
For more detailed information about GNN Sliding Graphs, look at our post <a href="http://sparklingdataocean.com/2024/05/25/slidingWindowGraph/">"Sliding Window Graph in GNN Graph Classification"</a> or refer to the paper <a href="https://dl.acm.org/doi/10.1145/3674029.3674059">GNN Graph Classification for Time Series: A New Perspective on Climate Change Analysis</a>.

<p></p>

For information about catching embedded graphs, look at our post <a href="http://sparklingdataocean.com/2024/07/04/vectorsGNN/">"Unlocking the Power of Pre-Final Vectors in GNN Graph Classification"</a> or refer to the paper <a href="https://mlg-europe.github.io/2024/">Utilizing Pre-Final Vectors from GNN Graph Classification for Enhanced Climate Analysis</a>.



<p></p>


<h2>Methods</h2>
<p></p>
<h3>Pipeline</h3>
<p></p>
<a href="#">
      <img src="{{ site.baseurl }}/img/slide1b.jpg" alt="Post Sample Image" width="808" >
</a>
<p></p>
Our pipeline for <strong>Graph Neural Network (GNN) Graph Classification</strong> consists of several stages.
<ul>

    <li>The process begins with <strong>data input</strong>, where EEG data representing brain activity during sleep and rest states is collected.</li>
    <li><strong>Graph construction:</strong>
        <ul>
            <li><strong>Sliding window method:</strong> Segments time series data into overlapping graphs to maintain temporal structure.</li>
            <li><strong>Virtual nodes:</strong> Act as central hubs, improving model accuracy and information flow.</li>
        </ul>
    </li>
    <li>The <strong>GNN model</strong> classifies these graphs based on detected patterns.</li>
    <li>To enhance interpretability, <strong>pre-final vectors</strong> are extracted from the model, capturing deeper structural information before classification.</li>
    <li><strong>Linear algebra analysis</strong> applies cosine similarity computations to these embeddings, uncovering connectivity trends over time.</li>
    <li>This approach enables effective modeling of long-duration EEG dynamics by integrating graph-based learning with temporal analysis techniques.</li>
</ul>


<p></p>
<p></p>

<p></p>
<p></p>

<p></p>


<h3>Sliding Graph Construction</h3>
<p></p>

In our previous study, <a href="https://dl.acm.org/doi/10.1145/3674029.3674059">GNN Graph Classification for Time Series: A New Perspective on Climate Change Analysis</a>, we introduced an approach to constructing graphs using the
<em>Sliding Window Method</em>.

<p></p>
<a href="#">
      <img src="{{ site.baseurl }}/img/slide3.jpg" alt="Post Sample Image" width="600" >
</a>
  <p></p>
  <p></p>
<h4>Sliding Window Method</h4>
  <ul>
    <li>
      <strong>Nodes</strong>: Represent data points within each sliding window, with features reflecting their respective values.
    </li>
    <li>
      <strong>Edges</strong>: Connect pairs of points to preserve the temporal sequence and structure.
    </li>
    <li>
      <strong>Labels</strong>: Assigned to detect and analyze patterns within the time series.
    </li>
  </ul>
<p></p>

<h3>Methodology for Sliding Window Graph Construction</h3>
<p></p>
<h4>Data to Graph Transformation</h4>
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
<h4>Node Calculation</h4>
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
<h4>Graph Calculation</h4>
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
<h4>Graph Construction</h4>
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
<h4>Graph Classification</h4>
<p></p>
We employ the <em>GCNConv</em> model from the PyTorch Geometric Library for GNN Graph Classification tasks. This model performs convolutional operations, leveraging edges, node attributes, and graph labels to extract features and analyze graph structures comprehensively.
<p></p>
By combining the sliding window technique with Graph Neural Networks, our approach offers a robust framework for analyzing time series data. It captures intricate temporal dynamics and provides actionable insights into both local and global patterns, making it particularly well-suited for applications such as EEG data analysis. This method allows us to analyze time series data effectively by capturing both local and global patterns, providing valuable insights into temporal dynamics.
<p></p>


<h3>Pairwise GNN Sliding Graph Classification</h3>
<p></p>
<a href="#">
      <img src="{{ site.baseurl }}/img/slide2b.jpg" alt="Post Sample Image" width="717" >
</a>
  <p></p>

This figure illustrates the process of pairwise GNN Sliding Graph Classification, where pairs of long time series are analyzed using graph-based methods to capture dynamic connectivity patterns. Channels F3-F4 during sleep are used as an illustrative example. Below is a breakdown of the methodology:
  <p></p>
  <ul>
      <li><strong>Input Time Series as Pairs:</strong> Two long time series (e.g., Sleep F3 and Sleep F4) are taken as input, each representing continuous data points over time.</li>
      <li><strong>Sliding Graph Construction:</strong> Each time series is segmented into overlapping windows, and graphs are created from these segments. These graphs are labeled according to their respective time series (e.g., F3 and F4).</li>
      <li><strong>GNN Graph Classification:</strong> The sliding graphs are processed by a GNN Graph Classification model. The model learns pairwise relationships between the graph labels, capturing interactions between the time series.</li>
      <li><strong>Pre-Final Vector Extraction:</strong> The GNN generates pre-final vectors (graph embeddings) for each segment. These embeddings are aligned with the time points of the original time series.</li>
      <li><strong>Cosine Similarity Computation:</strong> For each pair of embeddings at corresponding time points, cosine similarity is calculated to measure the relationship between the time series.</li>
      <li><strong>Temporal Analysis of Similarities:</strong> The cosine similarity values are plotted over time, revealing how the connectivity between the time series evolves dynamically.</li>
  </ul>
  <p>
  This approach bridges time series analysis and graph theory, offering a robust method to study pairwise relationships in applications like EEG connectivity or multi-channel sensor data.
  </p>


  <p></p>

<p></p>
<h2>Experiments Overview</h2>
<p></p>
<h3>Data Source: EEG Data</h3>
<p></p>
For this study, we utilized EEG data from the
<i><a href="https://github.com/OpenNeuroDatasets/ds003768/tree/master/sub-01/eeg" target="_blank">OpenNeuroDatasets</a></i>.
<p></p>
This dataset includes EEG data collected from 33 healthy participants using a 32-channel MR-compatible EEG system (Brain Products, Munich, Germany). The EEG data were recorded during two 10-minute resting-state sessions (before and after a visual-motor adaptation task) and multiple 15-minute sleep sessions.
<p></p>
For our analysis, we specifically focused on data from one resting-state session and one sleep session, using the raw EEG data for processing and comparative analysis of activity patterns during rest and sleep states.


<p></p>

We used the <code>mne</code> Python library to process EEG data. The dataset includes recordings in the BrainVision format, which were preloaded for analysis. Below is the Python code used for this preprocessing step:
<p></p>
{% highlight python %}
!pip install mne
import mne
vhdr_file_path1 = filePath+'sub-01_task-rest_run-1_eeg.vhdr'
vhdr_file_path2 = filePath+'sub-01_task-sleep_run-3_eeg.vhdr'
raw1 = mne.io.read_raw_brainvision(vhdr_file_path1, preload=True)
raw2 = mne.io.read_raw_brainvision(vhdr_file_path2, preload=True)
{% endhighlight %}
<p></p>
We specifically extracted EEG data from one resting-state session (<code>sub-01_task-rest_run-1_eeg.vhdr</code>) and one sleep session (<code>sub-01_task-sleep_run-3_eeg.vhdr</code>), which were recorded using a 32-channel MR-compatible EEG system (Brain Products, Munich, Germany). These raw EEG signals were prepared for further analysis and sliding graph construction.

<p></p>
After loading the EEG data, we transformed the raw signals into structured pandas DataFrames for ease of analysis. The following code snippet demonstrates this step:

<p></p>
{% highlight python %}
import pandas as pd
eeg_data1, times1 = raw1.get_data(return_times=True)
eeg_df1 = pd.DataFrame(eeg_data1.T, columns=channel_names1)
eeg_df1['Time'] = times1
eeg_data2, times2 = raw2.get_data(return_times=True)
eeg_df2 = pd.DataFrame(eeg_data2.T, columns=channel_names1)
eeg_df2['Time'] = times2
eeg_df1.shape,eeg_df2.shape
((4042800, 33), (4632500, 33))
{% endhighlight %}
<p></p>

The EEG signals from both the rest and sleep sessions were converted into DataFrames. Each DataFrame contains 32 EEG channels and a corresponding <code>Time</code> column, enabling a clear representation of time series data for further processing. The shapes of the resulting DataFrames are as follows:
  </p>
  <ul>
    <li><strong>Rest session:</strong> 4,042,800 rows × 33 columns</li>
    <li><strong>Sleep session:</strong> 4,632,500 rows × 33 columns</li>
  </ul>

This structured format facilitates segmentation, feature extraction, and the eventual construction of sliding graphs.

  <p></p>

Given the large size of the EEG datasets, we applied downsampling to reduce the number of rows while retaining the temporal structure of the signals. Specifically, every 20th row from each DataFrame was selected, effectively reducing the data size by a factor of 20.

<p></p>
{% highlight python %}
eeg_df1 = eeg_df1.iloc[::20, :].reset_index(drop=True)
eeg_df2 = eeg_df2.iloc[::20, :].reset_index(drop=True)
print(eeg_df1.shape, eeg_df2.shape)
(202140, 33) (231625, 33)
{% endhighlight %}
<p></p>

  <ul>
    <li><strong>Rest session:</strong> 202,140 rows × 33 columns</li>
    <li><strong>Sleep session:</strong> 231,625 rows × 33 columns</li>
  </ul>
  <p>
This step significantly reduced the computational overhead for subsequent processing steps while preserving meaningful patterns in the data.
<p></p>

To ensure compatibility during analysis, both EEG DataFrames were truncated to have the same number of rows. This step is essential to facilitate pairwise comparisons and maintain consistency across the datasets.

<p></p>
{% highlight python %}
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
min_rows = min(len(eeg_df1), len(eeg_df2))
eeg1df = eeg_df1.iloc[:min_rows]
eeg2df = eeg_df2.iloc[:min_rows]
eeg1df.shape,eeg2df.shape
((202140, 33), (202140, 33))
{% endhighlight %}
<p></p>

After truncation, both DataFrames contain:

<ul>
  <li><strong>Row count:</strong> 202,140</li>
  <li><strong>Column count:</strong> 33 EEG channels</li>
</ul>
<p></p>
This ensures that subsequent operations, such as similarity calculations or graph-based analysis, can be performed without inconsistencies in data alignment.

<p></p>

To prepare the EEG data for analysis, numerical columns were normalized to ensure consistent scaling across features. The 'Time' column was excluded during normalization and re-added afterward. This step helps improve the performance of subsequent analytical methods by standardizing the data.

<p></p>
{% highlight python %}
eeg1_features = eeg1df.drop(columns=['Time'])
eeg2_features = eeg2df.drop(columns=['Time'])
eeg1 = (eeg1_features - eeg1_features.mean()) / (eeg1_features.std() + 1e-5)
eeg2 = (eeg2_features - eeg2_features.mean()) / (eeg2_features.std() + 1e-5)
eeg1['Time'] = eeg1df['Time']
eeg2['Time'] = eeg2df['Time']
{% endhighlight %}
<p></p>

To enhance data tracking and processing, the 'Time' column was renamed, formatted as a string, and additional metadata columns were added:

<p></p>
{% highlight python %}
eeg1=eeg1.rename(columns={'Time':'date'})
eeg2=eeg2.rename(columns={'Time':'date'})
eeg1['dateStr'] =  '~' + eeg1['date'].astype(str)
eeg2['dateStr'] =  '~' + eeg2['date'].astype(str)
eeg1['rowIndex'] = range(len(eeg1))
eeg2['rowIndex'] = range(len(eeg2))
{% endhighlight %}
<p></p>

These steps ensure that the data is not only normalized but also organized with clear metadata, facilitating downstream analysis and visualization tasks.



<p></p>
<h3>Raw Data Analysis</h3>
<p></p>
This step of data analysis focuses on comparing the cosine similarity between EEG channels during sleep and rest states. The top bar chart visualizes the channel-wise differences, highlighting which brain regions exhibit notable variations in activity patterns. The bottom chart aggregates these comparisons region-wise (e.g., Central, Occipital, Temporal), providing a high-level view of how different brain regions behave in sleep versus rest.
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/eegSlide1.jpg" alt="Post Sample Image" width="678" >
</a>
<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/eegSlide2.jpg" alt="Post Sample Image" width="600" >
</a>
<p></p>
Since time measures in separate sections do not overlap, this comparison offers a broad overview, serving as a basis for more detailed studies on individual sessions.

<p></p>
<h4>Normalization and Preprocessing</h4>
<p></p>
In this step, we normalized the EEG data to ensure consistency across different sessions and reduce the impact of varying scales. The following processes were carried out:

<ul>
  <li>
    <strong>Numerical Column Selection:</strong>
    Excluded the 'Time' column to focus only on the numerical EEG data for normalization.
  </li>
  <li>
    <strong>Data Normalization:</strong>
    Each feature was normalized using z-score normalization:
    <br>
    <code>Normalized Value = (Value - Mean) / (Standard Deviation + 1e-5)</code>
    <br>
    This ensures the data has a mean of 0 and a standard deviation of 1, improving the stability of subsequent analyses.
  </li>
  <li>
    <strong>Reintegrating the Time Column:</strong>
    The 'Time' column was added back to the normalized dataset and renamed to <code>date</code> for easier readability and alignment with temporal analyses.
  </li>
  <li>
    <strong>String Representation for Dates:</strong>
    Created a <code>dateStr</code> column by prefixing the time values with a tilde (<code>~</code>), providing a textual representation of the timestamps.
  </li>
  <li>
    <strong>Index Assignment:</strong>
    Added a <code>rowIndex</code> column to assign a unique index to each row for tracking during further analysis.
  </li>
</ul>
<p></p>
This normalization step prepared the data for sliding window segmentation and graph construction, ensuring consistency and improving the robustness of the subsequent analyses.

{% highlight python %}
eeg1_features = eeg1df.drop(columns=['Time'])
eeg2_features = eeg2df.drop(columns=['Time'])
eeg1 = (eeg1_features - eeg1_features.mean()) / (eeg1_features.std() + 1e-5)
eeg2 = (eeg2_features - eeg2_features.mean()) / (eeg2_features.std() + 1e-5)
eeg1['Time'] = eeg1df['Time']
eeg2['Time'] = eeg2df['Time']
eeg1=eeg1.rename(columns={'Time':'date'})
eeg2=eeg2.rename(columns={'Time':'date'})
eeg1['dateStr'] =  '~' + eeg1['date'].astype(str)
eeg2['dateStr'] =  '~' + eeg2['date'].astype(str)
eeg1['rowIndex'] = range(len(eeg1))
eeg2['rowIndex'] = range(len(eeg2))
{% endhighlight %}
<p></p>

<h4>Channel Grouping by Brain Regions</h4>
<p></p>
To organize the EEG channels for our study, we grouped them based on their prefixes. This grouping helps us focus on specific brain regions for analysis and simplifies the selection process. Below are the steps and results of this process:
<p></p>
<ul>
  <li>
    <strong>Grouping Channels:</strong>
    Each EEG channel was categorized by its prefix, which corresponds to the brain region it represents. Channels ending with <code>'z'</code> were treated as central and grouped by removing the trailing <code>'z'</code>. For all other channels, their alphabetical prefix was used for grouping.
  </li>
  <li>
    <strong>Code Implementation:</strong>
    The grouping was performed programmatically using a dictionary structure where the keys represent brain region prefixes, and the values contain the corresponding EEG channels.
  </li>
</ul>
<p></p>
Below is the Python implementation used for channel grouping:
<p></p>
{% highlight python %}
from collections import defaultdict
channel_groups = defaultdict(list)
for channel in eeg1.columns:
    if channel.endswith('z'):
        prefix = channel[:-1]
    else:
        prefix = ''.join([char for char in channel if char.isalpha()])
    channel_groups[prefix].append(channel)
for group, channels in channel_groups.items():
    print(f"{group}: {channels}")
{% endhighlight %}
<p></p>

<h4>Grouped Channels</h4>
<p></p>
The resulting channel groups are as follows:
<ul>
  <li><strong>Fp:</strong> ['Fp1', 'Fp2']</li>
  <li><strong>F:</strong> ['F3', 'F4', 'F7', 'F8', 'Fz']</li>
  <li><strong>C:</strong> ['C3', 'C4', 'Cz']</li>
  <li><strong>P:</strong> ['P3', 'P4', 'P7', 'P8', 'Pz']</li>
  <li><strong>O:</strong> ['O1', 'O2', 'Oz']</li>
  <li><strong>T:</strong> ['T7', 'T8']</li>
  <li><strong>FC:</strong> ['FC1', 'FC2', 'FC5', 'FC6']</li>
  <li><strong>CP:</strong> ['CP1', 'CP2', 'CP5', 'CP6']</li>
  <li><strong>TP:</strong> ['TP9', 'TP10']</li>
  <li><strong>EOG:</strong> ['EOG']</li>
  <li><strong>ECG:</strong> ['ECG']</li>
  <li><strong>Time:</strong> ['Time']</li>
</ul>
<p></p>
These groups will guide our selection of brain regions and EEG channels for further analysis in the study.

<p></p>

<h3>Computing Cosine Similarities Within EEG Channel Groups</h3>
<p></p>
As part of our EEG analysis, we calculated cosine similarities between channel pairs within the same group. This step focuses on understanding relationships between channels in specific brain regions. Below are the details of the process and implementation:
<p></p>
<h4>Steps in Analysis</h4>
<p></p>
<ol>
  <li><strong>Channel Grouping:</strong> EEG channels were grouped based on their prefixes, corresponding to specific brain regions. Channels ending with <code>'z'</code> were adjusted by removing the trailing <code>'z'</code>, and other channels were grouped by their letter prefixes.</li>
  <li><strong>Sorting Channels:</strong> Channels within each group were sorted alphabetically to ensure consistent pairwise comparisons.</li>
  <li><strong>Cosine Similarity Calculation:</strong> Cosine similarities were computed for all possible pairs within each group using their numerical feature vectors.</li>
  <li><strong>Sorting Results:</strong> The cosine similarity pairs were sorted alphabetically for easy interpretation and analysis.</li>
</ol>
<p></p>
The following Python code was used to perform the analysis:

<p></p>
{% highlight python %}
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
channel_groups = defaultdict(list)
for channel in eeg_df1_truncated.columns:
    if channel.endswith('z'):
        prefix = channel[:-1]
    else:
        prefix = ''.join([char for char in channel if char.isalpha()])
    channel_groups[prefix].append(channel)
cosine_similarities = {}
for group, channels in channel_groups.items():
    channels = sorted(channels)
    for i, channel1 in enumerate(channels):
        for channel2 in channels[i + 1:]:
            vector1 = eeg_df1_truncated[channel1].to_numpy().reshape(1, -1)
            vector2 = eeg_df1_truncated[channel2].to_numpy().reshape(1, -1)
            similarity = cosine_similarity(vector1, vector2)[0][0]
            cosine_similarities[f"{channel1}-{channel2}"] = similarity
sorted_cosine_similarities = dict(sorted(cosine_similarities.items()))
{% endhighlight %}
<p></p>


<ul>
  <li>Cosine similarities provide insights into the relationships between EEG channels within the same brain region.</li>
  <li>The sorted similarity pairs offer a clear view of which channels are most or least correlated within each group.</li>
</ul>

<p></p>
This method helps isolate patterns within specific brain regions, contributing to our understanding of channel interactions during rest and sleep sessions.

<p></p>
<figure>
    <img src="{{ site.baseurl }}/img/eegSlide5.jpg" alt="Data Analysis: Cosine Similarities" style="width:50%; margin:auto;">
    <figcaption>The table summarizes cosine similarity values for EEG channel pairs during sleep and rest states, alongside the difference between these states (Sleep - Rest).</figcaption>
</figure>

<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/eegSlide7.jpg" alt="Post Sample Image" width="600" >
</a>
<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/eegSlide8.jpg" alt="Post Sample Image" width="600" >
</a>
<p></p>
<ul>
  <li><strong>Channel Pairs</strong>: EEG channel pairs analyzed for similarity.</li>
  <li><strong>Sleep Cos</strong>: Cosine similarity during the sleep session.</li>
  <li><strong>Rest Cos</strong>: Cosine similarity during the rest session.</li>
  <li><strong>Sleep-Rest</strong>: Difference in similarity between sleep and rest, showing how connectivity changes across states.</li>
</ul>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/eegSlide9.jpg" alt="Post Sample Image" width="600" >
</a>
<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/eegSlide10.jpg" alt="Post Sample Image" width="600" >
</a>
<p></p>
<p></p>

For our analysis, we selected the EEG channel pairs C4-Cz, F3-F4, and O1-O2. These pairs were chosen based on their relevance to brain region interactions and their notable differences in connectivity between sleep and rest states. These channels represent central, frontal, and occipital brain regions, providing a comprehensive view of neural activity across different areas of the brain.

<p></p>
<h3>Sliding Graph</h3>
<p></p>
This function, <code>create_segments_df</code>, is designed to process a time series DataFrame by creating overlapping segments for a specified column. It helps prepare data for sliding window analysis, which is essential for studying temporal patterns in EEG signals. Below is a high-level description of its workflow:

<ul>
  <li><strong>Inputs:</strong> The function takes the following parameters:
    <ul>
      <li><code>df</code>: The DataFrame containing the data.</li>
      <li><code>column_name</code>: The column to segment.</li>
      <li><code>window_size</code>: The size of each sliding window.</li>
      <li><code>shift</code>: The step size for sliding the window.</li>
      <li><code>columnLabel</code>: A label to annotate the segments.</li>
    </ul>
  </li>

  <li>Process:
    <ul>
      <li>Iterates over the DataFrame to extract overlapping windows of the specified size.</li>
      <li>Transposes each window to arrange its data as a single row for easier concatenation.</li>
      <li>Adds metadata to each segment, including:
        <ul>
          <li><code>start_date</code>: The start time of the segment.</li>
          <li><code>rowIndex</code>: The row index of the original DataFrame.</li>
          <li><code>theColumn</code>: The name of the column being segmented.</li>
          <li><code>columnLabel</code>: A label for the segment.</li>
        </ul>
      </li>
      <li>Appends each processed segment to a list.</li>
    </ul>
  </li>
  <li><strong>Output:</strong> Combines all segments into a single DataFrame for downstream analysis.</li>
</ul>
<p></p>
This function is particularly useful in EEG studies, enabling the division of continuous signals into manageable segments for sliding graph or time-series analysis.
<p></p>
{% highlight python %}
def create_segments_df(df, column_name, window_size, shift,columnLabel):
    segments = []
    for i in range(0, len(df) - window_size + 1, shift):
        segment = df.loc[i:i + window_size - 1,
          [column_name]].reset_index(drop=True)
        segment = segment.T  
        segment['start_date'] = df['date'][i]
        segment['rowIndex'] = df['rowIndex'][i]
        segment['theColumn'] = column_name
        segment['columnLabel'] = columnLabel
        segments.append(segment)
    return pd.concat(segments, ignore_index=True)
{% endhighlight %}
<p></p>

<p></p>
The function <code>group_segments</code> is designed to group smaller data segments into larger groups for graph-based analysis. This process is crucial for aggregating segments in sliding window studies, particularly for EEG analysis. Here’s a detailed explanation:

<ul>
  <li><strong>Inputs:</strong> The function takes the following parameters:
    <ul>
      <li><code>segments_df</code>: The DataFrame containing individual segments.</li>
      <li><code>group_size</code>: The number of segments in each group.</li>
      <li><code>group_shift</code>: The step size for sliding between groups.</li>
    </ul>
  </li>
<p></p>
  <li>Process:
    <ul>
      <li>Iterates over the DataFrame to extract overlapping groups of the specified size.</li>
      <li>Resets the index for each group to maintain consistent indexing.</li>
      <li>Adds a new column, <code>graphIndex</code>, to assign a unique identifier to each group.</li>
      <li>Appends each grouped segment to a list for aggregation.</li>
      <li>Increments the <code>group_index</code> after each group to ensure unique identifiers.</li>
    </ul>
  </li>
  <li><strong>Output:</strong> Combines all grouped segments into a single DataFrame for further analysis or graph construction.</li>
</ul>

<p></p>
This function facilitates efficient grouping of sliding window segments, enabling robust graph-based analysis for temporal patterns in EEG data.

<p></p>
{% highlight python %}
def group_segments(segments_df, group_size, group_shift):
    grouped_segments = []
    group_index = 0  
    for i in range(0, len(segments_df) - group_size + 1, group_shift):
        group = segments_df.loc[i:i + group_size - 1].reset_index(drop=True)
        group['graphIndex'] = group_index  
        grouped_segments.append(group)
        group_index += 1  
    return pd.concat(grouped_segments, ignore_index=True)
{% endhighlight %}
<p></p>

<h4>Preprocessing and Sliding Window Preparation</h4>

<p></p>
Parameters for Sliding Window and Grouping:
<p></p>
We defined the following parameters for creating sliding windows and grouping segments:
<p></p>
<ul>
  <li><em>Window size (W):</em> 32 data points per segment.</li>
  <li><em>Shift (S):</em> 16 data points between segments.</li>
  <li><em>Group size (G):</em> 32 segments per group.</li>
  <li><em>Group shift (S<sub>g</sub>):</em> 16 segments between groups.</li>
</ul>
<p></p>
{% highlight python %}
window_size=32
shift=16
group_size=32
group_shift=16
{% endhighlight %}
<p></p>
Data Scaling and Handling Missing Values:
<p></p>
We selected EEG channels (e.g., <code>O1</code> and <code>O2</code>) for analysis and processed them as follows:
<ul>
  <li>Missing values were replaced with the mean of the respective column.</li>
  <li>Min-Max Scaling was applied to normalize the data for consistency across features.</li>
</ul>
<p></p>
Sliding Window Segmentation and Grouping:
<p></p>
Using the defined parameters, sliding windows were created for each channel (e.g., <code>O1</code> and <code>O2</code>), with each segment assigned a unique node index. Segments were then grouped into larger units for graph analysis.

<p></p>
Dataset Creation:
<p></p>
The grouped segments for both channels were concatenated into a single dataset. Each group was assigned a unique graph index, resulting in a dataset with 787 graph groups, ready for graph-based processing and analysis.

<p></p>
{% highlight python %}
from sklearn.preprocessing import MinMaxScaler
pairColumns=['O1','O2']
col1 = pairColumns[0]
col2 = pairColumns[1]
scaler = MinMaxScaler()
fx_data=df
if col1 in fx_data.columns:
    fx_data[col1] = fx_data[col1].fillna(fx_data[col1].mean())
    fx_data[col1] = scaler.fit_transform(fx_data[[col1]])
if col2 in fx_data.columns:
    fx_data[col2] = fx_data[col2].fillna(fx_data[col2].mean())
    fx_data[col2] = scaler.fit_transform(fx_data[[col2]])
columnLabel=0
segments1 = create_segments_df(df, col1, window_size, shift, columnLabel)
columnLabel=1
segments2 = create_segments_df(df, col2, window_size, shift, columnLabel)  
segments1['nodeIndex']=segments1.index
segments2['nodeIndex']=segments2.index
grouped_segments1 = group_segments(segments1, group_size, group_shift)
grouped_segments2 = group_segments(segments2, group_size, group_shift)
dataSet= pd.concat([grouped_segments1, grouped_segments2], ignore_index=True)
graphMax = dataSet['graphIndex'].max()
graphMax
787  
{% endhighlight %}
<p></p>


<p></p>

<p></p>

<p></p>
<h4>Sliding Window Graph as Input for GNN Graph Classification</h4>
<p></p>
In this stage of our analysis, we prepared sliding window graphs as input for a Graph Neural Network (GNN) classification task. Below is a high-level description of the process:
<p></p>
Process Overview:
<p></p>
We iteratively constructed graphs for EEG data using the predefined sliding windows and grouped segments. Each graph corresponds to a unique segment of the EEG data, capturing temporal relationships within the window. For each graph:
<ul>
  <li>Features (<code>x</code>): Derived from EEG signal values within the segment, including the average of node features to enhance representation.</li>
  <li>Edges (<code>edge_index</code>): Created based on cosine similarity between node pairs, using a threshold (<code>cos &gt; 0.9</code>) to establish connections between nodes.</li>
  <li>Labels (<code>y</code>): Assigned based on the channel being analyzed (e.g., <code>O1</code> or <code>O2</code>).</li>
</ul>
<p></p>
Cosine Similarity Calculation:
<p></p>
Cosine similarity was computed for all node pairs within each graph to determine connectivity. Node pairs exceeding the threshold of 0.9 were added as edges. This ensures that only significant relationships within the EEG signals are represented in the graph structure.
<p></p>
DataLoader Preparation:
<p></p>
The resulting graphs were packaged into datasets for model training and testing:
<ul>
  <li><em>DatasetTest:</em> Contains graphs prepared for evaluation.</li>
  <li><em>DatasetModel:</em> Contains graphs ready for training the GNN model.</li>
</ul>
<p></p>
These datasets were loaded into PyTorch Geometric's <code>DataLoader</code> for efficient batch processing during model training and evaluation.

<p></p>
Outcome:
The constructed sliding window graphs provide a structured and efficient way to capture temporal EEG patterns for graph-based classification. This approach highlights the power of combining sliding window analysis with GNNs to study EEG signals.

<p></p>
{% highlight python %}
from torch_geometric.loader import DataLoader
cos=0.9
datasetTest=list()
datasetModel=list()
cosPairsUnion=pd.DataFrame()
for label in range(0,2):
  column=pairColumns[label]
  for graphIdx in range(0, graphMax):
    data1=dataSet[(dataSet['graphIndex']==graphIdx)
      & (dataSet['theColumn']==column)]
    values1=data1.iloc[:,:-7]
    fXValues1= values1.fillna(0).values.astype(float)
    fXValuesPT1=torch.from_numpy(fXValues1)
    fXValuesPT1avg=torch.mean(fXValuesPT1,dim=0).view(1,-1)
    fXValuesPT1union=torch.cat((fXValuesPT1,fXValuesPT1avg),dim=0)
    cosine_scores1 = pytorch_cos_sim(fXValuesPT1, fXValuesPT1)
    cosPairs1=[]
    score0=cosine_scores1[0][0].detach().numpy()
    for i in range(group_size):
      date1=data1.iloc[i]['start_date']
      datasetIdx=data1.iloc[i]['datasetIdx']
      cosPairs1.append({'cos':score0, 'graphIdx':graphIdx,
                        'label':label,'theColumn':column,
                        'k1':i, 'k2':window_size,
                        'date1':date1,
                        'date2':'XXX','datasetIdx': datasetIdx,
                        'score': score0})
      for j in range(group_size):
        if i<j:
          score=cosine_scores1[i][j].detach().numpy()
          if score>cos:
            date2=data1.iloc[j]['start_date']
            datasetIdx=data1.iloc[i]['datasetIdx']
            cosPairs1.append({'cos':cos, 'graphIdx':graphIdx,
                              'cos':score0, 'graphIdx':graphIdx,
                              'label':label,'theColumn':column,
                              'k1':i,
                              'k2':j,
                              'date1':date1,
                              'date2':date2,
                              'datasetIdx': datasetIdx,
                              'score': score})
    dfCosPairs1=pd.DataFrame(cosPairs1)
    edge1=torch.tensor(dfCosPairs1[['k1',	'k2']].T.values)
    dataset1 = Data(edge_index=edge1)
    dataset1.y=torch.tensor([label])
    dataset1.x=fXValuesPT1union
    datasetTest.append(dataset1)
    loader = DataLoader(datasetTest, batch_size=32)
    loader = DataLoader(datasetModel, batch_size=32)
    cosPairsUnion = pd.concat([cosPairsUnion, dfCosPairs1], ignore_index=True)
{% endhighlight %}
<p></p>

<p></p>   
<h4>GNN Graph Classification: Model Training.</h4>
<p></p>  

To classify EEG data using a graph neural network (GNN), we implemented a training pipeline that incorporates data splitting, model definition, and training steps. Below is an overview of the process:



<p></p>
{% highlight python %}
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
test_size = 0.17
train_dataset, test_dataset =
  train_test_split(graphInput, test_size=test_size, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
{% endhighlight %}
<p></p>
<strong>Dataset Splitting:</strong> The dataset was split into training and testing sets with a 17% test size. The data was prepared for training using PyTorch Geometric's DataLoader, ensuring efficient batch processing.

<p></p>

<strong>Model Architecture:</strong> A Graph Convolutional Network (GCN) was designed for EEG graph classification. The model includes:
<ul>
<li><strong>Node Embedding Steps:</strong> Three graph convolutional layers process node-level information.</li>
<li><strong>Graph Embedding Step:</strong> A global mean pooling layer aggregates node-level embeddings into graph-level embeddings.</li>
<li><strong>Classification Step:</strong> A fully connected layer classifies graphs into two categories.</li>
</ul>
<p></p>
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
        self.conv1 = GCNConv(window_size, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)
    def forward(self, x, edge_index, batch, return_graph_embedding=False):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        graph_embedding = global_mean_pool(x, batch)  
        if return_graph_embedding:
            return graph_embedding  
        x = F.dropout(graph_embedding, p=0.3, training=self.training)
        x = self.lin(x)
        return x
model = GCN(hidden_channels=16)
{% endhighlight %}
<p></p>
The model is now ready for training and evaluation using the prepared data loaders. This architecture leverages node-level and graph-level features for effective classification.

<p></p>
<h3>Model Training and Evaluation</h3>
<p></p>

The training and evaluation process for the GNN model involves key steps to optimize the parameters and assess performance. Below is an overview of the methodology:
<p></p>
<strong>Training Process:</strong>
<ul>
  <li>Perform a single forward pass over batches in the training dataset.</li>
  <li>Compute the loss using the cross-entropy loss function.</li>
  <li>Derive gradients using backpropagation.</li>
  <li>Update model parameters based on the computed gradients.</li>
  <li>Clear gradients after each step to prevent accumulation.</li>
</ul>
<p></p>
<strong>Evaluation Process:</strong>
<ul>
  <li>Iterate over the test dataset in batches.</li>
  <li>Perform forward passes to compute predictions.</li>
  <li>Use the class with the highest probability as the predicted label.</li>
  <li>Compare predictions with ground-truth labels to compute the accuracy.</li>
  <li>Return the ratio of correct predictions as the evaluation metric.</li>
</ul>
<p></p>

<p></p>
{% highlight python %}
from IPython.display import Javascript
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))
model = GCN(hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
{% endhighlight %}
<p></p>


This section details the training and evaluation process of the graph neural network (GNN) model for the EEG channel pair F3-F4 during the sleep session. The model was trained over 16 epochs, with accuracy metrics computed for both the training and test datasets at each epoch.


<p></p>
{% highlight python %}
for epoch in range(1, 17):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d},
      Train Acc: {train_acc:.4f},
      Test Acc: {test_acc:.4f}')
{% endhighlight %}
<a href="#">
    <img src="{{ site.baseurl }}/img/eegSlide6.jpg" alt="Post Sample Image" width="445" >
</a>
<p></p>
<ul>
  <li><b>Training Accuracy:</b> Indicates the model's ability to learn patterns from the training dataset. Accuracy steadily increased across epochs, reaching a peak of <strong>0.9502</strong>.</li>
  <li><b>Test Accuracy:</b> Reflects the model's performance on unseen test data, gradually improving and achieving a high value of <strong>0.9366</strong> by the final epoch.</li>
</ul>
<p></p>
The consistent improvement in both training and test accuracy demonstrates the model's capability to generalize well. This highlights its effectiveness in classifying EEG data based on sliding window graphs for the F3-F4 channel pair during sleep.

<p></p>

<p></p>
<p></p>
The table summarizes cosine similarity values and graph neural network (GNN) performance for selected EEG channel pairs across sleep and rest sessions. It provides insights into how these pairs interact during different states and how well the GNN model captures these patterns.



<p></p>
<h4>Analysis of Cosine Similarity and GNN Performance for Selected EEG Pairs</h4>
<p></p>


<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/eegSlide11.jpg" alt="Post Sample Image" width="657" >
</a>
<p></p>

This table presents cosine similarity values and GNN Graph Classification performance for selected EEG channel pairs across sleep and rest states, offering insights into connectivity patterns and classification accuracy.

<p></p>
<ul>


    <li>Cosine Similarity & Channel Interactions:
        <ul>
            <li><strong>F3-F4 (Frontal Lobe):</strong> Moderate similarity in both states with the highest training and test accuracy, indicating strong differentiation between sleep and rest.</li>
            <li><strong>C4-Cz (Central Region):</strong> Higher similarity during sleep, suggesting stronger functional connectivity in this state. However, its stable patterns across conditions resulted in moderate classification accuracy.</li>
            <li><strong>O1-O2 (Occipital Lobe):</strong> Consistently high similarity across both states, limiting classification performance due to minimal variation.</li>
        </ul>
    </li>

    <li>Brain Regions & Functional Roles:
        <ul>
            <li><strong>Frontal Activity (F3-F4):</strong> Notable differences in similarity between sleep and rest align with the frontal lobe’s role in cognitive processing, which decreases during sleep.</li>
            <li><strong>Visual Processing (O1-O2):</strong> The occipital lobe pair maintained stable interactions across states, reflecting consistent neural activity in visual regions.</li>
        </ul>
    </li>

    <li>Model Performance & Interpretation:
        <ul>
            <li><strong>Training Accuracy:</strong> The GNN effectively learned EEG patterns, with F3-F4 achieving the highest accuracy, reinforcing its distinct connectivity changes across states.</li>
            <li><strong>Test Accuracy:</strong> Performance varied across pairs; F3-F4 demonstrated strong generalization, while others showed moderate accuracy shifts.</li>
            <li><strong>High Similarity & Lower Accuracy:</strong> While strong cosine similarity suggests stable EEG interactions, it can reduce variability needed for classification. This is evident in O1-O2, where consistently high similarity limited the model’s ability to distinguish between sleep and rest.</li>
        </ul>
    </li>


</ul>

<p></p>
<p></p>
These findings highlight the complex dynamics of EEG signal relationships and the challenges of analyzing highly correlated data. The results also demonstrate how GNN-based approaches can capture distinct neural patterns, offering a powerful framework for studying sleep-state transitions and functional connectivity.
<p></p>
<p></p>
<h4>Note on O1-O2 Analysis</h4>
<p></p>
Although <strong>O1-O2</strong> was initially included as part of the analysis, its results have been excluded from the figures and detailed discussion due to the very low model training and testing accuracy observed for this channel pair. This suggests that the model failed to capture meaningful patterns or dynamics for O1-O2, likely due to insufficient signal quality or inherent limitations in the data for this pair.


<p></p>
<p></p>



<p></p>


<p></p>
<p></p>

<p></p>   

<p></p>
<p></p>   
<h3>Model Results Interpretation</h3>
<p></p>

The results interpretation phase analyzed the predictions and embeddings from the GNN Graph Classification model. A softmax function transformed the model’s outputs into probabilities, making classification predictions more interpretable. This process helped identify the most likely labels for each graph.

Process:
<ul>
  <li><i>Softmax Transformation:</i> The raw outputs of the GNN model were passed through a softmax function to convert them into probability distributions over the possible classes.</li>
  <li><i>Prediction Extraction:</i> The predicted labels for each graph were determined by identifying the class with the highest probability.</li>
  <li><i>Graph Embeddings:</i> The GNN model also generated graph-level embeddings for each graph, providing a compact vector representation of the patterns captured within the graph.</li>
  <li><i>Data Storage:</i> These embeddings, along with the predicted labels and probabilities, were stored in a structured DataFrame for further analysis and visualization.</li>
</ul>
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


The resulting DataFrame contains each graph's index, embedding vectors, and prediction results. The embeddings serve as high-dimensional representations of the EEG data, enabling further analysis of the underlying patterns and relationships identified by the GNN model.


<p></p>

<p></p>
{% highlight python %}
graphUnion_df=pd.DataFrame(graphUnion)
graphUnion_df.tail()
      index	vector
1569	1569	[[0.17810732, -0.19235992, -0.16263075, -0.167...
1570	1570	[[0.2913107, -0.073132396, -0.09579194, -0.039...
1571	1571	[[0.030929727, -0.10722159, -0.040990006, -0.0...
1572	1572	[[0.3690454, -0.014458519, 0.03268631, 0.04397...
1573	1573	[[0.123519175, -0.23811509, -0.22812074, -0.16.
{% endhighlight %}
<p></p>
This step bridges the gap between model training and interpretability, allowing for a deeper understanding of how the GNN processes and classifies EEG-based sliding window graphs.
<p></p>
<h4>Cosine Similarity Analysis for Graph Embeddings</h4>
<p></p>
This step evaluates the similarity between pre-final embedding vectors generated by the GNN model for sliding window graphs. By calculating cosine similarity, we gain insights into the relationships and connectivity patterns captured by the model.
<p></p>
Key Steps:
<ul>
    <li><strong>Graph Embedding Vectors:</strong> Each graph is represented by a vector derived from the GNN's pre-final embedding layer, summarizing temporal and spatial relationships within the EEG signal.</li>
    <li><strong>Middle Point Calculation:</strong> For each pair of graph embeddings, the middle point between their corresponding time windows is calculated to align temporal information with similarity analysis.</li>
    <li><strong>Cosine Similarity:</strong> Cosine similarity is computed between graph embedding vectors to quantify the relationship between graphs. This metric reveals how closely related the patterns in the two time segments are.</li>
    <li><strong>Result Compilation:</strong> The results include cosine similarity scores and metadata like the middle point of time windows. These scores provide a basis for exploring the relationships in EEG data.</li>
</ul>


<p></p>
{% highlight python %}
cosine_sim_pairs = []
for i in range(len(graphList_1)):
    datasetIdx_0=graphList_0['datasetIdx'][i]   
    datasetIdx_1=graphList_1['datasetIdx'][i]
    min = graphList_0['min'][i]
    max = graphList_1['max'][i]
    middle_point = (min+max)/2
    # cos_sim_value = cos_sim(datasetIdx_0, datasetIdx_1).numpy().flatten
    vector_0 = torch.tensor(graphUnion_df['vector'][datasetIdx_0])
    vector_1 = torch.tensor(graphUnion_df['vector'][datasetIdx_1])
    cos_sim_value = cos_sim(vector_0, vector_1).numpy().flatten()[0]
    cosine_sim_pairs.append({            
            'middle_point':middle_point,            
            'cos': cos_sim_value
        })
{% endhighlight %}
<p></p>

This analysis bridges the gap between model outputs and interpretability, offering a clearer understanding of how the GNN captures and distinguishes temporal patterns. By identifying regions of high and low similarity, this step enables further exploration of brain dynamics during sleep and rest states, paving the way for advanced graph-based analyses.
<p></p>




<p></p>
<h3>Analysis of Embedded Graphs: Statistics</h3>
<p></p>
<figure>
    <img src="{{ site.baseurl }}/img/slide4.jpg" alt="Cosine Similarity Statistics for Channel Pairs F3-F4 and C4-Cz" style="width:90%; margin:auto;">
    <figcaption>Cosine similarity statistics for EEG channel pairs F3-F4 and C4-Cz during sleep and rest sessions, highlighting shifts in connectivity patterns and variability across states.</figcaption>
</figure>

<p></p>
This process aligns embeddings with their corresponding time points, enabling a structured temporal analysis. Cosine similarity is computed between paired embeddings (e.g., F3 and F4) at each time point, generating a dynamic connectivity profile that reveals evolving relationships within the time series.
<p></p>
Table 2 presents cosine similarity statistics for EEG channel pairs F3-F4 and C4-Cz across sleep and rest sessions, highlighting shifts in connectivity behavior. The mean similarity for F3-F4 shifts from negative during sleep (-0.1529) to positive in rest (0.1522), indicating a state-dependent connectivity change. In contrast, C4-Cz remains positive in both sleep (0.1069) and rest (0.2889), suggesting stable connectivity with an increase in rest.
<p></p>
Higher standard deviations in rest for both pairs reflect greater variability, demonstrating that neural interactions fluctuate more during wakeful rest than in sleep. These findings further emphasize the distinct temporal dynamics of frontal and central brain regions across different states.

<p></p>
<h3>Temporal Analysis of Connectivity Within Sleep and Rest</h3>
<p></p>
Understanding how connectivity evolves within each state requires a detailed temporal analysis. While statistical comparisons provide an overview of differences between sleep and rest, examining connectivity patterns over time within each session offers deeper insights. Figures below present a time-resolved views of cosine similarities for F3-F4 and C4-Cz, capturing fluctuations in connectivity as they unfold. This approach helps identify transient changes, sustained trends, and potential transitions in neural activity, providing a more nuanced understanding of brain dynamics in sleep and rest.
<p></p>
<h4>Transforming Time Points</h4>
<p></p>
First, we converted the middle points of each sliding window into minutes and seconds to provide a clear temporal context. This was achieved by calculating the integer division and modulo of the middle points by 60 to derive minutes and seconds, respectively. These were then formatted into readable time labels (e.g., "12m 34.5s") for enhanced interpretability in our plots.
<p></p>
{% highlight python %}
cosine_sim_pairs_df['minutes'] = cosine_sim_pairs_df['middle_point'] // 60
cosine_sim_pairs_df['seconds'] = cosine_sim_pairs_df['middle_point'] % 60
cosine_sim_pairs_df['time_label'] = cosine_sim_pairs_df['minutes']
  .astype(int).astype(str) + 'm ' + cosine_sim_pairs_df['seconds']
  .round(3).astype(str) + 's'
{% endhighlight %}
<p></p>
<p></p>

<p></p>



<p></p>

<h4>Smoothing Cosine Similarity Values</h4>
<p></p>

Next, to reduce noise and highlight meaningful trends, we applied a Gaussian smoothing filter to the cosine similarity values. This technique helps clarify patterns by averaging adjacent points in the time series, resulting in smoother curves that better represent the underlying data.
<p></p>
<h4>Creating the Plot</h4>
<p></p>
The smoothed cosine similarity values for both channel pairs were plotted against their corresponding time points. Key details of the plot include:
<ul>
    <li><strong>X-axis:</strong> Time in minutes and seconds, with custom ticks to reduce clutter, ensuring a clear and focused visualization.</li>
    <li><strong>Y-axis:</strong> Cosine similarity values, representing the strength of connectivity between the selected EEG channels.</li>
    <li><strong>Curves:</strong> Separate lines for each channel pair (F3-F4 and C4-Cz) to allow for direct comparison of their temporal dynamics.</li>
</ul>
<p></p>
<h4>Insights and Observations</h4>
<p></p>
The resulting plot showcases how connectivity between specific brain regions changes over time. The F3-F4 pair, for instance, might exhibit distinct patterns compared to C4-Cz, reflecting differences in activity across these regions. This visualization provides a foundation for deeper analyses, such as correlating these dynamics with behavioral or physiological states.

<p></p>
<h4>Technical Details</h4>
<p></p>
The plot was created using Python libraries, including <code>matplotlib</code> for visualization and <code>scipy.ndimage</code> for smoothing. The data preparation involved grouping cosine similarity values, aligning them temporally, and ensuring consistency in the time axis for both channel pairs. This ensures an accurate and visually compelling comparison of the EEG data's temporal features.
<p></p>
By transforming, smoothing, and plotting the cosine similarity values, this analysis offers a detailed view of temporal connectivity dynamics in EEG data. It provides a vital step in understanding the intricate relationships between brain regions and their changes across different states, such as sleep and rest.

<p></p>

<p></p>
{% highlight python %}
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
cos_smoothed_sleep = gaussian_filter1d(cosine_sim_pairs_df1['cos'], sigma=2)
cos_smoothed_rest = gaussian_filter1d(cosine_sim_pairs_df2['cos'], sigma=2)
time_labels = cosine_sim_pairs_df1['time_label']  
step_size = 60
x_ticks = cosine_sim_pairs_df1['middle_point'][::step_size]
x_labels = [f"{int(m)}:{s:.1f}" for m,
  s in cosine_sim_pairs_df1[['minutes', 'seconds']].iloc[::step_size].values]
plt.figure(figsize=(12, 6))
plt.plot(
    cosine_sim_pairs_df1['middle_point'], cos_smoothed_sleep,
    label='F3-F4', color='brown', linewidth=1.5
)
plt.plot(
    cosine_sim_pairs_df2['middle_point'], cos_smoothed_rest,
    label='C4-Cz', color='green', linewidth=1.5
)
plt.xticks(x_ticks, x_labels, rotation=15, fontsize=10)
plt.xlabel('Time (minutes:seconds)', fontsize=12)
plt.ylabel('Cosine Similarity', fontsize=12)
plt.title('Cosine Similarity at Rest Time: F3-F4 vs. C4-Cz', fontsize=14)
plt.grid(alpha=0.3)
plt.legend()
plt.show()
{% endhighlight %}
<p></p>

<p></p>

<p></p>
<h4>Cosine Similarity at Sleep Time: F3-F4 vs. C4-Cz</h4>
<p></p>
This figure illustrates the temporal dynamics of cosine similarity for two EEG channel pairs, <strong>F3-F4</strong> and <strong>C4-Cz</strong>, during sleep. The x-axis represents time in minutes and seconds, while the y-axis shows the cosine similarity values. The red line corresponds to the F3-F4 channel pair, and the green line corresponds to the C4-Cz channel pair. The fluctuations in similarity values over time highlight differences in connectivity between these brain regions during sleep. This visualization offers a detailed view of how specific brain areas interact dynamically during sleep, capturing subtle connectivity changes.

<p></p>

<figure>
    <img src="{{ site.baseurl }}/img/eegSlide12.jpg" alt="Traditional EEG Graph Example" style="width:90%; margin:auto;">
    <figcaption>Temporal dynamics of cosine similarity during sleep for EEG channel pairs F3-F4 and C4-Cz, showcasing distinct connectivity patterns in brain regions associated with motor and sensory processing..</figcaption>
</figure>

<p></p>
<h4>Cosine Similarity at Rest Time: F3-F4 vs. C4-Cz</h4>
<p></p>
This figure depicts the cosine similarity for the same EEG channel pairs, <strong>F3-F4</strong> and <strong>C4-Cz</strong>, during rest. Similar to the sleep plot, the x-axis indicates time in minutes and seconds, and the y-axis represents cosine similarity values. The trends for F3-F4 (red) and C4-Cz (green) reveal distinct patterns of connectivity during rest, differing from the sleep state. These patterns reflect how brain activity and connectivity are modulated across different states.


<p></p>
<figure>
    <img src="{{ site.baseurl }}/img/eegSlide13.jpg" alt="Traditional EEG Graph Example" style="width:90%; margin:auto;">
    <figcaption>Temporal dynamics of cosine similarity during rest for EEG channel pairs F3-F4 and C4-Cz, highlighting connectivity differences in brain regions compared to the sleep state.</figcaption>
</figure>
<p></p>








<p></p>
<p></p>









<p></p>
<h2>Conclusion</h2>
<p></p>
<p>This study explores how <strong>sliding graph neural networks</strong> can help analyze <strong>EEG time series</strong>, capturing <strong>shifting connectivity patterns</strong> in the brain. By transforming EEG signals into overlapping graphs, <strong>GNN Graph Classification</strong> not only tracks how brain activity changes over time but also provides deeper insights into neural interactions beyond simple classification.</p>

<p>Our findings highlight clear differences between <strong>sleep and rest</strong>, especially in the <strong>frontal (F3-F4) and central (C4-Cz) regions</strong>. <strong>Cosine similarity analysis</strong> shows that while <strong>C4-Cz remains strongly connected during rest</strong>, <strong>F3-F4 shifts more between states</strong>, reflecting how different brain areas behave across conditions.</p>

<p>Bringing <strong>neuroscience and graph theory</strong> together opens exciting possibilities. <strong>Sliding graphs</strong> give neuroscientists a fresh way to uncover EEG patterns that might go unnoticed with traditional methods, while <strong>graph-based techniques</strong> gain new applications in sleep research. This collaboration isn’t just about analyzing data—it’s about connecting disciplines and discovering new ways to study the brain.</p>

<p>Beyond EEG, <strong>GNN Sliding Graph Classification</strong> has potential in many fields, from <strong>tracking climate trends</strong> to <strong>understanding financial markets</strong>. With opportunities to <strong>scale, improve interpretability, and tackle real-world challenges</strong>, this approach could offer fresh insights into complex systems far beyond neuroscience.</p>




<!-- This study demonstrates the effectiveness of sliding graph neural networks for EEG time series analysis. By transforming EEG signals into overlapping graph structures, GNN Graph Classification captures dynamic connectivity patterns over time, while pre-final vector embeddings provide deeper insights into neural interactions beyond classification.
<p></p>
Our analysis reveals distinct connectivity differences between rest and sleep states, particularly in the frontal (F3-F4) and central (C4-Cz) regions. Cosine similarity analysis shows that while C4-Cz exhibits stronger connectivity during rest, F3-F4 varies more across states, aligning with neurophysiological patterns.
<p></p>
Bridging neuroscience and graph theory presents both challenges and opportunities. Sliding graphs offer neuroscientists a new lens to uncover EEG patterns that conventional methods may miss, while graph experts see their tools applied to real-world challenges like sleep research. By merging these perspectives, we go beyond data analysis—building connections between disciplines and demonstrating how graph-based techniques can reshape brain research.
<p></p>
These findings suggest sliding graph-based GNNs as a powerful framework for modeling long-duration EEG dynamics. Future work will focus on scalability to larger datasets, clinical validation, and enhanced interpretability for sleep research and neurodiagnostics.
<p></p> -->







<p></p>

<p></p>

<p></p>
<p></p>
