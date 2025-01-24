---
layout:     post
title:      "Sliding Window Graph in GNN Graph Classification"
subtitle:   "GNN Graph Classification for Time Series: A New Perspective on Climate Change Analysis"
date:       2025-01-25 12:00:00
author:     "Melenar"
header-img: "img/page115e.jpg"
---

<p></p>
The use of Graph Neural Networks (GNNs) in time series analysis represents a rising field of study, particularly in the context of GNN Graph Classification, a technique traditionally applied in disciplines such as biology and chemistry. Our research repurposes GNN Graph Classification for the analysis of time series climate data, focusing on two distinct methodologies: the city-graph method, which effectively captures static temporal snapshots, and the sliding window graph method, adept at tracking dynamic temporal changes. This innovative application of GNN Graph Classification within time series data enables the uncovering of nuanced data trends.
<p></p>
We demonstrate how GNNs can construct meaningful graphs from time series data, showcasing their versatility across different analytical contexts. A key finding is GNNs’ adeptness at adapting to changes in graph structure, which significantly improves outlier detection. This enhances our understanding of climate patterns and suggests broader applications of GNN Graph Classification in analyzing complex data systems beyond traditional time series analysis. Our research seeks to fill a gap in current studies by providing an examination of GNNs in climate change analysis, highlighting the potential of these methods in capturing and interpreting intricate data trends.

<p></p>

<h2>GNN Sliding Graph Classification: Introduction</h2>

<p>
Understanding long time series data requires innovative approaches that can capture both temporal dynamics and topological patterns. In this blog, we introduce a novel methodology called <strong>GNN Sliding Graph Classification</strong>, designed to uncover deeper topological insights from time series data. This approach combines the power of sliding graph construction with the advanced capabilities of graph neural networks (GNNs).
</p>

<p>
The methodology consists of three key steps:
</p>

<ul>
  <li>
    <strong>Sliding Graph Construction:</strong> Transform time series data into graph structures by segmenting it into overlapping windows. Each graph captures localized temporal and spatial relationships, representing distinct patterns over the chosen time frame.
  </li>
  <li>
    <strong>GNN-Based Graph Classification:</strong> Utilize GNNs to classify these graphs, extracting high-level features from their topology while preserving the structural and temporal dependencies in the data.
  </li>
  <li>
    <strong>Pre-final Vectors:</strong> Obtain graph embeddings (pre-final vectors) from the GNN model during classification. These embeddings represent the learned topological features and are further analyzed to reveal temporal and structural patterns in the time series.
  </li>
</ul>

<p>
This approach bridges sliding window techniques and graph-based modeling, providing a powerful framework for analyzing complex temporal data. By doing so, it enables the discovery of both localized and global topological insights, advancing our understanding of time series dynamics.
</p>



<p></p>
<h2>Exploring EEG Through Graph-Based Methods: A Journey Through Our Studies</h2>
<p>
    Over the years, we’ve been on a mission to uncover the secrets of brain connectivity using <strong>EEG data</strong>.
    Our work has evolved from traditional graph analysis techniques to cutting-edge Graph Neural Networks (GNNs),
    each step uncovering deeper insights into neural dynamics. Let’s take a closer look at these studies.
</p>

<p><strong>Study 1: Traditional Graph Analysis</strong></p>
<p>
    Our journey began with a traditional graph analysis approach. In this study, we constructed connectivity graphs from EEG trials using
    <strong>cosine similarity</strong> between channels. Each graph’s nodes represented EEG electrodes, and the edges reflected their functional connectivity.
</p>
<p>Key Findings:</p>
<ul>
    <li>Differences in connectivity patterns emerged between the <strong>Alcoholic</strong> and <strong>Control</strong> groups, providing insights into altered neural activity.</li>
    <li>Graph features like clustering coefficients and edge density helped highlight these differences.</li>
    <li>However, traditional methods struggled to distinguish subtle variations, particularly in <strong>single-stimulus conditions</strong>, prompting the need for more advanced techniques.</li>
</ul>
<figure>
    <img src="{{ site.baseurl }}/img/dataSource5.jpg" alt="Traditional EEG Graph Example" style="width:70%; margin:auto;">
    <figcaption>Figure 1: A sample connectivity graph constructed from EEG data using cosine similarity.</figcaption>
</figure>
<p>
For a deeper dive into this work, check out our post <a href="http://sparklingdataocean.com/2020/08/19/brainGraphEeg/">"EEG Patterns by Deep Learning and Graph Mining"</a> or refer to the paper <a href="#">here</a>.
</p>
<p></p>

<p></p>
<p><strong>Study 2: Graph Neural Networks for Trial Classification</strong></p>
<p>
Building on our first study, we introduced <strong>Graph Neural Networks (GNNs)</strong> to analyze EEG data at the trial level. Each graph represented an entire EEG trial, encapsulating the connectivity across all channels.
</p>
<p>Why GNNs?</p>
<p>
GNNs brought a new level of sophistication by enabling the model to learn spatial relationships and connectivity dynamics within the graph.
</p>
<p>Key Findings:</p>
<ul>
    <li><strong>Improved Classification Accuracy:</strong> GNNs significantly outperformed traditional methods in differentiating between Alcoholic and Control groups.</li>
    <li><strong>Enhanced Connectivity Insights:</strong> Subtle variations in connectivity, previously missed, were captured.</li>
    <li><strong>Challenges:</strong> Misclassifications within the Control group highlighted the complexity of EEG connectivity patterns.</li>
</ul>

<p></p>
<figure>
    <img src="{{ site.baseurl }}/img/brain4.jpg" alt="Traditional EEG Graph Example" style="width:70%; margin:auto;">
    <figcaption>Figure 1: A sample connectivity graph constructed from EEG data using cosine similarity.</figcaption>
</figure>

YYYY
<p></p>
This approach is detailed further in our post <a href="http://sparklingdataocean.com/2023/05/08/classGraphEeg/">"GNN Graph Classification for EEG Pattern Analysis"</a>.
<p></p>

<p><strong>Study 3: Graph Neural Networks for Link Prediction</strong></p>
<p>
    In our third study, the focus shifted to <strong>link prediction</strong>, using GNNs to analyze node- and edge-level connectivity. A unified graph constructed from EEG electrode distances was used to predict connectivity dynamics.
</p>
<p>Key Findings:</p>
<ul>
    <li><strong>Revealing Hidden Connectivity:</strong> GNNs highlighted relationships between electrodes that were previously unobserved.</li>
    <li><strong>Node Importance:</strong> Certain electrodes emerged as more central to connectivity patterns.</li>
    <li><strong>Limitations:</strong> This method focused primarily on short-term EEG segments, leaving the dynamics of long-term recordings unexplored.</li>
</ul>
<figure>
    <img src="dataSource5.jpg" alt="Link Prediction Graph" style="width:80%; margin:auto;">
    <figcaption>Figure 3: A unified graph showcasing node- and edge-level EEG connectivity.</figcaption>
</figure>
<p>
    For more on this work, check out our <a href="http://sparklingdataocean.com/2024/11/09/GNN_timeSeries_EEG/">"Graph Neural Networks for EEG Connectivity Analysis"<a href="#"></a>.
<p></p>

<strong>Looking Ahead: Current Study</strong>
<p></p>
    While these studies advanced our understanding of EEG connectivity, they primarily focused on short-term EEG data. The current study takes a bold step forward, applying <strong>sliding graphs</strong> to analyze <strong>long-time EEG series</strong>.
<p></p>
    <strong>What’s New?</strong> This time, we focus on a single EEG channel and generate sliding graphs to explore how neural dynamics evolve over time during <strong>sleep</strong> and <strong>rest</strong>.
<p></p>
    <strong>Why It Matters?</strong> Understanding long-term EEG patterns opens the door to studying extended states of brain activity, offering insights into transitions and sustained neural processes.
<p></p>
    Stay tuned as we bring this next chapter to life, blending advanced graph techniques with long-term EEG data.



<p></p>




In this study, we expand on our previous research using Graph Neural Network (GNN) models to analyze climate data. Our earlier method categorized climate time series data into 'stable' and 'unstable' to identify unusual patterns in climate change.
<p></p>

<p></p>


  <h2>Methods</h2>
  <p><strong>Sliding Graph Construction</strong></p>
  <p>
    In our study, we introduce a novel approach to constructing graphs from EEG data using the
    <em>Sliding Window Method</em>.
  </p>
  <p></p>
  <a href="#">
      <img src="{{ site.baseurl }}/img/eegSlide3.jpg" alt="Post Sample Image" width="600" >
  </a>
  <p></p>
  <p></p>
  <h3>Sliding Window Method</h3>
  <ul>
    <li>
      <strong>Nodes</strong>: Represent data points within each sliding window, with features reflecting their respective values.
    </li>
    <li>
      <strong>Edges</strong>: Connect sequential points to preserve the temporal sequence and structure.
    </li>
    <li>
      <strong>Labels</strong>: Assigned to detect and analyze patterns within the time series.
    </li>
  </ul>

  <h3>Pipeline</h3>
  <p>Our pipeline for <strong>Graph Neural Network (GNN) Graph Classification</strong> consists of the following stages:</p>
  <ol>
    <li><strong>Data Input</strong>: For instance, EEG data representing brain activity during sleep and rest states.</li>
    <li>
      <strong>Graph Construction</strong>:
      <ul>
        <li><em>Sliding Window Method</em>: Segmenting time series data into smaller, overlapping graphs.</li>
        <li>
          <em>Virtual Nodes</em>: Acting as central hubs in small graphs, improving accuracy and enabling model tuning.
        </li>
      </ul>
    </li>
    <li><strong>GNN Model Application</strong>: Classifying graphs based on detected patterns using a GNN model.</li>
  </ol>

  <h3>Methodology for Sliding Window Graph Construction</h3>
  <p>
    <strong>Data to Graph Transformation</strong>: Time series data is segmented into overlapping windows using the sliding
    window technique. Each segment forms a unique graph, allowing for the analysis of local temporal dynamics.
  </p>
  <p>
    <strong>Graph Creation</strong>: In these graphs:
  </p>
  <ul>
    <li>
      <strong>Nodes</strong>: Represent data points within the window, with features derived from their values.
    </li>
    <li>
      <strong>Edges</strong>: Connect sequential nodes to maintain temporal relationships.
    </li>
  </ul>

  <p><strong>Key Parameters</strong></p>
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

    <strong>Graph Construction</strong>: Cosine similarity matrices are generated from the time series data and transformed into
    graph adjacency matrices.
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
    This framework effectively captures both local and global patterns within the time series, yielding valuable insights into
    temporal dynamics.
<p></p>

<p></p>
    <strong>Graph Classification</strong>: We employ the <em>GCNConv</em> model from the PyTorch Geometric Library for graph
    classification tasks. This model performs convolutional operations, leveraging edges, node attributes, and graph labels to
    extract features and analyze graph structures comprehensively.
<p></p>
    By combining the sliding window technique with Graph Neural Networks, our approach offers a robust framework for analyzing
    time series data. It captures intricate temporal dynamics and provides actionable insights into both local and global patterns,
    making it particularly well-suited for applications such as EEG analysis and classification tasks.

<p></p>







This method allows us to analyze time series data effectively by capturing both local and global patterns, providing valuable insights into temporal dynamics.
<p></p>
<h3>Model Training</h3>
<p></p>

Our methodology involves processing both city-centric and sliding window graphs. We start by generating cosine similarity matrices from time series data, which are then converted into graph adjacency matrices. This process includes creating edges for vector pairs with cosine values above a set threshold and adding a virtual node to ensure network connectivity, a critical step for preparing the graph structure.
<p></p>
For graph classification tasks, we use the GCNConv model from the PyTorch Geometric Library. This model excels in feature extraction through its convolutional operations, taking into account edges, node attributes, and graph labels for comprehensive graph analysis. The approach concludes with the training phase of the GNN model, applying these techniques to both types of graphs for robust classification.
<p></p>

<p></p>
<h2>Experiments Overview</h2>
<p></p>
<h3>Data Source: EEG Data</h3>
<p></p>
<p>
    For this study, we utilized EEG data from the
    <i><a href="https://github.com/OpenNeuroDatasets/ds003768/tree/master/sub-01/eeg" target="_blank">
      OpenNeuroDatasets
    </a></i>.
    This dataset includes EEG data collected from 33 healthy participants using a 32-channel MR-compatible EEG system
    (Brain Products, Munich, Germany). The EEG data were recorded during two 10-minute resting-state sessions (before and
    after a visual-motor adaptation task) and multiple 15-minute sleep sessions.
  </p>
  <p>
    For our analysis, we specifically focused on data from one resting-state session and one sleep session, using the raw
    EEG data for processing and comparative analysis of activity patterns during rest and sleep states.
  </p>

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
<p>
    The EEG signals from both the rest and sleep sessions were converted into DataFrames. Each DataFrame contains 32 EEG channels and a corresponding <code>Time</code> column, enabling a clear representation of time series data for further processing. The shapes of the resulting DataFrames are as follows:
  </p>
  <ul>
    <li><strong>Rest session:</strong> 4,042,800 rows × 33 columns</li>
    <li><strong>Sleep session:</strong> 4,632,500 rows × 33 columns</li>
  </ul>
  <p>
    This structured format facilitates segmentation, feature extraction, and the eventual construction of sliding graphs.
  </p>
  <p></p>
  <p>
      Given the large size of the EEG datasets, we applied downsampling to reduce the number of rows while retaining the temporal structure of the signals. Specifically, every 20th row from each DataFrame was selected, effectively reducing the data size by a factor of 20.
    </p>
<p></p>
{% highlight python %}
eeg_df1 = eeg_df1.iloc[::20, :].reset_index(drop=True)
eeg_df2 = eeg_df2.iloc[::20, :].reset_index(drop=True)
print(eeg_df1.shape, eeg_df2.shape)
(202140, 33) (231625, 33)
{% endhighlight %}
<p></p>
<p>
    After downsampling:
  </p>
  <ul>
    <li><strong>Rest session:</strong> 202,140 rows × 33 columns</li>
    <li><strong>Sleep session:</strong> 231,625 rows × 33 columns</li>
  </ul>
  <p>
    This step significantly reduced the computational overhead for subsequent processing steps while preserving meaningful patterns in the data.
  </p>
<p></p>  
<p>
  To ensure compatibility during analysis, both EEG DataFrames were truncated to have the same number of rows. This step is essential to facilitate pairwise comparisons and maintain consistency across the datasets.
</p>
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
<p>
  After truncation, both DataFrames contain:
</p>
<ul>
  <li><strong>Row count:</strong> 202,140</li>
  <li><strong>Column count:</strong> 33 EEG channels</li>
</ul>
<p>
  This ensures that subsequent operations, such as similarity calculations or graph-based analysis, can be performed without inconsistencies in data alignment.
</p>
<p></p>
<p>
  To prepare the EEG data for analysis, numerical columns were normalized to ensure consistent scaling across features. The 'Time' column was excluded during normalization and re-added afterward. This step helps improve the performance of subsequent analytical methods by standardizing the data.
</p>
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
<p>
    To enhance data tracking and processing, the 'Time' column was renamed, formatted as a string, and additional metadata columns were added:
  </p>
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
<p>
    These steps ensure that the data is not only normalized but also organized with clear metadata, facilitating downstream analysis and visualization tasks.
  </p>


<p></p>
<h3>Data Analysis</h3>
<p></p>
This step of data analysis focuses on comparing the cosine similarity between EEG channels during sleep and rest states. The top bar chart visualizes the channel-wise differences, highlighting which brain regions exhibit notable variations in activity patterns. The bottom chart aggregates these comparisons region-wise (e.g., Central, Occipital, Temporal), providing a high-level view of how different brain regions behave in sleep versus rest.
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/eegSlide1.jpg" alt="Post Sample Image" width="678" >
</a>
<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/eegSlide2.jpg" alt="Post Sample Image" width="500" >
</a>
<p></p>
Since time measures in separate sections do not overlap, this comparison offers a broad overview, serving as a basis for more detailed studies on individual sessions.

<p></p>
<h5>Normalization and Preprocessing</h5>
<p>In this step, we normalized the EEG data to ensure consistency across different sessions and reduce the impact of varying scales. The following processes were carried out:</p>

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

<p>This normalization step prepared the data for sliding window segmentation and graph construction, ensuring consistency and improving the robustness of the subsequent analyses.</p>

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
<p>To organize the EEG channels for our study, we grouped them based on their prefixes. This grouping helps us focus on specific brain regions for analysis and simplifies the selection process. Below are the steps and results of this process:</p>

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

<p>Below is the Python implementation used for channel grouping:</p>
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
<p>The resulting channel groups are as follows:</p>
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

<p>These groups will guide our selection of brain regions and EEG channels for further analysis in the study.</p>

<p></p>

<h3>Computing Cosine Similarities Within EEG Channel Groups</h3>
<p>As part of our EEG analysis, we calculated cosine similarities between channel pairs within the same group. This step focuses on understanding relationships between channels in specific brain regions. Below are the details of the process and implementation:</p>

<h4>Steps in Analysis</h4>
<ol>
  <li><strong>Channel Grouping:</strong> EEG channels were grouped based on their prefixes, corresponding to specific brain regions. Channels ending with <code>'z'</code> were adjusted by removing the trailing <code>'z'</code>, and other channels were grouped by their letter prefixes.</li>
  <li><strong>Sorting Channels:</strong> Channels within each group were sorted alphabetically to ensure consistent pairwise comparisons.</li>
  <li><strong>Cosine Similarity Calculation:</strong> Cosine similarities were computed for all possible pairs within each group using their numerical feature vectors.</li>
  <li><strong>Sorting Results:</strong> The cosine similarity pairs were sorted alphabetically for easy interpretation and analysis.</li>
</ol>

<p>The following Python code was used to perform the analysis:</p>

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

<p>This method helps isolate patterns within specific brain regions, contributing to our understanding of channel interactions during rest and sleep sessions.</p>

<p></p>
<figure>
    <img src="{{ site.baseurl }}/img/eegSlide5.jpg" alt="Data Analysis: Cosine Similarities" style="width:50%; margin:auto;">
    <figcaption>The table summarizes cosine similarity values for EEG channel pairs during sleep and rest states, alongside the difference between these states (Sleep - Rest).</figcaption>
</figure>

<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/eegSlide7.jpg" alt="Post Sample Image" width="500" >
</a>
<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/eegSlide8.jpg" alt="Post Sample Image" width="500" >
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
    <img src="{{ site.baseurl }}/img/eegSlide9.jpg" alt="Post Sample Image" width="500" >
</a>
<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/eegSlide10.jpg" alt="Post Sample Image" width="500" >
</a>
<p></p>
<p></p>

For our analysis, we selected the EEG channel pairs C4-Cz, F3-F4, and O1-O2. These pairs were chosen based on their relevance to brain region interactions and their notable differences in connectivity between sleep and rest states. These channels represent central, frontal, and occipital brain regions, providing a comprehensive view of neural activity across different areas of the brain.

<p></p>
<h3>Sliding Graph</h3>
<p>This function, <code>create_segments_df</code>, is designed to process a time series DataFrame by creating overlapping segments for a specified column. It helps prepare data for sliding window analysis, which is essential for studying temporal patterns in EEG signals. Below is a high-level description of its workflow:</p>

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

  <li><strong>Process:</strong>
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

<p>This function is particularly useful in EEG studies, enabling the division of continuous signals into manageable segments for sliding graph or time-series analysis.</p>
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
<p>The function <code>group_segments</code> is designed to group smaller data segments into larger groups for graph-based analysis. This process is crucial for aggregating segments in sliding window studies, particularly for EEG analysis. Here’s a detailed explanation:</p>

<ul>
  <li><strong>Inputs:</strong> The function takes the following parameters:
    <ul>
      <li><code>segments_df</code>: The DataFrame containing individual segments.</li>
      <li><code>group_size</code>: The number of segments in each group.</li>
      <li><code>group_shift</code>: The step size for sliding between groups.</li>
    </ul>
  </li>

  <li><strong>Process:</strong>
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

<p>This function facilitates efficient grouping of sliding window segments, enabling robust graph-based analysis for temporal patterns in EEG data.</p>

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

<p><strong>Parameters for Sliding Window and Grouping:</strong></p>
<p>We defined the following parameters for creating sliding windows and grouping segments:</p>
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
<p><strong>Data Scaling and Handling Missing Values:</strong></p>
<p>We selected EEG channels (e.g., <code>O1</code> and <code>O2</code>) for analysis and processed them as follows:</p>
<ul>
  <li>Missing values were replaced with the mean of the respective column.</li>
  <li>Min-Max Scaling was applied to normalize the data for consistency across features.</li>
</ul>

<p><strong>Sliding Window Segmentation and Grouping:</strong></p>
<p>Using the defined parameters, sliding windows were created for each channel (e.g., <code>O1</code> and <code>O2</code>), with each segment assigned a unique node index. Segments were then grouped into larger units for graph analysis.</p>

<p><strong>Dataset Creation:</strong></p>
<p>The grouped segments for both channels were concatenated into a single dataset. Each group was assigned a unique graph index, resulting in a dataset with 787 graph groups, ready for graph-based processing and analysis.</p>

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
{% highlight python %}
xxxx
{% endhighlight %}
<p></p>
<p></p>

<p></p>
<h4>Sliding Window Graph as Input for GNN Graph Classification</h4>
<p>In this stage of our analysis, we prepared sliding window graphs as input for a Graph Neural Network (GNN) classification task. Below is a high-level description of the process:</p>

<p><strong>Process Overview:</strong></p>
<p>We iteratively constructed graphs for EEG data using the predefined sliding windows and grouped segments. Each graph corresponds to a unique segment of the EEG data, capturing temporal relationships within the window. For each graph:</p>
<ul>
  <li>Features (<code>x</code>): Derived from EEG signal values within the segment, including the average of node features to enhance representation.</li>
  <li>Edges (<code>edge_index</code>): Created based on cosine similarity between node pairs, using a threshold (<code>cos &gt; 0.9</code>) to establish connections between nodes.</li>
  <li>Labels (<code>y</code>): Assigned based on the channel being analyzed (e.g., <code>O1</code> or <code>O2</code>).</li>
</ul>

<p><strong>Cosine Similarity Calculation:</strong></p>
<p>Cosine similarity was computed for all node pairs within each graph to determine connectivity. Node pairs exceeding the threshold of 0.9 were added as edges. This ensures that only significant relationships within the EEG signals are represented in the graph structure.</p>

<p><strong>DataLoader Preparation:</strong></p>
<p>The resulting graphs were packaged into datasets for model training and testing:</p>
<ul>
  <li><em>DatasetTest:</em> Contains graphs prepared for evaluation.</li>
  <li><em>DatasetModel:</em> Contains graphs ready for training the GNN model.</li>
</ul>
<p>These datasets were loaded into PyTorch Geometric's <code>DataLoader</code> for efficient batch processing during model training and evaluation.</p>

<p><strong>Outcome:</strong></p>
<p>The constructed sliding window graphs provide a structured and efficient way to capture temporal EEG patterns for graph-based classification. This approach highlights the power of combining sliding window analysis with GNNs to study EEG signals.</p>

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
<h5>GNN Graph Classification: Model Training.</h5>
<p></p>  
<p>To classify EEG data using a graph neural network (GNN), we implemented a training pipeline that incorporates data splitting, model definition, and training steps. Below is an overview of the process:</p>



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
<p><strong>Dataset Splitting:</strong> The dataset was split into training and testing sets with a 17% test size. The data was prepared for training using PyTorch Geometric's DataLoader, ensuring efficient batch processing.</p>

<p></p>

<p><strong>Model Architecture:</strong> A Graph Convolutional Network (GCN) was designed for EEG graph classification. The model includes:</p>
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
<p>The model is now ready for training and evaluation using the prepared data loaders. This architecture leverages node-level and graph-level features for effective classification.</p>


<h4>Model Training and Evaluation</h4>
<p></p>

<p>The training and evaluation process for the GNN model involves key steps to optimize the parameters and assess performance. Below is an overview of the methodology:</p>

<p><strong>Training Process:</strong></p>
<ul>
  <li>Perform a single forward pass over batches in the training dataset.</li>
  <li>Compute the loss using the cross-entropy loss function.</li>
  <li>Derive gradients using backpropagation.</li>
  <li>Update model parameters based on the computed gradients.</li>
  <li>Clear gradients after each step to prevent accumulation.</li>
</ul>

<p><strong>Evaluation Process:</strong></p>
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


<p>This section details the training and evaluation process of the graph neural network (GNN) model for the EEG channel pair F3-F4 during the sleep session. The model was trained over 16 epochs, with accuracy metrics computed for both the training and test datasets at each epoch.</p>


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
    <img src="{{ site.baseurl }}/img/eegSlide6.jpg" alt="Post Sample Image" width="345" >
</a>
<p></p>
<ul>
  <li><b>Training Accuracy:</b> Indicates the model's ability to learn patterns from the training dataset. Accuracy steadily increased across epochs, reaching a peak of <strong>0.9502</strong>.</li>
  <li><b>Test Accuracy:</b> Reflects the model's performance on unseen test data, gradually improving and achieving a high value of <strong>0.9366</strong> by the final epoch.</li>
</ul>

<p>The consistent improvement in both training and test accuracy demonstrates the model's capability to generalize well. This highlights its effectiveness in classifying EEG data based on sliding window graphs for the F3-F4 channel pair during sleep.</p>

<p></p>

<p></p>
<p></p>
<p>The table summarizes cosine similarity values and graph neural network (GNN) performance for selected EEG channel pairs across sleep and rest sessions. It provides insights into how these pairs interact during different states and how well the GNN model captures these patterns.</p>

<h4>Analysis of Cosine Similarity and GNN Performance for Selected EEG Pairs</h4>
<p>The table summarizes cosine similarity values and graph neural network (GNN) performance for selected EEG channel pairs across sleep and rest sessions. It provides insights into how these pairs interact during different states and how well the GNN model captures these patterns.</p>

<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/eegSlide11.jpg" alt="Post Sample Image" width="657" >
</a>
<p></p>

<p><strong>Key Observations:</strong></p>
<ul>
  <li><b>Channel Pairs and Cosine Similarity:</b> Cosine similarity values highlight the relationship between EEG signals. For example, the C4-Cz pair showed stronger similarity during sleep, while the O1-O2 pair maintained consistently high similarity across both states.</li>
  <li><b>Brain Regions Affected:</b> The F3-F4 pair, associated with frontal brain activity, showed notable differences in similarity between sleep and rest, reflecting the frontal lobe's role in decision-making and reduced cognitive activity during sleep. The occipital pair (O1-O2), responsible for visual processing, showed consistently high similarity across both states, indicating stable interactions in this region.</li>
  <li><b>Training Accuracy:</b> The GNN model effectively learned patterns from the training data, with the F3-F4 pair achieving the highest training accuracy during sleep.</li>
  <li><b>Test Accuracy:</b> Performance on test data varied across pairs and states. The F3-F4 pair demonstrated strong generalization during sleep, while other pairs showed moderate accuracy differences between states.</li>
</ul>

<p><strong>Interpreting High Cosine Similarity and Low Accuracy:</strong></p>
<p>While high cosine similarity values suggest stable and predictable relationships between EEG signals, they can reduce the variability necessary for effective machine learning classification. When similarity values are consistently high, the model struggles to differentiate patterns, resulting in lower classification accuracy. This is particularly evident in the O1-O2 pair, where consistently high cosine similarity across both states contributed to reduced GNN accuracy.</p>

<p>These results underscore the variability in EEG signal relationships and model performance, reflecting the distinct dynamics of sleep and rest states, as well as the challenges of analyzing highly correlated data.</p>

<p>These results underscore the variability in EEG signal relationships and model performance, reflecting the distinct dynamics of sleep and rest states.</p>

<p></p>


<p></p>
<p></p>

<p></p>   

<p></p>
<p></p>   
<h3>Model Results Interpretation</h3>
<p></p>

<p>The results interpretation phase of the study focused on analyzing the predictions and embeddings generated by the graph neural network (GNN) model. Using a softmax function, the model's outputs were transformed into probabilities to better understand the classification predictions and identify the most likely labels for each graph.</p>

<p><b>Process:</b></p>
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


<p>The resulting DataFrame contains each graph's index, embedding vectors, and prediction results. The embeddings serve as high-dimensional representations of the EEG data, enabling further analysis of the underlying patterns and relationships identified by the GNN model.


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
This step bridges the gap between model training and interpretability, allowing for a deeper understanding of how the GNN processes and classifies EEG-based sliding window graphs.</p>

<h4>Cosine Similarity Analysis for Graph Embeddings</h4>

<p>This step evaluates the similarity between pre-final embedding vectors generated by the GNN model for sliding window graphs. By calculating cosine similarity, we gain insights into the relationships and connectivity patterns captured by the model.</p>

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

<p>This analysis bridges the gap between model outputs and interpretability, offering a clearer understanding of how the GNN captures and distinguishes temporal patterns. By identifying regions of high and low similarity, this step enables further exploration of brain dynamics during sleep and rest states, paving the way for advanced graph-based analyses.</p>
<p></p>
<h4>Transforming Time Points</h4>
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



<h4>Smoothing Cosine Similarity Values</h4>
Next, to reduce noise and highlight meaningful trends, we applied a Gaussian smoothing filter to the cosine similarity values. This technique helps clarify patterns by averaging adjacent points in the time series, resulting in smoother curves that better represent the underlying data.
<p></p>
<h4>Creating the Plot</h4>
<p>The smoothed cosine similarity values for both channel pairs were plotted against their corresponding time points. Key details of the plot include:</p>
<ul>
    <li><strong>X-axis:</strong> Time in minutes and seconds, with custom ticks to reduce clutter, ensuring a clear and focused visualization.</li>
    <li><strong>Y-axis:</strong> Cosine similarity values, representing the strength of connectivity between the selected EEG channels.</li>
    <li><strong>Curves:</strong> Separate lines for each channel pair (F3-F4 and C4-Cz) to allow for direct comparison of their temporal dynamics.</li>
</ul>
<p></p>
<h4>Insights and Observations</h4>
<p>The resulting plot showcases how connectivity between specific brain regions changes over time. The F3-F4 pair, for instance, might exhibit distinct patterns compared to C4-Cz, reflecting differences in activity across these regions. This visualization provides a foundation for deeper analyses, such as correlating these dynamics with behavioral or physiological states.</p>
<p></p>
<h4>Technical Details</h4>
<p>The plot was created using Python libraries, including <code>matplotlib</code> for visualization and <code>scipy.ndimage</code> for smoothing. The data preparation involved grouping cosine similarity values, aligning them temporally, and ensuring consistency in the time axis for both channel pairs. This ensures an accurate and visually compelling comparison of the EEG data's temporal features.
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
<h4>Explanation and Suggested Descriptions for the Figures</h4>

<h5>Figure 1: Cosine Similarity at Sleep Time: F3-F4 vs. C4-Cz</h5>
<p>
This figure illustrates the temporal dynamics of cosine similarity for two EEG channel pairs, <strong>F3-F4</strong> and <strong>C4-Cz</strong>, during sleep. The x-axis represents time in minutes and seconds, while the y-axis shows the cosine similarity values. The red line corresponds to the F3-F4 channel pair, and the green line corresponds to the C4-Cz channel pair. The fluctuations in similarity values over time highlight differences in connectivity between these brain regions during sleep. This visualization offers a detailed view of how specific brain areas interact dynamically during sleep, capturing subtle connectivity changes.
</p>
<p></p>

<figure>
    <img src="{{ site.baseurl }}/img/eegSlide12.jpg" alt="Traditional EEG Graph Example" style="width:90%; margin:auto;">
    <figcaption>Temporal dynamics of cosine similarity during sleep for EEG channel pairs F3-F4 and C4-Cz, showcasing distinct connectivity patterns in brain regions associated with motor and sensory processing..</figcaption>
</figure>


<p></p>
<h5>Figure 2: Cosine Similarity at Rest Time: F3-F4 vs. C4-Cz</h5>
<p>
This figure depicts the cosine similarity for the same EEG channel pairs, <strong>F3-F4</strong> and <strong>C4-Cz</strong>, during rest. Similar to the sleep plot, the x-axis indicates time in minutes and seconds, and the y-axis represents cosine similarity values. The trends for F3-F4 (red) and C4-Cz (green) reveal distinct patterns of connectivity during rest, differing from the sleep state. These patterns reflect how brain activity and connectivity are modulated across different states.
</p>

<p></p>
<figure>
    <img src="{{ site.baseurl }}/img/eegSlide13.jpg" alt="Traditional EEG Graph Example" style="width:90%; margin:auto;">
    <figcaption>Temporal dynamics of cosine similarity during rest for EEG channel pairs F3-F4 and C4-Cz, highlighting connectivity differences in brain regions compared to the sleep state.</figcaption>
</figure>
<p></p>





<h5>Note on O1-O2 Analysis</h5>
<p>
Although <strong>O1-O2</strong> was initially included as part of the analysis, its results have been excluded from the figures and detailed discussion due to the very low model training and testing accuracy observed for this channel pair. This suggests that the model failed to capture meaningful patterns or dynamics for O1-O2, likely due to insufficient signal quality or inherent limitations in the data for this pair.
</p>



<p></p>
<p></p>


<p></p>






<p></p>
<h2>In Conclusion</h2>
<p></p>


<p></p>




<p></p>
{% highlight python %}
xxxx
{% endhighlight %}
<p></p>

<p></p>
{% highlight python %}
xxxx
{% endhighlight %}
<p></p>

<p></p>
{% highlight python %}
xxxx
{% endhighlight %}
<p></p>


<p></p>

<p></p>

<p></p>
<p></p>
