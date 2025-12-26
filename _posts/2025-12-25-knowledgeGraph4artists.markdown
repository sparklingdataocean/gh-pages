---
layout:     post
title:      "Giving AI a Shared Language with Unified Knowledge Graphs"
subtitle:   "Country Borders as a Lens on Multimodal Embeddings"
date:       2025-12-25 12:00:00
author:     "Melenar"
header-img: "img/ukgPage8f.jpg"
---

<section>
  <h2>Why AI Needs a Shared Language</h2>

  <p>
    Modern data doesn’t come in neat tables anymore. It shows up as text, time-series signals,
    dashboards, maps, logs, and relationships—often all describing the same entities from different
    angles. Countries, companies, patients, products: each is surrounded by stories, numbers, and
    connections that live in separate systems and are analyzed with separate tools.
  </p>

  <p>
    AI can handle each data type in isolation. Language models read text. Statistical models handle
    time series. Graph methods analyze relationships. But the real insight lives between these views.
    The hard part isn’t modeling text or numbers—it’s making them <em>comparable</em>,
    <em>relational</em>, and <em>interpretable together</em>.
  </p>

  <p>
    That’s where <strong>Unified Knowledge Graphs (UKGs)</strong> come in.
  </p>
Unified Knowledge Graphs help by acting as a relationship layer that can hold all these views together. And here’s the important clarification for anyone who hears “graph” and immediately thinks “graph database” or “Cypher/SPARQL”: you don’t need a graph database, and you don’t need graph query languages for this. A “graph” here is simply a lightweight way to express relationships—nodes and edges—even if your data still lives in SQL, Parquet, or JSON.
  <p>
    A UKG doesn’t just connect entities with edges. It aligns heterogeneous data into a common
    representation so text, time series, and relationships can be analyzed side by side. The
    challenge is that these modalities speak different mathematical languages. Simply attaching
    embeddings to nodes—or concatenating features late in the pipeline—often produces representations
    that look unified, but aren’t truly comparable.
  </p>

  <p>
    In this work, we explore a simple idea: use <strong>GNN link prediction</strong> as an alignment
    mechanism.
  </p>

  <p>
    Instead of forcing every data type into a single model, we let each modality keep its native
    representation—but train all modalities under the same relational objective. By learning
    embeddings that must explain observed relationships in the graph, each modality is effectively
    “compiled” into a shared vector space—a common coordinate system where similarities,
    differences, and outliers become meaningful across data types.
  </p>

  <p>
    We illustrate the approach using <strong>countries as entities</strong>,
    <strong>borders as relationships</strong>, <strong>Factbook text</strong> as narrative context,
    and <strong>World Bank time series</strong> (GDP, life expectancy, internet usage) as numeric
    signals. Borders act as an interpretable backbone—not because geography is the goal, but because
    it provides a clean way to test whether different modalities align (or fail to align) once
    embedded into the same space.
  </p>

  <p>
    The result is not one monolithic model, but a <strong>unified embedding space</strong> that gives
    you:
  </p>

  <ul>
    <li>a single coordinate system for text and numeric signals,</li>
    <li>embeddings shaped by real relationships, not just feature similarity,</li>
    <li>multimodal profiles that can be compared, fused, and analyzed consistently.</li>
  </ul>

  <a href="#">
    <img src="{{ site.baseurl }}/img/ukg7.jpg"
         alt="Unified knowledge graph as a multimodal integration layer"
         width="777">
  </a>

  <p>
    Although demonstrated here at the country level, the idea is domain-agnostic. Any setting with
    heterogeneous attributes and meaningful relationships—enterprises, products, biological
    entities, infrastructure systems—can benefit from treating a knowledge graph as an
    <strong>alignment layer</strong>, not just a storage layer.
  </p>

  <p>
    In short: unified knowledge graphs aren’t about adding more data—they’re about giving AI a
    <strong>common language</strong> for reasoning over relationships.
  </p>
</section>








<section id="methods">

  <h2>Methods: Turning Heterogeneous Data into a Shared Graph Language</h2>

  <p>
    The goal of this pipeline is not to train a single task-specific model, but to create a
    <strong>shared embedding space</strong> in which heterogeneous country attributes—text,
    time series, and relationships—become directly comparable.
    The unifying idea is simple: <em>every modality is aligned through the same relational task</em>.
  </p>

  <p>
    Instead of forcing different data types into one model, we let each modality keep its native
    representation, then use <strong>GNN link prediction</strong> as a common alignment mechanism.
    This produces relationship-aware embeddings that all live in the same vector space.
  </p>

  <a href="#">
    <img src="{{ site.baseurl }}/img/ukg4.jpg"
         alt="Unified Knowledge Graph pipeline overview"
         width="720">
  </a>

  <h3>Relational Backbone: Country Borders</h3>

  <p>
    We start by constructing a country-level graph. Each node represents a sovereign state.
    Edges encode geographic adjacency derived from two sources:
    land borders and maritime (sea) borders.
  </p>

  <p>
    The two graphs are merged into a single border topology, with each edge labeled as
    <code>land</code>, <code>sea</code>, or <code>both</code>.
    This border graph is <strong>fixed and shared across all modalities</strong>,
    acting as the relational backbone that aligns every embedding.
  </p>

  <h3>Multimodal Node Attributes</h3>

  <p>
    Each country node is associated with multiple, heterogeneous feature channels.
    These channels are intentionally kept separate through most of the pipeline.
  </p>

  <h4>Text: CIA World Factbook</h4>

  <p>
    For each country, we extract two narrative fields from the CIA World Factbook:
    <em>Background</em> and <em>Geography</em>.
    These are concatenated into a single document describing historical, political,
    and geographic context.
  </p>

  <p>
    The text is embedded using the <code>all-MiniLM-L6-v2</code> sentence transformer,
    producing a fixed-length vector per country. These vectors serve as the text feature channel
    for graph learning.
  </p>

  <h4>Time Series: World Bank Indicators</h4>

  <p>
    We use three socio-economic indicators from the World Bank:
    GDP per capita, life expectancy at birth, and internet usage.
    Each indicator is represented as a yearly sequence per country.
  </p>

  <p>
    Within each indicator, values are normalized, missing years are interpolated along the
    time axis, and countries with insufficient coverage are removed.
    The result is one fixed-length time-series vector per country, per indicator.
  </p>

  <h3>Alignment via GNN Link Prediction</h3>

  <p>
    Each modality—text and each time-series indicator—is aligned separately using the
    <strong>same GNN link-prediction objective</strong> on the shared border graph.
    We use GraphSAGE with identical architecture and training protocol across modalities.
  </p>

  <p>
    Link prediction is used deliberately: it forces embeddings to explain observed relationships.
    As a result, each modality is translated into a
    <em>relationship-aware representation</em>, shaped by the same topology.
  </p>

  <p>
    Every model outputs a <strong>64-dimensional embedding per country</strong>.
    Because all modalities share the same graph structure, objective, and embedding dimension,
    their vectors lie in a common latent space and can be compared directly.
  </p>

  <h3>From Modality-Specific to Unified Representations</h3>

  <p>
    At this point, each country has multiple embeddings:
    one from text and one from each time-series indicator.
    These vectors are already comparable—but still modality-specific.
  </p>

  <p>
    To construct a single unified representation, we concatenate the four modality vectors
    into a 256-dimensional multimodal profile and project it back to 64 dimensions.
    This projection is an explicit unification step, not a late-stage heuristic.
  </p>

  <p>
    The resulting unified embedding summarizes all modalities while remaining compatible
    with the original per-modality embeddings in the same space.
  </p>

  <h3>What the Pipeline Produces</h3>

  <p>
    The final output of the pipeline consists of:
  </p>

  <ul>
    <li>Relationship-aware 64-dimensional embeddings for each modality</li>
    <li>A unified 64-dimensional embedding per country</li>
  </ul>

  <p>
    Together, these representations support consistent similarity analysis,
    border-aware reasoning, and exploratory graph analysis across heterogeneous data—
    without forcing premature fusion or losing relational structure.
  </p>

</section>


<section id="experiments">
  <h2>Experiments</h2>
<p></p>


  <section id="graph-construction">
    <h3>Graph Construction: Country Borders</h3>

    <p>
      All experiments use a <strong>fixed, unweighted country border graph</strong>.
      Countries are nodes; edges encode geographic adjacency from two public geospatial sources.
    </p>

    <p>
      <strong>Land borders</strong> come from Natural Earth <em>Admin 0 – Countries</em> polygons:
      two countries are connected if their polygons share a land boundary.
      <strong>Sea borders</strong> come from the MarineRegions <em>World EEZ v12</em> dataset:
      two countries are connected if their EEZ regions touch or overlap, capturing maritime neighbors.
    </p>

    <p>
      We merge land and sea adjacency into a single undirected graph and label each edge as
      <code>land</code>, <code>sea</code>, or <code>both</code>. The topology is <strong>not learned</strong>;
      it is shared across modalities so differences in embeddings reflect the data, not the graph.
    </p>

    <p>
      Borders are used <strong>as a diagnostic lens</strong>, not as ground truth for similarity:
      they let us test how different modalities align with adjacency once embedded in a shared space.
    </p>
  </section>


<p></p>

<p>
    <strong>Land borders:</strong> Natural Earth. "Admin 0 - Countries." Version 5.0.1, Natural Earth, 2023.
    <a href="https://www.naturalearthdata.com" target="_blank">https://www.naturalearthdata.com</a>
</p>
<p></p>

<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/gkgMap1.jpg" alt="Post Sample Image" width="700" >
</a>


<p>
    <strong>Sea boundaries:</strong> Marineregions.org. (2023). World EEZ v12 [Dataset]. Version 12.
    <a href="https://www.marineregions.org/" target="_blank">https://www.marineregions.org/</a>.
</p>
<a href="#">
    <img src="{{ site.baseurl }}/img/gkgMap2.jpg" alt="Post Sample Image" width="700" >
</a>
<p></p>
<p></p>




  <p>
    We evaluate link prediction with AUC (Area Under the ROC Curve), which summarizes how well the model
    separates true border edges from non-edges across thresholds. For the text modality, we obtain
    <strong>AUC = 0.9202</strong>, suggesting that Factbook narratives align strongly with geographic adjacency
    once translated into relationship-aware embeddings.
  </p>
</section>

<section id="text-modality-factbook">
  <h3>Text Modality: CIA Factbook</h3>

  <p>
    We combine two CIA World Factbook fields—<em>Background</em> and <em>Geography</em>—into one document per country and embed it with <code>all-MiniLM-L6-v2</code> (384-dim).
  </p>

  <p>
    Using these vectors as node features, we train a GraphSAGE link-prediction model on the border graph and take the pre-final 64-dim embeddings as <strong>Factbook vectors</strong> (text representations shaped by geographic adjacency).
  </p>

  <p>
    Link prediction achieves <strong>AUC = 0.9202</strong>, indicating strong alignment between Factbook narratives and border structure.
  </p>
</section>




<p></p>


<section id="ts-modality">
  <h3>Time-Series Modality: World Bank Indicators</h3>

  <p>
    To capture change over time, we use three World Bank WDI indicators and treat each one as a separate channel, so we can compare how different numeric signals behave on the same border graph.
  </p>

  <p><strong>Indicators used:</strong></p>
  <ul>
    <li><strong>Life Expectancy</strong> (1960–2022)</li>
    <li><strong>GDP per Capita</strong> (1960–2023)</li>
    <li><strong>Internet Usage</strong> (1990–2023)</li>
  </ul>

  <p>
    For each country, each indicator is a yearly sequence. We normalize per-indicator, fill missing years via simple time-axis interpolation, and drop countries with excessive missingness. The resulting sequences are used as node features.
  </p>

  <p>
    We train a separate GraphSAGE link-prediction model for each indicator on the same border graph, then take the pre-final 64-dim embeddings as the indicator-specific vectors (time-series representations shaped by geography).
  </p>

  <p>
    AUC varies by indicator: <strong>Life Expectancy</strong> <strong>(0.8696)</strong>, <strong>GDP per Capita</strong> <strong>(0.7817)</strong>, and <strong>Internet Usage</strong> <strong>(0.8318)</strong>, reflecting how strongly each signal aligns with border structure once embedded.
  </p>
</section>






<section id="ukg">
  <h3>Unified Knowledge Graph</h3>

  <p>
    Training one model per modality yields four 64-dimensional embeddings per country (Factbook text, GDP, life expectancy, internet usage). Because all models are trained on the same border graph with the same link-prediction objective, these embeddings are directly comparable within a shared coordinate system.
  </p>

  <p>
    To produce a single “all-in-one” country representation, we fuse the four vectors by concatenating them into a 256-D profile and projecting back down to a compact 64-D embedding. We denote this unified vector as <code>concat64</code> (also <em>z<sup>uni</sup></em>).
  </p>

  <p>
    Conceptually, this shared 64-D space becomes a machine-readable language for multimodal data: text and time-series signals are mapped into a common representation layer, enabling consistent similarity and relationship analysis across modalities without modality-specific rules.
  </p>
</section>


<p></p>

<section id="global-border-type-structure">

<h3>Global Border-Type Structure</h3>

<p>
  We begin by analyzing the <strong>unified country embeddings</strong> obtained by fusing all modalities
  into a single 64-dimensional representation. To test whether geographic structure is preserved in this
  unified space, we compute cosine similarity for all country pairs and group the scores by border type:
  <code>land</code>, <code>sea</code>, <code>both</code>, and <code>none</code>. This provides a global,
  relationship-centric view of how adjacency is reflected after multimodal fusion.
</p>

<p>
  The resulting similarity distributions exhibit a clear and interpretable ordering. Countries sharing
  <code>land</code> borders or <code>both</code> land and sea borders are, on average, the most similar.
  <code>sea</code>-only neighbors occupy an intermediate regime, while <code>none</code> pairs (no border)
  show the greatest variability and lowest central tendency. This pattern indicates that fusion preserves
  meaningful geographic signal rather than washing it out.
</p>
<a href="#">
  <img src="{{ site.baseurl }}/img/ukg6.jpg" alt="Unified knowledge graph as a multimodal integration layer" width="589">
</a>
<p>
  Applying the same border-type analysis to <strong>individual modality embeddings</strong> reveals substantial
  differences in how modalities align with geographic structure. Text-based embeddings derived from the
  CIA Factbook show the strongest separation by border type, suggesting that historical, political, and
  geographic narratives are tightly coupled to physical proximity. Economic and infrastructure indicators
  (GDP and Internet usage) display moderate sensitivity, reflecting regional synchronization alongside
  notable cross-border divergence. In contrast, life expectancy shows minimal dependence on borders,
  consistent with its globally smooth temporal behavior.
</p>
<a href="#">
  <img src="{{ site.baseurl }}/img/ukg5.jpg" alt="Unified knowledge graph as a multimodal integration layer" width="777">
</a>
<p>
  Table below summarizes these trends numerically. Factbook text exhibits the strongest contrast between
  neighboring and non-neighboring countries, while life expectancy remains nearly invariant across border types.
  GDP and Internet usage occupy an intermediate regime, confirming that some—but not all—modalities encode
  geographic structure in a relationship-aware way.
</p>
<a href="#">
  <img src="{{ site.baseurl }}/img/ukg11.jpg" alt="Unified knowledge graph as a multimodal integration layer" width="600">
</a>
<p>
  Taken together, these results show that unified embeddings retain interpretable relational structure while
  still exposing modality-specific differences. Borders are not treated as ground truth for similarity, but as
  a diagnostic lens that reveals where multimodal representations align with geography—and where they meaningfully
  diverge.
</p>

</section>
  <p></p>

  <section id="relational-outliers">
    <h3>Relational outliers (pairwise view)</h3>

    <p>
      Aggregate plots show the overall trend, but the more interesting stories often live in the
      exceptions. To make those visible, we look for <strong>relational outliers</strong>—country pairs
      whose similarity sharply contradicts geographic expectations.
    </p>
    <a href="#">
      <img src="{{ site.baseurl }}/img/ukg12.jpg" alt="Unified knowledge graph as a multimodal integration layer" width="600">
    </a>
    <p>
      Among <strong>dissimilar neighbors</strong> (countries connected by <code>land</code> or
      <code>both</code> borders), the GDP embedding highlights borders where adjacent countries follow
      very different economic paths. The lowest-similarity pairs concentrate around a small set of regions,
      especially <strong>Israel and its neighbors</strong> (Syria, Egypt, Jordan, Lebanon), where structural
      divergence dominates despite direct adjacency. Other examples—like <strong>China–Hong Kong</strong> and
      <strong>Austria–Liechtenstein</strong>—suggest that differences in economic systems or scale effects can
      outweigh shared geography.
    </p>
    <a href="#">
      <img src="{{ site.baseurl }}/img/ukg13.jpg" alt="Unified knowledge graph as a multimodal integration layer" width="600">
    </a>

    <p>
      Text-based (Factbook) outliers behave differently. Here, low-similarity neighbor pairs often involve
      historically and politically prominent countries such as <strong>China</strong>, <strong>France</strong>,
      and the <strong>United States</strong>. This is consistent with the idea that national narratives
      (history, governance, geopolitical role) can diverge substantially even across borders, producing
      textual distance that is not purely geographic.
    </p>
    <a href="#">
      <img src="{{ site.baseurl }}/img/ukg14.jpg" alt="Unified knowledge graph as a multimodal integration layer" width="600">
    </a>
    <p>
      In the opposite direction, <strong>similar non-neighbors</strong> under GDP embeddings reveal distant
      pairs with near-identical trajectories. Examples like <strong>Egypt–Tunisia</strong> and
      <strong>United Kingdom–Sweden</strong> point to regional or developmental synchronization, while pairs
      like <strong>Austria–Denmark</strong> and <strong>India–Vietnam</strong> show that modality-specific
      similarity can emerge independently of proximity.
    </p>

    <p>
      Taken together, these rankings demonstrate the value of a relationship-aware embedding space: it can
      preserve geographic structure <em>and</em> surface meaningful deviations—borders where adjacency does not
      imply similarity, and distant pairs where strong alignment appears unexpectedly.
    </p>
  </section>


  <section id="country-neighborhood-diversity">
    <h3>Country-Centric Neighborhood Diversity</h3>

    <p>
      Pairwise outliers are helpful, but a country-centric view asks: does a country look like its
      neighborhood <em>on average</em>? For each country, we compute the mean cosine similarity between its
      embedding and the embeddings of its neighbors (using <code>land</code> + <code>both</code> edges).
      Lower averages indicate a more heterogeneous neighborhood in the embedding space.
    </p>



    <p>
      <strong>Unified embedding (<code>concat64</code>).</strong>
      The table below highlights countries whose neighborhoods are the most diverse under the fused
      multimodal profile. Syria is the strongest outlier, and countries like China, Iraq, and Iran also score low,
      suggesting regions where borders connect countries with very different multimodal signatures. This is not
      claiming borders imply similarity—rather, it flags where the unified space sees high local variation worth
      investigating.
    </p>
    <a href="#">
      <img src="{{ site.baseurl }}/img/ukg15.jpg" alt="Unified knowledge graph as a multimodal integration layer" width="600">
    </a>


    <p class="note">
  <em>Note:</em> Results are reported only for countries with <strong>at least 5 neighbors</strong>.
  Neighbors are defined using <code>land</code> + <code>both</code> border types. Lower average cosine = greater neighborhood heterogeneity.
</p>


    <p>
      <strong>Factbook-only embedding.</strong>
      To see whether these effects are specific to fusion or already present in one modality, we repeat the same
      metric using the Factbook embeddings. Values are higher overall (text tends to align with borders more strongly),
      but France, China, and Russia still appear among the lowest averages—suggesting their textual profiles diverge
      more from nearby neighbors than typical.
    </p>
    <a href="#">
      <img src="{{ site.baseurl }}/img/ukg16.jpg" alt="Unified knowledge graph as a multimodal integration layer" width="600">
    </a>


    <p>
      Taken together, this country-centric lens complements the global and pairwise analyses by highlighting border
      regions with systematically diverse neighborhoods—patterns that are hard to see when modalities are analyzed
      independently.
    </p>
  </section>

  <a href="#">
    <img src="{{ site.baseurl }}/img/ukg17.jpg" alt="Unified knowledge graph as a multimodal integration layer" width="800">
  </a>
  <p>
    This map shows how similar each country is to its neighbors in the <strong>GDP embedding space</strong>.
    Each country is colored by its <strong>average cosine similarity</strong> to its
    <strong>land</strong> and <strong>both</strong> border neighbors. We compute this statistic only for
    countries with <strong>at least 2 neighbors</strong>; countries that do not meet this threshold (or
    have missing data) are shown in <span style="color:#777;">gray</span>.
  </p>

  <ul>
    <li><strong>Lighter</strong> colors indicate countries that are <strong>more similar</strong> to their neighbors.</li>
    <li><strong>Darker</strong> colors indicate countries that are <strong>more different</strong> from their neighbors.</li>
  </ul>

  <p>
    Note: this is <strong>not</strong> GDP level. It reflects <strong>relationship-aware GDP similarity</strong>
    after embedding.
  </p>





<section id="conclusion">
  <h2>Conclusion</h2>
  <p>


  Unified knowledge graphs are not about storing more data — they are about giving AI systems a shared language for reasoning across text, numbers, and relationships. By using a common relational objective to align heterogeneous modalities, we obtain a single embedding space that is reusable, interpretable, and AI-ready.
  </p><p>
  While demonstrated here with countries and borders, the same idea applies wherever entities are described by mixed signals and connected by meaningful relationships — from enterprises and products to infrastructure and biological systems. The result is not a single task-specific model, but a stable representation layer that simplifies downstream analytics, retrieval, and reasoning.
  </p><p>
  The approach assumes an informative relational backbone, and extending it to noisier or weaker graphs is an important next step. But even in its current form, unified knowledge graphs offer a practical path from messy data to coherent, machine-readable understanding.
  </p><p>
