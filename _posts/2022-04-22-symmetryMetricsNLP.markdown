---
layout:     post
title:      "Find Unexpected Word Pairs by Symmetry Metrics"
subtitle:   "How to use CNN deep learning symmetry metrics for NLP"
date:       2022-04-22 04:00:00
author:     "Melenar"
header-img: "img/pagePicNlp2a.jpg"
---
<p><h3>From Unsupervised CNN Deep Learning Classification to Vector Similarity Metrics</h3>

Pairwise vector similarity measures play significant roles in various data mining tasks. Finding matching pairs supports solving problems such classification, clustering or community detection and finding diversified pairs problems like outlier detection.
</p><p>
One of the problems related to word pair dissimilarities is called in psychology 'free associations'. This is psychoanalysis method that is being used to get into unconscious process. In this study we will show how to find unexpected free associations by symmetry metrics.  
</p><p>
We introduced a nontraditional vector similarity measure, symmetry metrics in
our previous post <i><a href="http://sparklingdataocean.com/2022/02/22/symmetryMetrics/">"Symmetry Metrics for High Dimensional Vector Similarity"</a></i>. These metrics are based on transforming pairwise vectors to GAF images and classifying images through CNN image classifcation. In this post we will demonstrate how to use symmetry metrics to find dissimilar words pairs.
</p><p>
<p><h3>Introduction</h3>
</p><p>

</p><p>

<p><h4>Free Associations</h4>

</p>
<p>

</p>
<p>
Free Associations is a psychoanalytic technique that was developed by Sigmund Freud and still used by some therapists today. Patients relate to whatever thoughts come to mind in order for the therapist to learn more about how the patient thinks and feels. As Freud described it: "The importance of free association is that the patients spoke for themselves, rather than repeating the ideas of the analyst; they work through their own material, rather than parroting another's suggestions"
</p><p>
In our posts to detect semantically similar or dissimilar word pairs we experimented with data about Psychoanalysis taken from Wikipedia and used different techniques that all start with the following steps:
</p><p>
<ul>
<li>Tokenize text file and removed stop words </li>
<li>Transform words to embedded vectors through models like Word2Vec, Glove, BERT or other. </li>
<li>Select pairs of words that stay next to each other in the document.</li>
</ul>

<p>
In our post <i><a href="http://sparklingdataocean.com/2017/12/24/word2vec2graphPsychoanalysis/">"Word2Vec2Graph - Psychoanalysis Topics"</a></i> we showed how to find free associations using Word2Vec2Graph techniques. For vector similarity measures we used cosine similarities. To create Word2Vec2Graph model we selected pairs of words located next to each other in the document and built a direct graph on word pairs with words as nodes, word pairs as edges and vector cosine similarities as edge weights. This method was publiched in 2021:  
<i><a href="https://aircconline.com/ijdkp/V11N4/11421ijdkp01.pdf">"SEMANTICS GRAPH MINING FOR TOPIC
DISCOVERY AND WORD ASSOCIATIONS"</a></i>

</p><p>
In another post -
<i><a href="http://sparklingdataocean.com/2019/06/01/word2vec2CNN/">"Free Associations"</a></i>  -
we demonstrated a different method - word pair similarity based on unsupervised Convolutional Neural Network image classification. We joined word vector pairs reversing right vectors, tranformed joint vectors to GAF images and classified them as 'similar' and 'different'.  
</p><p>
In this post we will show how to predict word similarity measures using a novel technique -  symmetry metrics.
</p>
<p><h4>Symmetry Metrics</h4>
Vector similarity measures on large amounts of high-dimensional data has become essential in solving many machine learning problems such as classification, clustering or information retrieval. Vector similarity measures are being used for solving problems such as classification or clustering that usually looking for pairs that are closed to each other.
<p></p>

In the previous post <i><a href="http://sparklingdataocean.com/2022/02/22/symmetryMetrics/">"Symmetry Metrics for High Dimensional Vector Similarity"</a></i>
we introduced symmetry metrics for high dimensional vector similarity. These metrics are based on unsupervised pairwise vector classification model -
<i><a href="http://sparklingdataocean.com/2021/08/01/unsupervisdCityTempCNN/">"Unsupervised Deep Learning for Climate Data Analysis"</a></i> - that is implemented through transforming joint vectors to GAF images and classifying images as symmetric or asymmetric. Symmetry metric is defined as a probability of GAF image ran through trained model and get to the ’same’, i.e. 'symmetric' class.
<p></p>



<a href="#">
    <img src="{{ site.baseurl }}/img/nldl_img6.jpg" alt="Post Sample Image" width="333" >
</a>
</p><p>
<p></p>

To distinguish between similar and dissimilar vector pairs this mode classifies data to 'same' and 'different' classes. Trained data for the 'same' class consists of self-reflected, mirror vectors and 'different' class of non-equal pairs. Visually mirror vectors are represented as symmetric images and 'different' pairwise vector as asymmetric images. Similarity metric is defined as a probability of pairwise vectors to get into the 'same' class.
<p></p>
</p><p>
In this post we will show how to apply trained unsupervised GAF image classification model to find vector similarities for entities taken from different domain. We will experiment with model trained on daily temperature time series data and apply it to word pair similarities.
</p><p>



</p><p>
<p><h3>Methods</h3>
<p><h4>Unsupervised Vector Classification Model</h4>
In one of our previous posts we introduced a
<i><a href="http://sparklingdataocean.com/2021/08/01/unsupervisdCityTempCNN/">novel unsupervised time series classification model</a></i>.
For this model we are embedding entities to vectors and combining entity pairs to pairwise vectors. Pairwise vectors are transformed to Gramian Angular Fields (GAF) images and GAF images are classified to symmetric or asymmetric classes using transfer learning CNN image classification.
We examined how this model works for entity pairs with two-way and one-way relationships and indicated that it is not reliable for classification of entity pairs with two-way relationships.
</p><p>
In this post we will use this method for one-way related pairs of words that are located next to each other in the document. We will generate pairwise word vectors for left and right words, transform joint vectors to GAF images and run these images through trained model to predict word similaritites through symmetry metrics.
<p>

<p><h4>Data Preparation</h4>
For data processing, model training and interpreting the results we will use the following steps:

</p>
<ul>
<li>Tokenize text and transform tokens to vectors</li>
<li>Get pairs of co-located words and create joint vectors</li>
<li>Transform joint vectors to GAF images</li>
<li>Get similarity metrics based on interpretation of trained CNN image classification model </li>
</ul>

Data preparation, training and interpretation techniques are described in details in our previous posts.




</p><p>
<p><h4>Transform Vectors to Images</h4>
</p><p>
As a method of vector to image translation we used Gramian Angular Field (GAF) - a polar coordinate transformation based techniques. We learned this technique in fast.ai
<i><a href="https://course.fast.ai"> 'Practical Deep Learning for Coders'</a></i>
class and fast.ai forum   
<i><a href="https://forums.fast.ai/t/time-series-sequential-data-study-group/29686">'Time series/ sequential data'</a></i> study group.
This method is well described by Ignacio Oguiza in Fast.ai forum
<i><a href="https://forums.fast.ai/t/share-your-work-here/27676/367"> 'Time series classification: General Transfer Learning with Convolutional Neural Networks'</a></i>.
</p><p>
To describe vector to GAF image translation Ignacio Oguiza referenced to paper <i><a href="https://aaai.org/ocs/index.php/WS/AAAIW15/paper/viewFile/10179/10251">Encoding Time Series as Images for Visual Inspection and Classification Using Tiled Convolutional Neural Networks</a></i>.



</p><p>
<p><h4>Training of Unsupervised Vector Classification Model</h4>
</p><p>

For model training we used fast.ai CNN transfer learning image classification. To deal with comparatively small set of training data, instead of training the model from scratch, we followed ResNet-50 transfer learning: loaded the results of model trained on images from the ImageNet database and fine tuned it with data of interest. Python code for transforming vectors to GAF images and fine tuning ResNet-50 is described in fast.ai forum.

</p><p>

<p><h4>Use Results of Trained Models</h4>
To calculate how similar are vectors to each other we will combine them as joint vectors and transform to GAF images. Then we will run GAF images through trained image classification model and use probabilities of getting to the ’same’ class as symmetry metrics.
To predict vector similarities based on the trained model, we will use fast.ai function 'learn.predict'.


<h3>Experiments</h3>
<p></p>

<p><h4>Transform Text Data to Words</h4>

As a data source we will use data about Psychoanalysis taken from Wikipedia. First, we will tokenize text data and exclude stop words:
</p>
{% highlight python %}
modelWV = api.load("glove-wiki-gigaword-100")
tokenizer = RegexpTokenizer(r'\w+')
tokenizer =RegexpTokenizer(r'[A-Za-z]+')
tokens = tokenizer.tokenize(file_content)
STOPWORDS = set(stopwords.words('english'))
dfStopWords=pd.DataFrame (STOPWORDS, columns = ['words'])
dfTokens = pd.DataFrame (tokens, columns = ['token'])
dfTokens['token'] = dfTokens['token'].str.lower()
dfTokens['len']=dfTokens['token'].str.len()
dfTokens=dfTokens[dfTokens['len']>3]
dfTokens=dfTokens.replace({'token': {'/': ' '}}, regex=True)
df_excluded = dfTokens[~dfTokens['token'].isin(dfStopWords['words'].values)].reset_index(drop=True)
tokenz = df_excluded.filter(['token'], axis=1)
{% endhighlight %}

<p>
<p><h4>Transform Words to Vectors</h4>
Next, we will transform words to vectors through Glove model:
</p>
{% highlight python %}
modelWV = api.load("glove-wiki-gigaword-100")
token=tokenz['token'].values.tolist()
listWords=[]
listVectors=[]
for word in token:
  try:
    listVectors.append(modelWV[word])
    listWords.append(word)
  except KeyError:
    x=0
dfWords=pd.DataFrame(listWords,columns=['word'])
dfVectors=pd.DataFrame(listVectors)
dfWordVec=pd.concat([dfWords,dfVectors],axis=1)
{% endhighlight %}

<p><h4>Create Pairwise Vectors</h4>
In our previous posts we described the process of data preparation for pairwise vector method, model training and interpretation techniques are described in details in another post of our technical blog. In this post we followed the steps:
<ul>
<li>Create Left vectors with column 'word1' and Right vectors with column 'word2'</li>
<li>Delete the first row from Right vectors and reverse them</li>
<li>Delete the last row from Left vectors</li>
<li>Concatenate Left and Right vectors</li>
<li>Concatenate word1 and word2 to column pair as 'word1'~'word2'</li>
<li>Drop duplicates</li>
<li>Split to metadata [word1,word2,pair] and numeric arrays</li>
</ul>

{% highlight python %}
dfWordVec1=dfWordVec.rename({'word':'word1'}, axis=1)
dfWordVec2=dfWordVec.rename({'word':'word2'}, axis=1)
leftVecWV=dfWordVec1.iloc[:-1,:]
leftVecWV.reset_index(inplace=True, drop=True)
rightVecWV=dfWordVec2[dfWordVec2.columns[::-1]].reset_index(drop=True)
rightVecWV = rightVecWV.iloc[1: , :].reset_index(drop=True)
pairVecWV0=pd.concat([leftVecWV, rightVecWV], axis=1)
pairVecWV0['pair'] = pairVecWV0['word1'].astype(str) + '~' + pairVecWV0['word2'].astype(str)
pairVecWV=pairVecWV0.drop_duplicates()
pairVecWV=pairVecWV.reset_index(drop=True)
pairMetadata= pairVecWV[['word1','word2','pair']]
pairVecWVvalues=pairVecWV
pairVecWVvalues.drop(['word1','word2','pair'], axis=1, inplace=True)
fXpairWV=pairVecWVvalues.fillna(0).values.astype(float)
{% endhighlight %}

<p><h4>Transform Vectors to Images</h4>
</p><p>
As a method of vector to image translation we used Gramian Angular Field (GAF) - a polar coordinate transformation based techniques. We learned this technique in fast.ai
<i><a href="https://course.fast.ai"> 'Practical Deep Learning for Coders'</a></i>
class and fast.ai forum   
<i><a href="https://forums.fast.ai/t/time-series-sequential-data-study-group/29686">'Time series/ sequential data'</a></i> study group.
</p><p>

<p></p>
{% highlight python %}
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
from pyts.image import GramianAngularField as GASF
image_size = 200
gasf = GASF(image_size)
fXpair_gasfWV = gasf.fit_transform(fXpairWV)
plt.plot(fXpairWV[2073])
plt.imshow(fXpair_gasfWV[2073], cmap='rainbow', origin='lower')
{% endhighlight %}
<p></p>
Here are examples that show that self-reflected vectors are represented as symmetric plots and GAF images and semantically different joint word vectors are represented as asymmetric plots and GAF images.
</p><p>
<a href="#">
    <img src="{{ site.baseurl }}/img/symNlp1a.jpg" alt="Post Sample Image" width="314" >
</a>
<p></p>
<p></p>


<p><h4>Train the Model</h4>
Time series classification model training was done on fast.ai transfer learning method. This model is described in detail in
<i><a href="http://sparklingdataocean.com/2021/08/01/unsupervisdCityTempCNN/">"Unsupervised Deep Learning for Climate Data Analysis"</a></i> post.

Model was trained on data about daily temperature for 1980 to 2020 years from 1000 most populous cities in the world.  The training model accuracy metric was about 96.5 percent.


<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/nldl_img3.jpg" alt="Post Sample Image" width="444" >
</a>
<p></p>




<p></p>

<p><h4>Symmetry Metrics based on Results of Trained Models</h4>

To calculate symmetry metrics we will create pairwise vectors by concatenating word vector pairs. Then we ran GAF images through the trained image classification model and used probabilities of getting to the ’same’ class as symmetry metrics.

To predict vector similarities based on the trained model, we used fast.ai function 'learn.predict':
<p></p>
{% highlight python %}
PATH_IMG='/content/drive/My Drive/city2/img4'
data = ImageDataBunch.from_folder(PATH_IMG,  train=".", valid_pct=0.23,size=200)
learn = learner.cnn_learner(data2, models.resnet34, metrics=error_rate)
learn.load('stage-1a')
{% endhighlight %}
<p></p>

<p></p>

<p><h4>Symmetry Metrics on Pairs of Words</h4>

We calculated symmetry metrics on pairs of co-located words or 2-grams from Psychoanalysis Wikipedia data. The distribution of statistics on word pair similarities shows that there are much less dissimilar word pairs that similar word pairs:

<p></p>

<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/symNlp2a.jpg" alt="Post Sample Image" width="222" >
</a>

<p></p>
Here are some word pair examples of dissimilar and similar neighbors for the word 'infant' - {infant - word} pairs


<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/symNlp2d.jpg" alt="Post Sample Image" width="333" >
</a>

<p></p>
Here are word pairs taken in opposite direction: {word - infant} pairs:

<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/symNlp2c.jpg" alt="Post Sample Image" width="333" >
</a>

<p></p>


<p><h3>Conclusion</h3>
In this post we demonstrated how to use symmetry metrics for predictions of pairwise word semantic similarities. We mapped word to vectors through Glove model, transformed joint word vectors to GAF images and classifyed images as symmetric or asymmetric.
<p></p>
Model that we used for symmetry metrics was trained on different data domain - daily temperature time series data.

We demonstrated that pairwise vectors model trained on symmetric and asymmetric GAF images trained on some data domain can be used for other data domains.


<p></p>

In the future we are planning to experiment with building direct graphs through symmetry metrics for graph mining and Graph Neural Network research.


<p></p>


<p></p>
<p></p>
