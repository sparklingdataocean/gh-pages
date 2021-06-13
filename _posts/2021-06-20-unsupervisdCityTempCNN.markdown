---
layout:     post
title:      "Unsupervised Deep Learning for Climate Data Analysis"
subtitle:   "How to find climate data anomalies via unsupervised image classifier methods"
date:       2021-06-20 12:00:00
author:     "Melenar"
header-img: "img/pagePic5.jpg"
---
<p><h3>Convolutional Neural Network</h3>


</p><p>

In the last few years deep learning demonstrated great success outperforming state-of-the-art machine learning techniques in various domains, in particularly, in Convolutional Neural Network image classification.
CNN image classification methods are getting high accuracies but being based on supervised machine learning, they require labeling of huge volumes of data.
</p><p>
Supervised learning really helps to produce a data output from the previous experience but it does not help to understand unknown data. Unsupervised machine learning helps to find unknown patterns in data and features which can be useful for categorization.
</p><p>
In this study we will introduce unsupervised machine learning model that categorizes pairs of entities to classes of similar pairs and different pairs. It transforms pairs of entities to vectors, vectors to images and classifies images though CNN deep learning.
</p><p>
<p><h4>Our Method: CNN Classification of Daily Temperature Time Series</h4>
</p><p>

<p>In this post we will use another deep learning technique and make Time Series classification via CNN Deep Learning. We learned this technique in fast.ai
<i><a href="https://course.fast.ai"> 'Practical Deep Learning for Coders, v3'</a></i>
class and fast.ai forum   
<i><a href="https://forums.fast.ai/t/time-series-sequential-data-study-group/29686">'Time series/ sequential data'</a></i> study group.</p>
<p>
We employed this technique for Natural Language Processing in our two previous posts - <i><a href="
http://sparklingdataocean.com/2019/06/01/word2vec2CNN/">"Free Associations -
Find Unexpected Word Pairs via Convolutional Neural Network"</a></i>  and
<i><a href="
http://sparklingdataocean.com/2019/03/16/word2vec2graph2CNN/">"Word2Vec2Graph to Images to Deep Learning."</a></i>
<p><h4>Classification Method</h4>
<p>We will convert time series of EEG channels to images using Gramian Angular Field (GASF) - a polar coordinate transformation. This method is well described by Ignacio Oguiza in Fast.ai forum
<i><a href="https://forums.fast.ai/t/share-your-work-here/27676/367"> 'Time series classification: General Transfer Learning with Convolutional Neural Networks'</a></i>. He referenced to paper <i><a href="https://aaai.org/ocs/index.php/WS/AAAIW15/paper/viewFile/10179/10251">Encoding Time Series as Images for Visual Inspection and Classification Using Tiled Convolutional Neural Networks</a></i>.

For data processing we will use ideas and code from Ignacio Oguiza code is in his GitHub notebook
<i><a href="https://gist.github.com/oguiza/c9c373aec07b96047d1ba484f23b7b47"> Time series - Olive oil country</a></i>.
<p></p>
</p><p>
</p>
<p><h3>CNN Classification</h3>

<p></p>
<ul>
<li>Preprocess raw data to time series</li>
<li>Convert time series to embedding vectors</li>
<li>Create pairs of mirror vectors for CNN classification model training</li>
<li>Transform vectors to images for CNN classification and visualization</li>
<li>Train CNN image classification model</li>
<li>Transform vectors mirror vector images for data analysis</li>
<li>Run mirror vector images through the CNN classification model and analyze the results</li>
</ul>
<p></p>


<p>
<h3>Data Processing</h3>
<p></p>

<p><h4>Raw Data to Embedded Vectors</h4>

As data Source we will use climate data from kaggle data sets:
<i><a href="
https://www.kaggle.com/hansukyang/temperature-history-of-1000-cities-1980-to-2020">"Temperature History of 1000 cities 1980 to 2020"</a></i> - daily temperature for 1980 to 2020 years from 1000 most populous cities in the world.

</p><p>

In our previous post <i><a href="
http://127.0.0.1:4000/2021/04/04/cityTempCNN/">"CNN Image Classification for Climate Data"</a></i> we described the process of raw data transformation to {city, year} time series:

<ul>
<li>Metadata columns: city, latitude, longitude, country, zone, year</li>
<li>365 columns with average daily temperatures</li>
</ul>
<p></p>
<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/scr1b.jpg" alt="Post Sample Image" width="800" >
</a>
<p></p>
<p><h4>Distances Between City Pairs</h4>

To find outliers by temperature vector classification we will look for different pairs of vectors from geographically nearby cities and similar pairs for geographically far away cities. First, we will get metadata into cityMetadata table:

<a href="#">
    <img src="{{ site.baseurl }}/img/scr3a.jpg" alt="Post Sample Image" width="322" >
</a>
<p></p>
We will use the function to calculate the distance by geographic coordinates:
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
{% endhighlight %}
<p></p>

To calculate distance Between two cities we will use city metadata and the distance fuction:
<p></p>
{% highlight python %}
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
Examples:
<p></p>
{% highlight python %}
cityDist('Tokyo','Japan','Mexico City','Mexico')
11301.1797
cityDist('Paris','France','London','United Kingdom')
340.7889
{% endhighlight %}
<p></p>
<p><h3>Prepare Training Data for Unsupervised Learning Classification</h3>
For training data we will create a set of self-joined vectors as 'same' class and set of different pair of vectors as 'different' class.
<ul>
<li>For the 'same' class we will combine vectors with their mirrors</li>
<li>for the 'different' class we will combine random pairs of vectors with temperatures of different years and different cities.</li>
<li>In each pair the second vector will be reversed.</li>
</ul>

<p><h4>'Same' Class: Coalesce Vectors with their Mirrors</h4>
To generate training data images for the 'same' class we will combine vecors with reversed themselves. For each vector we will create a label 'city~year'. For vector pairs we will combine these labels to 'city~year~city~year' labels and will use these labels as file names.

<p></p>
<p></p>
{% highlight python %}
dataSet1 = dataSet.reset_index(drop=True)
dataSet2= dataSet1[dataSet1.columns[::-1]].reset_index(drop=True)
dataSet1['fileName1'] = dataSet1['city_ascii'] + '~' + dataSet1['year'].astype(str)
dataSet1.drop(['city_ascii', 'lat', 'lng','country','zone','year'],
  axis=1, inplace=True)
dataSet2['fileName2'] = dataSet2['city_ascii'] + '~' + dataSet2['year'].astype(str)
dataSet2.drop(['city_ascii', 'lat', 'lng','country','zone','year'],
  axis=1, inplace=True)
{% endhighlight %}
<p></p>
<ul>
<li>Concatenate vectors with reversed themselves</li>
<li>Concatenate 'city~year' labels to 'city~year~city~year' labels</li>
<li>Mark image class type as 'same'.</li>
</ul>

<p></p>
<p></p>
{% highlight python %}
dataSetSameValues=pd.concat([dataSet1, dataSet2], axis=1)
dataSetSameValues['fileName'] =
  dataSetSameValues['fileName1'] + '~' + dataSetSameValues['fileName2'].astype(str)
dataSetSameValues['type'] ='same'
dataSetSameValues.drop(['fileName1','fileName2'], axis=1, inplace=True)
{% endhighlight %}
<p></p>


<p><h4>'Different' Class: Coalesce Vectors with Reversed Other Vectors</h4>
To generate training data images for the 'different' class we will do the following steps:



<ul>
<li>To randomly select different pairs of vectors (dataSet1) we will shuffle vectors and reversed vectors (dataSet2)</li>
<li>For first vectors we will create a label 'city1~year1' and for second (reversed) vectors - label 'city2~year2'</li>
<li>Concatenate vector pairs</li>
<li>Concatenate 'city1~year1' and 'city2~year2' labels to 'city1~year1~city2~year2' labels</li>
<li>Mark image class type as 'different'.</li>
</ul>

<p></p>
<p></p>
{% highlight python %}
from sklearn.utils import shuffle
dataSet11=shuffle(dataSet1).reset_index(drop=True)
dataSet12=shuffle(dataSet2).reset_index(drop=True)
dataSetDiffValues=pd.concat([dataSet11, dataSet12], axis=1).reset_index(drop=True)
dataSetDiffValues['fileName'] = dataSetDiffValues['fileName1'] + '~' +
  dataSetDiffValues['fileName2'].astype(str)
dataSetDiffValues['type'] ='diff'
dataSetDiffValues.drop(['fileName1','fileName2'], axis=1, inplace=True)
{% endhighlight %}
<p></p>

<p></p>

<p><h4>Vectors to Images</h4>
<p></p>
Tthen we will transform daily temperature arrays to pictures - see more code details in our previous post.
<p></p>
{% highlight python %}
from pyts.image import GramianAngularField as GASF
image_size = 200
gasf = GASF(image_size)
fXcity_gasf = gasf.fit_transform(fXcity)
dplt.figure(figsize=(12, 7))

{% endhighlight %}


<p></p>
'Same' class example: daily weather in Auckland, New Zealand in 1983.
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/scr3g.jpg" alt="Post Sample Image" width="678" >
</a>
<p></p>
<p></p>
'Different' class example: daily weather in Quang Ha, Vietnam in 2007 and Lima, Peru in 2013.
<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/scr3f.jpg" alt="Post Sample Image" width="678" >
</a>
<p></p>
<p></p>
Store GASF pictures to '/content/drive/My Drive/city2/img4' directory with 'same'/'different' class type subdirectories. Image file names will be defined as row labels 'city1~year1~city2~year2'. File names we will use later for data analysis based on the model results.
<p></p>
{% highlight python %}
imgPath='/content/drive/My Drive/city2'
import os
import warnings
warnings.filterwarnings("ignore")
IMG_PATH = imgPath +'/'+ "img4"
if not os.path.exists(IMG_PATH):
    os.makedirs(IMG_PATH)
numRows=metadataSet.shape[0]
for i in range(numRows):
    if not os.path.exists(IMG_PATH):
        os.makedirs(IMG_PATH)  
    idxPath = IMG_PATH +'/' + str(f['type'][i])
    if not os.path.exists(idxPath):
        os.makedirs(idxPath)
    imgId = (IMG_PATH +'/' +  str(f['type'][i])) +'/' + str(f['fileName'][i])
    plt.imshow(fXpair_gasf[i], cmap='rainbow', origin='lower')  
    plt.savefig(imgId, transparent=True)
    plt.close()
{% endhighlight %}
<p></p>
<p><h3>Image Classification Model</h3>
<p></p>
<p><h4>Prepare the Data</h4>
<p></p>
<p></p>
Prepare data for training and show some 'same' and 'different' class examples.
<p></p>
{% highlight python %}
PATH_IMG=IMG_PATH
tfms = get_transforms(do_flip=False,max_rotate=0.0)
np.random.seed(41)
data = ImageDataBunch.from_folder(PATH_IMG,  train=".", valid_pct=0.20,size=200)
data.show_batch(rows=2, figsize=(9,9))
{% endhighlight %}
<p></p>
<p></p>
<p></p>
Examples: 'same' class:
<a href="#">
    <img src="{{ site.baseurl }}/img/scr3e.jpg" alt="Post Sample Image" width="374" >
</a>
<p></p>
<p></p>
Examples: 'different' class:
<a href="#">
    <img src="{{ site.baseurl }}/img/scr3d.jpg" alt="Post Sample Image" width="374" >
</a>
<p></p>


<p></p>
<p><h4>Train the Model</h4>
<p></p>
Time series classification model training was done on fast.ai transfer learning approach:
<p></p>
{% highlight python %}
from fastai.text import *
from fastai.vision import learner
learn = learner.cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(2)
learn.save('stage-1a')
interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(10,10))
{% endhighlight %}
<p></p>
Accuracy is about 96.5%:

<a href="#">
    <img src="{{ site.baseurl }}/img/scr3j.jpg" alt="Post Sample Image" width="311" >
</a>
<p></p>
<p></p>
<p><h3>Finding Outliers through the Model</h3>
<p></p>

<p></p>

<ul>
<li>Selected 66 cities from contitental West Europe.</li>
<li>For each city we've got daily temperature data for years 1992 and 2016.</li>
<li>Combined vectors with other vectors (reversed).</li>
<li>Transformed vectors to images.</li>
<li>Through the model to classify images as similar or different.</li>
<li>Find similar pairs on long distance and different pairs on short distance.</li>
</ul>
<p></p>
<p></p>
<p><h4>Select Daily Temperature Data for Years 1992 and 2016 for West European Cities.</h4>
<p></p>

We selected 66 cities from contitental West Europe and daily temperatures for years 1992 and 2016.
<p></p>
<p></p>
{% highlight python %}
countryList=['Belgium', 'Germany', 'France','Austria','Netherlands','Switzerland',
              'Spain','Italy','Denmark','Finland','Greece','Italy','Monaco',
              'Netherlands','Norway','Portugal','Sweden','Switzerland']
{% endhighlight %}
<p></p>
Select data for years 1992 and 2016 for cities in the contry list:
<p></p>
{% highlight python %}
dataSetEurope = dataSet[(dataSet['country'].isin(countryList))]
dataSet2016 = dataSetEurope[(dataSetEurope['year'].isin(['2016']))]
dataSet1992 = dataSetEurope[(dataSetEurope['year'].isin(['1992']))]
{% endhighlight %}
<p></p>
<p></p>
<p><h4>Combine Pairs of Vectors (for year 1992)</h4>
<p></p>
Get subsets of vectors (dataSet21) and transform to to reversed vectors (dataSet22).
<p></p>
{% highlight python %}
dataSet21 = dataSet1992.reset_index(drop=True)
dataSet22 = dataSet11[dataSet21.columns[::-1]].reset_index(drop=True)
{% endhighlight %}
<p></p>
Attach metadata label as 'city'~'country'~'year'. To not use the same column names when merging vectors, change dataSet22 column names by adding 'b' prefix.
<p></p>
{% highlight python %}
dataSet21['fileName1'] = dataSet21['city_ascii'] + '~' + dataSet21['country']
  + '~' + dataSet21['year'].astype(str)
dataSet21.drop(['city_ascii', 'lat', 'lng','country','datetime' ,'capital','zone','year'],
  axis=1, inplace=True)
dataSet22['fileName2'] = dataSet22['city_ascii'] + '~' + dataSet22['country']
  + '~' + dataSet22['year'].astype(str)
dataSet22.drop(['city_ascii', 'lat', 'lng','country','datetime', 'capital','zone','year'],
  axis=1, inplace=True)
dataSet23=dataSet22.add_prefix('b')
{% endhighlight %}
<p></p>
Merge data frame by keys, exclude self-joined vectors and merge label columns to 'city1'~'country1'~'year1'~'city2'~'country2'~'year2':
<p></p>
{% highlight python %}
dataSet21['key']=0
dataSet23['key']=0
pairDataSet = dataSet21.merge(dataSet23,on='key', how='outer')
pairDataSet2 = pairDataSet[pairDataSet['fileName1']!=pairDataSet['bfileName2']]
pairDataSet2['fileName'] = pairDataSet2['fileName1'].astype(str)
  + '~' + pairDataSet2['bfileName2'].astype(str)
pairDataSet2.drop(['fileName1','bfileName2','key'], axis=1, inplace=True)
{% endhighlight %}
<p></p>
Split data set to vectors and labels:
<p></p>
{% highlight python %}
mirrorDiffColumns = pairDataSet2[['fileName']]
dataSetPairValues = pairDataSet2.drop(pairDataSet3.columns[[730]],axis=1)
{% endhighlight %}
<p></p>
<p></p>
<p></p>
<p><h4>Transform Vectors to Images</h4>
<p></p>
<p></p>
Transform vectors to images and save images using label (fileName) as image name:
<p></p>
{% highlight python %}
fXpair=dataSetPairValues.fillna(0).values.astype(float)
image_size = 200
gasf = GASF(image_size)
fXpair_gasf = gasf.fit_transform(fXpair)
IMG_PATH = '/content/drive/My Drive/city2/img9'
numRows = mirrorDiffColumns.shape[0]
for i in range(numRows):
   imgId = (IMG_PATH +'/' +  str(f['fileName'][i]))
   plt.imshow(fXpair_gasf[i], cmap='rainbow', origin='lower')  
   plt.savefig(imgId, transparent=True)
   plt.close()
{% endhighlight %}
<p></p>
<p></p>
<p><h4>Vector Classification Based on the Model</h4>
<p></p>
To predict vector similarities based on the trained model, first, we will load the trained model:
<p></p>
{% highlight python %}
PATH_IMG='/content/drive/My Drive/city2/img4'
data = ImageDataBunch.from_folder(PATH_IMG,  train=".", valid_pct=0.23,size=200)
learn = learner.cnn_learner(data2, models.resnet34, metrics=error_rate)
learn.load('stage-1a')
{% endhighlight %}
<p></p>
To predict probabilities of similarity for an image we can use fast.ai function learn.predict. Here are two examples of different pairs:
<p></p>
{% highlight python %}
pred_class,pred_idx,out=learn.predict(open_image(str(
  '/content/drive/MyDrive/city2/img9/Marseille~France~1992~Paris~France~1992.png')))
pred_class,pred_idx,out

(Category tensor(0), tensor(0), tensor([0.8589, 0.1411]))

pred_class,pred_idx,out=learn.predict(open_image(str(
  '/content/drive/MyDrive/city2/img8/Bonn~Germany~2016~Salerno~Italy~2016.png')))
pred_class,pred_idx,out

(Category tensor(0), tensor(0), tensor([0.9125, 0.0875]))

{% endhighlight %}
<p></p>

Run images through the model to claasify:

<p></p>
{% highlight python %}
pathPair='/content/drive/MyDrive/city2/img9/'
files = os.listdir(pathPair)
resEI1992=[]
for file in files:
  pred_class,pred_idx,out=learn.predict(open_image(pathPair +  '/' +file))
  resEI1992.append((file,out[0].tolist(),out[1].tolist()))
dfRes1992EIdata = DataFrame (resEI1992,columns=['cityPair','probDiff','probSame'])

{% endhighlight %}
<p></p>
<p></p>
Split labeled column to {city, country, year}:

<p></p>
{% highlight python %}

dfRes1992EIdata[['city1','country1','year','city2','country2','xx']] =
  dfRes1992EIdata.cityPair.apply(lambda x: pd.Series(str(x).split("~")))
{% endhighlight %}
<p></p>
<p></p>
<p></p>
<p></p>

<p></p>
Read city metadata and calculate distances between cities and save the results:
<p></p>
{% highlight python %}
cityMetadataPath='/content/drive/MyDrive/city1/img6/cityMetadata.csv'
cityMetadata=pd.read_csv(cityMetadataPath,sep=',',header=0).
  drop(['Unnamed: 0','datetime','zone'],axis=1)
dfRes1992EIdata['dist'] = dfRes1992EIdata.
  apply(lambda x: cityDist(x.city1,x.country1,x.city2,x.country2), axis=1)
dfRes1992EIdata=dfRes1992EIdata.drop(['Unnamed: 0','cityPair','xx'],axis=1)
dfRes1992EIdata.head()
{% endhighlight %}
<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/scr3l.jpg" alt="Post Sample Image" width="600" >
</a>
<p></p>
<p></p>
<p></p>
<p></p>
<p></p>
Stable years: 2016, 2002
Unstable years: 2008, 2006, 1992

from matplotlib import pyplot as plt
plt.figure(figsize=(8, 8))

from pyts.image import MarkovTransitionField as MTF
image_size = 200
mtf = MTF(image_size)
mtf
dfRes1992EIdist.csv

dfRes1992EIdata.to_csv('/content/drive/MyDrive/city2/img14/dfRes1992EI.csv')

dfRes1992EIPath='/content/drive/MyDrive/city2/img14/dfRes1992EI.csv'
dfRes1992EIdata=pd.read_csv(dfRes1992EIPath,sep=',',header=0)
dfRes1992EIdata.tail(3)

cityMetadata=cityMetadata.drop(['Unnamed: 0','datetime','zone'],axis=1)
cityMetadata.tail(3)

Hey guys,
On May 5 at 10:30 we will participate in the Knowledge Graph Conference: Anna Zhang and Dmytro Dolgopolov will present 'Entity Disambiguation with Knowledge Graph'

 https://www.knowledgegraph.tech/kgc2021/program/#legend

 Hope to see you.
-Elena

pathPair='/content/drive/MyDrive/city2/img17/'
files = os.listdir( pathPair )

len(files)


from sklearn.utils import shuffle
dataSetSameValues
dataSetDiffValues

dataSetSameValues.drop(['fileName1','fileName2'], axis=1, inplace=True)
pairDataSet3=pairDataSet2.iloc[2500:, :]
pairDataSet3=pairDataSet3.reset_index(drop=True)
mirrorDiffColumns=pairDataSet3[['fileName']]
dataSetPairValues1=pairDataSet3.drop(pairDataSet3.columns[[730]],axis=1)
<p></p>
