---
layout:     post
title:      "Unsupervised Deep Learning for Climate Data Analysis"
subtitle:   "How to find climate data anomalies via unsupervised image classifier methods"
date:       2021-08-01 12:00:00
author:     "Melenar"
header-img: "img/pagePic5.jpg"
---
<p><h3>Unsupervised CNN Deep Learning Classification</h3>


</p><p>
Outstanding success of Convolutional Neural Network image classification in the last few years influenced application of this technique to extensive variety of objects. CNN image classification methods are getting high accuracies but they are based on supervised machine learning, require labeling of input data and do not help to understand unknown data.
</p><p>
In this post we apply unsupervised machine learning model that categorizes entity pairs to similar pairs and different pairs classes. This is done by transforming pairs of entities to vectors, then vectors to images and classifying images using CNN transfer learning classification.
</p><p>

</p><p>


</p><p>
Previously we used this technique for Natural Language Processing in our post - <i><a href="
http://sparklingdataocean.com/2019/06/01/word2vec2CNN/">"Free Associations -
Find Unexpected Word Pairs via Convolutional Neural Network"</a></i> to classify word pairs by word similarities. Based on this model we identified word pairs that are unexpected to be next to each other in documents or data streams.

Applying this model to NLP we demonstrated that it works well for classification of entity pairs with one-way relationships. In this post we will show that this model in not reliable to entity pairs with two-way relationships.

To demonstrate this we will use city temperature history and will compare temperatures between city pairs.

</p><p>
<p><h3>Methods</h3>

<p>
Steps that we used for data processing, model training and interpreting the results:
</p>
<ul>
<li>Transformed raw data to embedding vectors</li>
<li>Created pairs of mirror vectors</li>
<li>Transformed mirror vectors to images for CNN image classification</li>
<li>Trained CNN image classification model</li>
<li>Ran mirror vector images through the CNN classification model and analyzed the results</li>
</ul>
<p></p>

</p><p>
<p><h4>Transform Time Series Data to Mirror Vectors</h4>
</p><p>
As the first step of time series data processing we transformed data to embedded vectors that was used as basis for image classification.
</p><p>
Then we generated training data on vector pairs. Second vector in each pair was reversed and vecors were concatenated. We called these concatenated vectors 'mirror vecors'.


For ‘Different’ class training data we combined pairs of vectors of different entities and for the 'Same' class training data we combined pairs of vectors of the same entity.




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
We employed this technique for Natural Language Processing in our two previous posts - <i><a href="
http://sparklingdataocean.com/2019/06/01/word2vec2CNN/">"Free Associations -
Find Unexpected Word Pairs via Convolutional Neural Network"</a></i>  and
<i><a href="
http://sparklingdataocean.com/2019/03/16/word2vec2graph2CNN/">"Word2Vec2Graph to Images to Deep Learning."</a></i>

</p><p>
<p><h4>Training of Unsupervised Vector Classification Model</h4>
</p><p>

For this study we used fast.ai CNN transfer learning image classification. To deal with comparatively small set of training data, instead of training the model from scratch, we followed ResNet-50 transfer learning: loaded the results of model trained on images from the ImageNet database and fine tuned it with data of interest. Python code for transforming vectors to GAF images and fine tuning ResNet-50 is described in fast.ai forum.

</p><p>
Python code for transforming vectors to GAF images and fine tuning ResNet-50 is described in Ignacio Oguiza code in his GitHub notebook
<i><a href="https://gist.github.com/oguiza/c9c373aec07b96047d1ba484f23b7b47"> Time series - Olive oil country</a></i>.
<p></p>
</p><p>
</p>


<p>
<h3>Data Preparation for Classification</h3>
<p></p>

<p><h4>Raw Data</h4>

As data Source we will use climate data from Kaggle data sets:
<i><a href="
https://www.kaggle.com/hansukyang/temperature-history-of-1000-cities-1980-to-2020">"Temperature History of 1000 cities 1980 to 2020"</a></i> - daily temperature for 1980 to 2020 years from 1000 most populous cities in the world.


In our previous post <i><a href="
http://127.0.0.1:4000/2021/04/04/cityTempCNN/">"CNN Image Classification for Climate Data"</a></i> we described the process of raw data transformation to {city, year} time series:
<p></p>
<ul>
<li>Metadata columns: city, latitude, longitude, country, zone, year</li>
<li>365 columns with average daily temperatures</li>
</ul>
<p></p>

<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/scr1b.jpg" alt="Post Sample Image" width="800" >
</a>
<p></p>
<p><h4>Distances Between City Pairs</h4>

To analyze results of vector classification we will look for different pairs of vectors from geographically nearby cities and similar pairs for geographically far away cities. First, we will get metadata into cityMetadata table:
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/scr3a.jpg" alt="Post Sample Image" width="322" >
</a>
<p></p>
Function to calculate the distance in kilometers by geographic coordinates:
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

Distance between two cities by city metadata and distance fuction:
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
Examples of city distances in km:
<p></p>
{% highlight python %}
cityDist('Tokyo','Japan','Mexico City','Mexico')
11301.1797

cityDist('Paris','France','London','United Kingdom')
340.7889
{% endhighlight %}
<p></p>

<p><h4>Prepare Vector Pairs</h4>
For training data we will create a set 'same' class mirror vecors and 'different' class mirror vectors:
<ul>
<li>For the 'same' class we will combine vectors with their mirrors</li>
<li>For the 'different' class we will combine random pairs of vectors with temperatures of different years and different cities.</li>
<li>In each pair the second vector will be reversed.</li>
</ul>

<p><h5>'Same' Class: Coalesce Vectors with their Mirrors</h5>
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

<p><h5>'Different' Class: Coalesce Vectors with Reversed Other Vectors</h5>
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

<p><h4>Transform Vectors to Images</h4>
<p></p>
Then we will transform mirror vectors to pictures - see more code details in our previous post.
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

For CNN training we will store GAF pictures to 'same' and 'different' subdirectories. Image file names will be defined as labels 'city1~year1~city2~year2'. These file names we will use later for data analysis based on the model results.

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

<p><h4>Training Data</h4>
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
Accuracy was about 96.5%:

<a href="#">
    <img src="{{ site.baseurl }}/img/scr3j.jpg" alt="Post Sample Image" width="311" >
</a>
<p></p>
<p></p>

<p><h3>Pairs with Two-way Relationships</h3>
<p></p>

Transforming mirror vectors to GAF images technique is based on polar coordinates and therefore it might generate not similar results for turn around pairs. We will test this hypothesis based on the following data:

<ul>
<li>We will select 66 cities from contitental West Europe and for each city we will take daily temperature data for the year 1992.</li>
<li>For all city pairs we will create mirror vectors in both directions, i.e. Paris - Berlin and Berlin - Paris.</li>
<li>We will transform mirror vectors to GAF images.</li>
<li>Then we will run these images through the model and classify them as 'Similar' or 'Different'.</li>
<li>Finally, we will compare model results for reversed city pairs.</li>
</ul>
<p></p>

<p></p>
<p><h4>Select Daily Temperature Data for Year 1992 for West European Cities.</h4>
<p></p>

Get 1992 year daily temperature data for cities from the following country list:
<p></p>

<p></p>
{% highlight python %}
countryList=['Belgium', 'Germany', 'France','Austria','Netherlands','Switzerland',
              'Spain','Italy','Denmark','Finland','Greece','Italy','Monaco',
              'Netherlands','Norway','Portugal','Sweden','Switzerland']
{% endhighlight %}


<p></p>
{% highlight python %}
dataSetEurope = dataSet[(dataSet['country'].isin(countryList))]
dataSet1992 = dataSetEurope[(dataSetEurope['year'].isin(['1992']))]
{% endhighlight %}
<p></p>
<p></p>
<p><h4>Combine Pairs of Vectors</h4>
<p></p>
Get set of vectors (dataSet21) and transform to reversed vectors (dataSet22).
<p></p>
{% highlight python %}
dataSet21 = dataSet1992.reset_index(drop=True)
dataSet22 = dataSet11[dataSet21.columns[::-1]].reset_index(drop=True)
{% endhighlight %}
<p></p>
Attach metadata labels as 'city'~'country'~'year'. To not use the same column names when merging vectors, change dataSet22 column names by adding 'b' prefix.
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
Merge data frames by keys, exclude self-joined vectors and merge label columns to 'city1'~'country1'~'year1'~'city2'~'country2'~'year2':
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

<p></p>
<p><h4>Transform Vectors to GAF Images</h4>
<p></p>
Split data set to vectors and labels:
<p></p>
{% highlight python %}
mirrorDiffColumns = pairDataSet2[['fileName']]
dataSetPairValues = pairDataSet2.drop(pairDataSet3.columns[[730]],axis=1)
{% endhighlight %}
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
To predict vector similarities based on the trained model, we will use fast.ai function 'learn.predict':
<p></p>
{% highlight python %}
PATH_IMG='/content/drive/My Drive/city2/img4'
data = ImageDataBunch.from_folder(PATH_IMG,  train=".", valid_pct=0.23,size=200)
learn = learner.cnn_learner(data2, models.resnet34, metrics=error_rate)
learn.load('stage-1a')
{% endhighlight %}
<p></p>
Here are two examples of city pairs. The first number shows 'different' probability and second number - 'same' probability:
<p></p>
{% highlight python %}
pred_class,pred_idx,out=learn.predict(open_image(str(
  '/content/drive/MyDrive/city2/img9/Marseille~France~1992~Paris~France~1992.png')))
pred_class,pred_idx,out
(Category tensor(0), tensor(0), tensor([0.8589, 0.1411]))

pred_class,pred_idx,out=learn.predict(open_image(str(
  '/content/drive/MyDrive/city2/img9/Turin~Italy~1992~Monaco~Monaco~1992.png')))
pred_class,pred_idx,out
(Category tensor(0), tensor(0), tensor([0.5276, 0.4724]))
{% endhighlight %}
<p></p>

<p></p>
The following example shows that temperature similarities depend on city order. In the first result vectors in Naples and Porto are 'same' but vectors between Porto and Naples are 'different':
<p></p>
{% highlight python %}
pred_class,pred_idx,out=learn.predict(open_image(str(
  '/content/drive/MyDrive/city2/img9/Naples~Italy~1992~Porto~Portugal~1992.png')))
pred_class,pred_idx,out
(Category tensor(1), tensor(1), tensor([0.3950, 0.6050]))

pred_class,pred_idx,out=learn.predict(open_image(str(
  '/content/drive/MyDrive/city2/img9/Porto~Portugal~1992~Naples~Italy~1992.png')))
pred_class,pred_idx,out
(Category tensor(0), tensor(0), tensor([0.6303, 0.3697]))

{% endhighlight %}
<p></p>

To calculate how many inconsistent pairs we have, first, we will run all images through the model:

<p></p>
{% highlight python %}
pathPair='/content/drive/MyDrive/city2/img9/'
files = os.listdir(pathPair)
resEI1992=[]
for file in files:
  pred_class,pred_idx,out=learn.predict(open_image(pathPair +  '/' +file))
  resEI1992.append((file,out[0].tolist(),out[1].tolist()))
dfRes1992EIdata = DataFrame (resEI1992,columns=['cityPair','probDiff','probSame'])

dfRes1992EIdata.shape
(4290, 4)
{% endhighlight %}
<p></p>
<p></p>
Next, we will split 'cityPair' column to {city, country, year} pairs:

<p></p>
{% highlight python %}

dfRes1992EIdata[['city1','country1','year','city2','country2','xx']] =
  dfRes1992EIdata.cityPair.apply(lambda x: pd.Series(str(x).split("~")))
{% endhighlight %}
<p></p>
<p></p>

Calculate distances (in kilometers) between cities:
<p></p>
{% highlight python %}
cityMetadataPath='/content/drive/MyDrive/city1/img6/cityMetadata.csv'
cityMetadata=pd.read_csv(cityMetadataPath,sep=',',header=0).
  drop(['Unnamed: 0','datetime','zone'],axis=1)
dfRes1992EIdata['dist'] = dfRes1992EIdata.
  apply(lambda x: cityDist(x.city1,x.country1,x.city2,x.country2), axis=1)
dfRes1992EIdata=dfRes1992EIdata.drop(['Unnamed: 0','cityPair','xx'],axis=1)
{% endhighlight %}

<p></p>
Finally, the total number of city pairs we have is 4290 and there are 630 inconsisted city pairs, therefore about 14.7% of city pairs get inconsisted model results.
<p></p>
{% highlight python %}
dfRes1992EIdata['same']=(0.5-dfRes1992EIdata.probSame)*(0.5-dfRes1992EIdata.probSame2)

len(dfRes1992EIdata.loc[dfRes1992EIdata['same'] < 0])
630
{% endhighlight %}
<p></p>

<p></p>
Inconsistency of city pairs similarity predictions are not dependent of distances between the cities: this problem occurs for city pairs that are located nearby and faraway.
For example, here are city pairs with inconsisted similarity predictions and longest distances:
<p></p>
{% highlight python %}
dfRes1992EIdata.loc[dfRes1992EIdata['same'] < 0].sort_values('dist').tail(18)
{% endhighlight %}
<p></p>
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/scr3m.jpg" alt="Post Sample Image" width="700" >
</a>
<p></p>
Examples of city pairs with inconsisted similarity predictions and shortest distances:
<p></p>
<p></p>
{% highlight python %}
dfRes1992EIdata.loc[dfRes1992EIdata['same'] < 0].sort_values('dist').head(18)
{% endhighlight %}
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/scr3n.jpg" alt="Post Sample Image" width="700" >
</a>
<p></p>

<p></p>
<p><h3>Find Outliers</h3>
<p></p>

</p><p>
We proved the hypothesis that mirror vectors classification model is not reliable for similarity prediction of pairs with two-way relationships and therefore this model should be used only for classifcation of entity pairs with one-way relationships.
</p><p>
Here we will show scenarios of using this model to compare daily temperatures of pairs with one-way relationships. First, we will calculate average vector of all yearly temperatures vectors for cities in Western Europe and compare it with yearly temperature vectors for all cities. Second, we will find a city located in the center of Western Europe and compare this city temperatures for years 2008 and 2016 with other cities.
<p></p>

<p></p>
<p><h4>Compare {City, Year} Temperature Vectors with Average of All Yearly Temperatures</h4>
<p></p>
To find cities with yearly temperatures similar to average temperatures for years from 1980 to 2019 in Western Europe we calculated average time series for 2640 daily temperature time series (40 years and 66 cities).
<p></p>
As average of average temperature vectors provides a very smooth line, we don't expect that many city-year temperature vectors will be similar to it.
In fact only 33 of city-year temperature time series (1.25%) had higher than 0.5 'same' probability. Here are top 22 pairs with 'same' probability higher than 0.65:   
</p><p>
<a href="#">
    <img src="{{ site.baseurl }}/img/scr3o.jpg" alt="Post Sample Image" width="333" >
</a>
<p></p>
It's interesting that most of cities with high probabilities to be similar to average temperature are located on Mediterranean Sea not far from each other. Here is a clockwise city list: Marseille (France), Nice (France), Monako (Monako), Genoa (Italy), Rome (Italy), Naples (Italy), and Salerno (Italy).

<p></p>
<p></p>
<p><h4>Compare {City, Year} Temperature Vectors with Central City</h4>
<p></p>
<p>
Here is another scenario of using mirror vector model to compare daily temperatures of pairs with one-way relationships. From Western Europe city list the most cetrally located city is Stuttgart, Germany.
<p></p>

<p></p>
<ul>
<li>Concatenated temperature vector pairs {city, Stuttgart} for years 2008 and 2016</li>
<li>Transformed mirror vectors to GAF images</li>
<li>Analyzed 'same' or 'different' probabilities by running images through trained model.</li>
</ul>

Based on the model, for both years, cities located close to Stuttgart had high probability of similar temperature vectors and cities located far from Stuttgart had high probabilities of different temperature vectors. Here we show two cities that had very different temperatures in both 2008 and 2016 years:
</p><p>
</p><p>
<a href="#">
    <img src="{{ site.baseurl }}/img/scr3s.jpg" alt="Post Sample Image" width="628" >
</a>
<p></p>
Less easy was to predict which cities had temperature vector similarities on the border between 'different' and 'same'. Here are cities for the year 2008:
<p></p>
<a href="#">
    <img src="{{ site.baseurl }}/img/scr3q.jpg" alt="Post Sample Image" width="371" >
</a>


<p></p>
And here are cities for the year 2016:
</p><p>
<a href="#">
    <img src="{{ site.baseurl }}/img/scr3p.jpg" alt="Post Sample Image" width="371" >
</a>
<p></p>

Here are two cities that were 'on the border' in years 2008 and 2016. In both years Stockholm, Sweden was on the 'same' side and Nice, France was on 'different' side.


</p><p>
<a href="#">
    <img src="{{ site.baseurl }}/img/scr3t.jpg" alt="Post Sample Image" width="628" >
</a>
<p></p>


<p></p>
<p></p>
<p><h4>Stop here</h4>
<p></p>


<p></p>
