---
layout:     post
title:      "Deep Learning Language Model"
subtitle:   "Playing with fast.ai Language Model"
date:       2018-11-09 12:00:00
author:     "Melenar"
header-img: "img/deepLearning2.jpg"
---


<p>This post is our first step in finding connections between Word2Vec2Graph model and Deep Learning NLP. </p>
<p><h3>About fast.ai Language Model</h3>
<p>In this post we looked at Language Model that predicts the next word given its previous word. This post's Language Model was built based on Jerome Howard's Deep Leaning fast.ai library. We used Python code almost the same as the code described in Jeremy's lesson <i><a href="https://course.fast.ai/lessons/lesson10.html"> 10—NLP CLASSIFICATION AND TRANSLATION</a></i></p>

<p>Fantastic thing fast.ai Language Model approach provides is Transfer Learning for NLP models. Fast.ai transfer learning is well described in NLP fast.ai post <i><a href="http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html"> Introducing state of the art text classification with universal language models</a></i></p>

<p><h3>Slightly Modified fast.ai Language Model</h3>
<p>In Lesson 10 Jeremy is using Language Model pre-trained on Wikipedia data and then he fine-tunes it by IMDb movie reviews data. For our model instead of IMDb movie reviews we used a combination of text files that we used in our previous posts: Stress text file, Psychoanalysis text file, and Creativity and Aha Moments text file.</p>

<p>To randomly generate word sequences we used a function from fast.ai Lesson 4: </p>
{% highlight python %}
def sample_model(m, s, l=50):
    s_toks = Tokenizer().proc_text(s)
    s_nums = [stoi[i] for i in s_toks]
    s_var = V(np.array(s_nums))[None]
    m[0].bs=1
    m.eval()
    m.reset()
    res,  *_ = m(s_var)
    print('...', end='')
    for i in range(l):
        r = torch.multinomial(res[-1].exp(), 2)
        #r = torch.topk(res[-1].exp(), 2)[1]
        if r.data[0] == 0:
            r = r[1]
        else:
            r = r[0]
        word = itos[to_np(r)[0]]
        res, *_ = m(r[0].unsqueeze(0))
        print(word, end=' ')
    m[0].bs=bs
{% endhighlight %}

<p><p><h3>Language Model Examples</h3></p>

{% highlight scala %}
sample=sample_model(m, "freud", l=50)
{% endhighlight %}
<p><small><mark>freud</mark>
...had given a opinion that that society could support children rather than even women without further research . in many cases transcranial was among the psychoanalytic groups that sought to develop new and
</small></p>

{% highlight scala %}
sample=sample_model(m, "freud", l=33)
{% endhighlight %}
<p><small><mark>freud</mark>...'s thoughts on religious beliefs and the arguments his version of what he describes as " the intrinsically destructive drive " . tom martin explained how the " practicing of the mind "
</small></p>


{% highlight scala %}
sample=sample_model(m, "freud was", l=50)
{% endhighlight %}
<p><small><mark>freud was</mark>...recognized as the center of the research . analytically had entirely improves upon an everyday life with a formal idea by the north american affects of the previous year ; and the non - psychological perspective that it obtained by listening to the eye . the board also asked brentano
</small></p>

{% highlight scala %}
sample=sample_model(m, "stress", l=50)
{% endhighlight %}
<p><small><mark>stress</mark>
...disorder causing ques- eld diffuse patients to become more “ active ” informa- dishonesty or par- martindale time . s. frosch said : “ we can spend a lot of time looking at people 's memory . dr assets are about other things that are a very important side of
</small></p>

{% highlight scala %}
sample=sample_model(m, "stress", l=50)
{% endhighlight %}
<p><small><mark>stress</mark>
...and stress disorder and psychopathology . painful things might be evident when they are under a cold weather after being turned about by old gist jakob ◆ thinker ( nt ) whom he had tests in an reflective lab . an old object that results in a blocking effect is
</small></p>

{% highlight scala %}
sample=sample_model(m, "stress disorder", l=60)
{% endhighlight %}
<p><small><mark>stress disorder</mark>
...previously rejected " escape " program calls . it resulted in the appearance of a " trajectories " . the new magazine called this book " the universe of future men essays " an " air of openness " for beauty and creativity . david ptsd is some of whom had previously been considered a theorists . in 1999 he</small></p>


{% highlight scala %}
sample=sample_model(m, "stress disorder", l=50)
{% endhighlight %}
<p><small><mark>stress disorder</mark>
...in the to - bulb field ( including the second burst ) in a working group . the best carbon - based work or writing of this period is in the largest prospective sixteen members of european society . for most of the work how for the united states population ] the international business times described and criticized the language
</small></p>


{% highlight scala %}
sample=sample_model(m, "creative", l=50)
{% endhighlight %}
<p><small><mark>creative</mark>
...power . the presentation was notable for its use of decorative terms " eeg " to describe what are called " other understanding characteristics " . academy of humanities production 's code of conduct results in being written as follows : whether the work from the prior reading is an
</small></p>

{% highlight scala %}
sample=sample_model(m, "creativity", l=50)
{% endhighlight %}
<p><small><mark>creativity</mark>
...is from the three learn how to become an american . we know apparently the best way to develop this method of learning is through really specific courses and studies . the ability to learn provides basic skills . the creative framework requires creative information and intelligence to relax .
</small></p>


{% highlight scala %}
sample=sample_model(m, "insight", l=40)
{% endhighlight %}
<p><small><mark>insight</mark>
.... anxiety can be a crucial part of mental structure . science research group first 5 november 2012 concluded that the lovell experiment involved " remarkable performance " and noted that despite being highly genius worked out despite having a
</small></p>


{% highlight scala %}
sample=sample_model(m, "neural", l=50)
{% endhighlight %}
<p><small><mark>neural</mark>
...content needs to be biases and will be able to arise in the now episodic ways . “ the lessons are mundane and cause you to view the expectations . ” this is a goal used to explore mental abilities located on the oceans . in particular these systems interconnected
</small></p>

{% highlight scala %}
sample=sample_model(m, "neural network", l=50)
{% endhighlight %}
<p><small><mark>neural network</mark>
...of a new controlling interest next to the 2012 gold age . information obtained by scientists from various online strategic services has been given to aca- self- ... the most ’’ of these programs .
 xbos xfld 1 the war is not still widely known since the beginning of
</small></p>



{% highlight scala %}
sample=sample_model(m, "artificial", l=50)
{% endhighlight %}
<p><small><mark>artificial</mark>
...the transference knowledge which has been implemented as an illusion of the future . the assumption that ® wishes people are valuable is not originally written by hans - ernst wilhelm von perls but by von der psychologists who started using classical view as an explanation for how to solve
</small></p>



{% highlight scala %}
sample=sample_model(m, "intelligence", l=50)
{% endhighlight %}
<p><small><mark>intelligence</mark>
...experiments performed by the spontaneously detected latency stress disorder involving inserm at the control room . the patient had been given an extensive account of the surgery with the domains . once the surgery he had had his behavioral network developed to the point where it was able to travel
</small></p>

{% highlight scala %}
sample=sample_model(m, "artificial intelligence", l=50)
{% endhighlight %}
<p><small><mark>artificial intelligence</mark>
...never came into existence . this from history was only a matter of reason . despite more significant arlow — particularly to learn greater details — that key children would lose each year they needed more opportunities for a job . explained by the fact that a 2003 museum of
</small></p>

{% highlight scala %}
sample=sample_model(m, "big data", l=40)
{% endhighlight %}
<p><small><mark>big data</mark>
...from the beginning to end with a strongly natural message from them . a new list of intellectual hints such as those that did not necessarily name this system — one by david ernest duncan and another by daniel a.
</small></p>

{% highlight scala %}
sample=sample_model(m, "big data", l=40)
{% endhighlight %}
<p><small><mark>big data</mark>
...they were not able to speed up attempts to win phrase for a single objective . overall the writers sought to solve this problem by offering the task of creating additional reflective material for details . today these stuck largely
</small></p>

<h3>Compare Language Model with Word2Vec Model </h3>
<p>On his class Jeremy recommends his students to think about Language Model vs. Word2Vec model. He explains that Word2vec is a single embedding matrix that maps words to vectors and Language Model really understands English. We are absolutely agree with it. </p>
<p>
What is also fantastic in fast.ai library is an ability to use pre-trained Language Model and fine-tune it for specific data domains. In our model of this post we used a very small data file (about 900 KB). Using just this data to understand English would be impossible. However, building it on-top of pre-trained model allowed it to 'speak English' in psychology language. </p>  

<p>What are other differences between Language Model comparing to Word2Vec model? </p>
<ul>
<li>Geometrically these models are very different. Language Model creates a one dimensional line from a word - sequence of words, word2vec model creates multi-dimensional vectors.
</li>
<li>Vectors can be compared via cosine similarity to find similar or not similar words.</li>
<li>Cosine similarities within pairs of words can translate pairs of words to graph using words as nodes, word pairs as edges and cosine similarities as edge weights.</li>
<li>Word2Vec graphs can combine Word2Vec and graph functionalities - in our previous posts we introduced Word2Vec2Graph model.</li>
<li>Word2Vec2Graph gives us new insights about text: top words in text file (pageRank), word topics (communities), word neighbors ('find' function).
</li>
</ul>

<h3>More Questions</h3>
<ul>
<li>Can Language Model be built based embedding words to vectors? This would allow us to combine it with graph functionality.   
</li>

<li>Can we translate Language Model sequences to numbers (indexes) and combine the numbers to vectors? </li>
<li>Let's assume we use single words. For each word we can calculate a sequence of size X, map it to X numbers and combine with the initial word number. These sequences of (X+1) numbers can be transformed to vectors. How cosine similarities would work for finding similar words? </li>
<li>Can we get NLP insights from graph functionality built on top of Language   Model?
</li>
</ul>

<p><h3>Next Post - Language Model</h3>
In the next several posts we will continue analyzing connections between Word2Vec and Language Models.</p>
