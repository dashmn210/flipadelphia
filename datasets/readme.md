# In progress Datasets 

## **COURSE DESCRIPTIONS**
* **idea**: take course description, predict enrollement, control for professor, subject/dept, (school, reqs met)
* **progress**:
  * raw XML files stored at [https://drive.google.com/drive/folders/0B5DfdyRTPhIudjk5VXRMdDk3aEU](https://drive.google.com/drive/folders/0B5DfdyRTPhIudjk5VXRMdDk3aEU)
  * seems like crawling is our most likely bet. 
    * [example crawling script](https://github.com/rpryzant/flipadelphia/blob/master/datasets/course_catalog/Abhijeets_crawling_script.py)
    * [example crawler 2](https://github.com/rpryzant/SubCrawl/blob/master/code_release/corpus_generation/subscene_crawler.py)
    * [examples crawler 3](https://github.com/rpryzant/japanese_corpus/blob/master/crawlers/daddicts/d_addicts_crawler.ipynb)
  * but there's also an official api put out by explorecourses (it might be crappy though?)
    * [example usage](http://git.javadeploy.net/jimsproch/explorecourses-api-example/tree/master)
    * [docs](https://github.com/rpryzant/flipadelphia/tree/master/datasets/course_catalog/explorecourses_api)
  * there's also an XML api
    * To get all current active classes, you can simply call http://explorecourses.stanford.edu/search?view=xml-20140630&filter-coursestatus-Active=on&q=%25
    * Data from past years can be obtained by passing an additional flag such as "&academicYear=20162017"
    * This script converts it into JSON: https://gist.github.com/tummykung/cfa37c3be373a3da073ffb9b2de03d7e
    * reid meeting notes
      * throw away attributes
      * days => categorical
      * keep nulls
      * rm termid (dup of term)
      * course => course_number
      * make a new course_level feature (100, 200 etc)
      * make sure there aren't any within-example newlines
      * replace reqs with number of things the course satisfies
      * rename id ==> courseid, classid ==> sectionid
      * rm enrollStatus
      * change startTime/endTIme to some kind of int
      * look into mapping instructure to some kind of importance
* **papers**: 

## **FINANCE**
* **idea**: look at financial docs (10K, 8Q), predict stock price. controling for industry
* **progress**:
  * https://nlp.stanford.edu/pubs/stock-event.html
* **papers**: 
  * [https://web.stanford.edu/~jurafsky/pubs/lrec2014_stocks.pdf](https://web.stanford.edu/~jurafsky/pubs/lrec2014_stocks.pdf)


## **FOOD/RESTURAUNTS**
* **idea**: predict hygene independent of cuisine
* **progress**:
  * http://www3.cs.stonybrook.edu/~junkang/hygiene/ woot!!!!
  * 1) dataset, which we used to run classification including inspection information and review contents. Please note that this dataset is per inspection period as we described in the paper. 
  * 2) restaurant meta data, which lists the restaurants and their meta information.
  * 3) reviews, which lists all reviews with a link to each restaurant one was written for. 
* **papers**: 
  * [https://aclweb.org/anthology/D/D13/D13-1150.pdf](https://aclweb.org/anthology/D/D13/D13-1150.pdf)
  
## **MORE FOOD**
* **idea** predict rating from review, control by resturaunt type, price, etc
* **progress**
  * data is at https://nlp.stanford.edu/robvoigt/nis/, but unopenable
  * emailed rob about this



## **INTERNET COMMENTS**
* **idea**: feed in tweets?, predict sentiment, control for topic,party,region. 
* **progress**:
  * rob says the data is on the nlp cluster, in `/scr/nlp3/facebook`
  * verified: `$ ssh rpryzant@jacob.stanford.edu; cd /scr/nlp3/facebook`
* **papers**: 
  * robs lrec submission: [https://cs.stanford.edu/~rpryzant/msc/rtgender.pdf](https://cs.stanford.edu/~rpryzant/msc/rtgender.pdf)


## **REDDIT**
* **idea**: post title, predict upvotes while controling for subreddit, (time of day, author?)
* **progress**: 
  * spoke to will. the data is on the infoLAB machines, he will work to get us access (10/10)
  * verified data is in `madmax3:/dfs/dataset/infolab`
* **papers**: 





# In the pipeline...


## **BOOKS**
* **idea**: feed in book (summary, editorial, review) and predict sales, control for author, genre, year published. crawl goodreads, amazon?
* **progress**:
* **papers**: 





## **MUSIC**
* **idea**: predict length on chart from lyrics, control for genre/popularity of singer/(when released)
* **progress**:
* **papers**: 


## **MOVIES**
* **idea**: take movie reviews?, predict box office sales. control for ad budget, popular of actor, hype, year. 
* **progress**:
* **papers**: 


## **ASPECT ANALYSIS**
* **idea**: ?
* **progress**:
  * emailed jiwei
* **papers**: 


## **WIKIPEDIA**
* **idea**: predict page hits with first paragraph control for parent, domain, etc.
* **progress**:
* **papers**: 


## **RECIDIVISM**
* **idea**: feed in court hearing transcript —> predict parole decision. control race, ??
* **progress**:
  * emailed Vinod
  * vinod is going to get the data into the right format for us
  * utterance, cop/innmate, race of innmate, gender, parole decision, length of conversation, etc
* **papers**: 
