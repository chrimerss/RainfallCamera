# Rainfall Camera
---

<figure>
<img style="float: right;" src="images/h2i_header.jpg" width="40%"/>
<img style="float: middle;" src="images/Singapore.PNG" width="100%"/>
</figure>

## _Contents_

### 1. [Introduction](#introduction)
### 2. [Classifier](#classifier)
### 3. [Normal Rainfall Processing](#normal)
### 4. [Heavy Rainfall Processing](#heavy)
### 5. [Night Image Processing](#night)
### 6. [Reference](#reference)
### 7. [Misc](#misc)
#### 7.1 [To-do List](#todo)
#### 7.2 [Updates](#update)



## _Introduction_<a name='introduction'></a>

<figure>
    <img style="float: middle; " src="images/Flowchart.png" width="50%"/><br>
    <caption style="font-size:1em;"><center>**Fig.1 Flow Chart for Rainfall Camera**</center></caption>
</figure>

## _Classifier_<a name='classifier'></a>

1. Model Description:

> The model built for classifying rainfall images is [SVM](https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72), which in general, seperate datapoints with hyperplanes. The reason why we build a relatively light model is for the sake of computational time. The simplest and robust way is what we seek for.

2. Data Acquisition:

> To build such machine-learning model, we still need information to train it. With four categories we want to specify, 100 images inside each category are selected from the streaming videos. Image size to train is confined with (300,300).

3. Information to support:

> With data, how can we translate image into useful information that we can feed into SVM? By considering the characteristics during night, rainy days, heavy rainfall and sunny days, I came up with five informative descriptions: contrast, brightness, sharpness, hue and saturation.

4. Return:

> After implementing this classifier, we are able to get a assigned probability of each category for single image. 


## _Normal Rainfall Processing_<a name='normal'></a>
   **1. RNN**<hr style="height:10px; visibility:hidden">
   <figure>
   <img src="images/RNN.PNG" style="float: center"><br>
   <caption style="font-size:1em;"><center>**Fig.2 Overview of RNN model** (Progressive Image Deraining Networks: A Better and Simpler Baseline)</center></caption>
   </figure>
   <br>
   **2. Allamano Algorithm**<hr style="height:10px; visibility:hidden">
   <figure>
   <img src="https://raw.githubusercontent.com/chrimerss/RainProperty/master/Rainstreak.png" style="float: center"><br>
   <caption style="font-size:1em;"><center>**Fig.3 Example of delineated rain streaks** </center></caption>
   </figure>
   <br>
   
   > Allamano Algorithm is used for evaluating the rainfall intensity, the philosophi behind is control volume approach to count rain drops inside the defined bounding box, and calculate rainfall terminal velocity etc.

## _Heavy Rainfall Processing_<a name='heavy'></a>

    So far, we are limited by the data available to supervise a model towards the "correct" path
    will add once more data can be aquired

## _Night Image Processing_<a name='night'></a>

## _Reference_<a name='reference'></a>

R. Dongwei, Z. Wangmeng etc. (2019) _Progressive Image Deraining Networks: A Better and Simpler Baseline_  
R. Martin and M. Frank (2008.) _Classification of Weather Situations on Single Color Images_   
P. Allamano, A. Croci, and F. Laio1 (2015) _Toward the camera rain gauge_  

## _Miscellaneous_<a name='misc'></a>

### To-do list<a name='todo'></a>
- [x] build dask task manager
- [x] add classifier
- [ ] add regression model/convert to RGB image
- [ ] night image processing
- [ ] GUI
- [x] add visualization
- [ ] add computational time table
- [ ] GPU version(convert all dask array to torch array and hard code torch version SVM)

### Updates<a name='update'></a>
    
    2019.4.19 add visualization.py
    2019.4.18 optimized code and retrained model
    2019.4.17 dask implementation
    2019.4.15 trained a classifier model
  
