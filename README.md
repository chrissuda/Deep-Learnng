# Deep-Learning

## Abstract
This project aims to classify accessibility feature of storefront on sidewalk from google street view images and it can help visually impaired people to avoid dangerous obstacles in the street and allow them to access each store. We are conducting some experiments on Faster-RCNN, a popular architecture in Object Detection. It enables us to have better understanding of how this model actually “sees” a physical object. 

## Dataset
#### We aims to identify 4 type of categories(excluding Backgroudn) as following.  
Training Data:   928 images with labels  
Validation Data: 100 images with labels  
Category:
* Background(0)
* Door(1)
* Knob(2)
* Stairs(3)
* Ramp(4)


## Detection Result(Evaluating with 100 images)
#### Testing without using depth-filtering 
*************************Recall Precision **************************************
Door -> TP: 159  Predict: 350  Truth: 164  Precision:45.43%  Recall:96.95%
Knob -> TP: 58  Predict: 211  Truth: 77  Precision:27.49%  Recall:75.32%
Stairs -> TP: 94  Predict: 361  Truth: 96  Precision:26.04%  Recall:97.92%
Ramp -> TP: 1  Predict: 13  Truth: 10  Precision:7.69%  Recall:10.00%
*******************************************************************************

#### Testing using depth-filtering 
*************************Recall Precision **************************************
Door -> TP: 154  Predict: 283  Truth: 164  Precision:54.42%  Recall:93.90%
Knob -> TP: 58  Predict: 211  Truth: 77  Precision:27.49%  Recall:75.32%
Stairs -> TP: 94  Predict: 361  Truth: 96  Precision:26.04%  Recall:97.92%
Ramp -> TP: 1  Predict: 13  Truth: 10  Precision:7.69%  Recall:10.00%
*******************************************************************************

#### non-filtering vs depth-filtering

## Predict Doors' Location


## Demo
![image](https://github.com/chrissuda/Deep-Learnng/blob/master/result/predict_original_001217_1.jpg)

#### More Demos can be founded here:
[Demo](https://github.com/chrissuda/Deep-Learnng/tree/master/result)

#### Final Poster can be founded here:
[Poster](https://github.com/chrissuda/Deep-Learnng/blob/master/Bars_Poster.pdf)


## Future Work
We are still working and updating on it
