# Deep-Learning

## Abstract
This project aims to classify accessibility feature of storefront on sidewalk from google street view images and it can help visually impaired people to avoid dangerous obstacles in the street and allow them to access each store. We are conducting some experiments on Faster-RCNN, a popular architecture in Object Detection. It enables us to have better understanding of how this model actually “sees” a physical object. 

## Dataset
#### Category:
* Background(0)
* Door(1)
* Knob(2)
* Stairs(3)
* Ramp(4)

#### Dataset Size:
* Training Set:928 images with labels
* Validation Set: 100 images with labels 
<br />
<br />
<br />

## Detection Result
#### Testing without using depth-filtering 
*************************Recall Precision ************************************** <br />
Door -> TP: 159  Predict: 350  Truth: 164  Precision:45.43%  Recall:96.95% <br />
Knob -> TP: 58  Predict: 211  Truth: 77  Precision:27.49%  Recall:75.32% <br />
Stairs -> TP: 94  Predict: 361  Truth: 96  Precision:26.04%  Recall:97.92% <br />
Ramp -> TP: 1  Predict: 13  Truth: 10  Precision:7.69%  Recall:10.00% <br/>
*******************************************************************************  <br />

#### Testing using depth-filtering 
*************************Recall Precision ************************************** <br />
Door -> TP: 154  Predict: 283  Truth: 164  Precision:54.42%  Recall:93.90% <br />
Knob -> TP: 58  Predict: 211  Truth: 77  Precision:27.49%  Recall:75.32% <br />
Stairs -> TP: 94  Predict: 361  Truth: 96  Precision:26.04%  Recall:97.92% <br />
Ramp -> TP: 1  Predict: 13  Truth: 10  Precision:7.69%  Recall:10.00% <br />
*******************************************************************************
<br />
<br />
<br />

#### Non-filtering vs Depth-filtering
<table>
  <tr>
    <th>Non-filtering</th>
    <th>Depth-Filtering</th> 
  </tr>
  <tr>
    <td><img title="Non-filtering" src="https://github.com/chrissuda/Deep-Learnng/blob/master/Demo/_0UWVi_fhk1Tucg5-Z2qkg_1_predict.jpg" width="100%" /></td>
    <td><img title="Depth-Filtering" src="https://github.com/chrissuda/Deep-Learnng/blob/master/Demo/_0UWVi_fhk1Tucg5-Z2qkg_1_predict_filter.jpg" width="100%" /></td>
  </tr>
</table>
  

## Predicting Doors' Geolocation
#### Prediction's Accuracy
![Door's GeoLocation](https://github.com/chrissuda/Deep-Learnng/blob/master/Demo/doorOnMap.jpg)

<br />

#### Assiocated Stores' Names With Doors
![Doors with Store Names](https://github.com/chrissuda/Deep-Learnng/blob/master/Demo/maptrial.png)
<br />
<br />
<br />


## Still Updating!
