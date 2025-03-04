# THE DAY AFTER TOMORROW

Real-time tropical storm forecasting model with interactive jupyter notebook.

![111](https://github.com/acse-hst21/Weather_Forecast_Tool/blob/96afaa753181257367e4e44c6eaef884f50c07fa/img/output.gif)


## About

Hurricanes would kill up to 1,000 people and produce over $50 billion in damage in a single occurrence, and have killed far over 160,000 people in recent history. Humanitarian  response activities during a tropical cyclone rely on reliable risk approximation models that can assist anticipate optimal emergency strategy options.

FEMA (Federal Emergency Management Agency) in the United States has announced an open competition to strengthen its emergency processes in the event of hurricanes, which is open to teams of ML specialists who are able to solve the challenge of forecasting the the evolution of tropical storms in real time. Team Katrina participated the competition and implementing custom networks to solve the challenge.



## Data

- NASA Satellite images of tropical storms
- 494 storms around the Atlantic and East Pacific Oceans (precise locations undisclosed)
- Each with varied number of time samples (4 - 648, avg 142)
- Labelled by id, ocean (1 or 2) and wind speed

- [Data source](https://mlhub.earth/data/nasa_tropical_storm_competition) (To download it you must create an account with Radient MLHub)



## Installation Guide 

Clone the local repository by running :

```
git clone https://github.com/acse-hst21/Weather_Forecast_Tool
```

Then navigate to the root of the repository that you cloned and install all the required packages by running:

```
pip install -r requirements.txt
pip install -e .
```



## User instructions

Click the `surprise_storm.ipynb` file, then you are ready to go. 

All the interactive function is written inside the notebook, every function will trigger corresponding packaged utility scripts. The dataset used in `train.py` is provided by [NASA](https://www.nasa.gov/). The dataset used in `surprise_storm.ipynb ` is provied by Imperial College London ESE department. You can use the downloaded dataset by setting hyperparameter`download` as True.


**Training process based on the Windows platform. Under Linux system, if the GPU is used for training, there is a high probability of outputting Nan results. This may caused by the CUDA drive. Please use the Windows environment for training.**


## Documentation 

[Documentation](https://github.com/acse-hst21/Weather_Forecast_Tool/blob/96afaa753181257367e4e44c6eaef884f50c07fa/docs/storm_forcast.pdf) can be found inside the the `docs` folder in the `storm_forcast.pdf` which contains installation instructions and `storm_forcast` function APIs.





## References

1. Shi, X.; Chen, Z.; Wang, H.; Yeung, D.; Wong, W.; Woo, W. Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting.
2. Graves A. Generating Sequences With Recurrent Neural Networks 2022. 
3. J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation.
4. R. Panda, "Video Frame Prediction using ConvLSTM Network in PyTorch", Medium, 2022. [Online]. Available: https://sladewinter.medium.com/video-frame-prediction-using-convlstm-network-in-pytorch-b5210a6ce582. [Accessed: 26- May- 2022]
