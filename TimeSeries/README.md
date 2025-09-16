# Time Series Forecasting

I provide the Jupyter Notebook with the code in Tensorflow for a simple neural network designed for timeseries forecasting. I use a learning rate scheduler to identify 
the correct value for the learning rate. \
I attach the dataset as a csv file. It provides the daily minimum temperatures in Melbourne over a period of ten years. The dataset is taken from Kaggle. \
The network contains a CONV1D layer, two LSTMs and two Fully Connected layers. The forecasting is performed considering a window of 32 data points and predicting a single value. The training requires a few minutes on CPU.
