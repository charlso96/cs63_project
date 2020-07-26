# Final Project for CS63 course (Artificial Intelligence) at Swarthmore College.

Keonwoo Oh and Han Huang

* Please read our pdf file, instead of compiling the tex file. We used Overleaf
for editing & compiling, so there might be errors with compiling.

Our LSTM_Finance program trains its neural network on the given time series data
to give either 1) accurate predictions of the data point for each previous subsequence
2) the direction of change in the data point for each previous subsequence 3) iteratively
forecast the entire time series data. Then it appropriately plots the results.
The usage convention is as follows:

Usage: LSTM_Finance.py dataset --result --dataSet --numeric --forecast --CNN

1) dataset is the time series dataset the network is trained and tested on. There
are multiple csv files that store different time series data the program can run on.
2) if --result, the predictions made by the network and the actual data of the test
set are printed out.
3) if --dataSet, the data set is printed out.
4) if --numeric, the actual data is predicted at each time point. The default is
to predict the direction of change in the data.
5) if --forecast, the neural network iteratively forecasts an entire time series
6) if --CNN, CNN-LSTM network is used. By default, standard LSTM network is used.

After the network trains and is tested, the results will be plotted appropriately.
In case of change prediction, the plot represents whether the network made the
right prediction, 1 if it was correct and 0 if it was not.
