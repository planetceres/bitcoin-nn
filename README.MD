## Time Series Prediction with LSTM Neural Network

This is part of a project working with time series to build a predictive model around bitcoin prices. Stocks and cryptocurrency data is noisy and likely unpredictable with small amounts of data, so this is more an exercise in experimentation than a serious attempt to gain any advantage in the market.
  
Ticker data is collected on 15 second intervals, and predicted 5 minutes into the future. In addition to ticker data, a second data file contains features collected using the [Bitcoin Developer Websocket API](https://blockchain.info), which provides streaming data on new transactions and blocks. Features are organized into epochs and normalized to zero mean and unit variance.

Several variations of network architecture can be used to model the data with small modifications to the layers. In its current form, the model uses a Long Short Term Memory (LSTM) combined with a stacked neural network and dropout. LSTMs can learn long term dependencies, usually in the context of sequences of speech or text, and likewise seemed like a logical choice with which to build a time series predictor.   

The graph for this configuration is as follows:

![lstm graph] (img/lstm_dnn_graph.png)

This uses the development branch of Tensorflow, currently version [0.11.0rc0](https://www.tensorflow.org/)

More data is needed to adequately assess results, but for now, results look something like this:

```
Training Error: 0.85 RMSE
Test Error: 0.82 RMSE

Loss: 0.665656
done in 4.848s.
Overall time: 794.409s
```

![results graph] (img/time_series_LSTM_001.png)

Directories and scripts:

time_series_lstm.py - main program
utils/cleaner.py - cleans the two datasets and fills in missing data with redundant data source
utils/data_loader.py - loads and formats raw data into dataframes
utils/data_splitter.py - spilts into training, test, and optional validation sets

To run:

install dependencies
- Tensorflow
- Numpy
- Pandas
- Matplotlib
    
Then run:

```
python3 time_series_lstm.py
```

Then send donations to:
```
3DrXNQydchoC7aaNHa4F4BSNCL2YyZJAjn
``` 