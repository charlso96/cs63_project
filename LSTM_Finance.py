
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, LSTM, SimpleRNN, ConvLSTM2D, Flatten, TimeDistributed, Conv1D, MaxPooling1D
import numpy as np
import pandas as pd
import random
from argparse import ArgumentParser

def parse_args():
    p = ArgumentParser()
    p.add_argument("dataset", type=str,help="The dataset to run on.")
    p.add_argument('--result', action='store_true',
                    help='Print out prediction results.')
    p.add_argument('--dataSet',action='store_true',
                    help='Print out dataSet.')
    p.add_argument('--numeric',action='store_true',
                    help='Do continuous prediction.')
    p.add_argument('--forecast',action='store_true',
                    help='forecasting Mode.')
    p.add_argument('--CNN',action='store_true',
                    help='Use CNN LSTM.')
    return p.parse_args()

def create_sequences_vector(data, seqlen):
    X = [] #input
    Y = [] #target
    for i in range(len(data)-seqlen):
        X.append(data[i:i+seqlen])
        Y.append(data[i+seqlen][:5])
    return np.array(X), np.array(Y)

#Only use Adjusted close as both the input and output
def create_sequences(data, seqlen):
    X = [] #input
    Y = [] #target
    for i in range(len(data)-seqlen):
        X.append(data[i:i+seqlen][:,4])
        Y.append(data[i+seqlen][4])
    return np.array(X).reshape(-1,seqlen,1), np.array(Y).reshape(-1,1)

#include up and down vectors.
def create_sequences_UD(data, seqlen):
    X = [] #input
    Y = [] #target
    for i in range(len(data)-seqlen):
        X.append(data[i:i+seqlen][:,:-2])
        Y.append(data[i+seqlen][-2:])
    return np.array(X), np.array(Y)

def create_CNN_sequences(data, unroll, seqlen):
    X = [] #input
    Y = [] #target
    for i in range(len(data)-unroll):
        X.append(data[i:i+unroll][:,4])
        Y.append(data[i+unroll][-2:])
    X=np.asarray(X)
    print(X.shape)
    return np.array(X).reshape(X.shape[0], seqlen, unroll//seqlen, X.shape[2]), \
    np.array(Y).reshape(-1,2)

def create_CNN_sequences_vector(data, unroll, seqlen):
    X = [] #input
    Y = [] #target
    for i in range(len(data)-unroll):
        X.append(data[i:i+unroll])
        Y.append(data[i+unroll])
    X=np.asarray(X)
    print(X.shape)
    return np.array(X).reshape(X.shape[0], seqlen, unroll//seqlen, X.shape[2]), \
    np.array(Y)


def forecast(model,length,data):
    if (len(data) == 0):
        return

    unroll = data.shape[1]
    num_features = data.shape[2]

    #get the initial data that we want to predict
    data_to_predict = data[0].reshape(1,unroll,num_features)
    forecasts = data_to_predict.reshape(unroll,num_features)

    for i in range(1,length):
        #make the next prediction
        next_forecast = model.predict(data_to_predict)
        #add the next prediction to the time series data
        forecasts = np.concatenate((forecasts,next_forecast), axis = 0)
        #get the next data we want to predict
        data_to_predict = forecasts[i:i+unroll,:].reshape(1,unroll,num_features)

    return forecasts


def forecast_CNN(model,length,data,data_3D):
    if (len(data) == 0):
        return

    seqlen = data.shape[1]
    timelen = data.shape[2]
    num_features = data.shape[3]

    #get the initial data that we want to predict
    data_to_predict = data[0].reshape(1,seqlen,timelen,num_features)
    forecasts = data_3D[0].reshape(seqlen*timelen,num_features)

    for i in range(1,length):
        #make the next prediction
        next_forecast = model.predict(data_to_predict)
        #add the next prediction to the time series data
        forecasts = np.concatenate((forecasts,next_forecast), axis = 0)
        #get the next data we want to predict
        data_to_predict = forecasts[i:i+seqlen*timelen,:].reshape(1,seqlen,timelen,num_features)

    return forecasts


def build_lstm_model(input_size, output_size, unrolling_steps):
    model = Sequential()
    model.add(LSTM(input_size, use_bias=True, dropout=0.2,
                   batch_input_shape=(None, unrolling_steps, input_size)))
    model.add(Dense(output_size))
    model.add(Activation('relu'))
    model.compile(loss='mse', optimizer='adam')
    print(model.summary())
    return model


def build_lstm_model_UD(input_size, output_size, unrolling_steps):
    model = Sequential()
    model.add(LSTM(input_size, use_bias=True, dropout=0.2,
                   batch_input_shape=(None, unrolling_steps, input_size)))
    model.add(Dense(output_size))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def build_CNN_lstm_model_UD(input_size, output_size, unrolling_steps, X_train, Y_train, epochs):
    num_features = X_train.shape[3]
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=128, kernel_size=1, activation='relu'),\
     input_shape=(None,unrolling_steps, num_features)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(50, activation='relu'))
    model.add(Dense(output_size))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    print(model.summary())
    history = model.fit(X_train, Y_train, epochs=epochs, verbose=1)
    return model, history


def build_CNN_lstm_model(input_size, output_size, unrolling_steps, X_train, Y_train, epochs):
    num_features = X_train.shape[3]
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=128, kernel_size=1, activation='relu'),\
     input_shape=(None,unrolling_steps, num_features)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(output_size))
    model.add(Activation('relu'))
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())
    history = model.fit(X_train, Y_train, epochs=epochs, verbose=1)
    return model, history


def main():
    #Read in data from a csv file
    args = parse_args()
    filename = args.dataset
    print_result = args.result
    print_dataSet = args.dataSet
    numerical_model = args.numeric
    enable_forecast = args.forecast
    CNN_model = args.CNN

    #Read in data from a csv file
    dataSet = np.genfromtxt(filename, delimiter=',')
    #clean up the data.
    cols = dataSet.shape[1]
    dataSet = dataSet[1:,1:cols-1]
    print ("Shape of dataSet: ", dataSet.shape)

    #Normalize the data
    Max = []
    for i in range(dataSet.shape[1]):
        Max.append(max(dataSet[:,i]))
        dataSet[:,i] /= max(dataSet[:,i])

    n = 0
    unroll = 0
    epochs = 0
    X_train, Y_train, X_test, Y_test = None,None,None,None
    lstm = None

    '''
    The Categorical Model
    '''
    if (not numerical_model):
        #Generate categorical vector for Up/Down
        #They're the last two columns
        up = np.asarray([0]*len(dataSet))
        down = np.asarray([0]*len(dataSet))

        #First value is bogus
        up[0] = 1
        down[0] = 0

        for i in range(1,len(dataSet)):
            #Adj. close is the 5th entry
            if (dataSet[i,4] > dataSet[i-1,4]):
                up[i] = 1
                down[i] = 0
            else:
                up[i] = 0
                down[i] = 1

        #stack it onto dataSet
        dataSet = np.column_stack((dataSet,up))
        dataSet = np.column_stack((dataSet,down))
        if (not CNN_model):
            ##############Hyperparameters####################
            #80-20 split of training and testing data
            n = int(0.8 * len(dataSet))
            #length of temporal sequence as inputs
            unroll = 30
            epochs = 100

            #Process the time series data into training and testing.
            X_train, Y_train = create_sequences_UD(dataSet[:n], unroll)
            print("Shape of training inputs", X_train.shape)
            print("Shape of training targets", Y_train.shape)
            X_test, Y_test = create_sequences_UD(dataSet[n:], unroll)
            print("Shape of testing inputs", X_test.shape)
            print("Shape of testing targets", Y_test.shape)

            # Build the model
            input_size = 5
            output_size = 2
            lstm = build_lstm_model_UD(input_size, output_size, unroll)

            # Train the model
            history = lstm.fit(X_train, Y_train, batch_size=32, epochs=epochs)

            # Display training information
            import matplotlib
            import matplotlib.pyplot as plt
            f = plt.figure(1)
            plt.plot(history.history["loss"], label="loss")
            plt.plot(history.history["acc"], label="accuracy")
            f.suptitle('Loss and Accuracy')

            # Test the model
            loss, accuracy = lstm.evaluate(X_test, Y_test)
            print("\nloss", loss, "accuracy", accuracy)

            predictions_test = lstm.predict(X_test)

            prediction_plot = [0]*len(predictions_test)

            for i in range(len(predictions_test)):
                toPrint = [0,0]
                #convert likelihood to one-hot
                if (predictions_test[i][0] < predictions_test[i][1]):
                    toPrint[0] = 0
                    toPrint[1] = 1
                else:
                    toPrint[0] = 1
                    toPrint[1] = 0

                if (toPrint == list(Y_test[i])):
                    prediction_plot[i] = 1


                if (print_result):
                    print ("Data: ", X_test[i], " prediction: ", toPrint, " actual: ", Y_test[i])

            w = plt.figure(4)
            plt.plot(prediction_plot)
            w.suptitle('Prediction')

            plt.show()

        #CNN model
        else:
            ##############Hyperparameters####################
            #80-20 split of training and testing data
            n = int(0.8 * len(dataSet))
            #length of temporal sequence as inputs
            unroll = 30
            epochs = 200

            seqlen = 6
            #Process the time series data into training and testing.
            X_train, Y_train = create_CNN_sequences(dataSet[:n], unroll, seqlen)
            print("Shape of training inputs", X_train.shape)
            print("Shape of training targets", Y_train.shape)
            X_test, Y_test = create_CNN_sequences(dataSet[n:], unroll, seqlen)
            print("Shape of testing inputs", X_test.shape)
            print("Shape of testing targets", Y_test.shape)

            # Build the model
            input_size = 1
            output_size = 2
            lstm, history = build_CNN_lstm_model_UD(input_size, output_size, unroll//seqlen, X_train, Y_train, epochs)

            # Display training information
            import matplotlib
            import matplotlib.pyplot as plt
            f = plt.figure(1)
            plt.plot(history.history["loss"], label="loss")
            f.suptitle('Loss')

            predictions_test = lstm.predict(X_test)

            prediction_plot = [0]*len(predictions_test)

            counter = 0
            for i in range(len(predictions_test)):
                if (i < 3):
                    print (predictions_test[i])

                toPrint = [0,0]
                #convert likelihood to one-hot
                if (predictions_test[i][0] < predictions_test[i][1]):
                    toPrint[0] = 0
                    toPrint[1] = 1
                else:
                    toPrint[0] = 1
                    toPrint[1] = 0

                if (toPrint == list(Y_test[i])):
                    prediction_plot[i] = 1
                    counter +=1

                if (print_result):
                    print ("Data: ", X_test[i], " prediction: ", toPrint, " actual: ", Y_test[i])

            print("accuracy: ", float(counter)/len(predictions_test))

            w = plt.figure(4)
            plt.plot(prediction_plot)
            w.suptitle('Prediction')

            plt.show()



    else:
        if (not CNN_model):

            ##############Hyperparameters####################
            #80-20 split of training and testing data
            n = int(0.8 * len(dataSet))
            #length of temporal sequence as inputs
            unroll = 30
            epochs = 100

            #Process the time series data into training and testing.
            X_train, Y_train = create_sequences_vector(dataSet[:n], unroll)
            print("Shape of training inputs", X_train.shape)
            print("Shape of training targets", Y_train.shape)
            X_test, Y_test = create_sequences_vector(dataSet[n:], unroll)
            print("Shape of testing inputs", X_test.shape)
            print("Shape of testing targets", Y_test.shape)

            # Build the model
            input_size = 5
            output_size = 5
            lstm = build_lstm_model(input_size, output_size, unroll)

            # Train the model
            history = lstm.fit(X_train, Y_train, batch_size=32, epochs=epochs)

            # Display training information
            import matplotlib
            import matplotlib.pyplot as plt
            f = plt.figure(1)
            plt.plot(history.history["loss"], label="loss")
            f.suptitle('Loss')

            loss = lstm.evaluate(X_test, Y_test)
            print("\nTesting Loss", loss)

            predictions_test = lstm.predict(X_test)

            for i in range(len(predictions_test)):
                if (print_result):
                    print ("Data: ", X_test[i], " prediction: ", predictions_test[i],\
                     " actual: ", Y_test[i])

            predictions_train = lstm.predict(X_train)

        #CNN model
        else:
            ##############Hyperparameters####################
            #80-20 split of training and testing data
            n = int(0.8 * len(dataSet))
            #length of temporal sequence as inputs
            unroll = 30
            epochs = 200

            seqlen = 6

            #Process the time series data into training and testing.
            X_train, Y_train = create_CNN_sequences_vector(dataSet[:n], unroll, seqlen)
            print("Shape of training inputs", X_train.shape)
            print("Shape of training targets", Y_train.shape)
            X_test, Y_test = create_CNN_sequences_vector(dataSet[n:], unroll, seqlen)
            print("Shape of testing inputs", X_test.shape)
            print("Shape of testing targets", Y_test.shape)

            # Build the model
            input_size = 5
            output_size = 5
            lstm,history = build_CNN_lstm_model(input_size, output_size, unroll//seqlen, X_train, Y_train, epochs)

            # Display training information
            import matplotlib
            import matplotlib.pyplot as plt
            f = plt.figure(1)
            plt.plot(history.history["loss"], label="loss")
            f.suptitle('Loss')

            loss = lstm.evaluate(X_test, Y_test)
            print("\nTesting Loss", loss)

            predictions_test = lstm.predict(X_test)

            for i in range(len(predictions_test)):
                if (print_result):
                    print ("Data: ", X_test[i], " prediction: ", predictions_test[i],\
                     " actual: ", Y_test[i])

            predictions_train = lstm.predict(X_train)

            errors = [sum(abs(Y_test[i]-predictions_test[i])) for i in range(len(predictions_test))]
            print("maximum error found:", max(errors))


        # Test the model
        if (not enable_forecast):
            #plot how well LSTM fits to the test data.
            g = plt.figure(2)
            #only plot the Adjusted Close
            plt.plot(predictions_test*Max[4],'-C0', label='Predictions')
            plt.plot(Y_test*Max[4],'-C1', label='Test Data')
            plt.legend(loc='upper left')
            g.suptitle('Test Data Fitness')

            h = plt.figure(3)
            plt.plot(predictions_train*Max[4],'-C0', label='Predictions')
            plt.plot(Y_train*Max[4],'-C1', label='Train Data')
            plt.legend(loc='upper left')
            h.suptitle('Train Data Fitness')

            plt.show()

        #forecast Mode
        else:
            if (not CNN_model):
                T = 30
                future_forecasts = forecast(lstm,T,X_test)

                #plot how well LSTM forecasts
                g = plt.figure(2)
                #only plot the Adjusted Close
                plt.plot(future_forecasts[unroll:T+unroll,4]*Max[4],'-C0', label='forecasts')
                plt.plot(Y_test[:T,4]*Max[4],'-C1', label='Reality')
                plt.legend(loc='upper left')
                g.suptitle('forecast vs. Reality')
                plt.show()

            #Use CNN to forecast
            else:
                #For technicality reasons.
                X_test_3D, Y_test_3D = create_sequences_vector(dataSet[n:], unroll)

                T = 30
                future_forecasts = forecast_CNN(lstm,T,X_test,X_test_3D)

                #plot how well LSTM forecasts
                g = plt.figure(2)
                #only plot the Adjusted Close
                plt.plot(future_forecasts[unroll:T+unroll,4]*Max[4],'-C0', label='forecasts')
                plt.plot(Y_test[:T,4]*Max[4],'-C1', label='Reality')
                plt.legend(loc='upper left')

                g.suptitle('forecast vs. Reality')

                plt.show()

if __name__ == "__main__":
    main()
