import os
import glob

import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM

merge_files = True   # Enable to merge all csvs into one
create_total_mean = True  # Enable to create mean of each timestamp
fill_missing_values = True  # Enable to replace missing values in merged csv
create_day_average = True   # Enable to calculate average values of each day and export into a csv
create_day_average_supply_sum = True   # Enable to sum the different types of average power supply of each day into one
clustering = True  # Enable to cluster average demand and supply values
lstm = True
extension = 'csv'

# Save the filenames in arrays to use in merging
os.chdir("demand")
demand_filenames = [i for i in glob.glob('*.{}'.format(extension))]
os.chdir("../sources")
sources_filenames = [i for i in glob.glob('*.{}'.format(extension))]

filenum = len(demand_filenames)

os.chdir("..")

if merge_files:
    i = 0
    exists = 1

    # first stage of merging, combine source and demand files
    while i < filenum:  # while there are files

        try:  # try to read a demand files
            a = pd.read_csv("demand/" + demand_filenames[i])  # save it in temporary variable a
            a = a.drop(index=288)  # remove the extra 0:00 timestamp at the end
        except pd.errors.EmptyDataError:  # if day is non-existent (eg. 31/02/2020)
            exists = 0
        try:  # repeat for source file
            b = pd.read_csv("sources/" + sources_filenames[i])
            if len(b) == 289:
                b = b.drop(index=288)
            b = b.drop(columns='Time')  # remove Time column so that it doesn't duplicate in merged file
            b = b.rename(columns={"Natural Gas": "Natural gas", "Large Hydro": "Large hydro"})  # fix inconsistency
        except pd.errors.EmptyDataError:
            exists = 0
        if exists == 1:  # if files/dates existed
            combined_csv = pd.concat([a, b], axis=1)  # merge
            combined_csv.insert(0, 'Date', demand_filenames[i][:9])  # Insert date/filename as a column
            # export as csv
            combined_csv.to_csv("combined/combined_csv_" + demand_filenames[i], index=False, encoding='utf-8-sig')
        i += 1
        exists = 1

    # second stage of merging, merge combined source and demand files into one
    i = 0
    mega_csv = pd.read_csv("combined/combined_csv_" + demand_filenames[0])
    i += 1
    while i < filenum:
        try:
            temp = pd.read_csv("combined/combined_csv_" + demand_filenames[i])
            mega_csv = pd.concat([mega_csv, temp], axis=0)
        except FileNotFoundError:
            print(demand_filenames[i])
        i += 1
    mega_csv.to_csv("mega/mega.csv", index=False, encoding="utf-8-sig")
    print("merge ok")

if create_total_mean:
    # load data
    mega_csv = pd.read_csv("mega/mega.csv")
    # create a blank csv with only the columns and their names
    mean_csv = mega_csv
    mean_csv = pd.DataFrame(columns=mean_csv.columns)

    index1 = 0  # index for hour
    index2 = 0  # index for minutes
    label = "00:00"  # variable to store timestamp
    while index1 < 24:  # loops used to parse through timestamps
        while index2 < 60:
            label = "00:00"
            if index2 < 10 and index1 < 10:
                label = "0" + str(index1) + ":0" + str(index2)
            elif index1 < 10:
                label = "0" + str(index1) + ":" + str(index2)
            elif index2 < 10:
                label = str(index1) + ":0" + str(index2)
            else:
                label = str(index1) + ":" + str(index2)
            fil = (mega_csv["Time"] == label)  # for all dates for this timestamp
            temp = mega_csv.loc[fil].mean(axis=0)  # calculate mean
            mean_csv = mean_csv.append(temp, ignore_index=True)  # insert the row into the dataframe
            mean_csv['Time'] = mean_csv['Time'].replace([np.nan], label)  # replace empty value with the timestamp
            index2 += 5  # + 5 minutes
        index2 = 00
        index1 += 1  # + 1 hour

    mean_csv.drop(mean_csv.columns[0], axis=1, inplace=True)  # remove Date column, not useful
    mean_csv = mean_csv.round(2)  # round the result to 2 decimal digits
    mean_csv.to_csv("mega/mean.csv", index=False, encoding="utf-8-sig")  # export
    print("mean ok")

if fill_missing_values:
    # read files
    mega_csv = pd.read_csv("mega/mega.csv")
    mean_csv = pd.read_csv("mega/mean.csv")

    # parse through whole file and for every missing value fill it with the mean of it's timestamp + column which is
    # already calculated in the mean.csv file
    for i in range(len(mega_csv.columns)):
        for j in range(len(mega_csv)):
            if pd.isnull(mega_csv.iloc[j, i]):
                time = mega_csv.iloc[j, 1]
                print(time)
                mega_csv.iloc[j, i] = mean_csv.loc[mega_csv["Time"] == time].iloc[0, i - 1]

    mega_csv.to_csv("mega/mega.csv", index=False, encoding="utf-8-sig")

if create_day_average:
    # create a csv with the average values of every timestamp of a single day, for each day

    # read file
    mega_csv = pd.read_csv("mega/mega.csv")

    # create a blank csv with only the columns and their names
    day_average_csv = mega_csv
    day_average_csv = pd.DataFrame(columns=day_average_csv.columns)

    index1 = 0
    while index1 < len(mega_csv):  # parse through whole file
        temp = mega_csv.iloc[index1:(index1 + 289)].mean(axis=0)  # for the 290 timestamps of every day, calculate mean
        day_average_csv = day_average_csv.append(temp, ignore_index=True)  # insert into dataframe
        index1 += 290  # show at 00:00 of next day

    day_average_csv.drop(day_average_csv.columns[1], axis=1, inplace=True)  # remove Time column, useless
    day_average_csv = day_average_csv.round(0)  # remove decimals
    day_average_csv.to_csv("mega/day_average.csv", index=False, encoding="utf-8-sig")  # export
    print("day_average_csv ok")

if create_day_average_supply_sum:
    # read files
    day_average_csv = pd.read_csv("mega/day_average.csv")
    # create new supply sum column
    day_average_csv.insert(4, 'Supply sum', 0)

    # indexes to parse through all different power supply types
    index1 = 0
    index2 = 5
    sums = 0

    while index1 < len(day_average_csv):  # for every timestamp
        while index2 < len(day_average_csv.columns):  # parse through all different power supply types
            sums += day_average_csv.iloc[index1, index2]  # sum
            index2 += 1  # next type/column
        day_average_csv.iloc[index1, 4] = sums  # store
        index1 += 1  # next row/timestamp
        index2 = 5  # reset column index
        sums = 0  # reset sum
    day_average_csv.to_csv("mega/day_average.csv", index=False, encoding="utf-8-sig")  # export

if clustering:
    day_average_csv = pd.read_csv("mega/day_average.csv")
    data = []
    for j in range(len(day_average_csv)):  # Store energy demand of every day in an indexed table
        data.insert(j, [j, day_average_csv.iloc[j, 3]])
    data = np.array(data)  # Modify data into a numpy array
    x = np.arange(len(data))  # Store values of x axis
    print(x)
    algorithm = DBSCAN(eps=400, min_samples=4).fit(data)  # run DBSCAN with these parameters
    clusters = algorithm.labels_  # Store the labels of the clusters
    print(len(set(clusters)))
    print(clusters)
    df = DataFrame(dict(x=data[:, 0], y=data[:, 1], label=clusters))  # save results  to plot
    colors = {-1: 'red', 0: 'green', 1: 'blue', 2: 'black', 3: 'yellow', 4: 'purple', 5: 'brown', 6: 'pink'}  # Colour of clusters
    fig, ax = plt.subplots(figsize=(8, 8))  # Plot
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    plt.show()

    day_average_csv = pd.read_csv("mega/day_average.csv")
    data = []
    for j in range(len(day_average_csv)):  # Store energy supply of every day in an indexed table
        data.insert(j, [j, day_average_csv.iloc[j, 4]])
    data = np.array(data)  # Modify data into a numpy array
    x = np.arange(len(data))  # Store values of x axis
    print(x)
    algorithm = DBSCAN(eps=400, min_samples=4).fit(data)  # run DBSCAN with these parameters
    clusters = algorithm.labels_  # Store the labels of the clusters
    print(len(set(clusters)))
    print(clusters)
    df = DataFrame(dict(x=data[:, 0], y=data[:, 1], label=clusters))  # save results  to plot
    colors = {-1: 'red', 0: 'green', 1: 'blue', 2: 'black', 3: 'yellow', 4: 'purple', 5: 'brown', 6: 'pink'}  # Colour of clusters
    fig, ax = plt.subplots(figsize=(8, 8))  # Plot
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    plt.show()


if lstm:
    mega_csv = pd.read_csv("mega/mega.csv")  # Calculate total renewable energy and remove it from current demand
    mega_csv['Renewable'] = mega_csv[['Solar', 'Wind', 'Geothermal', 'Biomass', 'Biogas', 'Small hydro', 'Large hydro', 'Batteries']].sum(axis=1)
    mega_csv['Non-Renewable demand'] = mega_csv['Current demand'] - mega_csv['Renewable']
    mega_csv.loc[mega_csv['Non-Renewable demand'] < 0, 'Non-Renewable demand'] = 0
    print(mega_csv)
    data = mega_csv.filter(['Non-Renewable demand']).iloc[:]
    dataset = data.values
    training_data_len = 15000  # Set training data length
    z = 300

    scaler = MinMaxScaler(feature_range=(0, 1))  # Scale data for better results
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:training_data_len]  # Store the training data
    x_train = []
    y_train = []
    for i in range(z, len(train_data)):
        x_train.append(train_data[i-z:i, 0])
        y_train.append(train_data[i])

    x_train, y_train = np.array(x_train), np.array(y_train)  # Change into numpy array
    print(x_train.shape)
    print(x_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # Reshape data
    print(x_train.shape)

    model = Sequential()  # Create LSTM model
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))  # 50 neurons, first layer
    model.add(LSTM(50, return_sequences=False))  # 50 neurons, second layer
    model.add(Dense(25))  # Third layer, density layer
    model.add(Dense(1))  # Fourth layer, density layer

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, y_train, batch_size=1, epochs=1)  # Train the model

    test_data = scaled_data[training_data_len - z:, :]

    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(z, len(test_data)):
        x_test.append(test_data[i-z:i, 0])

    x_test = np.array(x_test)  # Change into numpy array

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  # Reshape data

    predictions = model.predict(x_test)  # Predict values
    predictions = scaler.inverse_transform(predictions)  # Restore values after scaling

    train = data[:training_data_len]  # Create tables for plotting
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    # model.save('my_model.h5')  # Save model to avoid re-training

    plt.figure(figsize=(16, 8))  # Plot results
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Power demand')
    plt.plot(train['Non-Renewable demand'])
    plt.plot(valid['Non-Renewable demand'], color='red', alpha=0.5)
    plt.plot(valid['Predictions'], color='blue', alpha=0.5)
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()

    print(valid['Non-Renewable demand'])
    print(valid['Predictions'])
    mae = mean_absolute_error(valid['Non-Renewable demand'], valid['Predictions'])
    print('mae:' + str(mae))
    mse = mean_squared_error(valid['Non-Renewable demand'], valid['Predictions'])
    print('mse :' + str(mse))
    rmse = np.sqrt(np.mean(predictions - y_test)**2)
    print('rmse :' + str(rmse))
    r2 = r2_score(valid['Non-Renewable demand'], valid['Predictions'])
    print('r2:' + str(r2))
