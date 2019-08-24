import bs4 as bs
import pandas as pd
import pickle
import requests
import datetime as dt
import os
import pandas_datareader.data as web
import matplotlib.pyplot as plot
from matplotlib import style
import numpy as np
from collections import Counter
from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

style.use('ggplot')

# creates a graph which visualizes the correlation between companies in the S&P500 index
def visualize_data():
    df = pd.read_csv('datasets/sp500_joined_closes.csv')
    df_corr = df.corr()

    # define variables to build heatmap
    data1 = df_corr.values
    fig1 = plot.figure()
    ax1 = fig1.add_subplot(111)
    heatmap1 = ax1.pcolor(data1, cmap=plot.cm.RdYlGn)
    # add a color bar as a scale
    fig1.colorbar(heatmap1)
    # create axes ticks
    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    # inverts y axis and flips x axis to the top of the graph
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    # this will add the company names to our ticks
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    # rotate the x axis labels 90 degrees
    plot.xticks(rotation=90)
    heatmap1.set_clim(-1,1)
    plot.tight_layout()
    plot.show()

# function designed to get S&P500 list from wikipedia
def save_sp500_tickers():
    # use requests to get the source code from wikipedia
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    # convert the source code into 'soup' object, basically allows us to work with the source code as a python object
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    # iterate through the table and generate a list of tickers and save them to a pickle object..?
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.replace('.', '-')
        tickers.append(ticker[:-1])
        with open('sp500tickers.pickle', 'wb') as f:
            pickle.dump(tickers, f)
    return tickers

# new function to either reload or store the pickle file into a tickers variable
def get_data_from_yahoo(reload_sp500=False):
    # if prompted we will reload the data from the web
    if reload_sp500:
        tickers = save_sp500_tickers()
    # else we just load our pickle file
    else:
        with open('sp500tickers.pickle', 'rb') as f:
            tickers = pickle.load(f)

    # we need to create a directory to store our data in before we manipulate it
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    # now we will pull our data as we did in the GetData.py script (part 1)
    start = dt.datetime(2019, 6, 8)
    end = dt.datetime.now()

    for ticker in tickers:
        # save progress incase connection breaks
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            # load stock data from yahoo
            df = web.DataReader(ticker, 'yahoo', start, end)
            # reset the index of the dataframe and copy it
            df.reset_index(inplace=True)
            # set the index to date
            df.set_index('Date', inplace=True)
            # drop the ticker symbol column
            # df = df.drop('Symbol', axis=1)
            # save a copy of the stock data csv
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

# compile the ticker data and read into a new dataframe
def compile_data():
    # load the pickled tickers and create an emply DataFrame
    with open('sp500tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)
    main_df = pd.DataFrame()

    # iterate throught the list of tickers
    for count, ticker in enumerate(tickers):
        # read in the csv data for the specified ticker
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)
        df.rename(columns={'Adj Close': ticker}, inplace=True)
        # drop all colums but Adj Close
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)
        #append this df's data to the main dataframe
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')
        # print the current count modulo 10
        if count % 10 == 0:
            print(count)

    # print the main_df
    print(main_df.head())
    main_df.to_csv('datasets/sp500_joined_closes.csv')

# process the data for creating labels
def process_data_for_labels(ticker):
    #read data for 7 days
    hm_days = 7
    df = pd.read_csv('datasets/sp500_joined_closes.csv', index_col=0)
    # create a list of the column values in the dataframe
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    # creates 7 new columns of data which respresent the percent change between current day and price i-days from today 
    for i in range(1, hm_days + 1):
        # formats a new column in the dataframe for each ticker, shift(-i) will shift the column UP by i rows (so this calculates the difference in the closing price and the closing price i-days from now and then divides by the closing price again to get a percent difference) 
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i)-df[ticker])/df[ticker]
    # fill empty data in with 0's and return tickers and the dataframe 
    df.fillna(0, inplace=True)
    return tickers, df

# create a function that creates our 'label' or basically tells us whether to buy or sell a commodity
def buy_sell_hold(*args):
    # list of the columns we have passed to the function
    cols = [c for c in args]
    requirement = 0.02
    # this will be used to map columns to 'labels' in a new pandas dataframe
    for col in cols:
        if col > requirement:
            # buy
            return 1
        if col < -requirement:
            # sell
            return -1

# 
def extract_featuresets(ticker):
    # the processed data for creating our labels
    tickers, df = process_data_for_labels(ticker)
    # creates a target column, has either 1, 0, or -1 for each day we have processed our data for, uses the buy_sell_hold function to map a day to a value (buy/sell) in the target column, each column has 7 rows (days) of data. 
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold, 
                                             df['{}_1d'.format(ticker)],
                                             df['{}_2d'.format(ticker)],
                                             df['{}_3d'.format(ticker)],
                                             df['{}_4d'.format(ticker)],
                                             df['{}_5d'.format(ticker)],
                                             df['{}_6d'.format(ticker)],
                                             df['{}_7d'.format(ticker)]))
    # gets the distribution of our data
    # outputs the results of our buy/sell mapping to a list
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print()
    print('Data Spread: ', Counter(str_vals))

    # remove any infinite data from the tables and change 'missing' data to 0
    df.fillna(0, inplace=True)
    # replaces infinite data with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    # stores the percent change for each stock
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    # finally we create our labels
    X = df_vals.values
    y = df['{}_target'.format(ticker)].values
    # X contains the 'featuresets' the daily pct change in our data, y is our buy/sell/hold label or target, now we be ready for some machine learning
    return X, y, df

# machine learning function
def do_machine_learning(ticker):
    # extract the featuresets for the specified ticker
    X, y, df = extract_featuresets(ticker)

    # this shuffles data to avoid biases, and creates training and testing samples for our machine learning algorithms
    X_train, X_test, y_train, y_test, = cross_validation.train_test_split(X,
                                                                         y,
                                                                       test_size=0.25)
    # choose a classifier
    clf = neighbors.KNeighborsClassifier()
    # fit (train) the classifier to the data, fits X data to y data for each X,y pair
    clf.fit(X_train, y_train)
    # takes some featuresets (X_test), makes a prediction and sees if it matches our labels (y_test) and then returns a percent accuracy in decimal form
    confidence = clf.score(X_test, y_test)

    # prints our results for chosen ticker data
    print('accuracy: ', confidence)
    predictions = clf.predict(X_test)
    print('predicted class counts: ', Counter(predictions))
    print()
    print()

do_machine_learning('ATVI')
do_machine_learning('AAPL')
do_machine_learning('ABT')

# visualize_data()
# save_sp500_tickers()
# get_data_from_yahoo()
# compile_data()
