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

style.use('ggplot')

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


visualize_data()

# save_sp500_tickers()
get_data_from_yahoo()
compile_data()
