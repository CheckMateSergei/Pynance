import datetime as dt
import matplotlib.pyplot as plot
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates

# set style for graphs
style.use('ggplot')

start = dt.datetime(2010, 1, 1)
end = dt.datetime.now()

# create a dataframe to store the data in (using pandas_datareader)
# df = web.DataReader('TSLA', 'yahoo', start, end)
# df.to_csv('datasets/tsla.csv')
# saved data to csv to speed up process, parse_dates converts the 'Date' column values into datetime objects to work with easier
df = pd.read_csv('datasets/tsla.csv', parse_dates=['Date'])

# set index to date column
df.reset_index(inplace=True)
df.set_index('Date', inplace=True)
df.drop('index', axis=1, inplace=True)  # why was this even here

# moving average: calculates the average Adj Close price over the selected day
# and the past 99, then moves the window over by one day and repeats
# here we are adding a new column and labeling it '100ma'
# note: first 100 days will NOT have a value unless min_periods is changed to 0
df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()

# creates a slightly more complex graph. create 2 subplots, both act like they are on  a 6x1 grid, the first subplot starts at 0,0 on the 6x1 grid and spans 5 rows and 1 column, the second subplot starts a 5,0 and spans 1 row and 1 column, share=ax1 in the second subplot just means the two subplots stayed aligned #
# ax1 = plot.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
# ax2 = plot.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
# plots adjusted close prices on first subplot
# ax1.plot(df.index, df['Adj Close'])
# plots the 100 day moving average on the first subplot
# ax1.plot(df.index, df['100ma'])
# plots the volume as a bar graph on the second subplot
# ax2.bar(df.index, df['Volume'])
# plot.show()

# resamples the adjusted close and volume columns by a 10 day timeframe, resampling is like grouping but by a time period isntead of a value, note that the index 'Date' must be converted to a datetime object to resample
df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()  # sum of trades over 10 days

# not sure if this is necessary anymore
df_ohlc = df_ohlc.reset_index()
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

# set up new plot with ohlc data
fig = plot.figure()
# same plot as before
ax1 = plot.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plot.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
ax1.xaxis_date()  # converts the raw mdate numbers to dates on the axis

# graphs the candlestick chart on the first subplot
candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
# graphs the volume on the second subplot (lookup fill_between later)
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
# plot the graph
plot.show()


# plot the data
# df.plot()
# plot just data you are interested in (adjusted closing price)
# df[['Low', 'High']].plot()
# show the plot you have generated
# plot.show()

print(df_ohlc.head())
print(df.tail())
