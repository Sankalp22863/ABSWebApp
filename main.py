# pip install streamlit pystan = 2.19.9.1 fbprophet yfinance plotly
import streamlit as st
from datetime import date

from threading import Thread
import multiprocessing

import altair as alt

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
from matplotlib.animation import FuncAnimation

from youtubesearchpython import VideosSearch

from streamlit_player import st_player

import os
import logging
from itertools import count

from PIL import Image

import speedtest

import time

SPEEDTEST_CMD = 'python speedtest-cli/speedtest.py'

LOG_FILE = 'speedtest.log'


def setup_logging():
  logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt = "%Y-%m-%d %H:%M",
)

def get_speedtest_results():
  '''
    Run test and parse results.
    Returns tuple of ping speed, download speed, and upload speed,
    or raises ValueError if unable to parse data.
  '''
  ping = download = upload = None
  with os.popen(SPEEDTEST_CMD + ' --simple') as speedtest_output:
        for line in speedtest_output:
            label, value, unit = line.split()
            if 'Ping' in label:
                ping = float(value)

            elif 'Download' in label:
                download = float(value)
                
            elif 'Upload' in label:
                upload = float(value)
                
        if all((ping, download, upload)): # if all 3 values were parsed
            return ping, download, upload
        else:
            raise ValueError('TEST FAILED')



@st.cache
def animate(x_vals, y_vals):
    x_vals.append(next(index))

    plt.cla()

    plt.plot(x_vals, y_vals)

    plt.tight_layout()


# Load Packages
import numpy as np
import pandas as pd
from pandas_datareader import data
from datetime import date

import matplotlib.pyplot as plt


import VideoPlaylist


st.title('Adaptive Bit-rate Scheme.')


# Create a page dropdown 
page = st.sidebar.selectbox("Choose your page", ["View Lecture.", "Future Stock price prediction.", "Portfolio Reccomendations."]) 

vid_search = st.text_input("Search a Youtube Video")

# link = ['https://www.youtube.com/watch?v=cjUc2ih8FMs', 'https://www.youtube.com/watch?v=7fhakgay9Eg']

videosSearch = VideosSearch(vid_search, limit = 10)

url = []
Vids = []
t = {}

def vid():
	global url 
	global t
	global Vids
	global vid_search
	
	if vid_search != '':
		for i in range(10):
			Vids.append(videosSearch.result()['result'][i]['title'])
			url.append(videosSearch.result()['result'][i]['link'])
		
		for i in range(10):
			t[Vids[i]] = url[i]

		selected_video = st.selectbox('Select the Video', Vids)

		prev = selected_video

		url = t[selected_video]
		st.subheader(selected_video)

		# Embed a youtube video
		st_player(url)
	return 



def disp():
	x_vals = []
	
	if vid_search != '':
		bd_monitor = st.checkbox('Monitor Band Width') # Monitors the Bandwidth.

	
		ping_list = []
		download_list = []
		upload_list = []


		df = pd.DataFrame({'Ping':[],
							'Download':[],
							'Upload':[]})

		first = True

		draw = st.line_chart(download_list)
		# data = st.sidebar.table(download_list)

		disp = st.sidebar.dataframe(df)

		# col1, col2, col3 = st.columns(3)

		# col1.metric(label = "Ping", value = "%.2f"%ping_list[-1] + " ms", delta = str(round(ping_list[-1] - ping_list[-2], 2)))
		# col2.metric(label = "Download", value = "%.2f"%download_list[-1]+ " Mbps " , delta = str(round(download_list[-1] - download_list[-2], 2)))
		# col3.metric(label = "Upload", value =  "%.2f"%upload_list[-1] + " Mbps", delta = str(round(upload_list[-1] - upload_list[-2], 2)))
		

		while bd_monitor:

			try:
				# Waiting for the Test Result
				# st.heading('Waiting for the Test Result.')
				if first:
					st.subheader('Waiting for the Test Result.')
				ping, download, upload = get_speedtest_results()
				# ping
				ping_list.append(ping)
				download_list.append(download)
				upload_list.append(upload)
				
			except ValueError as err:
				logging.info(err)
				download_list.append(0)
				upload_list.append(0)
				ping_list.append(ping_list[-1])
			else:
				logging.info("%5.1f %5.1f %5.1f", ping, download, upload)

			disp.dataframe(df.tail())

			# plt.style.use('fivethirtyeight')
			# index = count()

			draw.line_chart(download_list)
			df.loc[len(df.index)] = [ping_list[-1], download_list[-1], upload_list[-1]]


		

		
			first = False
	return 

if page == "View Lecture.":
	# p1 = multiprocessing.Process(name='p1', target=vid)
	# p = multiprocessing.Process(name='p', target=disp)
	# p1.run()
	# p.run()
	# p1.join()
	# p.join()

	# Using Multiple Threading for Running both the functions simultaneously.
	thread1 = Thread( target = vid)
	thread2 = Thread( target = disp)
	thread3 = Thread(target = disp)
	thread4 = Thread(target = disp)
	thread1.run()
	thread2.run()
	thread3.run()
	thread4.run()
	# thread1.join()
	thread2.join()
	thread3.join()
	thread4.join()
	
    
	# ani = FuncAnimation(plt.gcf(), animate, interval=1000)
	# animate(x_vals, download_list)

	# plt.tight_layout()
	# plt.show()



	# st.pyplot(plt.gcf())
	
	# time.sleep(0.5)

	

	
		
	
	


# if page == "Future Stock price prediction.":

# 	selected_stock = st.sidebar.selectbox('Select dataset for prediction', stocks)
# 	data = load_data(selected_stock)

# 	n_years = st.sidebar.slider('Years of prediction:', 1, 4)
# 	period = n_years * 365

# 	# Predict forecast with Prophet.
# 	df_train = data[['Date','Close']]
# 	df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
# 	df_train.dropna()

# 	m = Prophet()
# 	m.fit(df_train)
# 	future = m.make_future_dataframe(periods=period)
# 	forecast = m.predict(future)

# 	# Show and plot forecast
# 	st.subheader('Forecast data')
# 	st.write(forecast.tail())
		
# 	st.write(f'Forecast plot for {n_years} years')
# 	fig1 = plot_plotly(m, forecast)
# 	st.plotly_chart(fig1)

# 	st.write("Forecast components")
# 	fig2 = m.plot_components(forecast)
# 	st.write(fig2)

# if page == "Portfolio Reccomendations.":

# 	t = {"Oracle":'OFSS.NS', "Zee" :  'ZEEL.NS', "IDBI" : 'IDBI.NS', "Airtel" : 'BHARTIARTL.NS'}
# 	n_years = st.sidebar.slider('The Peirod of investment :', 1, 4)
# 	period = n_years * 365
# 	amount = st.sidebar.text_input('Enter the Amount you want to invest(Rs./anum).')

# 	metrics = st.sidebar.multiselect("What Stocks would you like to invest in?", ("Oracle", "Zee", "IDBI", "Airtel"))

# 	"So the Optimal portfolio for the Target Investment of ", amount*n_years, "is being calculated."

# 	data_load_state = st.text('Predicting the Data...')
# 	portfolio = pd.DataFrame([])
# 	for i in metrics:
# 		data = load_data(t[i])
# 		df_train = data[['Date','Close']]
# 		df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
# 		m = Prophet()
# 		m.fit(df_train)
# 		future = m.make_future_dataframe(periods=period)
# 		forecast = m.predict(future)
# 		portfolio["Date"] = forecast['ds']
# 		portfolio[i] = forecast['trend']

		
# 	if metrics != []:
# 		data_load_state.text('Predicting data... done!')
# 		portfolio.set_index('Date', inplace=True)
# 		portfolio.index = pd.to_datetime(portfolio.index, errors='coerce')
# 		# st.write(portfolio.head())

# 		# Finding the Covariance Matrix.
# 		cov_matrix = portfolio.pct_change().apply(lambda x: np.log(1+x)).cov()

# 		st.write("Covariance Matrix (x10^-6) :")
# 		st.write(cov_matrix*10**6)

# 		# Finding the Corellation Matirx.
# 		corr_matrix = portfolio.pct_change().apply(lambda x: np.log(1+x)).corr()


# 		st.write("Corellation Matrix :")
# 		st.write(corr_matrix)


# 		# Yearly returns for individual companies
# 		ind_er = portfolio.resample('Y').last().pct_change().mean()

# 		# # Portfolio returns
# 		# w = [0.1, 0.2, 0.5, 0.2]
# 		# port_er = (w*ind_er).sum()

# 		# Volatility is given by the annual standard deviation. We multiply by 250 because there are 250 trading days/year.
# 		ann_sd = portfolio.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))

# 		assets = pd.concat([ind_er, ann_sd], axis=1) # Creating a table for visualising returns and volatility of assets
# 		assets.columns = ['Returns', 'Volatility']

# 		p_ret = [] # Define an empty array for portfolio returns
# 		p_vol = [] # Define an empty array for portfolio volatility
# 		p_weights = [] # Define an empty array for asset weights

# 		num_assets = len(portfolio.columns)
# 		num_portfolios = 10000

# 		for _ in range(num_portfolios):
# 			weights = np.random.random(num_assets)
# 			weights = weights/np.sum(weights)
# 			p_weights.append(weights)
# 			returns = np.dot(weights, ind_er) # Returns are the product of individual expected returns of asset and its 
# 											# weights 
# 			p_ret.append(returns)
# 			var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
# 			sd = np.sqrt(var) # Daily standard deviation
# 			ann_sd = sd*np.sqrt(250) # Annual standard deviation = volatility
# 			p_vol.append(ann_sd)

# 		data = {'Returns':p_ret, 'Volatility':p_vol}

# 		for counter, symbol in enumerate(portfolio.columns.tolist()):
# 			#print(counter, symbol)
# 			data[symbol+' weight'] = [w[counter] for w in p_weights]

# 		portfolios  = pd.DataFrame(data)

# 		st.write(portfolios.head())

# 		# Plot efficient frontier
		
# 		# portfolios.plot.scatter(x='Volatility', y='Returns', marker='o', s=10, alpha=0.3, grid=True, figsize=[10,10])

# 		# Plot raw data
# 		def plot_data():
# 			# Plotting optimal portfolio

# 			fig = plt.figure(figsize=(10, 10))
# 			ax = fig.gca()
# 			ax.scatter(portfolios['Volatility'], portfolios['Returns'], marker='o', s=10, alpha=0.3)
# 			ax.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)
# 			ax.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=500)
# 			st.pyplot(fig)

# 		min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
# 		# idxmin() gives us the minimum value in the column specified.  
# 		"Minimum Volume Portfolio :"                             
# 		min_vol_port*100
		

# 		# Finding the optimal portfolio
# 		rf = 0.01 # risk factor
# 		optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()]
# 		"Optimal Risky Portfolio : "
# 		optimal_risky_port*100


# 		"So the Money investment portfolio will be as follows : "
# 		"For Minimum Volume Portfolio :"
# 		"Investing total of ", amount

# 		for i in metrics:
# 			i, ": Rs. ", float(min_vol_port[str(i) + ' weight'])*int(amount), " per Year"

# 		"For Optimal Risky Portfolio :"
# 		"Investing total of ", amount

# 		for i in metrics:
# 			i, ": Rs. ", float(optimal_risky_port[str(i) + ' weight'])*int(amount), " per Year"
		

# 		plot_data()



# 	else:
# 		data_load_state.text('Please Select at least one stock.')

