# pip install streamlit pystan = 2.19.9.1 fbprophet yfinance plotly
import streamlit as st
from datetime import date

from threading import Thread
import threading
import multiprocessing

# Load Packages
import numpy as np
import pandas as pd
from pandas_datareader import data
from datetime import date

import matplotlib.pyplot as plt

import VideoPlaylist

try:
    from streamlit.ReportThread import add_report_ctx
    from streamlit.server.Server import Server
except Exception:
    # Streamlit >= 0.65.0
    from streamlit.report_thread import add_report_ctx
    from streamlit.server.server import Server

import altair as alt


from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
# from matplotlib.animation import FuncAnimation

from youtubesearchpython import VideosSearch

from streamlit_player import st_player

import os
import logging
from itertools import count

from PIL import Image

import speedtest

import time


# Setting up the logging file.
def setup_logging():
  logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt = "%Y-%m-%d %H:%M",
)

# Findiing the speed result.
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

# Reading in the log file.
@st.cache
def read_data(path):
  df = pd.io.parsers.read_csv(
    path,
    names='date time ping download upload'.split(),
    header=None,
    sep=r'\s+',
    parse_dates={'timestamp':[0,1]},
    na_values=['TEST','FAILED'],
  )
  return df

@st.cache
def animate(x_vals, y_vals):
    x_vals.append(next(index))

    plt.cla()

    plt.plot(x_vals, y_vals)

    plt.tight_layout()

def vid():
	global url 
	global t
	global Vids
	global vid_search
	global displayed
	
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
		
		displayed = True

	return 




def disp(draw):
	global x_vals
	global ping_list
	global download_list
	global upload_list
	global df
	global bd_monitor
	
	if vid_search != '':

		first = True

		# draw = st.line_chart(download_list)
		# data = st.sidebar.table(download_list)

		# col1, col2, col3 = st.columns(3)

		# col1.metric(label = "Ping", value = "%.2f"%ping_list[-1] + " ms", delta = str(round(ping_list[-1] - ping_list[-2], 2)))
		# col2.metric(label = "Download", value = "%.2f"%download_list[-1]+ " Mbps " , delta = str(round(download_list[-1] - download_list[-2], 2)))
		# col3.metric(label = "Upload", value =  "%.2f"%upload_list[-1] + " Mbps", delta = str(round(upload_list[-1] - upload_list[-2], 2)))
		
		while bd_monitor:

			try:
				# Waiting for the Test Result
				# st.heading('Waiting for the Test Result.')
				# if first:
					# st.subheader('Waiting for the Test Result.')
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

			disp_df.dataframe(df.tail())

			# plt.style.use('fivethirtyeight')
			# index = count()

			draw.line_chart(download_list)
			df.loc[len(df.index)] = [ping_list[-1], download_list[-1], upload_list[-1]]


		

		
			first = False
	return 





# All the required Variables.

url = []
Vids = []
t = {}
displayed = False

SPEEDTEST_CMD = 'python speedtest-cli/speedtest.py'

LOG_FILE = 'speedtest.log'

x_vals = []
ping_list = []
download_list = []
upload_list = []
df = pd.DataFrame({'Ping':[],
							'Download':[],
							'Upload':[]})



# Starting with the Main App.

st.title('Adaptive Bit-rate Scheme.')

# Create a page dropdown 
page = st.sidebar.selectbox("Choose your page", ["View Lecture.", "Future Bandwidth prediction."]) 


if page == "View Lecture.":

	# Displaying the df.

	disp_df = st.sidebar.dataframe(df)

	vid_search = st.text_input("Search a Youtube Video")
	videosSearch = VideosSearch(vid_search, limit = 10)
	

	# Using Multiple Threading for Running both the functions simultaneously.
	thread1 = Thread( target = vid)
	thread2 = Thread(target=lambda: disp(draw))
	thread3 = Thread(target=lambda: disp(draw))
	thread4 = Thread(target=lambda: disp(draw))
	thread5 = Thread(target=lambda: disp(draw))
	thread6 = Thread(target=lambda: disp(draw))

	bd_monitor = st.checkbox('Monitor Band Width') # Monitors the Bandwidth.
	add_report_ctx(thread1)
	thread1.start()
	# thread1.join()
	

	if bd_monitor:
		st.subheader('Monitoring Bandwidth.')
		draw = st.line_chart(download_list)
		thread2.start()
		add_report_ctx(thread2)
		time.sleep(4)
		add_report_ctx(thread3)
		thread3.start()
		time.sleep(4)
		add_report_ctx(thread4)
		thread4.start()
		time.sleep(4)
		add_report_ctx(thread5)
		thread5.start()
		time.sleep(4)
		add_report_ctx(thread6)
		thread6.start()
		thread = st.subheader('The Threads are being initiated.')
		# thread.subheader('Thread2 Running.')
		thread2.join()
		# thread.subheader('Thread3 Running.')
		thread3.join()
		# thread.subheader('Thread4 Running.')
		thread4.join()
		thread5.join()
		thread6.join()



if page == "Future Bandwidth prediction.":
	data = read_data('speedtest.log')
	data = data[6500:7000]
	data = data.fillna(0)
	# st.dataframe(data)
	time = st.sidebar.slider('Hours of Prediction:', 1, 4)
	period = time * 60

	draw = st.line_chart(data['download'])

	# Predict forecast with Prophet.
	df_train = data[['timestamp', 'download']]
	st.sidebar.dataframe(df_train)
	df_train.rename(columns = {'timestamp':'ds', 'download':'y'}, inplace = True)
	# df.columns = ['ds', 'y']
	df_train.dropna()

	m = Prophet()
	m.fit(df_train)
	future = m.make_future_dataframe(periods=period)
	forecast = m.predict(future)

	# Show and plot forecast
	st.subheader('Forecast data')
	st.write(forecast.tail())
		
	st.write(f'Forecast plot for {time} hours')
	fig1 = plot_plotly(m, forecast)
	st.plotly_chart(fig1)

	st.write("Forecast components")
	fig2 = m.plot_components(forecast)
	st.write(fig2)

