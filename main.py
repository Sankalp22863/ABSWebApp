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
page = st.sidebar.selectbox("Choose your page", ["View Video.", "Future Bandwidth prediction."]) 


if page == "View Video.":
	st.sidebar.markdown("Bandwidth monitoring Web App built using Streamlit")
	st.sidebar.markdown(" ")

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

# Importing the Necessary Libraries.
import math
import tensorflow.compat.v1 as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Dense,Dropout

# Functions.

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=10):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

# preparing the dataset.
def prep_data(df, look_back):
	df = pd.DataFrame(df['download'].to_list())
	dataset = df.values
	dataset = dataset.astype('float32')
	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)
	# split into train and test sets
	train_size = int(len(dataset) * 0.8)
	test_size = len(dataset) - train_size
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
	# print(len(train), len(test))
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)
	# reshape input to be [samples, time steps, features]
	trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

	return trainX, trainY, testX, testY, scaler, dataset

def generate_model(hidd_layers, hidd_nodes, look_back):
	model = Sequential()

	# Adding the first LSTM layer and some Dropout regularisation.
	model.add(LSTM(units = hidd_nodes, return_sequences = True, input_shape = (1, look_back)))
	model.add(Dropout(0.2))

	for i in range(hidd_layers - 1):
		# Adding a hidden LSTM layer and some Dropout regularisation.
		model.add(LSTM(units = hidd_nodes, return_sequences = True))
		model.add(Dropout(0.2))
	# Adding a fourth LSTM layer and some Dropout regularisation
	model.add(LSTM(units = hidd_nodes))
	model.add(Dropout(0.2))

	# Adding the output layer
	model.add(Dense(units = 1))

	# Compiling the RNN
	model.compile(optimizer = 'adam', loss = 'mean_squared_error')

	return model

def predict(model, trainX, testX, trainY, testY, scaler):
	# make predictions
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	# print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	# print('Test Score: %.2f RMSE' % (testScore))
	col1, col2 = st.columns(2)

	col1.metric(label = "Train Score", value = "%.2f"%trainScore)
	col2.metric(label = "Test Score", value = "%.2f"%testScore)
	
	return trainPredict, testPredict
	
def disp_res(trainPredict, testPredict, dataset, look_back):
	# shift train predictions for plotting
	trainPredictPlot = np.empty_like(dataset)
	trainPredictPlot[:, :] = np.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
	# shift test predictions for plotting
	testPredictPlot = np.empty_like(dataset)
	testPredictPlot[:, :] = np.nan
	testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
	# plot baseline and predictions
	fig = plt.figure()
	# Create 1 3D subplot:
	ax = fig.add_subplot(111)
	ax.plot(scaler.inverse_transform(dataset))
	ax.plot(trainPredictPlot)
	ax.plot(testPredictPlot)
	ax.set_xlabel('Time')
	ax.set_ylabel('Bandwidth(Mbps)')
	# ax.title('Model Performance'))
	st.pyplot(fig)
	return

# Diaplaying the Progress to the User.

class CustomCallback(keras.callbacks.Callback):
	def __init__(self, no_of_epochs):
		self.progress = 0
		self.disp = ''
		self.pred = ''
		self.prog_disp = st.progress(self.progress)
		self.no_of_epochs = no_of_epochs
		return
	
	def on_train_begin(self, logs=None):
		self.disp = st.markdown("The Training has been initiated.")
		return
		
	def on_train_end(self, logs=None):
		self.disp.markdown("The Training Process has Ended.")
		return
	
	def on_epoch_begin(self, epoch, logs=None):
		self.progress += 1/self.no_of_epochs
		try:
			self.prog_disp.progress(self.progress)
		except:
			self.prog_disp.progress(1.0)
		# st.progress(self.progress)
		return 
	

if page == "Future Bandwidth prediction.":

	st.sidebar.markdown("Future Bandwidth prediction Web App built using Streamlit")
	st.sidebar.markdown(" ")

	# Choose the number of the hidden nodes.
	hidd_layers = st.sidebar.number_input('Choose the Number of the hidden Layers', min_value = 1, max_value = 5, value = 3)
	# Choose the Number of hidden nodes.
	hidd_nodes = st.sidebar.number_input('Select the Number of the Hidden Nodes', min_value = 10, max_value = 100, value = 50, step = 10)
	# Choose the Memory of the system.
	look_back = st.sidebar.number_input('Select the Memory of the system', min_value = 5, max_value = 15, value = 10)
	# Choose the Number of Epochs.
	no_of_epochs = st.sidebar.number_input('Select the No. of Epochs you have to train the model for', min_value = 50, max_value = 150, value = 100, step = 10)
	# st.dataframe(data)
	# time = st.sidebar.slider('Hours of Prediction:', 1, 4)
	# period = time * 60

	# importing the Data to work with.

	choose_dataset = st.sidebar.radio("Choose a dataset to work with :", ["Airtel Wifi Dataset.", "Mumbai Wifi Rajesh Dataset.", "Upload Your own Dataset."])

	# file_upload = st.sidebar.checkbox("Use Your Own Log File for precition.")
	file_upload = False
	if choose_dataset == "Upload Your own Dataset.":
		file_upload = True
	wrong_format = not file_upload

	if file_upload:
		try:
			uploaded_file = st.sidebar.file_uploader("Choose a file", ['.log', '.csv'])
			if uploaded_file is not None:
				data = read_data(uploaded_file)
				data = data.fillna(0)
				draw = st.line_chart(data['download'])
			else:
				st.sidebar.error("Please Choose a file.")
		except:
			st.sidebar.error("The Format of the file uploaded is wrong")
			wrong_format = True	
	if file_upload and wrong_format:
		data = read_data('Mobile_Data_4G_Airtel.log')
		data = data.fillna(0)
		draw = st.line_chart(data['download'])
	
	if choose_dataset == "Airtel Wifi Dataset.":
		data = read_data('Mobile_Data_4G_Airtel.log')
		data = data.fillna(0)
		draw = st.line_chart(data['download'])
	elif choose_dataset == "Mumbai Wifi Rajesh Dataset.":
		data = read_data('speedtest.log')
		data = data[6500:]
		data = data.fillna(0)
		draw = st.line_chart(data['download'])
	
	# Start the Computation.
	compute = st.sidebar.checkbox('Start Training the Model.')

	if compute:
		trainX, trainY, testX, testY, scaler, dataset = prep_data(data, look_back)

		model = generate_model(hidd_layers, hidd_nodes, look_back)
		# Training the Data.
		with st.spinner(text='Model Training in progress'):
			model.fit(trainX, trainY, epochs=no_of_epochs, batch_size=32, verbose=1, callbacks=[CustomCallback(no_of_epochs)])
			st.balloons()
			st.success("Model Trained Succesfully.")
		disp = st.markdown("Displaying the Resutls.")
		trainPredict, testPredict = predict(model, trainX, testX, trainY, testY, scaler)
		disp_res(trainPredict, testPredict, dataset, look_back)
		disp.markdown("Results")


	# # Predict forecast with Prophet.
	# df_train = data[['timestamp', 'download']]
	# st.sidebar.dataframe(df_train)
	# df_train.rename(columns = {'timestamp':'ds', 'download':'y'}, inplace = True)
	# # df.columns = ['ds', 'y']
	# df_train.dropna()

	# m = Prophet()
	# m.fit(df_train)
	# future = m.make_future_dataframe(periods=period)
	# forecast = m.predict(future)

	# # Show and plot forecast
	# st.subheader('Forecast data')
	# st.write(forecast.tail())
		
	# st.write(f'Forecast plot for {time} hours')
	# fig1 = plot_plotly(m, forecast)
	# st.plotly_chart(fig1)

	# st.write("Forecast components")
	# fig2 = m.plot_components(forecast)
	# st.write(fig2)

