import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import calendar
import math
import torch
import torch.nn as nn
import numpy as np
import scipy as sc
from sklearn.preprocessing import MinMaxScaler
from skimage.measure import block_reduce
from sklearn.metrics import mean_absolute_error
import io
import requests
from metpy.calc import specific_humidity_from_dewpoint
from metpy.units import units

import mysql.connector
from datetime import datetime, timedelta
from sqlalchemy import create_engine

mydb = mysql.connector.connect(
host="database-1.cccp1zhjxtzi.ap-southeast-1.rds.amazonaws.com",
user="admin",
password="Nath1234",
database= "rivercast"
)

class initiate_model():
    #IMPORTING
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # configure GPU    utilization
    device

    mydb._open_connection()
    query = "SELECT * FROM rivercast.modelData;"
    result_dataFrame = pd.read_sql(query, mydb)

    print(result_dataFrame.tail(10))

    # Specify the column to exclude (change 'column_to_exclude' to the actual column name)
    column_to_exclude = ['Date_Time', 'RF-Intensity.1']

    # Exclude the specified column
    df = result_dataFrame.drop(column_to_exclude, axis=1, errors='ignore')

    # Print the DataFrame without the excluded column

    # Now 'df' can be used as 'mainDataToDB' or for further processing

    # convert month name to integer

    # create datetime column
    df[['Year', 'Month', 'Day', 'Hour']] = df[['Year', 'Month', 'Day', 'Hour']].astype(int)
    df['Hour'] = df['Hour'].apply(lambda x: x if x < 24 else 0)

    # convert year, month, day, and hour columns into timestamp
    df['Datetime'] = df[['Year', 'Month', 'Day', 'Hour']].apply(lambda row: datetime(row['Year'], row['Month'], row['Day'], row['Hour']).isoformat(), axis=1)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format='ISO8601')

    # assign timestamps as the data frame index
    df.index = df["Datetime"]
    df = df.drop(['Datetime'], axis=1)

    # select the parameters
    df = df[['Waterlevel', 'Waterlevel.1', 'Waterlevel.2', 'Waterlevel.3', 'RF-Intensity', 'RF-Intensity.2', 'RF-Intensity.3', 'Precipitation', 'Precipitation.1', 'Precipitation.2', 'Humidity', 'Humidity.1', 'Humidity.2', 'Temperature', 'Temperature.1', 'Temperature.2']] 
    df = df.astype(np.float64)  # convert parameters into a double precision floating number

    # fill in missing values using linear interpolation
    df = df.interpolate(method='linear', limit_direction='forward')
    df = df.resample('6H').max()  # resample dataset using the max value for each 24-hours
    df = df.rolling(120).mean().dropna()  # perform moving average smoothing


    rawData = df

    # scale data
    scaler = MinMaxScaler()
    scaler.fit(df)
    # train label scaler
    label_scaler = MinMaxScaler()
    label_scaler.fit(df[['Waterlevel', 'Waterlevel.1', 'Waterlevel.2', 'Waterlevel.3']])

    scaled_ds = scaler.transform(df)
    df = pd.DataFrame(scaled_ds, columns=df.columns, index=df.index)

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len, step):
        self.data = data
        self.seq_len = seq_len
        self.step = step
        
    def __getitem__(self, index):
        in_start = index
        in_end = in_start + self.seq_len
        out_start = index + self.step
        out_end = out_start + self.seq_len
        
        inputs = self.data[in_start:in_end]
        labels = self.data[out_start:out_end, :4]
        
        return inputs, labels
    
    def __len__(self):
        return len(self.data) - (self.seq_len + self.step) + 1
    
BATCH_SIZE = 128
SEQ_LEN = 60
SEQ_STEP = 60
PRED_SIZE = 4
D_MODEL = 16
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = 2048 
DROPOUT = 0.10


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        
        return attn_probs, output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_scores, attn_output = self.scaled_dot_product_attention(Q, K, V)
        output = self.W_o(self.combine_heads(attn_output))
        return attn_scores, output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=2048):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_scores, attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return attn_scores, x

class Transformer(nn.Module):
    def __init__(self, pred_size, d_model, num_heads, num_layers, d_ff, dropout):
        super(Transformer, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, pred_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt):
        tgt_embedded = self.dropout(self.positional_encoding(tgt))

        enc_output = tgt_embedded
        for enc_layer in self.encoder_layers:
            attn_scores, enc_output = enc_layer(enc_output)

        output = self.sigmoid(self.fc(enc_output))
        return attn_scores, output

# define the model
decomposer = Transformer(
    pred_size=PRED_SIZE,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    d_ff=D_FF,
    dropout=DROPOUT
).float()

decomposer.to(initiate_model.device)


# test if the model is working properly using random values
decomposer.eval()

sample = np.random.rand(1, SEQ_LEN, D_MODEL)
output = torch.from_numpy(sample).float().to(initiate_model.device)
scores, output = decomposer(output)
output = output.detach().cpu().numpy()
scores = scores.detach().cpu().numpy()

print(output.shape)
print(scores.shape)


test_data = initiate_model.df['2021-01-01':'2023-01-01'].values

test_dataset = TimeSeriesDataset(test_data, seq_len=SEQ_LEN, step=SEQ_STEP)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=len(test_data),
    shuffle=False,
    drop_last=False
)

print(test_data.shape)

decomposer.load_state_dict(torch.load('C:\\Users\\Admin - Jordan\\Documents\\ModelAPI\\bidirectionalAPI\\transformer.pth', map_location=torch.device('cpu')))
  # load the trained model

decomposer.eval()  # set model on test mode

inputs, labels = [(inputs, labels) for _, (inputs, labels) in enumerate(test_dataloader)][0]  # fetch the test dataset

print(inputs.shape)
print(labels.shape)

mydb.close()

def forecast():
    test_data = initiate_model.df[-180:].values
    test_dates = initiate_model.df[-180:].index
    test_dates = test_dates[60:240]

    x_test = test_data[:180]
    y_label = test_data[60:]
    y_label = initiate_model.label_scaler.inverse_transform(y_label[:, :4])

    x_test = np.reshape(x_test, (1, x_test.shape[0], x_test.shape[1]))

    decomposer.eval()  # set model on test mode

    x_test = torch.from_numpy(x_test).float().to(initiate_model.device)
    attn_scores, y_test = decomposer(x_test)  # make forecast
    y_test = y_test.detach().cpu().numpy()
    y_test = np.reshape(y_test, (y_test.shape[1], y_test.shape[2]))
    y_test = initiate_model.label_scaler.inverse_transform(y_test[:, :4])


    time_steps_per_day = 4  # Assuming 4 time steps per day (6 hours per time step)
    forecast_days = 15
    
    mydb._open_connection()
    cursor = mydb.cursor()
    cursor.execute("SELECT DateTime FROM rivercast.bidirectional_waterlevel_prediction order by DateTime DESC LIMIT 1")
    lastPredDT = cursor.fetchone()[0]
    formatted_lastPredDT = lastPredDT.strftime('%Y-%m-%d %H:%M:%S')

    # Extract the forecast for the next 15 days
    forecast_values = y_test[:forecast_days * time_steps_per_day]

    # Create a DataFrame with the forecasted values and dates
    forecast_dates = pd.date_range(test_dates[-1], periods=forecast_days * time_steps_per_day + 1, freq='6H')[1:]
    forecast_df = pd.DataFrame(data=forecast_values, columns=['P.Waterlevel', 'P.Waterlevel-1', 'P.Waterlevel-2', 'P.Waterlevel-3'])
    forecast_df.insert(0, "DateTime", forecast_dates)

    matches_and_following_rows_pred = forecast_df[forecast_df['DateTime'] >= formatted_lastPredDT]



    cursor.execute("SELECT DateTime FROM rivercast.bidirectional_waterlevel_obs order by DateTime DESC LIMIT 1")
    lastTrueDT = cursor.fetchone()[0] + timedelta(hours=6)

    # Extract the forecast for the next 15 days
    true_values = y_label[-120:]

    true_dates = pd.date_range(test_dates[-120], periods=120, freq='6H')[:]
    true_df = pd.DataFrame(data=true_values ,columns=['T.Waterlevel', 'T.Waterlevel-1', 'T.Waterlevel-2', 'T.Waterlevel-3']) #converting numpy to dataframe
    true_df.insert(0, "DateTime", true_dates) #adding DateTime column

    puirpose = pd.DataFrame(data=y_label ,columns=['T.Waterlevel', 'T.Waterlevel-1', 'T.Waterlevel-2', 'T.Waterlevel-3'])

    formatted_lastTrueDT = lastTrueDT.strftime('%Y-%m-%d %H:%M:%S')

    mydb.close()

    matches_and_following_rows = true_df[true_df['DateTime'] >= formatted_lastTrueDT]

    print(matches_and_following_rows_pred[1:2])

    print(matches_and_following_rows)

    return matches_and_following_rows_pred[1:2], matches_and_following_rows



def getAttnScores():
    test_data = initiate_model.reduced_df['2023-09-27':].values
    test_dates = initiate_model.reduced_df['2023-09-27':].index
    test_dates = test_dates[60:240]

    x_test = test_data[:180]
    y_label = test_data[60:180]
    y_label = initiate_model.label_scaler.inverse_transform(y_label[:, :4])

    x_test = np.reshape(x_test, (1, x_test.shape[0], x_test.shape[1]))

    decomposer.eval()  # set model on test mode

    x_test = torch.from_numpy(x_test).float().to(initiate_model.device)
    attn_scores, y_test = decomposer(x_test)  # make forecast
    y_test = y_test.detach().cpu().numpy()
    y_test = np.reshape(y_test, (y_test.shape[1], y_test.shape[2]))
    y_test = initiate_model.label_scaler.inverse_transform(y_test[:, :4])

        # plot predictions
    for i in [0, 1, 2, 3]:
        plt.plot(np.convolve(y_test[:, i], np.ones(30), 'valid') / 30)
        plt.plot(y_label[30:, i], color='k', alpha=0.3)
        plt.show()

    # plot attention scores
    attn_scores = torch.squeeze(attn_scores, dim=0)
    attn_scores = attn_scores.detach().cpu().numpy()  # transfer output from GPU to CPU
    
    
    attention_score_images = []

    for idx, attention in enumerate(attn_scores):
        selected_attention = attention[10:]
        selected_attention = block_reduce(selected_attention, (15, 15), np.max)

        fig, ax = plt.subplots()
        ax.matshow(selected_attention, cmap='viridis')

        # Save the plot to a BytesIO object
        image_stream = io.BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)

        # Append the image stream to the list
        attention_score_images.append(image_stream)

    # Return the list of attention score images
    return attention_score_images