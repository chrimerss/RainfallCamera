import os
import matplotlib.pyplot as plt
import plotly.tools as pls
import matplotlib.animation as animation
from glob import glob
import pandas as pd
import numpy as np
import datetime
import cv2
import argparse
import matplotlib.dates as mdates

parser= argparse.ArgumentParser('settings')
parser.add_argument('--date',type=str,default='20181208', help='specify the date want to produce .gif')
opt= parser.parse_args()

df= pd.read_csv('./validation/all_data_4_events_new.csv')
# df.columns=['time','rain']
df.date= df.date.map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
df.set_index('date', inplace=True)
df[df<0]= np.nan
date= opt.date
df_date= df[df.index.strftime('%Y%m%d')==date]

def read_video(video):
    cap= cv2.VideoCapture(video)
    Frames=[]
    ind=-1
    while True:
        ret, frame= cap.read()
        ind+=1
        if not ret:
            break
        frame= frame[:,:,::-1] # convert bgr to rgb

        yield ind,frame


def frame():
    # prepare dataframe
    # date: str, "20180401"-like
    if not isinstance(date, str) or len(date)!=8:
        raise ValueError('input format is not correct, expect format %Y%m%d')
    video_path= os.path.join('videos',date,'')
    videos_path= glob(video_path+'*.mkv')
    ind=-1
    video_ind=-1
    tot_videos= len(videos_path)
    while True:
        video_ind+=1
        print(video_ind,'/',tot_videos)
        if video_ind>= 2:
            break
        video= videos_path[video_ind]
        for i, frame in read_video(video):
            print(ind,'/', len(df_date))
            ind+=1
            yield cv2.resize(frame,(480,270)), df_date.iloc[:ind+1,:]

def mv(series, window=30):
    """
    Applies simple moving average
    Input: 

    series: pd.series
    window: seconds, determine the stride

    Return:

    results: pd.series
    """
    if not isinstance(series, pd.Series):
        raise ValueError('expect input pandas series dataframe, but get %s'%(type(series)))

    return series.rolling(window=window, min_periods=1).mean()

def update(args):
    img= args[0]
    _df= args[1]
    # two axes that represent image and corresponding rainfall intensity respectively
    frame_axes.imshow(img)
    frame_axes.axis('off')
    frame_axes.set_title(_df.index[-1].strftime("%Y-%m-%d %H:%M:%S"))

    ts_axes.plot(_df.index, _df.camera, label='camera', color='red', marker='_')
    ts_axes.plot(_df.index, _df.gauge, label='gauge', color='blue')
    ts_axes.plot(_df.index, mv(_df.camera),label='moving average', color='black')
    ts_axes.legend(['camera','gauge','moving average'],prop={'size': 5})
    ts_axes.set_ylim([min(df_date.min()[['camera','gauge']]), max(df_date.max()[['camera','gauge']])])
    # ax[1].set_ylim([_df.rain.min(), _df.rain.max()])
    ts_axes.set_xlim([df_date.index[0], df_date.index[-1]])
    # ax[1].set_xlim([df_date.index[0], df_date.index[310]])
    ts_axes.set_xlabel('datetime', fontsize=6)
    ts_axes.set_title('rainfall intensity (mm/hour)',fontsize=8)
    ts_axes.yaxis.set_label_coords(0,-0.1)
    ts_axes.xaxis.set_major_formatter(myFmt)
    ts_axes.tick_params(axis='both', which='major', labelsize=5)
    
    cts_axes.plot(_df.index,_df.camera.cumsum()/3600.,label='camera',color='red')
    cts_axes.plot(_df.index,_df.gauge.cumsum()/3600.,label='gauge',color='blue')
    cts_axes.legend(['camera','gauge'],prop={'size': 5})
    cts_axes.set_ylim([min((df_date.camera.cumsum()/3600.).min(),(df_date.gauge.cumsum()/3600.).min()),
                    max((df_date.camera.cumsum()/3600.).max(),(df_date.gauge.cumsum()/3600.).max())])
    cts_axes.set_xlim([df_date.index[0], df_date.index[-1]])
    cts_axes.set_xlabel('datetime', fontsize=6)
    cts_axes.set_title('cumulative rainfall intensity (mm)', fontsize=8)
    cts_axes.yaxis.set_label_coords(0,-0.1)
    cts_axes.xaxis.set_major_formatter(myFmt)
    cts_axes.tick_params(axis='both', which='major', labelsize=5)

myFmt = mdates.DateFormatter('%H:%M:%S')
fig= plt.figure(figsize=(10,6))
grid= plt.GridSpec(2,8,hspace=0.6,wspace=0.4)
frame_axes= fig.add_subplot(grid[:,0:5])
ts_axes= fig.add_subplot(grid[:1,5:])
cts_axes= fig.add_subplot(grid[1:,5:])
# fig, ax= plt.subplots(3,1,figsize=(8,6),gridspec_kw = {'height_ratios':[3, 1, 1]})
ani = animation.FuncAnimation(fig,update,frame,save_count=len(df_date), interval=0)
# ani2 = animation.FuncAnimation(fig,update_ts,ts,interval=50)

# ani.save(f'{date}-demo.gif', writer='imagemagick', fps=30,bitrate=10)
ani.save(f'{date}-demo.mp4', writer='ffmpeg', fps=30)
# plt.show()
