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
parser.add_argument('--date',type=str,default='20180401', help='specify the date want to produce .gif')
opt= parser.parse_args()

df= pd.read_csv('rainfall.csv')
df.columns=['time','rain']
df.time= df.time.map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
df.set_index('time', inplace=True)
df[df<0]= np.nan
date= opt.date
df_date= df[df.index.strftime('%Y%m%d')==date]

def read_video(video):
    cap= cv2.VideoCapture(video)
    Frames=[]
    while True:
        ret, frame= cap.read()
        if not ret:
            break
        frame= frame[:,:,::-1] # convert bgr to rgb
        yield frame


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
        if video_ind>= tot_videos:
            break
        video= videos_path[video_ind]
        for frame in read_video(video):
            print(ind,'/', len(df_date))
            ind+=1
            yield cv2.resize(frame,(480,270)), df_date.iloc[:ind+1]


def update(args):
    img= args[0]
    _df= args[1]
    # two axes that represent image and corresponding rainfall intensity respectively
    ax[0].imshow(img)
    ax[1].plot(_df.index, _df.rain, color='red')
    ax[1].set_ylim([df_date.rain.min(), df_date.rain.max()])
    # ax[1].set_ylim([_df.rain.min(), _df.rain.max()])
    ax[1].set_xlim([df_date.index[0], df_date.index[-1]])
    # ax[1].set_xlim([df_date.index[0], df_date.index[310]])
    ax[1].set_xlabel('datetime')
    ax[1].set_ylabel('rainfall intensity (mm/hour')
    ax[1].xaxis.set_major_formatter(myFmt)
    
myFmt = mdates.DateFormatter('%H:%M:%S')
fig, ax= plt.subplots(2,1,figsize=(8,6),gridspec_kw = {'height_ratios':[3, 1]})
ani = animation.FuncAnimation(fig,update,frame,save_count=len(df_date), interval=0)
# ani2 = animation.FuncAnimation(fig,update_ts,ts,interval=50)

ani.save(f'{date}-demo.gif', writer='imagemagick', fps=30,bitrate=50)
# plt.show()
