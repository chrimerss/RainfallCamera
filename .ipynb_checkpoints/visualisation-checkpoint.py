import os
import matplotlib.pyplot as plt
import plotly.tools as pls
import matplotlib.animation as animation
from glob import glob
import pandas as pd
import numpy as np
import datetime
import cv2
df= pd.read_csv('rainfall.csv')
df.columns=['time','rain']
df.time= df.time.map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
df.set_index('time', inplace=True)
df[df<0]= np.nan
date= '20180401'
df_date= df[df.index.strftime('%Y%m%d')==date]

def read_video(video):
    cap= cv2.VideoCapture(video)
    while True:
        ret, frame= cap.read()
        frame= frame[:,:,::-1] # convert bgr to rgb
        if not ret:
            break
        yield frame

def frame():
    # prepare dataframe
    # date: str, "20180401"-like
    if not isinstance(date, str) or len(date)!=8:
        raise ValueError('input format is not correct, expect format %Y%m%d')
    video_path= os.path.join('videos',date,'')
    videos_path= glob(video_path+'*.mkv')
    ind=0
    video_ind=-1
    tot_videos= len(video_path)
    while True:
        video_ind+=1
        video= videos_path[video_ind]
        if video_ind> tot_videos:
            break
        for frame in read_video(video):
            yield frame, df_date.iloc[:ind+1]

            
def update(args):
    img= args[0]
    _df= args[1]
    # two axes that represent image and corresponding rainfall intensity respectively
    ax[0].imshow(img)
    ax[1].plot(_df.index, _df.rain, color='red')
    ax[1].set_ylim([df_date.rain.min(), df_date.rain.max()])
    ax[1].set_xlim([df_date.index[0], df_date.index[-1]])
    

fig, ax= plt.subplots(2,1,figsize=(10,5))
ani = animation.FuncAnimation(fig,update,frame, interval=0)
# ani2 = animation.FuncAnimation(fig,update_ts,ts,interval=50)

ani.save('20180401.gif', writer='imagemagick', fps=30)
# plt.show()
