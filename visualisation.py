import os
import matplotlib.pyplot as plt
import plotly.tools as pls
import matplotlib.animation as animation
from glob import glob
import pandas as pd
df= pd.read_excel('results/20180401.xlsx')[:100]
df.columns=['rain']


def update(args):
    img= args[0]
    _df= args[1]
    ax[0].imshow(img)
    ax[1].plot(_df.index, _df.rain, c='r')
    ax[1].set_xlim([df.index[0], df.index[-1]])
    ax[1].set_ylim([df.rain.min(), df.rain.max()])

def frame():
    imgs= glob(os.path.join('/Users/allen/Documents/Python/AI/DeepLearning/FaceRecognition/datasets/allen','*.png'))
    ind=-1
    while True:
        ind+=1
        if ind>=100:
            break
        yield (plt.imread(imgs[ind]),df.iloc[:ind+1])


fig, ax= plt.subplots(2,1,figsize=(10,5))
ani1 = animation.FuncAnimation(fig,update,frame, interval=60)
# ani2 = animation.FuncAnimation(fig,update_ts,ts,interval=50)
plt.show()
