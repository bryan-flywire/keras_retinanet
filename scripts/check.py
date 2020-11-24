#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

if __name__=="__main__":
    data=pd.read_csv('annotations.csv', header=None, names=['vid_path', 'x1','y1', 'x2','y2', 'species'])
    for idx,row in data.iterrows():
        rel = os.path.relpath(row.vid_path, '/data')
        image_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), rel)
        image = plt.imread(image_path)
        plt.imshow(image)
        x=np.array([row.x1, row.x2])
        y=np.array([row.y1, row.y2])
        plt.plot(x,y, 'r')
        plt.title(row.vid_path)
        plt.show()
