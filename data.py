import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

from bokeh.io import curdoc, push_notebook, output_notebook
from bokeh.layouts import column, layout
from bokeh.models import ColumnDataSource, Div, Select, Slider, TextInput, HoverTool
from bokeh.plotting import figure, show
from ipywidgets import interact, interactive, fixed, interact_manual

import numpy as np


from bing_image_urls import bing_image_urls


print(bing_image_urls("bear", limit=2))





