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

from sklearn.pipeline import Pipeline




product = pd.read_csv('cosmetic_p.csv')

for i in range(len(product['ingredients'])):
    product['ingredients'].iloc[i] = str(product['ingredients'].iloc[i]).replace('[', '').replace(']', '').replace("'", '').replace('"', '')


all_ingreds = []

for i in product['ingredients']:
    ingreds_list = i.split(', ')
    for j in ingreds_list:
        all_ingreds.append(j)

all_ingreds.remove('')

for i in range(len(all_ingreds)):
    if all_ingreds[i] == '':
        continue 
    if all_ingreds[i][-1] == ' ':
        all_ingreds[i] = all_ingreds[i][0:-1]

    if all_ingreds[i][0:2] == '* ' or all_ingreds[i][0:2] == '**':
        all_ingreds[i] = all_ingreds[i][2:]


    if all_ingreds[i][0:3] == '***' or all_ingreds[i][0:3] == '** ':
        all_ingreds[i] = all_ingreds[i][3:] 

    if all_ingreds[i][0] == '*':
        all_ingreds[i] = all_ingreds[i][1:]

        
all_ingreds = sorted(set(all_ingreds))


one_hot_list = [[0] * 0 for i in range(len(all_ingreds))]

for i in product['ingredients']:
    k=0
    for j in all_ingreds:
        if j in i:
            one_hot_list[k].append(1)
        else:
            one_hot_list[k].append(0)
        k+=1
        
ingred_matrix = pd.DataFrame(one_hot_list).transpose()
ingred_matrix.columns = [sorted(set(all_ingreds))]

estimators = [('SVD', TruncatedSVD(n_components=150, n_iter = 1000, random_state = 6))]
pipe = Pipeline(estimators)

#svd = TruncatedSVD(n_components=150, n_iter = 1000, random_state = 6) # firstly reduce features to 150 with truncatedSVD - this suppresses some noise
svd_features = pipe.fit_transform(ingred_matrix)
tsne = TSNE(n_components = 2, n_iter = 1000000, random_state = 6) # reduce 150 features to 2 using t-SNE with exact method
tsne_features = tsne.fit_transform(svd_features)

product['X'] = tsne_features[:, 0]
product['Y'] = tsne_features[:, 1]

product.to_csv('product.csv')
ingred_matrix.to_csv('matrix.csv') 






    #plot(product=product) 

cs_list = []
brands = []
output = []
binary_list = []

items = product[product['Label'] == 'Moisturizer']
items = items[items['price'] <= 100]
items = items[items['Combination'] == 0]
items = items[items['Dry'] == 1]
items = items[items['Normal'] == 0]
items = items[items['Oily'] == 0]
items = items[items['Sensitive'] == 1]


idx = items.index.to_list()


for index in idx:
    for i in ingred_matrix.iloc[index][1:]:
        binary_list.append(i)  

point1 = np.array(binary_list).reshape(1, -1)
point1 = [val for sublist in point1 for val in sublist]
point1 = point1[:6522]


prod_type = product['Label'][product['price'] <= 100].iat[0]
brand_search = product['brand'][product['price'] <= 100].iat[0]
product_by_type = product[product['Label'] == prod_type]

for j in range(product_by_type.index[0], product_by_type.index[0] + len(product_by_type)):
    binary_list2 = []
    for k in ingred_matrix.iloc[j][1:]:
        binary_list2.append(k)
    point2 = np.array(binary_list2).reshape(1, -1)
    point2 = [val for sublist in point2 for val in sublist]
    idx = min(len(point1), len(point2))
    point1 = point1[:idx]
    point2 = point2[:idx] 
    dot_product = np.dot(point1, point2)
    norm_1 = np.linalg.norm(point1)
    norm_2 = np.linalg.norm(point2)
    cos_sim = dot_product / (norm_1 * norm_2)
    cs_list.append(cos_sim)
    

product_by_type = pd.DataFrame(product_by_type)
product_by_type['cos_sim'] = cs_list
product_by_type = product_by_type.sort_values('cos_sim', ascending=False)
#product_by_type = product_by_type[product_by_type.name != search] 
l = 0
for m in range(len(product_by_type)):
    brand = product_by_type['brand'].iloc[l]
    if len(brands) == 0:
        if brand != brand_search:
            brands.append(brand)
            output.append(product_by_type.iloc[l])
    elif brands.count(brand) < 2:
        if brand != brand_search:
            brands.append(brand)
            output.append(product_by_type.iloc[l])
    l += 1
    
print('\033[1m', 'Recommending products similar to',':', '\033[0m'), print(pd.DataFrame(output)[['name', 'cos_sim']].head(5))
print(pd.DataFrame(output)[['name', 'cos_sim']].head(5))



