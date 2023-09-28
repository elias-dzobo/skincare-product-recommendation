import streamlit as st 
import pandas as pd
import numpy as np 
from PIL import Image
import requests 
from streamlit_card import card 
from bing_image_urls import bing_image_urls


def get_img(name):


    img = bing_image_urls(name, limit=1)

    return img 

def recommend_product(product_type, price, skin_type, condition):
    cs_list = []
    brands = []
    output = []
    binary_list = []

    product = pd.read_csv('product.csv')
    ingred_matrix = pd.read_csv('matrix.csv')

    items = product[product['Label'] == product_type]
    items = items[items['price'] <= price]
    items = items[items['Combination'] == skin_type[0]]
    items = items[items['Dry'] == skin_type[1]]
    items = items[items['Normal'] == skin_type[2]]
    items = items[items['Oily'] == skin_type[3]]
    items = items[items['Sensitive'] == condition]


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
        
    #print('\033[1m', 'Recommending products similar to',':', '\033[0m'), print(pd.DataFrame(output)[['name', 'cos_sim']].head(5))
    return(pd.DataFrame(output)[['name', 'cos_sim']].head(5))








st.title('Skincare Product Recommendation') 

product_options = ['Moisturizer', 'Cleanser', 'Treatment', 'Face Mask', 'Eye cream','Sun protect']

binary_types = []
binary_condition = 0

st.write('What Skincare Product do you want?')
product_choice = st.selectbox('Choose a Product', product_options) 


skin_type = ['Combination', 'Dry', 'Normal', 'Oily']
skin_choice = st.selectbox('Choose your skin type', skin_type)

if skin_choice == 'Combination':
    st.write('Write your skin combination from the choices {Dry, Normal, Oily}')

    types = st.text_input('Enter your combination from the options and separate by commas') 

    types_list = types.split(',')
    binary_types.append(1)

    for val in skin_type[1:]:
        if val not in types_list:
            binary_types.append(0)
        else:
            binary_types.append(1)

else:
    if skin_choice == 'Dry':
        binary_types = [0, 1, 0, 0]
    elif skin_choice == 'Normal':
        binary_types = [0, 0, 1, 1]
    elif skin_choice == 'Oily':
        binary_types = [0, 0, 0, 1] 
    

sensitive_skin = ['Yes', 'No']
skin_sensitive = st.selectbox('Do you have sensitive skin?', sensitive_skin)

if skin_sensitive == 'Yes':
    binary_condition = 1
else:
    binary_condition = 0

price = st.number_input('What is your budget?')
price = int(price)



if st.button('Recommend'):
    data = recommend_product(product_choice, price, binary_types, binary_condition)

    data['images'] = data.apply(lambda row: get_img(row['name']), axis=1)

    
    for i in range(len(list(data['name']))):
        new_card = card(
            title= data.iloc[i]['name'],
            text='',
            image= data.iloc[i]['images'] 
        )


