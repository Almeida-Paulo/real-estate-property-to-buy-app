from datetime import datetime
import pandas as pd
import streamlit as st
import numpy as np
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import geopandas
import plotly.express as px

st.set_page_config(layout='wide')

@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv(path)
    return data

@st.cache(allow_output_mutation=True)
def get_geofile(url):
    geofile = geopandas.read_file(url)
    return geofile

def set_date(data):
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
    return data

def set_feature(data):
    data['price_m^2'] = data['price'] / data['sqft_lot']
    return data

def data_overview(data):
    st.sidebar.title('Data Overview')
    f_attributes = st.sidebar.multiselect('Enter columns', data.columns) 

    f_zipcode = st.sidebar.multiselect(
        'Enter Zipcode', 
        data['zipcode'].unique())

    st.title('Data Overview')
    st.header('Dataset')

    if (f_zipcode != []) & (f_attributes != []): 
        data_zip = data.loc[data['zipcode'].isin(f_zipcode), f_attributes] 

    elif (f_zipcode != []) & (f_attributes == []): 
        data_zip = data.loc[data['zipcode'].isin(f_zipcode), :] 

    elif (f_zipcode == []) & (f_attributes != []): 
        data_zip = data.loc[:, f_attributes] 

    else:
        data_zip = data.copy()

    st.dataframe(data_zip)
    

    st.sidebar.title('Attributes Options')
    st.title('House Attributes')
    
    
    f_waterview = st.sidebar.checkbox('Only Houses with Waterview')
    if f_waterview:
        df_wtv = data_zip[data_zip['waterfront'] == 1]
    else:
        df_wtv = data_zip.copy()

    f_basement = st.sidebar.checkbox('Only Houses with Basement')
    df_wtv['basement'] = df_wtv['sqft_basement'].apply(lambda x : 1 if x!=0 else 0)
    if f_basement:
        df_bs = df_wtv[df_wtv['basement'] == 1]
    else:
        df_bs = df_wtv.copy()

    f_renovated = st.sidebar.checkbox('Only Houses Renovated')
    df_bs['renovated'] = df_bs['yr_renovated'].apply(lambda x : 1 if x!=0 else 0)
    if f_renovated:
        df_rv = df_bs[df_bs['renovated'] == 1]
    else:
        df_rv = df_bs.copy()

    st.subheader('Price mean per bedrooms')
    options_bed = sorted(set(df_rv['bedrooms'].unique()))
    f_bedrooms = st.sidebar.multiselect('Number of bedrooms', options_bed)
    if (f_bedrooms != []): 
        df_bed = df_rv.loc[df_rv['bedrooms'].isin(f_bedrooms)]  
    else:
        df_bed = df_rv.copy()
    bed_by_price = df_bed[['price', 'bedrooms']].groupby('bedrooms').mean()
    bed_frame = pd.DataFrame(bed_by_price)
    st.bar_chart(bed_frame, use_container_width=True)

    st.subheader('Price mean per bathrooms')
    options_bth = sorted(set(df_bed['bathrooms'].unique()))
    f_bathrooms = st.sidebar.multiselect('Number of bathrooms', options_bth)
    if (f_bathrooms != []): 
        df_bth = df_bed.loc[df_bed['bathrooms'].isin(f_bathrooms)]  
    else:
        df_bth = df_bed.copy() 
    bth_by_price = df_bth[['price', 'bathrooms']].groupby('bathrooms').mean()
    bth_frame = pd.DataFrame(bth_by_price)
    st.bar_chart(bth_frame, use_container_width=True)

    st.subheader('Price mean per floors')
    options_flr = sorted(set(df_bth['floors'].unique()))
    f_floors = st.sidebar.multiselect('Number of floors', options_flr)
    if (f_floors != []): 
        df_flr = df_bth.loc[df_bth['floors'].isin(f_floors)]  
    else:
        df_flr = df_bth.copy() 
    flr_by_price = df_flr[['price', 'floors']].groupby('floors').mean()
    flr_frame = pd.DataFrame(flr_by_price)
    st.bar_chart(flr_frame, use_container_width=True)

    options_cond = sorted(set(df_flr['condition'].unique()))
    f_condition = st.sidebar.multiselect('Condition', options_cond)
    if (f_condition != []): 
        df_cond = df_flr.loc[df_flr['condition'].isin(f_condition)]  
    else:
        df_cond = df_flr.copy()

    options_grd = sorted(set(df_cond['grade'].unique()))
    f_grade = st.sidebar.multiselect('Grade', options_grd)
    if (f_grade != []): 
        df_grd = df_cond.loc[df_cond['grade'].isin(f_grade)]  
    else:
        df_grd = df_cond.copy()
    
    options_vw = sorted(set(df_grd['view'].unique()))
    f_view = st.sidebar.multiselect('View', options_vw)
    if (f_view != []): 
        df_vw = df_grd.loc[df_grd['view'].isin(f_view)]  
    else:
        df_vw = df_grd.copy()
    

    c1, c2 = st.columns((1, 1))
    c1.header('Summary')
    df_summary = df_vw.copy()
    df_summary = df_summary.drop(['id', 'yr_renovated', 'long', 'lat', 'basement', 'renovated', 'zipcode', 'yr_renovated', 'yr_built', 'waterfront'], axis=1)
    num_attributes = df_summary.select_dtypes(include=['int64', 'float64'])
    media = pd.DataFrame(num_attributes.apply(np.mean))
    mediana = pd.DataFrame(num_attributes.apply(np.median))
    desvio = pd.DataFrame(num_attributes.apply(np.std))
    max_ = pd.DataFrame(num_attributes.apply(np.max))
    min_ = pd.DataFrame(num_attributes.apply(np.min))

    df_stats = pd.concat([media, max_, min_, desvio, mediana], axis=1).reset_index()
    df_stats.columns = ['attributes', 'mean', 'max', 'min', 'std', 'median']
    
    c1.dataframe(df_stats)

    c2.header('Summary by Zipcode')
    df1 = df_vw[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = df_vw[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = df_vw[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4 = df_vw[['price_m^2', 'zipcode']].groupby('zipcode').mean().reset_index()

    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    df = pd.merge(m2, df4, on='zipcode', how='inner') 

    df.columns = ['zipcode', 'houses', 'price mean', 'living  mean', 'price/m^2']

    c2.dataframe(df)

    st.header('Houses to buy')
    st.dataframe(df_vw.reset_index())

    return None

def commercial_distribution(data):
    st.sidebar.title('Commercial Options')
    st.title('Commercial Attributes')

    min_yr_built = int(data['yr_built'].min()) 
    max_yr_built = int(data['yr_built'].max()) 

    st.sidebar.subheader('Select max year built')
    f_yr_built = st.sidebar.slider('Year Built:', min_yr_built, max_yr_built, max_yr_built) 

    st.header('Average Price per Year Built')

    df = data[data['yr_built'] < f_yr_built] 
    df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

    fig = px.line(df, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)

    st.header('Average Price per Day')
    st.sidebar.subheader('Select max date')

    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')

    min_date = datetime.strptime(data['date'].min(), '%Y-%m-%d')
    max_date = datetime.strptime(data['date'].max(), '%Y-%m-%d')

    f_date = st.sidebar.slider('Date:', min_date, max_date, max_date)

    data['date'] = pd.to_datetime(data['date'])
    df = data[data['date'] < f_date]
    df = df[['date', 'price']].groupby('date').mean().reset_index()

    fig = px.line(df, x='date', y='price')
    st.plotly_chart(fig, use_container_width=True)

    st.header('Price Distribution')
    st.sidebar.subheader('Select max price')

    min_price = int(data['price'].min())
    max_price = int(data['price'].max())

    f_price = st.sidebar.slider('Price:', min_price, max_price, max_price)
    df = data[data['price'] < f_price]

    fig = px.histogram(df, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)

    return None

def portfolio_density(data, geofile):
    st.title('Region Overview')
    df1 = data.sample(10)
    st.header('Portfolio Density')
    density_map = folium.Map(location = [data['lat'].mean(),
                                        data['long'].mean()], 
                                        default_zoom_start = 8)

    cluster_ = MarkerCluster().add_to(density_map) 
    for name, row in df1.iterrows(): 
        folium.Marker([row['lat'], row['long']],
            popup=
            'Sold ${0} on: {1}. Features: {2} squarefeet, {3} bedrooms, {4} bathrooms, year built: {5}'.format(
                                                                                                                row['price'], 
                                                                                                                row['date'], 
                                                                                                                row['sqft_living'], 
                                                                                                                row['bedrooms'], 
                                                                                                                row['bathrooms'], 
                                                                                                                row['yr_built'])).add_to(cluster_)

    folium_static(density_map)

    st.header('Price Density')

    df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df2.columns = ['ZIP', 'PRICE']

    geofile = geofile[geofile['ZIP'].isin(df2['ZIP'].tolist())]

    region_price_map = folium.Map(location = [data['lat'].mean(),
                                            data['long'].mean()],
                                            default_zoom_start = 8)

    folium.Choropleth(data = df2,
                            geo_data = geofile,
                            columns= ['ZIP', 'PRICE'],
                            key_on = 'feature.properties.ZIP',
                            fill_color ='YlOrRd',
                            fill_opacity = 0.7,
                            line_opacity = 0.2,
                            legend_name = 'Avg Price').add_to(region_price_map)

    folium_static(region_price_map)
    
    return None




#ETL
if __name__ == "__main__":
    # Extraction
    path = 'kc_house_data.csv'
    url ='https://opendata.arcgis.com/api/v3/datasets/83fc2e72903343aabff6de8cb445b81c_2/downloads/data?format=geojson&spatialRefId=4326'

    data = get_data(path)
    geofile = get_geofile(url)

    #Trasformation
    set_date(data)
    data = set_feature(data)
    data_overview(data)
    commercial_distribution(data)
    portfolio_density(data, geofile)