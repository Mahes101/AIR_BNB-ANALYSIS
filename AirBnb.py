import streamlit as st
from plotly import express as px
from streamlit_option_menu import option_menu
import seaborn as sns
import matplotlib.pyplot as plt
from shapely.geometry import Point,Polygon
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#matplotlib inline

from wordcloud import STOPWORDS, WordCloud

import warnings as wr
wr.filterwarnings('ignore')

import PIL
from PIL import Image


#["PYTHON LIBRARIES FOR TABULAR DATA AND FILE HANDLING"]
import pandas as pd
import numpy as np

st.set_option('deprecation.showPyplotGlobalUse', False)



# ***** STREAMLIT PAGE ICON ***** 

icon = Image.open("C:/Users/mahes/Downloads/R.png")
# SETTING PAGE CONFIGURATION...........
st.set_page_config(page_title='AIRBNB',page_icon=icon,layout="wide")
style = "<style>h2 {text-align: center;}</style>"
style1 = "<style>h3 {text-align: left;}</style>"


coll1,coll2,coll3= st.columns(3)
with coll1:
        st.header(':red[AIR BNB]')
        st.markdown(style, unsafe_allow_html=True)
with coll2:
        
        st.image(Image.open("C:\\Users\\mahes\\Downloads\\568374.png"),width=150)
        
with coll3:
        st.header(':red[ANALYSIS]')
        st.markdown(style, unsafe_allow_html=True)
# OPTION MENU FOR CHOICES OF VIEWS
selected = option_menu(None,
                       options = ["Home","EDA Analysis","Insights","Dashboard"],
                       icons = ["house-door-fill","bar-chart-line-fill","lightbulb","rocket-takeoff-fill"],
                       default_index=0,
                       orientation="horizontal",
                       styles={"container": {"width": "100%"},
                               "icon": {"color": "white", "font-size": "24px"},
                               "nav-link": {"font-size": "24px", "text-align": "center", "margin": "-2px"},
                               "nav-link-selected": {"background-color": "#ff3800"}})


def home_menu():
        col1,col2 = st.columns(2)
        with col1:
                st.image(Image.open("C:\\Users\\mahes\\Downloads\\Airbnb-profile_new.jpg"),width=600)
                st.markdown("## :red[Done by] : UMAMAHESWARI S")
                st.markdown(style,unsafe_allow_html=True)
                st.markdown(":red[Githublink](https://github.com/mahes101)")
                
                
        with col2:
                st.title(':red[AIRBNB ANALYSIS]')  
                st.header(':red[AIRBNB]')  
                st.markdown(style, unsafe_allow_html=True)    
                st.write("Airbnb, Inc is an American company operating an online marketplace for short- and long-term homestays and experiences. The company acts as a broker and charges a commission from each booking.")
                st.markdown(style1, unsafe_allow_html=True)
                st.header(':red[SKILLS OR TECHNOLOGIES]')
                st.markdown(style, unsafe_allow_html=True)
                st.write("Python scripting, Data Preprocessing, Visualization, EDA, Streamlit, MongoDb, PowerBI or Tableau")
                st.markdown(style1, unsafe_allow_html=True)
                st.header(':red[DOMAIN]')
                st.markdown(style, unsafe_allow_html=True)
                st.write("Travel Industry, Property Management and Tourism")
                st.markdown(style1, unsafe_allow_html=True)
def load_data():
        df_air=pd.read_csv('C:\\Users\\mahes\\OneDrive\\Desktop\\data set\\Air bnb\\Airbnb_data_from_db.csv')
        df_air=pd.DataFrame(df_air)  
        return df_air
df = load_data()   
def univariate_analysis(df):
        opt = ["Numerical columns","Categorical columns"]
        option = st.sidebar.selectbox("Select an option",opt)
        #Getting numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if option == "Numerical columns":        
                # Univariate Analysis for numerical column        
                for Price in numeric_cols:
                        print('Skew :', round(df[Price].skew(), 2))
                        fig = plt.figure(figsize = (15, 4))
                        plt.subplot(1, 2, 1)
                        df[Price].hist(grid=False)
                        plt.ylabel('count')
                        plt.subplot(1, 2, 2)
                        sns.boxplot(x=df[Price])
                        st.pyplot(fig)

        elif option == "Categorical columns":        
                #univariate Analysis for categorical column
                fig, axes = plt.subplots(3, 2, figsize = (18, 18))
                fig.suptitle('Bar plot for all categorical variables in the dataset')
                sns.countplot(ax = axes[0, 0], x = 'Property_type', data = df, color = 'blue', 
                        order = df['Property_type'].value_counts().index)
                sns.countplot(ax = axes[0, 1], x = 'Room_type', data = df, color = 'blue', 
                        order = df['Room_type'].value_counts().index)
                sns.countplot(ax = axes[1, 0], x = 'Bed_type', data = df, color = 'blue', 
                        order = df['Bed_type'].value_counts().index)
                sns.countplot(ax = axes[1, 1], x = 'Country', data = df, color = 'blue', 
                        order = df['Country'].value_counts().index)
                sns.countplot(ax = axes[2, 0], x = 'Street', data = df, color = 'blue', 
                        order = df['Street'].head(20).value_counts().index)

                axes[1][1].tick_params(labelrotation=45)
                axes[2][0].tick_params(labelrotation=90)
                axes[2][1].tick_params(labelrotation=90);  
                st.pyplot(fig)      
                        
def bivariate_analysis():
        opt1 = ["Numerical columns","Categorical columns"]
        option1 = st.sidebar.selectbox("Select an option",opt1)
        if option1 == "Numerical columns":
                #bivariate analysis for numerical features.
                #Plotting a pair plot for bivariate analysis
                g = sns.PairGrid(df,vars=['Price','Min_nights','Max_nights','Availability_365','No_of_reviews','Review_scores'])
                #setting color
                g.map_upper(sns.scatterplot, color='crimson')
                g.map_lower(sns.scatterplot, color='limegreen')
                g.map_diag(plt.hist, color='orange')
                #show figure
                st.pyplot(g)  
        elif option1 == "Categorical columns":
                #Categorical Features Analysis.
                #setting the figure size size and fontsize
                fig = plt.figure(figsize=(14,8))
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                ax = plt.axes()
                ax.set_facecolor("black")
                #Plotting violin graph to show the relationship between catgorical feature vs price numeric column
                sns.violinplot(x=df['Country'],y=df['Price'],hue=df['Room_type'])   
                st.pyplot(fig)
        
def correlation_plot():
        #setting the figure size and fontsize
        fig = plt.figure(figsize=(14,8))
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        #visualzing the corelation between all numeric features
        sns.heatmap(df.drop(['Id','Host_id','Is_location_exact','Longitude','Latitude'],axis=1).corr(),annot=True,cmap='RdPu')   
        st.pyplot(fig)
def geospacial_visualization():
        # GETTING USER INPUTS
        country = st.sidebar.multiselect('Select a Country',sorted(df.Country.unique()),sorted(df.Country.unique()))
        property_type = st.sidebar.multiselect('Select Property_type',sorted(df.Property_type.unique()),sorted(df.Property_type.unique()))
        room_type = st.sidebar.multiselect('Select Room_type',sorted(df.Room_type.unique()),sorted(df.Room_type.unique()))
        price = st.sidebar.slider('Select Price',df.Price.min(),df.Price.max(),(df.Price.min(),df.Price.max()))
        
        # CONVERTING THE USER INPUT INTO QUERY
        query = f'Country in {country} & Room_type in {room_type} & Property_type in {property_type} & Price >= {price[0]} & Price <= {price[1]}'
        
        # AVG PRICE IN COUNTRIES SCATTERGEO
        country_df = df.query(query).groupby('Country',as_index=False)['Price'].mean()
        fig = px.scatter_geo(data_frame=country_df,
                                locations='Country',
                                color= 'Price', 
                                hover_data=['Price'],
                                locationmode='country names',
                                size='Price',
                                title= 'Avg Price in each Country',
                                color_continuous_scale='agsunset',
                                projection='equirectangular'
                        )
        st.plotly_chart(fig,use_container_width=True)
        
        # AVG AVAILABILITY IN COUNTRIES SCATTERGEO
        country_df = df.query(query).groupby('Country',as_index=False)['Availability_365'].mean()
        country_df.Availability_365 = country_df.Availability_365.astype(int)
        fig = px.scatter_geo(data_frame=country_df,
                                        locations='Country',
                                        color= 'Availability_365', 
                                        hover_data=['Availability_365'],
                                        locationmode='country names',
                                        size='Availability_365',
                                        title= 'Avg Availability in each Country',
                                        color_continuous_scale='agsunset',
                                        projection='equirectangular'
                                )
        st.plotly_chart(fig,use_container_width=True)
            
        
def house_rules(df):
        house_mask = np.array(Image.open('C:/Users/mahes/Downloads/house1.jpg'))
        text = df['House_rules'].dropna()
        text = text.apply(lambda x: x.lower())
        new_words = ['please','available','will',"don't"]
        text_1 = ' '.join(i for i in text)
        stopwords = list(STOPWORDS) + new_words
        wordcloud = WordCloud(stopwords=stopwords, background_color='#1C1C1C', mask=house_mask, colormap = 'twilight_shifted_r').generate(text_1)
        g1 = plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(g1)
def amenities(df):
        house_mask = np.array(Image.open('C:/Users/mahes/Downloads/house1.jpg'))
        text = df['Amenities'].dropna()
        text = text.apply(lambda x: x.lower())
        new_words = ['please','available','will',"don't"]
        text_1 = ' '.join(i for i in text)
        stopwords = list(STOPWORDS) + new_words
        wordcloud = WordCloud(stopwords=stopwords, background_color='#1C1C1C', mask=house_mask, colormap = 'twilight_shifted_r').generate(text_1)
        g2 = plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(g2) 
def cancellation_policy(df):
        house_mask = np.array(Image.open('C:/Users/mahes/Downloads/house1.jpg'))
        text = df['Cancellation_policy'].dropna()
        text = text.apply(lambda x: x.lower())
        new_words = ['please','available','will',"don't"]
        text_1 = ' '.join(i for i in text)
        stopwords = list(STOPWORDS) + new_words
        wordcloud = WordCloud(stopwords=stopwords, background_color='#1C1C1C', mask=house_mask, colormap = 'twilight_shifted_r').generate(text_1)
        g3 = plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(g3) 
def word_cloud():
        opt2 = ["House rules","Cancellation policy","Amenities"]
        option2 = st.sidebar.selectbox("Select an option",opt2)
        if option2 == "House rules":
                house_rules(df)
        elif option2 == "Cancellation policy":
                cancellation_policy(df)
        elif option2 == "Amenities":
                amenities(df)                        
                       
## STREAMLIT CODING FOR EDA ANALYSIS MENU.....                                        
def EDA_Analysis_menu():
        st.header(":red[EDA ANALYSIS]")
        menu = ['Univariate Analysis','Bivariate Analysis','Multivariate Analysis','Geographical Visualization','Word Cloud']
        
        choice = st.sidebar.selectbox("Choose EDA Analysis Option",menu)
        with st.expander("DATA VIEW"):
                st.dataframe(df)
        
        if choice == 'Univariate Analysis':
                univariate_analysis(df)
        elif choice == 'Bivariate Analysis':
                bivariate_analysis()
        elif choice == 'Multivariate Analysis':
                correlation_plot()
        elif choice == 'Geographical Visualization':
                geospacial_visualization()
        elif choice == 'Word Cloud':
                word_cloud()
        else:
                pass   
def dashboard_menu():
        opt3 = st.sidebar.selectbox("Dashboard options",['Dashboard1','Dashboard2']) 
        if opt3 == 'Dashboard1':
                st.image(Image.open("C:/Users/mahes/Downloads/Dashboard 1.png"))   
        elif opt3 == 'Dashboard2':
                st.image(Image.open("C:/Users/mahes/Downloads/Dashboard 2.png")) 
        else:
                pass        
                                 
def insights():
        #copy of dataframe .
        df1 = df.copy()
        
        ques = ['Question-1','Question-2','Question-3','Question-4','Question-5','Question-6','Question-7','Question-8']
        opt4 = st.sidebar.selectbox('Select an option',ques)
        if opt4 == 'Question-1':
                st.header("NUMBER OF HOST WHICH HAS 365 DAYS AVAILABILITY WITH TOP 10 ")
                #Fetching data host_id and availability_365 columns
                availability_365_days=df[['Host_id','Host_name','Availability_365']].copy()
                #Filtering out maximum availability of 365 days
                host_available_365_days=availability_365_days[availability_365_days['Availability_365']>364]
                how_many_host_available_for_365_days=host_available_365_days['Host_id'].nunique()
                st.write(f"There are totally {how_many_host_available_for_365_days} which has availability of 365 days")
                fig = plt.figure(figsize=(10,8))
                ax = sns.countplot(data=host_available_365_days,y=host_available_365_days.Host_name.values,order=host_available_365_days.Host_name.value_counts().index[:10])
                ax.set_title("Host available in 365 top list")
                st.pyplot(fig)
        elif opt4 == 'Question-2':
                st.header("HIGHEST NUMBER OF REVIEWS HAVING HOSTS")   
                #Grouping host_id and host_name and perform sum aggregation function
                groupby_host_id_and_host_name=df1.groupby(['Host_id','Host_name']).sum()[['No_of_reviews']].reset_index()
                highest_number_of_reviews=groupby_host_id_and_host_name.sort_values('No_of_reviews',ascending=False).head(10)
                #Replacing hostid with hypen symbol at end of each id. Because Plotly assumes it has integer
                highest_number_of_reviews['Host_id']=highest_number_of_reviews['Host_id'].astype('string').apply(lambda x:x+"_")
                #plotting bar graph and styling with pattern shape
                fig = px.bar(highest_number_of_reviews, y='No_of_reviews', x='Host_name', text='No_of_reviews',
                        color='Host_id',opacity=.8,pattern_shape="Host_id", 
                        pattern_shape_sequence=['|', '/', '\\', 'x', '-', '|', '+', '.'],color_discrete_sequence=px.colors.qualitative.Prism)
                #Updating traces and layout to beautify the plot and setting the font size
                fig.update_traces(textfont=dict(size=15,color='White'))
                fig.update_layout(title='Host has the highest number of reviews',xaxis=dict(titlefont = dict(size=15),tickfont = dict(size=14)),yaxis=dict(titlefont = dict(size=15),tickfont = dict(size=13),
                        showgrid=True,gridcolor='rgb(26, 173, 102)',
                        showticklabels=True),plot_bgcolor='black')
                #show figure
                st.pyplot(fig)
        elif  opt4 == 'Question-3':   
                st.header("TOP 10 ROOM TYPE") 
                fig = plt.figure(figsize=(10,8))
                sns.countplot(data=df1,x=df1.Room_type.values,order=df1.Room_type.value_counts().index[:10])
                plt.title("Room types")
                st.pyplot(fig) 
        elif   opt4 == 'Question-4':  
                st.header("TOP 10 PROPERTY TYPE")  
                fig = plt.figure(figsize=(15,8))
                sns.countplot(data=df1,x=df1.Property_type.values,order=df1.Property_type.value_counts().index[:10])
                plt.title("Top 10 Property Types available") 
                st.pyplot(fig)     
        elif  opt4 == 'Question-5':
                st.header("CHEAPEST PRICE OF COUNTRY")     
                country_list=df1.groupby(["Country","Country_code"])["Price"].mean().reset_index()
                st.dataframe(country_list.min())
                #plotting bar graph and styling with pattern shape
                fig = px.bar(country_list, y='Price', x='Country', text='Price',
                        color='Country_code',opacity=.8,pattern_shape="Country_code", 
                        pattern_shape_sequence=['|', '/', '\\', 'x', '-', '|', '+', '.'],color_discrete_sequence=px.colors.qualitative.Prism)
                #Updating traces and layout to beautify the plot and setting the font size
                fig.update_traces(textfont=dict(size=15,color='White'))
                fig.update_layout(title='Cheapest Price having Country',xaxis=dict(titlefont = dict(size=15),tickfont = dict(size=14)),yaxis=dict(titlefont = dict(size=15),tickfont = dict(size=13),
                        showgrid=True,gridcolor='rgb(26, 173, 102)',
                        showticklabels=True),plot_bgcolor='black')
                #show figure
                st.pyplot(fig)
        elif    opt4 == 'Question-6':
                st.header("CHEAPEST ROOM TYPE")
                room_type_list=df1.groupby(["Room_type"])["Price"].mean().reset_index()
                st.dataframe(room_type_list.min())
                fig = plt.subplots(figsize=(20, 5))
                sns.lineplot(x=room_type_list.Room_type, y=room_type_list.Price, data=room_type_list)
                st.pyplot(fig)
        elif    opt4 == 'Question-7':
                st.header("MINIMUM NIGHTS STAYED PROPERTY")
                proper_type_list=df1.groupby(["Property_type"])["Min_nights"].sum().reset_index()
                st.dataframe(proper_type_list.min())
                fig = plt.subplots(figsize=(20, 5))
                sns.lineplot(x=proper_type_list.Property_type, y=proper_type_list.Min_nights, data=proper_type_list, marker='*', markerfacecolor='limegreen', markersize=20).set(title='Property_type which have Min_night stayed ', xlabel='Property_type', ylabel='Min_nights')
                sns.set_theme(style='white', font_scale=3)
                st.pyplot(fig)
        elif    opt4 == 'Question-8':  
                st.header("MAXIMUM NIGHTS STAYED PROPERTY")
                proper_type_list1=df1.groupby(["Property_type"])["Max_nights"].sum().reset_index()
                st.dataframe(proper_type_list1.max())
                df_1 = proper_type_list1.sort_values("Max_nights",ascending=False)
                df_2 = df_1.head(10)
                fig = plt.subplots(figsize=(20, 5))
                sns.lineplot(x=df_2.Property_type, y=df_2.Max_nights, data=df_2, marker='*', markerfacecolor='limegreen', markersize=20).set(title='Property type which have Max_nights stayed', xlabel='Property_type', ylabel='Max_nights')
                sns.set_theme(style='white', font_scale=3)
                st.pyplot(fig)
                
                            
                         
                                        
                
        
                        
    
    
####STREAMLIT FRAME WORK CODING.....####

###HOME MENU...
if selected == 'Home':
        home_menu()
if selected == 'EDA Analysis':
        EDA_Analysis_menu()   
if selected == 'Insights':
        insights()
if selected == 'Dashboard':
        dashboard_menu()             
        