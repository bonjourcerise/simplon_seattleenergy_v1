# import
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn import metrics

import seaborn as sns



# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Seattle Energy", page_icon=":high_brightness:", layout="wide")




def main():

# header with image and title
    #st.image("seattle.png")
     # ---- MAINPAGE ----
    st.image("img/seattle.png")
    st.title(":bulb: Seattle Energy")
    st.markdown("##")

    @st.cache
    def load_data():
       
        data=pd.read_csv("data_model.csv")
        return data
  



    @st.cache
    def load_infos_gen(data):
        lst_infos = [data.shape[0],
                     round(data["BuildingAge"].mean()),
                     round(data["harvesine_distance"].max())]

        nb_batiments = lst_infos[0]
        age_moy = lst_infos[1]
        distance_moy = lst_infos[2]

        type_batiment = data.BuildingType.value_counts()

        return nb_batiments, age_moy, distance_moy , type_batiment

    @st.cache
    def identite_batiment(data, id):
        data_batiment = data[data.index == int(id)]
        return  data_batiment

    @st.cache
    def load_age_batiment(data):
        data_age_bati = data["BuildingAge"]
        return data_age_bati 


     #Loading data……
    data = load_data()
    id_bati = data.index.values


     #bati selection
    st.sidebar.header("**General Info**")

    #Loading selectbox
    chk_id = st.sidebar.selectbox("Building ", id_bati)

    #Loading general info
    nb_batiments, age_moy, distance_moy , type_batiment = load_infos_gen(data)
    star_rating = ":star:" * int(round(5 ,0))
    
    left_column, middle_column, right_column = st.columns(3)

    with left_column:
        st.subheader("Number of Buildings :")
        st.subheader(f"{nb_batiments:,}")

    with middle_column:
        st.subheader("Average Building Age :")
        st.subheader(f"{age_moy}")
    with right_column:
        st.subheader("Maximum Distance:")
        st.subheader(f"{distance_moy}")

    st.markdown("""---""")


 #PieChart pour le batiment
   
    # #st.sidebar.markdown("<u>......</u>", unsafe_allow_html=True)
    # fig, ax = plt.subplots(figsize=(5,5))
    # fig = px.bar(data, x="BuildingType", barmode='group',height=400)
    # #fig.set_facecolor("#FFFFFF")
    # st.sidebar.pyplot(fig)

#Customer information display : Customer Gender, Age, Family status, Children, …
    st.header("**Building information display**")

    if st.checkbox("Show Building information ?"):

        infos_bati = identite_batiment(data, chk_id)
        st.write("**Type : **", infos_bati["BuildingType"].values[0])
        st.write("**Age : **{:.0f} ans".format(int(infos_bati["BuildingAge"])))
        st.write("**Number of floors : **{:.0f}".format(infos_bati ["NumberofFloors"].values[0]))

        #Age distribution plot
        data_age = load_age_batiment(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_age, edgecolor = 'k', color="goldenrod", bins=20)
        ax.axvline(int(infos_bati["BuildingAge"]), color="green", linestyle='--')
        ax.set(title='Building age', xlabel='Age(Year)', ylabel='')
        st.pyplot(fig)


    @st.cache
    def dataXY(data):
        y=data[['TotalGHGEmissions','SiteEnergyUse(kBtu)']]
        X=data.drop(['TotalGHGEmissions','SiteEnergyUse(kBtu)'],axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state= 2022)
        return X_train, X_test, y_train, y_test 
    X_train, X_test, y_train, y_test=dataXY(data) 


    st.title("Energy Prediction")

    # # welcome info
    st.info(":high_brightness: Welcome here !  Hop, upload your .csv file & you can predict the energy use by the building in Seattle and the GHG emissions by year")
        
        
    with open('log_best_model.pkl' , 'rb') as f:
        model= pickle.load(f)
    
    
    y_pred=model.predict(X_test)
   
    errors = abs(y_pred - y_test)
    mae = 100 * np.mean(errors / y_test)
    accuracy = 100 - mae

    st.markdown("Model Performance")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(":city_sunrise:Energy use by kg/yard**")
        st.write("R2: ",round(metrics.r2_score(y_test.iloc[:,1], y_pred[:,1]),2))
        st.write("Percent MAE: {:.2f} %".format(mean_absolute_percentage_error(y_test.iloc[:,1], y_pred[:,1])))
        st.write("Accuracy for Energy use : {:.2f} %".format(accuracy[1]))


    with col2:
        st.markdown("**:city_sunset:GHG emissions by year**")
        st.write("R2: ", round(metrics.r2_score(y_test.iloc[:,0], y_pred[:,0]),2))
        #st.write("Percent MAE: ,{round(mean_absolute_percentage_error(y_test.iloc[:,0], y_pred[:,0]),2)
        st.write("Percent MAE: {:.2f} %".format(mean_absolute_percentage_error(y_test.iloc[:,0], y_pred[:,0])))
        st.write("Accuracy for GHG emissions: {:.2f} %".format(accuracy[0]))
    
   

   
 

if __name__ == '__main__':
    main()