from streamlit import *
import folium
from streamlit_folium import folium_static
import pandas as pd
from datetime import date
import plotly.graph_objs as go
import plotly.figure_factory as ff
import calendar
import xgboost as xgb
import requests
import seaborn as sns
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu
from funcs import *
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
#weather = pd.read_csv('city_temperature.csv')
df = pd.read_csv('weather_timeseries.csv', index_col='Timestamp')
df.index = pd.to_datetime(df.index)

timeseries=df

cities_list=["Amsterdam","Athens","Barcelona","Berlin","Budapest","Lisbon","London","Paris","Rome","Vienna",]
set_page_config(page_title = 'Airbnb',
                page_icon = 'Active',
                layout = 'wide')

data = pd.read_csv('airbnb.csv')

print(data.tail())


def barpot_1d():
    realnames=['Superhost','Weekend','Room type','Person capacity','Bedrooms']
    colnames=['host_is_superhost', 'type','room_type','person_capacity','bedrooms']
    fig = make_subplots(rows=1, cols=len(realnames), subplot_titles=realnames)


    for k,(col,name) in enumerate(zip(colnames, realnames)):
        series = data.value_counts(col)

        fig.add_trace(go.Bar(x=series.index, y=series.to_list(), name=name), 1,k+1)

    fig.update_layout(barmode='relative',  bargap=0.05, width=900, height=500)
    return fig

def add_categorical_legend(folium_map, title, colors, labels):
    if len(colors) != len(labels):
        raise ValueError("colors and labels must have the same length.")

    color_by_label = dict(zip(labels, colors))

    legend_categories = ""
    for label, color in color_by_label.items():
        legend_categories += f"<li><span style='background:{color}'></span>{label}</li>"

    legend_html = f"""
    <div id='maplegend' class='maplegend'>
      <div class='legend-title'>{title}</div>
      <div class='legend-scale'>
        <ul class='legend-labels'>
        {legend_categories}
        </ul>
      </div>
    </div>
    """
    script = f"""
        <script type="text/javascript">
        var oneTimeExecution = (function() {{
                    var executed = false;
                    return function() {{
                        if (!executed) {{
                             var checkExist = setInterval(function() {{
                                       if ((document.getElementsByClassName('leaflet-top leaflet-right').length) || (!executed)) {{
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].style.display = "flex"
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].style.flexDirection = "column"
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].innerHTML += `{legend_html}`;
                                          clearInterval(checkExist);
                                          executed = true;
                                       }}
                                    }}, 100);
                        }}
                    }};
                }})();
        oneTimeExecution()
        </script>
      """

    css = """

    <style type='text/css'>
      .maplegend {
        z-index:9999;
        float:right;
        background-color: rgba(255, 255, 255, 1);
        border-radius: 5px;
        border: 2px solid #bbb;
        padding: 10px;
        font-size:12px;
        positon: relative;
      }
      .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
      .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
      .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
      .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 0px solid #ccc;
        }
      .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
      .maplegend a {
        color: #777;
        }
    </style>
    """

    folium_map.get_root().header.add_child(folium.Element(script + css))

    return folium_map

def color_producer(prijs):
    if prijs < 50:
        return 'lightgreen'
    elif (prijs < 125) & (prijs >= 50):
        return 'darkgreen'
    elif (prijs < 250) & (prijs >= 125):
        return 'yellow'
    elif (prijs >= 250) & (prijs < 350):
        return 'darkorange'
    elif (prijs >= 350) & (prijs < 500):
        return 'saddlebrown'
    elif (prijs >= 500) & (prijs < 1000):
        return 'red'
    else: return 'black'

def slider_dicer(stad,max_pers,min_prijs, max_prijs,aantal_kamers,min_rating,superhosts,max_afstand,type):
    subdata = data[(data.city == stad) &
                (data.person_capacity >= max_pers) &
                (data.Prijs >= min_prijs) &
                (data.Prijs <= max_prijs) &
                (data.bedrooms >= aantal_kamers) &
                (data.guest_satisfaction_overall >= min_rating) &
                (data.dist <= max_afstand) &
                (data.type == type)]
    if superhosts:
        subdata = subdata[data.host_is_superhost == True]

    return subdata.dropna()



#with sidebar:
hoofdkeuze = option_menu(None, ["Vind jouw Airbnb",'Jouw locatie', '2D-Analyse','1D-Analyse','Bronnen'],orientation='horizontal',
                           icons=['pin-map','thermometer-sun', 'bar-chart','pie-chart-fill','book'], menu_icon="cast", default_index=0,
                       styles={
                           "container": {"padding": "0!important", "background-color": "white"},
                           "icon": {"color": "lightblue", "font-size": "25px"},
                           "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px",
                                        "--hover-color": "#eee"},
                           "nav-link-selected": {"background-color": "grey  "},
                       })

if hoofdkeuze == "Vind jouw Airbnb":
    vja_l,vja_r = columns(2)
    with vja_l:
        l1,l2=columns(2)
        stad = l1.selectbox('Zoek de stad',options=cities_list)
        slidvar = l1.number_input('Max prijs aangepast',value=600)
        superhosts = l2.checkbox('Alleen superhosts')
        go = l2.button('Zoek accomodaties')
        types = l2.selectbox('Dag',['Weekdag', 'Weekend'])
        types_dict = {'Weekdag': 'weekdays', 'Weekend': 'weekends'}
        def get_stuff():
            return types_dict[types]
        types=get_stuff()
        min_prijs, max_prijs = slider('Selecteer prijs', value=[0,slidvar])

        l3,l4=columns(2)
        with l3:
            aantal_kamers = number_input('Aantal slaapkamers', min_value=0, max_value=4, step=1)
            min_rating = slider('Minimale rating', min_value=0, max_value=10)
            # aantal_kamers = slider('Aantal slaapkamers', min_value=0, max_value=4, step=1)
        with l4:
            max_pers = number_input('Aantal personen', min_value=1, max_value=6, step=1)
            max_afstand = slider('Max afstand tot centrum',min_value=0, max_value=20,value=10)
            # max_pers = slider('aantal personen', min_value=1, max_value=6,step=1)



    #vja_r=container()
    with vja_r:
        if not go:
            folium_static(folium.Map(),width=545,height=410)
        if go:

            subdata = slider_dicer(stad.lower(),max_pers,min_prijs, max_prijs,aantal_kamers,min_rating,superhosts,max_afstand, types)
            write(f'{len(subdata)} zoekresultaten')
       #     metric('10',1)
            try:
                mapa = folium.Map(location=[subdata.lat.mean(), subdata.lng.mean()], zoom_start=12)
                for grp_name, df_grp in subdata.groupby('room_type'):
                    feature_group = folium.FeatureGroup(grp_name)
                    for row in df_grp.itertuples():
                        folium.CircleMarker(location=[row.lat, row.lng], popup='Prijs: ' + str(row.realSum) + ' Rating: ' +
                                                                               str(row.guest_satisfaction_overall) +
                                                                               ' Hygiëne-score: ' + str(
                            row.cleanliness_rating),
                                            tooltip='Klik voor informatie',
                                            color=color_producer(row.Prijs),
                                            radius=1).add_to(feature_group)
                    feature_group.add_to(mapa)
                # mapa = add_categorical_legend(mapa, 'Prijsklasse',
                #                               colors=['lightgreen', 'darkgreen', 'yellow', 'darkorange', 'saddlebrown',
                #                                       'red', 'black'],
                #                               labels=['Onder €50', '€50-€125', '€125-€250', '€250-€350',
                #                                       '€350-€500', '€500-€1000', 'boven €1000'])

                folium.LayerControl('topleft', collapsed=True).add_to(mapa)
                folium_static(mapa,width=545,height=410)
                from PIL import Image

                Heatmapcorrie = Image.open('legenda_appeltaart.png')
                image(Heatmapcorrie)
            except:
                write('Geen zoekresultaten')

if hoofdkeuze == '2D-Analyse':
    left, gggggg,right = columns((2,0.3,2))
    xxxlimmm = left.slider('Filter max prijs €', min_value=0,value=5000,max_value=5000)

    data=data.query(f'Prijs < {xxxlimmm}')
    left.plotly_chart(px.box(data_frame=data,x='city',y='Prijs').update_layout(yaxis_title='Prijs €', xaxis_title='Stad'), use_container_width=True)

    corr,scat1,scat2=right.tabs(['Correlatie kaart','Hygiëne vs rating', 'Metro afstand vs centrum afstand'])
    with corr:
        test = data[['Prijs', 'dist', 'host_is_superhost', 'guest_satisfaction_overall', 'bedrooms', 'metro_dist',
                         'cleanliness_rating']]
        fig,axi=plt.subplots()
        sns.heatmap(test.corr(),ax=axi)
        pyplot(fig)
    with scat1:
        checkie = checkbox('Trendline ')
        fig = px.scatter(data, x="guest_satisfaction_overall", y="cleanliness_rating", trendline=("ols" if checkie else None),
                         trendline_color_override=('black' if checkie else None),
                         title="Scatterplot met regressielijn", labels={
                         "guest_satisfaction_overall": "Gemiddelde rating van de gasten",
                            "cleanliness_rating": "Hygiëne cijfer"})
        plotly_chart(fig, use_container_width=True)
    with scat2:
        checkie = checkbox('Trendline')
        fig = px.scatter(data, x="metro_dist", y="dist", trendline=("ols" if checkie else None), trendline_color_override='black', labels={
            "dist": "Afstand tot centrum (km)",
            "metro_dist": "Afstand tot dichtsbijzijnde metro (km)"},
                         title="Scatterplot met regressielijn")
        plotly_chart(fig, use_container_width=True)


if hoofdkeuze == 'Jouw locatie':
    write()
    l,m,r,rr=columns((2,2,2,2))
    date1=pd.to_datetime(l.date_input('Start datum',value=date.today()))
    date2=pd.to_datetime(m.date_input('End datum',value=date1))
    targ_temp=r.number_input('Gewenste temperatuur',value=10)
    rr.header('')
    button=rr.button('Zoek mijn bestemming')
    if not button:
        write('Nothing to show')
    if button:
        temp_df = get_prediction_df(date1,date2,df)
        stats = pd.DataFrame(temp_df.median(axis=0))
        stats.rename(columns={0: 'Temp'}, inplace=True)
        stats['Off'] = np.abs(stats['Temp'] - targ_temp)
        stats=stats.sort_values('Off')
        best,temp_best=(stats[stats['Off'].min() == stats['Off']].index[0],stats[stats['Off'].min() == stats['Off']]['Temp'][0])
        leftcol,midcol,rightcol=columns((1.5,1.5,3))
        with leftcol:
            plotly_chart(statplot(timeseries,best).update_layout(plot_bgcolor='#07B2D9'),use_container_width=True)
        with rightcol:
            plotly_chart(linechart_choice(temp_df,best,targ_temp).update_layout(plot_bgcolor='#07B2D9'),use_container_width=True)
        with midcol:
            plotly_chart(px.bar(pd.DataFrame({'City':stats.iloc[::-1].index,'Temp':stats.iloc[::-1].Temp}), x='Temp', y='City', orientation='h').add_vline(x=targ_temp).update_layout(plot_bgcolor='#07B2D9',title='Temperatuur per stad',xaxis_title='Temperatuur °C'), use_container_width=True)



if hoofdkeuze == 'Bronnen':
    bronleft, bronright = columns(2)
    with bronleft:
        header('De Airbnb Data')
        programcode="""
#Initialisatie
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

#Dataset downloaden
!kaggle datasets download -d zgrcemta/airbnb-prices-in-european-cities
api.dataset_download_file('airbnb-prices-in-european-cities',
                         folder_name = 'airbnb-prices-in-european-cities')
!kaggle datasets download -d zgrcemta/airbnb-prices-in-european-cities

#Uitpakken
import zipfile

with zipfile.ZipFile("airbnb-prices-in-european-cities.zip","r") as file:
    file.extractall("Airbnb")
"""
        subheader('Kaggle API')
        code(programcode,language='python')
        subheader('De data')
        write(data.head())
    with bronright:
        header('De temperatuur data')
        subheader\
            ('De rauwe data')
        write(pd.read_csv('city_temperature.csv').head(3))
        subheader('Na manipulatie en filteren:')
        write(timeseries.head(3))
        plotly_chart(px.scatter(x=timeseries.index, y=timeseries.mean(axis=1), trendline='ols').update_layout(title='Gemiddelde temperatuur door de jaren',xaxis_title='Tijd',yaxis_title='Temp °C').update_layout(template='gridon', plot_bgcolor='lightgrey'),use_container_width=True)

if hoofdkeuze=='1D-Analyse':
    appeltaart_left,appeltaart_right = columns((2,1))
    xxxlimmm = appeltaart_left.slider('Filter max prijs €', min_value=0,value=1000, max_value=5000)

    data = data.query(f'Prijs < {xxxlimmm}')


    appeltaart_left.plotly_chart(barpot_1d().update_layout(title='Barplots categorische variabelen',yaxis_title='Aantal'),use_container_width=True)

    def distplot_price(filterdf):
        fig = ff.create_distplot([filterdf[filterdf.type == 'weekends']['Prijs'].to_list(),
                                  filterdf[filterdf.type == 'weekdays']['Prijs'].to_list()],
                                 ['weekends', 'weekdays'],show_rug=False)
        fig.update_layout(title='Verdeling prijs per type', height=550,xaxis_title='Prijs €',yaxis_title='Dichtheid')
        return fig

    tab1,gggt = appeltaart_right.tabs(['Histogram', 'Boxplot'])

    with tab1:
        plotly_chart(distplot_price(data),use_container_width=True)
    with gggt:
        plotly_chart(px.box(data, x='Prijs', y="type").update_layout(title='Verdeling prijs per type', height=550,xaxis_title='Prijs €'),use_container_width=True)
