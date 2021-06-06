import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
import streamlit as st

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://images.unsplash.com/photo-1538928456919-0eef34d8e400?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1650&q=80");
        background-size: 100%;
        background-repeat: no-repeat;
        background-position: center;
    </style>
    """,
    unsafe_allow_html=True
)


st.title('Formula 1 Race winner Predictor')


data = pd.read_csv('final_df.csv')
#st.dataframe(data)

df = data.copy()
df.podium = df.podium.map(lambda x: 1 if x == 1 else 0)

train = df[df.season <2019]
X_train = train.drop(['driver','podium'], axis = 1)
y_train = train.podium

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)

@st.cache
def score_classification(model):
    score = 0
    for circuit in df[df.season == 2019]['round'].unique():

        test = df[(df.season == 2019) & (df['round'] == circuit)]
        X_test = test.drop(['driver','podium'], axis = 1)
        y_test = test.podium

        #scaling
        X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)

        # make predictions
        prediction_df = pd.DataFrame(model.predict_proba(X_test), columns = ['proba_0', 'proba_1'])
        prediction_df['actual'] = y_test.reset_index(drop = True)
        prediction_df.sort_values('proba_1', ascending = False, inplace = True)
        prediction_df.reset_index(inplace = True, drop = True)
        prediction_df['predicted'] = prediction_df.index
        prediction_df['predicted'] = prediction_df.predicted.map(lambda x: 1 if x == 0 else 0)

        score += precision_score(prediction_df.actual, prediction_df.predicted)

    model_score = score / df[df.season == 2019]['round'].unique().max()
    
    return model_score

@st.cache
def race_winner_test(model,year,circuit):
    list_f = pd.DataFrame()
    test = df[(df.season == year) & (df['round'] == circuit)]
    X_test = test.drop(['driver','podium'], axis = 1)
    y_test = test.podium

        #scaling
    X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)

        # make predictions
    list_f['driver'] = test['driver'].reset_index(drop = True)
    prediction_df = pd.DataFrame(model.predict_proba(X_test), columns = ['proba_0', 'proba_1'])
    prediction_df['actual'] = y_test.reset_index(drop = True)
    list_f['actual'] = prediction_df['actual']
    prediction_df.sort_values('proba_1', ascending = False, inplace = True)
    prediction_df.reset_index(inplace = True, drop = True)
    prediction_df['predicted'] = prediction_df.index
    prediction_df['predicted'] = prediction_df.predicted.map(lambda x: 1 if x == 0 else 0)
    list_f['predicted'] = prediction_df['predicted']
    list_f['Winning_prob'] = prediction_df['proba_1']
    list_f['Lossing_prob'] = prediction_df['proba_0']
    return list_f

model = MLPClassifier(hidden_layer_sizes = (75,25,50,10), activation = 'identity', 
                      solver = 'lbfgs', alpha = 1.6238e-02, random_state = 1)
model.fit(X_train, y_train)

gp = st.slider('Select Round using the slider below',1,20,1,1)
year_c = st.slider('Select year using the slider below',1983,2019,2010,1)
btn = st.button('Predict')

if btn:
    st.dataframe(race_winner_test(model,year_c,gp))
else:
    pass

btn_pred = st.button('Calculate the score')

if btn_pred:
    st.text(score_classification(model))
else:
    pass






