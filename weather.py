import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.linear_model import Ridge
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
app = dash.Dash(__name__)
from dash.dependencies import Input


# Sample data for testing
df = pd.read_csv("pune.csv")
df_num = df.loc[:,['maxtempC','mintempC','cloudcover','humidity','tempC', 'sunHour','HeatIndexC', 'precipMM', 'pressure','windspeedKmph']]
df_num.index = pd.to_datetime(df.date_time)
df_num["selected_predictor"] = df_num.shift(-1)["maxtempC"]
df_num = df_num.iloc[:-1,:].copy()
model = Ridge(alpha=.1)
Predictor = ['maxtempC','mintempC','cloudcover','humidity','tempC', 'sunHour','HeatIndexC', 'precipMM', 'pressure','windspeedKmph']
# Create date picker component
date_picker = dcc.DatePickerSingle(
    id="date-picker",
    min_date_allowed=df_num.index.min(),
    max_date_allowed=df_num.index.max(),
    date=df_num.index.min(),
    style={'width': '50%', 'margin-left':'10px'}
)
# Define layout
app.layout = html.Div([
    html.H1("Weather Prediction App"),
    dcc.Dropdown(
        id='predictor-dropdown',
        options=[{'label': col, 'value': col} for col in Predictor],
        value='sunHour',  # Default predictor variable
        style={'width': '50%'}
    ),
    html.Button('Predict', id='predict-button', n_clicks=0, style={'height':'46px','width':'80px','margin-top':'10px'}),date_picker,
    dcc.Graph(id='weather-graph')
])

# Define callback to update graph based on user input
@app.callback(
    Output('weather-graph', 'figure'),
    [Input('predict-button', 'n_clicks')],
    [dash.dependencies.State('predictor-dropdown', 'value')],
    [Input("date-picker", "date")]
)
def update_graph(n_clicks, selected_predictor, selected_date):
    # Train the model
    train = df_num.loc[:"2020-12-31"]
    test = df_num.loc[selected_date:selected_date]
    model.fit(train[Predictor], train[selected_predictor])
    predictions = model.predict(test[Predictor])

    # Create a DataFrame with actual and predicted values
    results_df = pd.DataFrame({
        'Date': test.index,
        'Actual': test[selected_predictor].values,
        'Predicted': predictions
    })

    # Create separate traces for actual and predicted values
    trace_actual = go.Bar(
        x=results_df['Date'],
        y=results_df['Actual'],
        name='Actual',
        marker_color='blue'
    )

    trace_predicted = go.Bar(
        x=results_df['Date'],
        y=results_df['Predicted'],
        name='Predicted',
        marker_color='orange'
    )

    # Create a layout with side-by-side bars
    layout = go.Layout(
        xaxis=dict(
            title="Date"
        ),
        yaxis=dict(
            title=selected_predictor
        ),
        barmode='group',
        scattergap=1
        
    )

    # Combine traces and layout into a figure
    fig = go.Figure(data=[trace_actual, trace_predicted], layout=layout)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
