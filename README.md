# Weather-Prediction-using-csv-dataset
## Project Title: Weather Prediction App

## Project Description:

This project aims to develop a web application that allows users to predict weather conditions for a specific location and date. Users can select a date from a calendar interface and choose a weather variable they are interested in, such as temperature, precipitation, or cloud cover. The app then uses a machine learning model to analyze historical data and generate a prediction for the chosen date.

**Features:**

* **Interactive calendar interface:** Users can select a specific date to view predictions.
* **Variable selection:** Users can choose the weather variable they are interested in.
* **Prediction visualization:** Predictions are presented through a clear and easy-to-understand graph.
* **Historical data analysis:** Users can explore historical weather data for the chosen location.

**Benefits:**

* Gain insights into future weather conditions for specific dates.
* Make informed decisions based on weather predictions.
* Plan outdoor activities or events with greater confidence.
* Learn about weather patterns and historical trends.

**Target Audience:**

This application is intended for anyone who wants to access weather predictions for specific dates and locations. It is particularly useful for individuals who plan outdoor activities, travel, or work in weather-sensitive industries.


## Block 1 : Import libraries
`Python`
```import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.linear_model import Ridge
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
```

**Explanation:**
This block imports various libraries needed for building the web application:
* **Pandas**: Used for data manipulation and analysis.
* **dash**: Used to create interactive web applications with Python.
* **dcc**: Dash component library for interactive elements like dropdowns and graphs.
* **html**: Used for static HTML elements like headings and paragraphs.
* **dash.dependencies**: Used for defining callbacks that update the UI based on user interaction.
* **plotly.express**: Used for creating visualizations.
* **sklearn.linear_model**: Used for training a Ridge regression model for prediction.
* **dash_bootstrap_components**: Used for styling the layout with Bootstrap components.
These libraries provide the necessary functionality for data handling, user interface development, visualization, and machine learning.
## Block 2 : Load and prepare data
`Python`
```# Sample data for testing
df = pd.read_csv("pune.csv")
df_num = df.loc[:,['maxtempC','mintempC','cloudcover','humidity','tempC', 'sunHour','HeatIndexC', 'precipMM', 'pressure','windspeedKmph']]
df_num.index = pd.to_datetime(df.date_time)
df_num["selected_predictor"] = df_num.shift(-1)["maxtempC"]
df_num = df_num.iloc[:-1,:].copy()
```

**Explanation:**
This block focuses on data loading and preparation:
* **Reading Data:** It reads the weather data from a CSV file named pune.csv using pandas.read_csv.
* **Selecting Features:** It extracts only numerical features related to weather conditions and stores them in df_num.
* **Setting Datetime:** It converts the date-time column to datetime format for easier analysis.
* **Shifting Data:** It creates a new column named selected_predictor by shifting the maxtempC values one step forward. This ensures that the model predicts future values based on past data.
* **Dropping Last Row:** It removes the last row of the data to avoid inconsistencies during prediction.
This process prepares the data for training and testing the machine learning model.
## Block 3: Define the model
`Python`
```
model = Ridge(alpha=.1)
Predictor = ['maxtempC','mintempC','cloudcover','humidity','tempC', 'sunHour','HeatIndexC', 'precipMM', 'pressure','windspeedKmph']
```

**Explanation:**
This block defines the model for weather prediction:
* **Model Definition:** It creates a Ridge regression model using sklearn.linear_model.Ridge.
* **Regularization:** It sets the alpha value to 0.1 for regularization. This helps to prevent overfitting and improve the model'sgeneralizability.
* **Predictor Variables:** It defines a list named Predictor containing all the weather features used for prediction.
This block configures the machine learning model that will learn the relationship between different weather variables and predict future values.

## Block 4 : Define the layout
`Python`
```
date_picker = dcc.DatePickerSingle(
    id="date-picker",
    min_date_allowed=df_num.index.min(),
    max_date_allowed=df_num.index.max(),
    date=df_num.index.min(),
)
# Define layout
app.layout = html.Div([
    html.H1("Weather Prediction App"),
    dcc.Dropdown(
        id='predictor-dropdown',
        options=[{'label': col, 'value': col} for col in Predictor],
        value='sunHour', # Default predictor variable
        style={'width': '50%'}
    ), date_picker,
    html.Button('Predict', id='predict-button', n_clicks=0, tabIndex='100'),
    dcc.Graph(id='weather-graph')
])
```


**Explanation:**
This block defines the layout of the web application:
* **Heading:** It displays the heading "Weather Prediction App" to inform the user about the application's purpose.
* **Dropdown Menu:** It creates a dropdown element with the ID predictor-dropdown. This allows users to select the variable they want to predict.
* **Default Option:** The default value of the dropdown is set to sunHour.
* **Button:** It adds a button with the ID predict-button to trigger the prediction process.
* **date_picker:** date_picker" allows you to select a specific date to see predicted weather information for that day. You can choose any date within the available range.
* **Graph:** It includes a dcc.Graph element with the ID.

## Block 5: Define callback function
`Python`
```
@app.callback(
    Output('weather-graph', 'figure'),
    [Input('predict-button', 'n_clicks')],
    [Input('date_picker', 'date')],
    [dash.dependencies.State('predictor-dropdown', 'value')]
)
def update_graph(n_clicks, selected_predictor):
    # ...
```

**Explanation:**
This callback function is triggered whenever the user clicks the "Predict" button:
* **Inputs:** It receives two inputs:
    * n_clicks: This indicates the number of times the "Predict" button has been clicked.
    * selected_predictor: This stores the variable selected by the user from the dropdown menu.
* **Output:** It updates the figure displayed in the weather-graph element with the prediction results.
* **Function body:** The function body contains the logic for training the model, making predictions, and creating the visualization. This will be explained in detail in the next post.
This callback function ensures that the graph updates dynamically according to the user's selected predictor variable and triggers prediction.

## Block 6: Update the graph based on user interaction
`Python`
```
@app.callback(
    Output('weather-graph', 'figure'),
    [Input('predict-button', 'n_clicks')],
    [dash.dependencies.State('predictor-dropdown', 'value')]
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
```
**Explanation:**
This block implements the main logic of the application:
* **Train-test split:** The code separates the data into training (before 2021) and testing (after 2021) sets. We can choose the date for a particular day to compare the actual and predicted values.
* **Model training:** The Ridge regression model is trained on the training set using the selected predictor variable and the actual values of the chosen variable.
* **Prediction:** The model predicts the values of the selected variable for the testing set.
* **Results:** The actual and predicted values are stored in a DataFrame for visualization.
* **Graph creation:** The code creates two separate bar traces for actual and predicted values.
* **Layout configuration:** The layout defines the axes, titles, and bar grouping.
* **Figure creation:** The traces and layout are combined into a single figure object.
* **Output:** The function returns the figure to update the graph on the web page.
This block demonstrates how the user interaction triggers prediction, model training, and visualization of the results.
