# App to predict the chances of admission using a pre-trained ML model in Streamlit

# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
from mapie.regression import MapieRegressor

from mapie.metrics import regression_coverage_score

warnings.filterwarnings('ignore')

# Set up the app title and image
st.title('Traffic Volume Predictor')
st.write("Utilize our advanced Machine Learning application to predict traffic volume.") 
st.image('traffic_image.gif', use_column_width = True)

# Reading the pickle file that we created before 
# xgb_pickle = open('xgb_tvp.pickle', 'rb') 
# xgb_reg = pickle.load(xgb_pickle) 
# xgb_pickle.close()
with open('xgb_tvp.pickle', 'rb') as xgb_pickle:
    xgb_reg = pickle.load(xgb_pickle)

mapie = MapieRegressor(xgb_reg)

# Create a sidebar for input collection
st.sidebar.header('Input Features')
st.sidebar.write('You can either upload your data file or manually enter input features.')

# Load the default dataset
base_df = pd.read_csv('Traffic_Volume.csv')
base_df['holiday'].fillna('None', inplace=True)
base_df['date_time'] = pd.to_datetime(base_df['date_time'])
base_df['month'] = base_df['date_time'].dt.strftime('%B')
base_df['weekday'] = base_df['date_time'].dt.strftime('%A')
base_df['hour'] = base_df['date_time'].dt.hour
base_df.drop(columns=['date_time'], inplace=True)
base_df['month'] = base_df['month'].astype('category')
base_df['weekday'] = base_df['weekday'].astype('category')
sample_df = pd.read_csv('traffic_data_user.csv')
sampledf = sample_df.head()
    
# Option 1: CSV
with st.sidebar.expander("**Option 1: Upload a CSV File**", expanded = False):
    st.write('Upload a CSV file containing traffic details.')
    file_upload = st.file_uploader("Choose a CSV file", type="csv")
    st.header("Sample Data Format for Upload")
    st.write(sampledf)
    st.warning("⚠️&nbsp;&nbsp; Ensure your file has the same column names and data types as shown above")

# Option 2: Form
###Used ChatGPT to fix order and initial value of the inputs
with st.sidebar.expander("Option 2: Fill Out Form", expanded = False):
    st.write("Enter the traffic details manually using the form below")
    holiday = st.selectbox('Choose whether today is a designated holiday or not', options = base_df['holiday'].unique())
    temp = st.number_input('Average temperature in Kelvin', min_value = base_df['temp'].min(), max_value = base_df['temp'].max(), value = 281.21)
    rain = st.number_input('Amount in mm of rain that occured in the hour', min_value = base_df['rain_1h'].min(), max_value = base_df['rain_1h'].max(), value = .33)
    snow = st.number_input('Amount in mm of snow that occured in the hour', min_value = base_df['snow_1h'].min(), max_value = base_df['snow_1h'].max(), value = 0.00)
    cloud = st.number_input('Average temperature in Kelvin', min_value = base_df['clouds_all'].min(), max_value = base_df['clouds_all'].max(), value = 49)
    weather = st.selectbox('Choose the current weather', options = base_df['weather_main'].unique())
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    month = st.selectbox('Choose month', options=sorted(base_df['month'].unique(), key=lambda x: month_order.index(x)))
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday = st.selectbox('Choose weekday', options=sorted(base_df['weekday'].unique(), key=lambda x: weekday_order.index(x)))
    hour = st.selectbox('Choose hour', options=sorted(base_df['hour'].unique()))
    form_button = st.button('Submit Form Data')

if file_upload is None or form_button is None:
    st.info('ℹ️&nbsp;&nbsp; *Please choose a data input method to proceed.*')
else:
    pass

alpha = st.slider('Select alpha value for prediction intervals', min_value=0.01, max_value=0.5)

# If a file is uploaded
if file_upload is not None:
    # st.success("CSV file uploaded successfully.")
    # upload_f = pd.read_csv(file_upload)
    # upload_f['holiday'].fillna('None', inplace=True)
    # cat_var = ['hour', 'month', 'weekday', 'holiday', 'weather_main']
    # u_df = pd.get_dummies(upload_f, columns=cat_var)
    # order = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23', 'month_April', 'month_August', 'month_December', 'month_February', 'month_January', 'month_July', 'month_June', 'month_March', 'month_May', 'month_November', 'month_October', 'month_September', 'weekday_Friday', 'weekday_Monday', 'weekday_Saturday', 'weekday_Sunday', 'weekday_Thursday', 'weekday_Tuesday', 'weekday_Wednesday', 'holiday_Christmas Day', 'holiday_Columbus Day', 'holiday_Independence Day', 'holiday_Labor Day', 'holiday_Martin Luther King Jr Day', 'holiday_Memorial Day', 'holiday_New Years Day', 'holiday_State Fair', 'holiday_Thanksgiving Day', 'holiday_Veterans Day', 'holiday_Washingtons Birthday', 'weather_main_Clear', 'weather_main_Clouds', 'weather_main_Drizzle', 'weather_main_Fog', 'weather_main_Haze', 'weather_main_Mist', 'weather_main_Rain', 'weather_main_Smoke', 'weather_main_Snow', 'weather_main_Squall', 'weather_main_Thunderstorm']
    # u_df = u_df[order]
    # st.write(u_df)
    # u_df['prediction'] = xgb_reg.predict(u_df)
    # upload_f['prediction'] = u_df['prediction']

    # ci = (1 - alpha) * 100
    # st.write(f'## Prediction Result with a {ci:.0f}% Confidence Interval')
    # st.write(upload_f)
    cat_var = ['hour', 'month', 'weekday', 'holiday', 'weather_main']
    base_df_dummies = pd.get_dummies(base_df, columns=cat_var)

    # Get the columns to use as a reference order
    all_columns = base_df_dummies.columns.tolist()

    # Process the uploaded file
    st.success("CSV file uploaded successfully.")
    upload_f = pd.read_csv(file_upload)
    upload_f['holiday'].fillna('None', inplace=True)

    # Create dummies for upload_f
    u_df = pd.get_dummies(upload_f, columns=cat_var)

    # Align u_df to match the columns in all_columns (filling missing columns with 0)
    u_df = u_df.reindex(columns=all_columns, fill_value=0)

    # Display the processed DataFrame
    st.write(u_df)

    u_df['prediction'] = xgb_reg.predict(u_df)
    upload_f['prediction'] = u_df['prediction']

    ci = (1 - alpha) * 100
    st.write(f'## Prediction Result with a {ci:.0f}% Confidence Interval')
    st.write(upload_f)
else:
    # Success message when form is submitted
    st.success("Form data submitted successfully.")
    # Encode the inputs for model prediction
    encode_df = base_df.copy()
    encode_df = encode_df.drop(columns=['traffic_volume'])
    encode_df.loc[len(encode_df)] = [holiday, temp, rain, snow, cloud, weather, month, weekday, hour]

    # Create dummies for categorical variables
    cat_var = ['hour', 'month', 'weekday', 'holiday', 'weather_main']
    encode_dummy_df = pd.get_dummies(encode_df, columns=cat_var)

    order = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23', 'month_April', 'month_August', 'month_December', 'month_February', 'month_January', 'month_July', 'month_June', 'month_March', 'month_May', 'month_November', 'month_October', 'month_September', 'weekday_Friday', 'weekday_Monday', 'weekday_Saturday', 'weekday_Sunday', 'weekday_Thursday', 'weekday_Tuesday', 'weekday_Wednesday', 'holiday_Christmas Day', 'holiday_Columbus Day', 'holiday_Independence Day', 'holiday_Labor Day', 'holiday_Martin Luther King Jr Day', 'holiday_Memorial Day', 'holiday_New Years Day', 'holiday_State Fair', 'holiday_Thanksgiving Day', 'holiday_Veterans Day', 'holiday_Washingtons Birthday', 'weather_main_Clear', 'weather_main_Clouds', 'weather_main_Drizzle', 'weather_main_Fog', 'weather_main_Haze', 'weather_main_Mist', 'weather_main_Rain', 'weather_main_Smoke', 'weather_main_Snow', 'weather_main_Squall', 'weather_main_Thunderstorm']
    encode_dummy_df = encode_dummy_df[order]

    # Extract the encoded user data (the last row added)
    user_encoded_df = encode_dummy_df.tail(1)

    ### Used ChatGPT because I couldn't understand why I couldn't use alpha in my prediction

    n_bootstrap = 100  # Number of bootstrap iterations

    # Initialize an empty list to store predictions
    predictions = []

    # Perform bootstrapping
    for _ in range(n_bootstrap):
        # Sample the data with replacement
        bootstrap_sample = user_encoded_df.sample(frac=1, replace=True)
        
        # Get the predictions for the current bootstrap sample
        bootstrap_preds = xgb_reg.predict(bootstrap_sample)
        
        # Store predictions
        predictions.append(bootstrap_preds)

    # Convert list of predictions to a numpy array
    predictions = np.array(predictions)

    # Calculate the 2.5th and 97.5th percentiles for the confidence interval
    lower_limit = np.percentile(predictions, .5 * alpha, axis=0)
    upper_limit = np.percentile(predictions, 1-(.5 * alpha), axis=0)

    # Get the mean prediction value (final prediction)
    pred_value = np.mean(predictions, axis=0)

    # Display the prediction and confidence interval
    st.write("## Predicting Traffic Volume...")

    ci = (1 - alpha) * 100  # Confidence interval percentage
    st.metric(label="Predicted Traffic Volume:", value=f"{pred_value[0]:.0f} vehicles")
    st.write(f"**Confidence Interval** ({ci:.0f}%): [{lower_limit[0]:.0f}, {upper_limit[0]:.0f}] vehicles")

# Additional tabs for DT model performance
st.subheader("Model Performance and Inference")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", 
                            "Histogram of Residuals", 
                            "Predicted Vs. Actual", 
                            "Coverage Plot"])
with tab1:
    st.write("### Feature Importance")
    st.image('tvp_fi.svg')
    st.caption("Relative importance of features in prediction.")
with tab2:
    st.write("### Histogram of Residuals")
    st.image('tvp_res.svg')
    st.caption("Distribution of residuals to evaluate prediction quality.")
with tab3:
    st.write("### Plot of Predicted Vs. Actual")
    st.image('tvp_pva.svg')
    st.caption("Visual comparison of predicted and actual values.")
with tab4:
    st.write("### Coverage Plot")
    st.image('tvp_cp.svg')
    st.caption("Range of predictions with confidence intervals.")
