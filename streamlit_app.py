# import streamlit as st
# import pandas as pd
# import psycopg2
# import plotly.express as px
# import os
# from src.LeadGen.pipelines.pip_07_prediction_pipeline import CustomData, PredictionPipeline, ConfigurationManager
# from src.LeadGen.exception import CustomException
# from src.LeadGen.logger import logging

# # Database configuration
# DATABASE_URL = os.getenv('DATABASE_URL', "postgresql://minich:K2mQ4l4T8loWiVJ6KBGSoQ@minich-db-15346.7tt.aws-us-east-1.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full")

# @st.cache_resource
# def get_connection():
#     try:
#         conn = psycopg2.connect(DATABASE_URL)
#         print("Database connection successful")
#         return conn
#     except psycopg2.OperationalError as e:
#         st.error(f"Error connecting to database: {e}")
#         return None

# def update_database(df, conn):
#     # Convert DataFrame columns to lowercase
#     df.columns = [col.lower() for col in df.columns]

#     table_name = "new_leads_data"

#     # Create table query with correct column names
#     create_table_query = f"""
#     CREATE TABLE IF NOT EXISTS {table_name} (
#         lead_origin VARCHAR(255),
#         lead_source VARCHAR(255),
#         do_not_email VARCHAR(255),
#         do_not_call VARCHAR(255),
#         total_visits FLOAT,
#         total_time_spent_on_website FLOAT,
#         page_views_per_visit FLOAT,
#         last_activity VARCHAR(255),
#         country VARCHAR(255),
#         specialization VARCHAR(255),
#         how_did_you_hear_about_x_education VARCHAR(255),
#         current_occupation VARCHAR(255),
#         reason_for_choosing_course VARCHAR(255),
#         search VARCHAR(255),
#         newspaper_article VARCHAR(255),
#         x_education_forums VARCHAR(255),
#         newspaper VARCHAR(255),
#         digital_advertisement VARCHAR(255),
#         through_recommendations VARCHAR(255),
#         tags VARCHAR(255),
#         lead_quality VARCHAR(255),
#         lead_profile VARCHAR(255),
#         city VARCHAR(255),
#         asymmetric_activity_index VARCHAR(255),
#         asymmetric_profile_index VARCHAR(255),
#         asymmetric_activity_score FLOAT,
#         asymmetric_profile_score FLOAT,
#         a_free_copy_of_mastering_the_interview VARCHAR(255),
#         last_notable_activity VARCHAR(255),
#         lead_convertion INT
#     );
#     """

#     with conn.cursor() as cur:
#         cur.execute(create_table_query)
#         conn.commit()

#         # Filter DataFrame to include only columns present in the table
#         columns = ', '.join([f'"{col}"' for col in df.columns])
#         values = ', '.join(['%s' for _ in df.columns])
#         insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({values})"

#         # Insert DataFrame rows into the database
#         for row in df.itertuples(index=False, name=None):
#             try:
#                 cur.execute(insert_query, row)
#             except psycopg2.Error as e:
#                 print(f"Error inserting row: {row} - {e}")

#         conn.commit()

# def main():
#     st.title("Lead Conversion Prediction")
#     st.write("Upload a CSV file to predict lead conversion and update the database.")

#     conn = get_connection()
#     if conn is None:
#         st.stop()

#     uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)
        
#         # Initialize prediction pipeline
#         config_manager = ConfigurationManager()
#         prediction_config = config_manager.get_prediction_pipeline_config()
#         prediction_pipeline = PredictionPipeline(config=prediction_config)

#         # Make predictions
#         predictions = prediction_pipeline.make_predictions(df)

#         # Add predictions to DataFrame
#         df['lead_convertion'] = predictions

#         # Update the database
#         update_database(df, conn)
        
#         st.write("Predictions made and database updated successfully!")

#         st.subheader("Predictions Data")
#         st.write(df)

#         # Create bar chart
#         st.subheader("Bar Chart of Conversions")
#         df['conversion_status'] = df['lead_convertion'].map({0: 'Not Converted', 1: 'Converted'})
#         conversion_counts = df['conversion_status'].value_counts().reset_index()
#         conversion_counts.columns = ['conversion_status', 'count']
#         bar_chart = px.bar(conversion_counts, x='conversion_status', y='count', title="Conversion Counts")
#         st.plotly_chart(bar_chart)

#         # Create pie chart
#         st.subheader("Pie Chart of Conversions")
#         pie_chart = px.pie(conversion_counts, names='conversion_status', values='count', title="Conversion Proportions")
#         st.plotly_chart(pie_chart)

# if __name__ == '__main__':
#     main()

import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
import os
from src.LeadGen.pipelines.pip_07_prediction_pipeline import CustomData, PredictionPipeline, ConfigurationManager
from src.LeadGen.exception import CustomException
from src.LeadGen.logger import logging

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL', "postgresql://minich:K2mQ4l4T8loWiVJ6KBGSoQ@minich-db-15346.7tt.aws-us-east-1.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full")

@st.cache_resource
def get_connection():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        print("Database connection successful")
        return conn
    except psycopg2.OperationalError as e:
        st.error(f"Error connecting to database: {e}")
        return None

def update_database(df, conn):
    # Convert DataFrame columns to lowercase
    df.columns = [col.lower() for col in df.columns]

    table_name = "new_leads_data"

    # Create table query with correct column names
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        lead_origin VARCHAR(255),
        lead_source VARCHAR(255),
        do_not_email VARCHAR(255),
        do_not_call VARCHAR(255),
        total_visits FLOAT,
        total_time_spent_on_website FLOAT,
        page_views_per_visit FLOAT,
        last_activity VARCHAR(255),
        country VARCHAR(255),
        specialization VARCHAR(255),
        how_did_you_hear_about_x_education VARCHAR(255),
        current_occupation VARCHAR(255),
        reason_for_choosing_course VARCHAR(255),
        search VARCHAR(255),
        newspaper_article VARCHAR(255),
        x_education_forums VARCHAR(255),
        newspaper VARCHAR(255),
        digital_advertisement VARCHAR(255),
        through_recommendations VARCHAR(255),
        tags VARCHAR(255),
        lead_quality VARCHAR(255),
        lead_profile VARCHAR(255),
        city VARCHAR(255),
        asymmetric_activity_index VARCHAR(255),
        asymmetric_profile_index VARCHAR(255),
        asymmetric_activity_score FLOAT,
        asymmetric_profile_score FLOAT,
        a_free_copy_of_mastering_the_interview VARCHAR(255),
        last_notable_activity VARCHAR(255),
        lead_convertion INT
    );
    """

    with conn.cursor() as cur:
        cur.execute(create_table_query)
        conn.commit()

        # Filter DataFrame to include only columns present in the table
        columns = ', '.join([f'"{col}"' for col in df.columns])
        values = ', '.join(['%s' for _ in df.columns])
        insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({values})"

        # Insert DataFrame rows into the database
        for row in df.itertuples(index=False, name=None):
            try:
                cur.execute(insert_query, row)
            except psycopg2.Error as e:
                print(f"Error inserting row: {row} - {e}")

        conn.commit()
    # ... (your existing update_database function)

def main():
    st.title("Lead Conversion Prediction")
    st.write("Upload a CSV file to predict lead conversion and update the database.")

    conn = get_connection()
    if conn is None:
        st.stop()

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Initialize prediction pipeline
        config_manager = ConfigurationManager()
        prediction_config = config_manager.get_prediction_pipeline_config()
        prediction_pipeline = PredictionPipeline(config=prediction_config)

        # Make predictions
        predictions = prediction_pipeline.make_predictions(df)

        # Add predictions to DataFrame
        df['lead_convertion'] = predictions

        # Update the database
        update_database(df, conn)
        
        st.write("Predictions made and database updated successfully!")

        st.subheader("Predictions Data")
        st.write(df)

        # Create bar chart
        st.subheader("Bar Chart of Conversions")
        conversion_counts = df['lead_convertion'].value_counts().reset_index()
        conversion_counts.columns = ['conversion_status', 'count']
        bar_chart = px.bar(conversion_counts, x='conversion_status', y='count', title="Conversion Counts")
        st.plotly_chart(bar_chart)

        # Create pie chart
        st.subheader("Pie Chart of Conversions")
        pie_chart = px.pie(conversion_counts, names='conversion_status', values='count', title="Conversion Proportions")
        st.plotly_chart(pie_chart)

if __name__ == '__main__':
    main()