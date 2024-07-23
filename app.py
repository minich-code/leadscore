from flask import Flask, request, jsonify, render_template
import pandas as pd
import psycopg2
import os
from src.LeadGen.pipelines.pip_07_prediction_pipeline import CustomData, PredictionPipeline, ConfigurationManager   # Ensure correct import paths
from src.LeadGen.exception import CustomException
from src.LeadGen.logger import logging

# Create a Flask app
app = Flask(__name__)

os.environ['DATABASE_URL'] = "postgresql://minich:K2mQ4l4T8loWiVJ6KBGSoQ@minich-db-15346.7tt.aws-us-east-1.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full"

# Ensure environment variables are set correctly before running the application
# export DATABASE_URL="postgresql://minich:K2mQ4l4T8loWiVJ6KBGSoQ@minich-db-15346.7tt.aws-us-east-1.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full"
#export DATABASE_URL="postgresql://minich:K2mQ4l4T8loWiVJ6KBGSoQ@minich-db-15346.7tt.aws-us-east-1.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full"


# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL')

try:
    conn = psycopg2.connect(DATABASE_URL)
    print("Database connection successful")
except psycopg2.OperationalError as e:
    print(f"Error connecting to database: {e}")

    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_data_point():
    if request.method == 'GET':
        return render_template('home.html')
    
    if request.method == 'POST':
        try:
            if 'csv_file' not in request.files:
                logging.error("No csv_file key in request.files")
                return jsonify({"error": "No file part in the request"}), 400

            csv_file = request.files['csv_file']

            if csv_file.filename == '':
                logging.error("No selected file")
                return jsonify({"error": "No file selected"}), 400

            df = pd.read_csv(csv_file)

            # Initialize prediction pipeline
            config_manager = ConfigurationManager()
            prediction_config = config_manager.get_prediction_pipeline_config()
            prediction_pipeline = PredictionPipeline(config=prediction_config)

            # Make predictions
            predictions = prediction_pipeline.make_predictions(df)

            # Add predictions to DataFrame
            df['Lead_Convertion'] = predictions

            # Update the database
            update_database(df)

            return render_template('results.html')


        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            return jsonify({"error": "File not found"}), 404
        except (ValueError, TypeError) as e:
            logging.error(f"Invalid data provided: {e}")
            return jsonify({"error": "Invalid data provided"}), 400
        except CustomException as e:
            logging.error(f"Prediction pipeline error: {e}")
            return jsonify({"error": "Prediction pipeline error"}), 500
        except Exception as e:
            logging.exception(e)
            return jsonify({"error": "An error occurred during prediction."}), 500


def update_database(df):
    # Convert DataFrame columns to lowercase
    df.columns = [col.lower() for col in df.columns]

    # Ensure 'converted' column is not in DataFrame
    if 'converted' in df.columns:
        df = df.drop(columns=['converted'])

    # Print the columns of the DataFrame for debugging
    print("DataFrame columns:", df.columns)

    conn = psycopg2.connect(DATABASE_URL)
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

        # Check the table columns
        cur.execute(f"""
            SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS WHERE table_name = '{table_name}'
        """)
        table_columns = cur.fetchall()
        print("Database table columns:", [col[0] for col in table_columns])

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

    conn.close()



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)