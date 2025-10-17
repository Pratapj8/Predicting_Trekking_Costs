#  change app name when deploying to cloud (aws)
# python app.py


from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route("/")
def home_page():
    return render_template("index.html")


# http://127.0.0.1:5000/
@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            gender=request.form.get("gender"),
            age=request.form.get("age"),
            height=request.form.get("height"),
            weight=request.form.get("weight"),
            duration=request.form.get("duration"),
            transportation_cost=request.form.get("transportation_cost"),
            accommodation_cost=request.form.get("accommodation_cost"),
            food_cost=request.form.get("food_cost"),
            activity_cost=request.form.get("activity_cost"),
            miscellaneous_cost=request.form.get("miscellaneous_cost"),
            season=request.form.get("season"),
            destination=request.form.get("destination"),
            number_of_people=request.form.get("number_of_people"),
            guide_needed=request.form.get("guide_needed"),
            difficulty_level=request.form.get("difficulty_level"),
            previous_trek_experience=request.form.get("previous_trek_experience"),
            medical_history=request.form.get("medical_history"),
            emergency_contact_provided=request.form.get("emergency_contact_provided"),
            insurance_availed=request.form.get("insurance_availed"),
            fitness_level=request.form.get("fitness_level"),
            weather_conditions=request.form.get("weather_conditions"),
            terrain_type=request.form.get("terrain_type"),
            equipment_rented=request.form.get("equipment_rented"),
            local_guide_available=request.form.get("local_guide_available"),
            peak_season=request.form.get("peak_season"),
            payment_method=request.form.get("payment_method"),
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")

        return render_template("home.html", results=results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
