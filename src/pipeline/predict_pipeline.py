import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "/Users/apple/Downloads/Data_science_file/Agent8/Projects/Treking_cost_predictor/artifacts/model.pkl"
            preprocessor_path = "/Users/apple/Downloads/Data_science_file/Agent8/Projects/Treking_cost_predictor/artifacts/preprocessor.pkl"

            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            print("Input columns:", features.columns.tolist())
            print("Expected columns:", preprocessor.feature_names_in_.tolist())

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        age: int,
        height: float,
        weight: float,
        duration: int,
        transportation_cost: int,
        accommodation_cost: int,
        food_cost: int,
        activity_cost: int,
        miscellaneous_cost: int,
        season: str,
        destination: str,
        number_of_people: int,
        guide_needed: str,
        difficulty_level: str,
        previous_trek_experience: str,
        medical_history: str,
        emergency_contact_provided: str,
        insurance_availed: str,
        fitness_level: str,
        weather_conditions: str,
        terrain_type: str,
        equipment_rented: str,
        local_guide_available: str,
        peak_season: str,
        payment_method: str,
    ):
        self.gender = gender
        self.age = age
        self.height = height
        self.weight = weight
        self.duration = duration
        self.transportation_cost = transportation_cost
        self.accommodation_cost = accommodation_cost
        self.food_cost = food_cost
        self.activity_cost = activity_cost
        self.miscellaneous_cost = miscellaneous_cost
        self.season = season
        self.destination = destination
        self.number_of_people = number_of_people
        self.guide_needed = guide_needed
        self.difficulty_level = difficulty_level
        self.previous_trek_experience = previous_trek_experience
        self.medical_history = medical_history
        self.emergency_contact_provided = emergency_contact_provided
        self.insurance_availed = insurance_availed
        self.fitness_level = fitness_level
        self.weather_conditions = weather_conditions
        self.terrain_type = terrain_type
        self.equipment_rented = equipment_rented
        self.local_guide_available = local_guide_available
        self.peak_season = peak_season
        self.payment_method = payment_method

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "age": [self.age],
                "height": [self.height],
                "weight": [self.weight],
                "duration": [self.duration],
                "transportation_cost": [self.transportation_cost],
                "accommodation_cost": [self.accommodation_cost],
                "food_cost": [self.food_cost],
                "activity_cost": [self.activity_cost],
                "miscellaneous_cost": [self.miscellaneous_cost],
                "season": [self.season],
                "destination": [self.destination],
                "number_of_people": [self.number_of_people],
                "guide_needed": [self.guide_needed],
                "difficulty_level": [self.difficulty_level],
                "previous_trek_experience": [self.previous_trek_experience],
                "medical_history": [self.medical_history],
                "emergency_contact_provided": [self.emergency_contact_provided],
                "insurance_availed": [self.insurance_availed],
                "fitness_level": [self.fitness_level],
                "weather_conditions": [self.weather_conditions],
                "terrain_type": [self.terrain_type],
                "equipment_rented": [self.equipment_rented],
                "local_guide_available": [self.local_guide_available],
                "peak_season": [self.peak_season],
                "payment_method": [self.payment_method],
            }

            df = pd.DataFrame(custom_data_input_dict)

            # List of expected columns from your preprocessor
            expected_cols = [
                "Permit_fee_rupees",
                "identification documents (for permits)",
                "Currency",
                "city",
                "Max_altitude_m",
                "traveler_name",
                "state",
                "Lighting",
                "trekking_shoes",
                "payment information",
                "Footwear",
                "Accommodation",
                "Duration_days",
                "zip_code",
                "Industry",
                "Best_Season",
                "trek_type",
                "Region",
                "Cash/online",
                "emergency contact details",
                "Group_size",
                "payment status",
                "health_check_up",
                "Trek_Location",
                "Guide_cost_rupees",
                "Weather",
                "Backpack",
                "waterproof/windproof jackets",
                "Accessibility",
                "Best_Month",
                "Country",
                "Hotel_Price_per_day",
                "booking_status",
                "City",
                "Elevation_gain_m",
                "Season",
                "Difficulty",
                "surname",
                "traveller_profession",
            ]

            # Add missing columns with default values
            for col in expected_cols:
                if col not in df.columns:
                    # Choose default based on expected type (string as empty, numeric as zero)
                    if col in [
                        "Duration_days",
                        "Permit_fee_rupees",
                        "Guide_cost_rupees",
                        "Hotel_Price_per_day",
                        "Group_size",
                        "Elevation_gain_m",
                        "age",
                        "number_of_people",
                    ]:  # example numeric fields
                        df[col] = 0
                    else:
                        df[col] = ""

            # Ensure columns order matches preprocessor
            df = df[preprocessor.feature_names_in_]

            return df

        except Exception as e:
            raise CustomException(e, sys)
