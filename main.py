from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load the dataset and train a model
df = pd.read_csv("final.csv").drop(columns=["Unnamed: 0"])  # Drop the Unnamed: 0 column
X = df.drop("price", axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_price():
    try:
        data = request.form
        print(f"Received data: {data}")  # Debug statement
        beds = int(data["beds"])
        baths = float(data["baths"])
        size = float(data["size"])
        zip_code = int(data["zip_code"])

        # Create a new dataframe with the input data
        input_data = pd.DataFrame({
            "beds": [beds],
            "baths": [baths],
            "size": [size],
            "zip_code": [zip_code]
        })

        # Make a prediction
        predicted_price = model.predict(input_data)[0]
        print(f"Predicted price: {predicted_price}")  # Debug statement

        return jsonify({"predicted_price": predicted_price})
    except Exception as e:
        print(f"Error during prediction: {e}")  # Debug statement
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
