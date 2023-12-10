from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import haversine
import os
import requests
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()
app.config["ALLOWED_EXTENSIONS"] = set(["png", "jpg", "jpeg"])
app.config["UPLOAD_FOLDER"] = "static/uploads/"
app.config["MODEL_FILE"] = "GoBatik.h5"
api_key = os.environ.get("GOOGLE_PLACES_API_KEY")


def allowed_file(filename):
    return (
        "." in filename
        and filename.split(".", 1)[1] in app.config["ALLOWED_EXTENSIONS"]
    )


model = load_model(app.config["MODEL_FILE"], compile=False)


def predict_batik_type(image):
    img = Image.open(image).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.asarray(img)
    img_array = np.expand_dims(img_array, axis=0)
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    predictions = model.predict(data)
    # index = np.argmax(predictions)
    # class_name = labels[index]
    # confidence_score = predictions[0][index]
    return predictions


@app.route("/", methods=["GET"])
def index():
    return "hello gobatik guys"


@app.route("/gobatik/v1/batik_store", methods=["GET"])
def batik_store():
    try:
        location = request.args.get("location")

        if location is None:
            return jsonify({"error": "location are required."}), 400

        url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query=batik%20store%20in%20{location}&key={api_key}&rankby=prominence"

        response = requests.get(url)

        if response.status_code == 200:
            api_data = response.json()

            return jsonify(api_data["results"])
    except Exception as e:
        print(e)
        return jsonify({"error": "Internal Server Error", "message": e}), 500


@app.route("/gobatik/v1/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        image = request.files["image"]
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            prediction = model.predict(image_path)
            print(prediction)
            return jsonify({"status": "sukses", "data": prediction})

        else:
            return (
                jsonify(
                    {
                        "status": {
                            "code": 400,
                            "message": "Invalid file format. Please upload a JPG, JPEG, or PNG image.",
                        },
                        "data": None,
                    }
                ),
                400,
            )
    else:
        return (
            jsonify(
                {
                    "status": {"code": 405, "message": "Method not allowed"},
                    "data": None,
                }
            ),
            405,
        )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
