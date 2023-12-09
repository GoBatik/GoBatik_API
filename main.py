from crypt import methods
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import requests
import haversine

app = Flask(__name__)
app.config["ALLOWED_EXTENSIONS"] = set(["png", "jpg", "jpeg"])
app.config["UPLOAD_FOLDER"] = "static/uploads/"


def allowed_file(filename):
    return (
        "." in filename
        and filename.split(".", 1)[1] in app.config["ALLOWED_EXTENSIONS"]
    )


@app.route("/", methods=["GET"])
def index():
    return "hello gobatik guys"


@app.route("/gobatik/v1/batik_store", methods=["GET"])
def batik_store():
    try:
        location = request.args.get("location")

        if location is None:
            return jsonify({"error": "location are required."}), 400

        url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query=batik%20store%20in%20{location}&key=AIzaSyD43mDPRg4B-RanFfR3pGBF9Jmj1RHqByM&rankby=prominence"

        response = requests.get(url)

        if response.status_code == 200:
            api_data = response.json()

            return jsonify(api_data["results"])
    except Exception as e:
        print(e)
        return jsonify({"error": "Internal Server Error"}), 500


@app.route("/gobatik/v1/store_image", methods=["GET", "POST"])
def store_image():
    if request.method == "POST":
        # Ensure 'image' is in the request files
        if "image" not in request.files:
            return (
                jsonify(
                    {
                        "status": {
                            "code": 400,
                            "message": "No image part in the request",
                        },
                        "data": None,
                    }
                ),
                400,
            )

        image = request.files["image"]

        # Check if the file is allowed
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            # Specify the path to save the image
            # upload_path = os.path.join("static/uploads", filename)

            # Save the image
            # image.save(upload_path)

            return (
                jsonify(
                    {
                        "status": {
                            "code": 200,
                            "message": "Image successfully uploaded",
                        },
                        "data": {"filename": filename},
                    }
                ),
                200,
            )
        else:
            return (
                jsonify(
                    {
                        "status": {
                            "code": 400,
                            "message": "Invalid file type or extension",
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
                    "status": {
                        "code": 405,
                        "message": "Method not allowed",
                    },
                    "data": None,
                }
            ),
            405,
        )


if __name__ == "__main__":
    app.run(debug=True)
