from flask import Flask, render_template,request

app = Flask(__name__)

# Define routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/services")
def services():
    return render_template("services.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/sms", methods=['GET', 'POST'])
def project_2():
    prediction = None
    if request.method == 'POST':
        message = request.form['message']
        # Preprocess the message and use the loaded model for predictions
        prediction = random_forest.predict([message])[0]

    return render_template('sms_prediction.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
