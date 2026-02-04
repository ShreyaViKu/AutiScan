from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open("model/autism_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/screening')
def screening():
    return render_template("screening.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = []

    # AQ-10 answers (already mapped to 0/1 from form)
    for i in range(1, 11):
        data.append(float(request.form[f"q{i}"]))

    # Age group mapped to numeric age
    data.append(float(request.form['age']))

    # Other details
    data.append(float(request.form['gender']))
    data.append(float(request.form['jundice']))
    data.append(float(request.form['austim']))

    prob = model.predict_proba([data])[0][1]
    percent = round(prob * 100, 2)
    aq_score = sum(data[:10])

    if percent < 30:
        spectrum = "Low Risk"
    elif percent < 60:
        spectrum = "Moderate Risk"
    else:
        spectrum = "High Risk"

    return render_template(
        "result.html",
        percent=percent,
        spectrum=spectrum,
        aq_score=aq_score
    )

@app.route('/awareness')
def awareness():
    return render_template("awareness.html")

@app.route('/doctors')
def doctors():
    return render_template("doctors.html")

@app.route('/community')
def community():
    return render_template("community.html")

if __name__ == "__main__":
    app.run(debug=True)
