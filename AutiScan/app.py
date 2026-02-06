from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
lr_model, nn_model = pickle.load(open("model/autism_model.pkl", "rb"))


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/screening')
def screening():
    return render_template("screening.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = []

    for i in range(1, 11):
        data.append(float(request.form[f"q{i}"]))

    data.append(float(request.form["age"]))
    data.append(float(request.form["gender"]))
    data.append(float(request.form["jundice"]))
    data.append(float(request.form["austim"]))

    # AI Ensemble prediction
    lr_prob = lr_model.predict_proba([data])[0][1]
    nn_prob = nn_model.predict_proba([data])[0][1]

    final_prob = (lr_prob + nn_prob) / 2
    percent = round(final_prob * 100, 2)

    aq_score = sum(data[:10])

    if percent < 30:
        spectrum = "Low Risk"
    elif percent < 60:
        spectrum = "Moderate Risk"
    else:
        spectrum = "High Risk"

    confidence = round(abs(final_prob - 0.5) * 200, 2)

    return render_template(
        "result.html",
        percent=percent,
        spectrum=spectrum,
        aq_score=aq_score,
        confidence=confidence
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
