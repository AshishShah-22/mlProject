from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,predictPipeline


application= Flask(__name__)

app = application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdate',methods=['GET','POST'])

def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score')),
        )


        pred_df = data.get_data_as_data_frame()

        pred_df.rename(columns={
            "race_ethnicity": "race/ethnicity",
            "parental_level_of_education": "parental level of education",
            "test_preparation_course": "test preparation course",
            "reading_score": "reading score",
            "writing_score": "writing score"
        }, inplace=True)

        predict_pipeline = predictPipeline()
        results = predict_pipeline.predict(pred_df)


        if results is None:
            return "prediction failed"

        if isinstance(results, (list, np.ndarray)):
            result_value = results[0]
        else:
            result_value = results

        return render_template('home.html', results=result_value)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)