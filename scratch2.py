import joblib
pipeline = joblib.load('artifacts/model_pipeline.pkl')
print(pipeline.classes_)
