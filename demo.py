# Batch Prediction
# training Pipeline
from Insurance.pipeline.batch_prediction import start_batch_prediction
from Insurance.pipeline.training_pipeline import start_training_pipeline

# file_path=r"C:\Users\GAUTAM\Desktop\Data Science End to End Project\Data Science Pipeline\Project_1\insurance.csv"

if __name__=='main':
    try:
        # output=start_batch_prediction(input_file_path=file_path)
        output=start_training_pipeline()
        print(output)
    except Exception as e:
        print(e)