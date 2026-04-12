import time
from member4_model import run_pipeline

while True:
    try:
        run_pipeline()
        print("Pipeline executed successfully")
    except Exception as e:
        print("Error:", e)

    time.sleep(60)