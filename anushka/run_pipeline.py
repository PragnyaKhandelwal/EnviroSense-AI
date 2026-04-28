import time
from member4_model import run_pipeline


def main():
    while True:
        try:
            run_pipeline()
            print("Pipeline executed successfully")
        except Exception as e:
            print("Error:", e)

        time.sleep(60)


if __name__ == "__main__":
    main()