## Day 4: Video Classification

### Setup
1. Train  model
    ```
    cd train

    # You must complete the _ANS_ portions so that you can train the model
    python train.py
    ```
   
2. Create a file in `app/.flaskenv` that contains this:
   ```
    FLASK_APP=run.py
    FLASK_ENV=development
    FLASK_RUN_PORT=8080
   ```     
   Run the flask app:
   ```   
    cd app
    flask run
   ```
3. Go to http://localhost:8080. Upload a video to be classified.

![form](form.png)

![result](result.png)
