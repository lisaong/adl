## Day 1: Sentiment Classification

### Walkthrough video
https://youtu.be/w4dj3MfjM-8

### Setup
1. Install Python dependencies
   ```
    pip install -r ../requirements.txt
   ```
2. Train model to predict sentiment of text.
   ```
    cd train

    # You must complete the _ANS_ portions so that you can train the model
    python train.py
   ```
3. Create a file in `app/.flaskenv` that contains this:
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
4. Go to http://localhost:8080. Type your chat responses, and the bot will reply with "You seem *sentiment*".

    ![example](example.png)
