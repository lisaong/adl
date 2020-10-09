## Day 2: Basic Text Generation

### Setup
1. Train model to generate text (GPU preferred).
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
3. Go to http://localhost:8080. Type your chat responses, and the bot will reply with a sentence that is generated from choosing a random-selected seed word from your chat message.

    <img src="example.png" width=50%></img>
