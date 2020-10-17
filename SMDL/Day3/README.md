## Day 3: Neural Machine Translation (Seq2Seq)

### Setup
1. Train translation model
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
3. Go to http://localhost:8080. Follow the prompt to type an input text in English, and the bot will reply with a translated sentence in German.

The example below is after training for 500 epochs.

![example](example1.png)

Possible meaning?

![google](google_translate.png)