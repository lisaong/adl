from flask import render_template, request, Response, session
from demo import app
from demo import ml
import os
import cv2

# https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/
from werkzeug.utils import secure_filename

model_dir = os.path.join(os.getcwd(), 'demo', 'model')
model = ml.TFModel(model_dir=model_dir)


@app.route('/', methods=['GET'])
def default():
    return render_template("default.html")


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filename = secure_filename(file.filename)
    complete_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(complete_filename)

    # call the TFModel class to predict
    prediction, probability = model.predict(complete_filename)
    print(prediction, probability)

    # save the full video path on the client session to be rendered
    # by the video_feed route
    session['video_path'] = complete_filename
    return render_template('result.html',
                           prediction=f'{prediction} {probability:.3f}')


# https://blog.miguelgrinberg.com/post/video-streaming-with-flask
class VideoRenderer(object):
    def __init__(self, video_path):
        # capture video
        self.video = cv2.VideoCapture(video_path)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        # extract a frame
        ret, frame = self.video.read()
        if ret:
            # encode raw frame to JPEG and display it
            frame = cv2.resize(frame, (480, 480), interpolation=cv2.INTER_LINEAR)
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        else:
            return None


def gen(camera):
    # generator for the video feed
    while True:
        # render each frame
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            # no more frames, stop
            break


@app.route('/video_feed')
def video_feed():
    video_path = session.get('video_path', None)
    return Response(gen(VideoRenderer(video_path)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
