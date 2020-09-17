const express = require('express');
const app = express();
const fs = require('fs')
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

// functions
async function loadModel(url) {
  return await tf.loadLayersModel(url);
}

function loadModelArtifacts(path) {
  return JSON.parse(fs.readFileSync(path, 'utf-8'));
}

function predict(model, artifacts) {
  console.log(artifacts['X_test'][0]);
}

// load model and artifacts from relative path
let model = null;
let artifacts = null;
loadModel('file://model/model.json').then((result) => {
    artifacts = loadModelArtifacts('model_artifacts.json');
    model = result;
});

app.get('/', function (req, res) {
  if (model != null) {
    // do a test prediction
    predict(model, artifacts);

    res.send('<div>' + JSON.stringify(model) + '</div>');
  } else {
    res.send('Hello world');
  }
});

const port = process.env.PORT || 3000;
app.listen(port, function () {
  console.log('myapp listening on port ' + port);
});