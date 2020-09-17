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

async function predict(model, data) {
  let input = tf.tensor(data);
  return await model.predict(input).array();
}

// load model and artifacts from relative path
let predictions = null;
let data = null;
loadModel('file://model/model.json').then(model => {
    let artifacts = loadModelArtifacts('model_artifacts.json');
    data = artifacts['X_test'].slice(0, 5);
    predict(model, data).then(result => {
      predictions = result;
    })
});

app.get('/', function (req, res) {
  if (predictions != null) {
    var response = '';
    for (var i=0; i<predictions.length && i < data.length; i++) {
      response += '<div>' + JSON.stringify(data[i]) + ': '
               + JSON.stringify(predictions[i]) + '</div>';
    }
    res.send(response);
  } else {
    res.send('Hello world');
  }
});

const port = process.env.PORT || 3000;
app.listen(port, function () {
  console.log('myapp listening on port ' + port);
});