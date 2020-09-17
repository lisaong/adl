const express = require('express');
const app = express();
const fs = require('fs')
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

// to parse POST requests
const bodyParser = require('body-parser');
app.use(bodyParser.urlencoded({extended: true})); // parse form data
app.use(bodyParser.json()) // parse json data

// functions
async function loadModel(url) {
  return await tf.loadLayersModel(url);
}

function loadModelArtifacts(path) {
  return JSON.parse(fs.readFileSync(path, 'utf-8'));
}

function predict(model, data) {
  let input = tf.tensor(data);
  return model.predict(input).arraySync();
}

// load model and artifacts from relative path
let model = null;
let artifacts = null;
loadModel('file://model/model.json').then(m => {
    artifacts = loadModelArtifacts('model_artifacts.json');
    model = m;
});


app.get('/', (req, res) => {
  // form to get predictions
  res.send('<form action="/" method="post">' +
   '<div>5-day stock quotes for TSLA (comma-separated):<div/>' +
   '<input type="text" size=50 name="data" value="70.12,71.114,71.136,70.92,71.118" />' +
   '<input type="submit" value="Predict day 6"/>' +
   '</form>');
});


app.post('/', (req, res) => {
  // convert to floats
  const ar = req.body.data.split(',');
  let data = Array();
  for (var i=0; i<ar.length; i++) {
    data[i] = parseFloat(ar[i]);
  }
  let predictions = predict(model, [data]);
  return res.send('TSLA day 6: ' +
    JSON.stringify(predictions[0]));
});

const port = process.env.PORT || 3000;
app.listen(port, function () {
  console.log('myapp listening on port ' + port);
});