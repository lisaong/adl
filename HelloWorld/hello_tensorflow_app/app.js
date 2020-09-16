const express = require('express');
const app = express();
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

async function loadModel(url) {
  return await tf.loadLayersModel(url);
}

// load model from relative path
var model;
loadModel('file://model/model.json').then((result) => {
    model = result;
});

app.get('/', function (req, res) {
  if (model != null) {
    res.send('<div>' + JSON.stringify(model) + '</div>');
  } else {
    res.send('Hello world');
  }
});

const port = process.env.PORT || 3000;
app.listen(port, function () {
  console.log('myapp listening on port ' + port);
});