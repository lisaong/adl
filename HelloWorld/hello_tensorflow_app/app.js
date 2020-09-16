const express = require('express');
const app = express();
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

app.get('/', function (req, res) {
  res.send('Hello World!');
});

const port = process.env.PORT || 3000;
app.listen(port, function () {
  console.log('myapp listening on port ' + port);
});