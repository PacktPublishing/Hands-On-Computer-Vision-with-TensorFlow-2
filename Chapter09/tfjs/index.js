import * as tf from '@tensorflow/tfjs';
import * as faceapi from 'face-api.js';

const isDev = process.env.NODE_ENV === 'development';

const MOBILENET_MODEL_PATH = isDev ? 'http://localhost:1235/emotion_detection/model.json' : './emotion_detection/model.json';
const DETECTION_MODEL_PATH = isDev ? 'http://localhost:1235/face_detection': './face_detection';
const FACE_EXPRESSIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]


const IMAGE_SIZE = 48;

let mobilenet;
const mobilenetDemo = async () => {
  status('Loading model...');

  mobilenet = await tf.loadGraphModel(MOBILENET_MODEL_PATH);

  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  mobilenet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1])).dispose();

  status('');
};

/**
 * Given an image element, makes a prediction through mobilenet returning the
 * probabilities of the top K classes.
 */
async function predict(imgElement) {
  let img = await tf.browser.fromPixels(imgElement, 3).toFloat();

  const logits = tf.tidy(() => {
    // tf.fromPixels() returns a Tensor from an image element.
    img = tf.image.resizeBilinear(img, [IMAGE_SIZE, IMAGE_SIZE]);
    img = img.mean(2);
    const offset = tf.scalar(127.5);
    // Normalize the image from [0, 255] to [-1, 1].
    const normalized = img.sub(offset).div(offset);

    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 1]);

    // Make a prediction through mobilenet.
    return mobilenet.predict(batched);
  });

  return logits
}

/**
 * Computes the probabilities of the topK classes given logits by computing
 * softmax to get probabilities and then sorting the probabilities.
 * @param logits Tensor representing the logits from MobileNet.
 * @param topK The number of top predictions to show.
 */
export async function getTopClass(values) {

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });

  return valuesAndIndices[0]
}

//
// UI
//


const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;
const predictionsElement = document.getElementById('predictions');



window.onPlay = async function onPlay() {
  const video = document.getElementById('video');
  const overlay = document.getElementById('overlay');

  const detection = await faceapi.detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())

  if (detection) {

    const faceCanvases = await faceapi.extractFaces(video, [detection])

    const prediction = await predict(faceCanvases[0]);

    const values = await prediction.data();
    const topClass = await getTopClass(values)

    // TODO(eliot): fix this hack. we should not use private properties
    detection._className = FACE_EXPRESSIONS[topClass.index]
    detection._classScore = topClass.value
    drawDetections(video, overlay, detection)

  }

  setTimeout(window.onPlay, 100)
};

function resizeCanvasAndResults(dimensions, canvas, results) {
  const { width, height } = dimensions instanceof HTMLVideoElement
    ? faceapi.getMediaDimensions(dimensions)
    : dimensions
  canvas.width = width
  canvas.height = height

  // resize detections (and landmarks) in case displayed image is smaller than
  // original size
  return faceapi.resizeResults(results, { width, height })
}

function drawDetections(dimensions, canvas, detections) {
  const resizedDetections = resizeCanvasAndResults(dimensions, canvas, detections)
  faceapi.drawDetection(canvas, resizedDetections)
}

async function init() {
  var video = document.getElementById('video');

  await faceapi.loadTinyFaceDetectorModel(DETECTION_MODEL_PATH)
  const stream = await navigator.mediaDevices.getUserMedia({ video: {} })
  video.srcObject = stream
};

window.onload = async function () {
  await mobilenetDemo()
  init()
}
