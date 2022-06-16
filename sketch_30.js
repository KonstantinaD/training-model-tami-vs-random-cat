let tamiImages = [];
let otherCatImages = [];

function preload() {
    for (let i = 0; i < 30; i++) {
        tamiImages[i] = loadImage(`data/tami${i + 1}.png`);
        otherCatImages[i] = loadImage(`data/cat${i + 1}.png`);
    }
}

let imageClassifier;

function setup() {
    createCanvas(400, 400);
    // image(tamiImages[0], 0, 0, width, height);

const IMAGE_WIDTH = 64;
const IMAGE_HEIGHT = 64;
const IMAGE_CHANNELS = 4;

let options = {
    inputs: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
    // outputs: ['label'],
    task: 'imageClassification',
    debug: true
};

imageClassifier = ml5.neuralNetwork(options);

for (let i = 0; i < tamiImages.length; i++) {
    imageClassifier.addData({ image: tamiImages[i] }, { label: 'Tami' });
    imageClassifier.addData({ image: otherCatImages[i] }, { label: 'A Random Cat' });
}

imageClassifier.normalizeData();
console.log(imageClassifier);
imageClassifier.train({ epochs: 50 }, finishedTraining);
}

function finishedTraining() {
    console.log('Finished training!');
    imageClassifier.save();
}