const express = require('express');
const fetch = require('node-fetch');
const { RSI, SMA } = require('technicalindicators');
const tf = require('@tensorflow/tfjs-node'); // TensorFlow.js per Node
const path = require('path');

// Parametri
const SYMBOL = 'DOGE'; 
const INTERVAL = '5m';
const PORT = 3000;

const app = express();
app.use(express.static(path.join(__dirname, 'public')));

// Funzione per ottenere i dati da Binance
async function fetchCryptoTimeSeries(symbol, interval) {
    const url = `https://api.binance.com/api/v3/klines?symbol=${symbol}USDT&interval=${interval}`;
    try {
        const response = await fetch(url);
        const data = await response.json();

        const timeSeriesArray = data.map(entry => ({
            date: entry[0],
            open: parseFloat(entry[1]),
            high: parseFloat(entry[2]),
            low: parseFloat(entry[3]),
            close: parseFloat(entry[4]),
            volume: parseFloat(entry[5]),
            rsi: 0,
            sma: 0
        }));

        // Calcolo SMA (periodo 14)
        const smaArray = SMA.calculate({
            period: 14,
            values: timeSeriesArray.map(entry => entry.close)
        });

        // Calcolo RSI (periodo 14)
        const rsiArray = RSI.calculate({
            values: timeSeriesArray.map(entry => entry.close),
            period: 14
        });

        timeSeriesArray.forEach((entry, index) => {
            if (index >= 13) {
                entry.sma = smaArray[index - 13];
            }
        });

        timeSeriesArray.forEach((entry, index) => {
            if (index >= 14) {
                entry.rsi = rsiArray[index - 14];
            }
        });

        return timeSeriesArray;
    } catch (error) {
        console.error('Errore nel recupero delle time series:', error);
        return [];
    }
}

// Funzione per addestrare il modello e generare previsioni
async function trainAndPredictLSTM(timeSeriesData) {
    // Rimuoviamo eventuali righe incomplete (ad es. senza RSI o SMA)
    const filteredData = timeSeriesData.filter(entry => !isNaN(entry.sma) && !isNaN(entry.rsi));

    if (filteredData.length < 30) {
        throw new Error('Dati insufficienti per l\'addestramento.');
    }

    const maxOpen = Math.max(...filteredData.map(e => e.open));
    const maxHigh = Math.max(...filteredData.map(e => e.high));
    const maxLow = Math.max(...filteredData.map(e => e.low));
    const maxClose = Math.max(...filteredData.map(e => e.close));
    const maxSMA = Math.max(...filteredData.map(e => e.sma));
    const maxVolume = Math.max(...filteredData.map(e => e.volume));
    const maxRSI = Math.max(...filteredData.map(e => e.rsi));

    const normalizedData = filteredData.map(entry => ({
        date: new Date(entry.date),
        open: entry.open / maxOpen,
        high: entry.high / maxHigh,
        low: entry.low / maxLow,
        close: entry.close / maxClose,
        sma: (maxSMA === 0 ? 0 : entry.sma / maxSMA),
        volume: entry.volume / maxVolume,
        rsi: (maxRSI === 0 ? 0 : entry.rsi / maxRSI)
    }));

    const featureCount = Object.keys(normalizedData[0]).length - 1;
    const splitIndex = Math.floor(normalizedData.length * 0.8);
    const trainingData = normalizedData.slice(0, splitIndex);
    const testingData = normalizedData.slice(splitIndex);

    const inputSize = 7;
    const trainInputs = [];
    const trainLabels = [];

    for (let i = 0; i < trainingData.length - inputSize; i++) {
        const inputSequence = trainingData.slice(i, i + inputSize).map(e =>
            Object.keys(e).filter(k => k !== 'date').map(k => e[k])
        );
        const label = trainingData[i + inputSize].close;
        trainInputs.push(inputSequence);
        trainLabels.push(label);
    }

    const model = tf.sequential();
    model.add(tf.layers.lstm({
        units: 200,
        inputShape: [inputSize, featureCount],
        returnSequences: false
    }));
    model.add(tf.layers.dense({ units: 1 }));
    model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

    const xs = tf.tensor3d(trainInputs, [trainInputs.length, inputSize, featureCount]);
    const ys = tf.tensor2d(trainLabels, [trainLabels.length, 1]);

    const history = await model.fit(xs, ys, {
        epochs: 15,
        batchSize: 32,
        verbose: 0
    });

    // Previsioni sul set di test
    const testInputs = [];
    const testLabels = [];
    for (let i = 0; i < testingData.length - inputSize; i++) {
        const inputSequence = testingData.slice(i, i + inputSize).map(e => 
            Object.values(e).filter((_, index) => index !== 0)
        );
        const label = testingData[i + inputSize].close;
        testInputs.push(inputSequence);
        testLabels.push(label);
    }

    const testXs = tf.tensor3d(testInputs, [testInputs.length, inputSize, featureCount]);
    const testYs = tf.tensor2d(testLabels, [testLabels.length, 1]);

    const testPredictions = model.predict(testXs).dataSync();
    const denormalizedPredictions = Array.from(testPredictions).map(pred => pred * maxClose);

    const testDates = testingData.slice(inputSize).map(entry => entry.date);
    const testActual = testingData.slice(inputSize).map(entry => entry.close * maxClose);

    // Predizione futura
    let lastSequence = testingData.slice(-inputSize).map(e => 
        Object.values(e).filter((_, index) => index !== 0)
    );
    const inputTensor = tf.tensor3d([lastSequence], [1, inputSize, featureCount]);
    const futurePred = model.predict(inputTensor).dataSync()[0];
    const futurePrediction = futurePred * maxClose;

    const lastDenormalizedPrediction = denormalizedPredictions[denormalizedPredictions.length - 1];
    const percentageDifference = ((futurePrediction - lastDenormalizedPrediction) / lastDenormalizedPrediction) * 100;

    // Risultati da inviare al frontend
    return {
        // Dati storici (per grafico principale)
        historical: filteredData.map(d => ({
            date: d.date,
            close: d.close * maxClose
        })),
        // Dati di training (non necessari al frontend se non per debug)
        // Dati di test e predizioni
        testDates: testDates,
        testActual: testActual,
        testPredictions: denormalizedPredictions,
        futureDate: testDates[testDates.length - 1], 
        futurePrediction: futurePrediction,
        percentageDifference: percentageDifference,
        lossHistory: history.history.loss // trend della loss
    };
}

let cachedResults = null;
let lastFetchTime = 0;

// Endpoint per fornire i dati al frontend
app.get('/api/results', async (req, res) => {
    try {
        const now = Date.now();
        // Possiamo implementare una cache per non ricalcolare tutto ogni volta
        if (!cachedResults || (now - lastFetchTime) > 5 * 60 * 1000) {
            const data = await fetchCryptoTimeSeries(SYMBOL, INTERVAL);
            const results = await trainAndPredictLSTM(data);
            cachedResults = results;
            lastFetchTime = now;
        }
        res.json(cachedResults);
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Errore interno del server' });
    }
});

app.listen(PORT, () => {
    console.log(`Server in ascolto sulla porta ${PORT}`);
});