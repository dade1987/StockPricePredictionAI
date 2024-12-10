const express = require('express');
const fetch = require('node-fetch');
const { RSI, SMA } = require('technicalindicators');
const tf = require('@tensorflow/tfjs-node');
const path = require('path');

// Parametri di default
const DEFAULT_SYMBOL = 'BTC';
const DEFAULT_INTERVAL = '1d';

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
            smaFast: 0,
            smaSlow:0,
            smaSignal:0
        }));

        // Calcolo SMA
        const smaFastArray = SMA.calculate({
            period: 5,
            values: timeSeriesArray.map(entry => entry.close)
        });
        const smaSlowArray = SMA.calculate({
            period: 8,
            values: timeSeriesArray.map(entry => entry.close)
        });
        const smaSignalArray = SMA.calculate({
            period: 3,
            values: timeSeriesArray.map(entry => entry.close)
        });

        // Calcolo RSI (periodo 14)
        const rsiArray = RSI.calculate({
            values: timeSeriesArray.map(entry => entry.close),
            period: 14
        });

        timeSeriesArray.forEach((entry, index) => {
            if (index >= 4) {
                entry.smaFast = smaFastArray[index - 4];
            }
            if (index >= 7) {
                entry.smaSlow = smaSlowArray[index - 7];
            }
            if (index >= 2) {
                entry.smaSignal = smaSignalArray[index - 2];
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
    const filteredData = timeSeriesData;


    const maxOpen = Math.max(...filteredData.map(e => e.open));
    const maxHigh = Math.max(...filteredData.map(e => e.high));
    const maxLow = Math.max(...filteredData.map(e => e.low));
    const maxClose = Math.max(...filteredData.map(e => e.close));
    const maxSMAFast = Math.max(...filteredData.map(e => e.smaFast));
    const maxSMASlow = Math.max(...filteredData.map(e => e.smaSlow));
    const maxSMASignal = Math.max(...filteredData.map(e => e.smaSignal));
    const maxVolume = Math.max(...filteredData.map(e => e.volume));
    const maxRSI = Math.max(...filteredData.map(e => e.rsi));

    const normalizedData = filteredData.map(entry => ({
        date: new Date(entry.date),
        open: entry.open / maxOpen,
        high: entry.high / maxHigh,
        low: entry.low / maxLow,
        close: entry.close / maxClose,
        //smaFast: (maxSMAFast === 0 ? 0 : entry.smaFast / maxSMAFast),
        //smaSlow: (maxSMASlow === 0 ? 0 : entry.smaSlow / maxSMASlow),
        //smaSignal: (maxSMASignal === 0 ? 0 : entry.smaSignal / maxSMASignal),
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
        units: 100,
        returnSequences: true,
        inputShape: [inputSize, featureCount]
    }));
    model.add(tf.layers.lstm({
        units: 50,
        returnSequences: false
    }));
    model.add(tf.layers.dense({ units: 1 }));
    model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

    const xs = tf.tensor3d(trainInputs, [trainInputs.length, inputSize, featureCount]);
    const ys = tf.tensor2d(trainLabels, [trainLabels.length, 1]);

    const history = await model.fit(xs, ys, {
        epochs: 50,
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

    return {
        historical: filteredData.map(d => ({
            date: d.date,
            close: d.close * maxClose
        })),
        testDates: testDates,
        testActual: testActual,
        testPredictions: denormalizedPredictions,
        futureDate: testDates[testDates.length - 1],
        futurePrediction: futurePrediction,
        percentageDifference: percentageDifference,
        lossHistory: history.history.loss
    };
}

// Endpoint per fornire i dati al frontend con parametri
app.get('/api/results', async (req, res) => {
    const symbol = req.query.symbol || DEFAULT_SYMBOL;
    const interval = req.query.interval || DEFAULT_INTERVAL;
    try {
        const data = await fetchCryptoTimeSeries(symbol, interval);
        const results = await trainAndPredictLSTM(data);
        res.json(results);
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Errore interno del server' });
    }
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server in ascolto sulla porta ${PORT}`);
});
