// IT: Funzione per ottenere le serie temporali delle criptovalute dal servizio Binance.
// EN: Function to retrieve cryptocurrency time series data from the Binance API.
async function getCryptoTimeSeries(symbol) {
    // IT: Costruisce l'URL dell'endpoint Binance. Qui utilizziamo l'intervallo a 1 giorno.
    // EN: Construct the Binance API URL endpoint. We use a 1-day interval here.
    const url = `https://api.binance.com/api/v3/klines?symbol=${symbol}USDT&interval=5m`;
    try {
        // IT: Effettua la richiesta fetch per ottenere i dati.
        // EN: Make a fetch request to retrieve the data.
        const response = await fetch(url);
        const data = await response.json();

        // IT: Mappa i dati restituendo un array di oggetti con i valori necessari.
        // EN: Map the data to return an array of objects with the required values.
        const timeSeriesArray = data.map(entry => ({
            date: entry[0],  // IT: Converte il timestamp in una data ISO // EN: Convert the timestamp to an ISO date string
            open: parseFloat(entry[1]),
            high: parseFloat(entry[2]),
            low: parseFloat(entry[3]),
            close: parseFloat(entry[4]),
            volume: parseFloat(entry[5]),
            rsi: 0, // IT: Inizialmente 0, verrà calcolato poi // EN: Initially 0, will be computed later
            sma: 0  // IT: Inizialmente 0, verrà calcolato poi // EN: Initially 0, will be computed later
        }));

        // IT: Calcola la SMA (Simple Moving Average) con periodo 5.
        // EN: Calculate the SMA (Simple Moving Average) with a period of 5.
        const smaArray = SMA.calculate({
            period: 14, 
            values: timeSeriesArray.map(entry => entry.close)
        });

        // IT: Calcola l'RSI (Relative Strength Index) con periodo 14.
        // EN: Calculate the RSI (Relative Strength Index) with a period of 14.
        const rsiArray = RSI.calculate({
            values: timeSeriesArray.map(entry => entry.close), 
            period: 14
        });

        // IT: Assegna i valori SMA a partire dal 5° elemento.
        // EN: Assign the SMA values starting from the 5th element.
        timeSeriesArray.forEach((entry, index) => {
            if (index >= 13) {
                entry.sma = smaArray[index - 13];
            }
        });

        // IT: Assegna i valori RSI a partire dal 15° elemento.
        // EN: Assign the RSI values starting from the 15th element.
        timeSeriesArray.forEach((entry, index) => {
            if (index >= 14) {
                entry.rsi = rsiArray[index - 14];
            }
        });

        console.log(timeSeriesArray); // IT: Mostra i dati in console // EN: Log the data to console

        return timeSeriesArray; // IT: Ritorna l'array con i dati della serie temporale // EN: Return the time series array
    } catch (error) {
        console.error('Errore nel recupero delle time series:', error); // IT: Stampa errore // EN: Log error
        return []; // IT: Ritorna array vuoto in caso di errore // EN: Return empty array in case of an error
    }
}

// IT: Funzione per addestrare un modello LSTM sui dati storici e per prevedere i prezzi futuri.
// EN: Function to train an LSTM model on historical data and predict future prices.
async function trainAndPredictLSTM(timeSeriesData) {

    // IT: Normalizza i dati dividendo ciascun valore per il massimo dell'intero set.
    // EN: Normalize the data by dividing each value by the maximum value of the entire set.
    const normalizedData = timeSeriesData.map(entry => ({
        date: new Date(entry.date),
        open: entry.open / Math.max(...timeSeriesData.map(e => e.open)),
        high: entry.high / Math.max(...timeSeriesData.map(e => e.high)),
        low: entry.low / Math.max(...timeSeriesData.map(e => e.low)),
        close: entry.close / Math.max(...timeSeriesData.map(e => e.close)),
        //sma: entry.sma / Math.max(...timeSeriesData.map(e => e.sma)),
        //volume: entry.volume / Math.max(...timeSeriesData.map(e => e.volume)),
        //rsi: entry.rsi / Math.max(...timeSeriesData.map(e => e.rsi))
    }));

    console.log(Math.max(...timeSeriesData.map(e => e.rsi))); // IT: Mostra il massimo RSI // EN: Log max RSI
    console.log(normalizedData); // IT: Mostra i dati normalizzati // EN: Log normalized data

    // IT: Determina il numero di caratteristiche escludendo la data.
    // EN: Determine the number of features excluding the date.
    const featureCount = Object.keys(normalizedData[0]).length - 1;

    // IT: Suddivide i dati in set di training (80%) e testing (20%).
    // EN: Split the data into training set (80%) and testing set (20%).
    const splitIndex = Math.floor(normalizedData.length * 0.8);
    const trainingData = normalizedData.slice(0, splitIndex);
    const testingData = normalizedData.slice(splitIndex);

    // IT: Imposta la dimensione della finestra di input (7 giorni).
    // EN: Set the input window size (7 days).
    const inputSize = 7;
    const trainInputs = [];
    const trainLabels = [];

    // IT: Crea sequenze di input e relativi label (valore successivo di "close").
    // EN: Create input sequences and corresponding labels (the next "close" value).
    for (let i = 0; i < trainingData.length - inputSize; i++) {
        const inputSequence = trainingData.slice(i, i + inputSize).map(e => {
            // IT: Per ogni elemento, prende tutte le proprietà tranne la data.
            // EN: For each element, take all properties except the date.
            return Object.keys(e).filter(key => key !== 'date').map(key => e[key]);
        });

        // IT: L'etichetta è il valore "close" del giorno successivo.
        // EN: The label is the "close" value of the next day.
        const label = trainingData[i + inputSize].close;

        trainInputs.push(inputSequence);
        trainLabels.push(label);
    }

    // IT: Definisce il modello LSTM.
    // EN: Define the LSTM model.
    const model = tf.sequential();
    model.add(tf.layers.lstm({
        units: 100,
        inputShape: [inputSize, featureCount],
        returnSequences: false
    }));
    // IT: Aggiunge uno strato denso finale per la previsione.
    // EN: Add a final dense layer for prediction.
    model.add(tf.layers.dense({ units: 1 }));

    model.compile({
        optimizer: 'adam',
        loss: 'meanSquaredError'
    });

    // IT: Converte i dati in tensori per TensorFlow.js.
    // EN: Convert the data into tensors for TensorFlow.js.
    const xs = tf.tensor3d(trainInputs, [trainInputs.length, inputSize, featureCount]);
    const ys = tf.tensor2d(trainLabels, [trainLabels.length, 1]);

    // IT: Addestra il modello LSTM.
    // EN: Train the LSTM model.
    const history = await model.fit(xs, ys, {
        epochs: 15,
        batchSize: 32,
        verbose: 1
    });

    // IT: Traccia l'andamento della loss durante l'addestramento.
    // EN: Plot the training loss over epochs.
    const lossTrace = {
        x: Array.from({ length: history.epoch.length }, (_, i) => i + 1),
        y: history.history.loss,
        type: 'scatter',
        mode: 'lines+markers',
        marker: { color: 'green' },
        name: 'Errore di Addestramento' // IT: Training Error // EN: Training Error
    };

    const lossLayout = {
        title: 'Andamento Errore di Addestramento',  // IT: Training Loss Trend
        xaxis: { title: 'Epoca' }, // IT: Epoch
        yaxis: { title: 'Errore (Loss)' } // IT: Error (Loss)
    };

    // IT: Visualizza il grafico della loss.
    // EN: Display the loss chart.
    Plotly.newPlot('lossChart', [lossTrace], lossLayout);

    // IT: Prepara i dati di testing per la previsione.
    // EN: Prepare the testing data for prediction.
    const testInputs = [];
    const testLabels = [];

    // IT: Crea sequenze di test e label corrispondenti.
    // EN: Create test sequences and their corresponding labels.
    for (let i = 0; i < testingData.length - inputSize; i++) {
        const inputSequence = testingData.slice(i, i + inputSize).map(e => 
            Object.values(e).filter((_, index) => index !== 0) // IT: Rimuove la data // EN: Remove the date field
        );
        const label = testingData[i + inputSize].close;
        testInputs.push(inputSequence);
        testLabels.push(label);
    }

    const testXs = tf.tensor3d(testInputs, [testInputs.length, inputSize, featureCount]);
    const testYs = tf.tensor2d(testLabels, [testLabels.length, 1]);

    // IT: Prevede i risultati sui dati di test.
    // EN: Predict results on the test data.
    const testPredictions = model.predict(testXs).dataSync();

    // IT: Denormalizza le predizioni moltiplicandole per il max dei prezzi di chiusura originali.
    // EN: Denormalize predictions by multiplying them by the max of the original closing prices.
    const denormalizedPredictions = Array.from(testPredictions).map(pred => 
        pred * Math.max(...timeSeriesData.map(e => e.close))
    );

    // IT: Prepara i dati per la visualizzazione dei risultati di testing.
    // EN: Prepare data for displaying test results.
    const testDates = testingData.slice(inputSize).map(entry => entry.date);
    const testTrace = {
        x: testDates,
        y: testingData.slice(inputSize).map(entry => entry.close * Math.max(...timeSeriesData.map(e => e.close))),
        type: 'scatter',
        mode: 'lines+markers',
        marker: { color: 'blue' },
        name: 'Dati di Testing' // IT: Testing Data // EN: Testing Data
    };

    const predictionTrace = {
        x: testDates,
        y: denormalizedPredictions,
        type: 'scatter',
        mode: 'markers',
        marker: { color: 'red', size: 10 },
        name: 'Predizioni' // IT: Predictions // EN: Predictions
    };


    // IT: Calcola una predizione futura aggiuntiva oltre i dati di testing.
    // EN: Calculate one additional future prediction beyond the test data.
    let lastSequence = testingData.slice(-inputSize).map(e => 
        Object.values(e).filter((_, index) => index !== 0)
    );
    const inputTensor = tf.tensor3d([lastSequence], [1, inputSize, featureCount]);
    const prediction = model.predict(inputTensor).dataSync()[0];
    const futurePrediction = prediction * Math.max(...timeSeriesData.map(e => e.close));

    // IT: Calcola la differenza percentuale tra l'ultima predizione denormalizzata e la predizione futura.
    // EN: Calculate the percentage difference between the last denormalized prediction and the future prediction.
    const lastDenormalizedPrediction = denormalizedPredictions[denormalizedPredictions.length - 1];
    const percentageDifference = ((futurePrediction - lastDenormalizedPrediction) / lastDenormalizedPrediction) * 100;

    console.log(`Differenza percentuale predetta: ${percentageDifference.toFixed(2)}%`); // IT: Mostra la differenza percentuale // EN: Log the percentage difference

    // IT: Calcola la data futura (il giorno successivo all'ultimo dato di test).
    // EN: Compute the future date (one day after the last test date).
    const lastDate = new Date(testDates[testDates.length - 1]);
    const futureDate = lastDate;

    const futureTrace = {
        x: [futureDate],
        y: [futurePrediction],
        type: 'scatter',
        mode: 'markers',
        marker: { color: 'orange', size: 10 },
        name: 'Predizione Futura' // IT: Future Prediction // EN: Future Prediction
    };

    const testLayout = {
        title: 'Predizioni sui Dati di Testing e Future', // IT: Predictions on Testing and Future Data
        xaxis: { title: 'Data' },
        yaxis: { title: 'Prezzo di Chiusura (USD)' }, // IT: Closing Price (USD)
        showlegend: true,
        legend: { x: 0, y: 1.1, orientation: 'h' }
    };

    // IT: Visualizza il grafico dei dati di test, predizioni e predizione futura.
    // EN: Display the chart for test data, predictions, and the future prediction.
    Plotly.newPlot('testChart', [testTrace, predictionTrace, futureTrace], testLayout);
}

// IT: Al caricamento dello script, recupera i dati storici di BTC, visualizzali e addestra il modello.
// EN: On script load, fetch BTC historical data, display it, and train the model.
getCryptoTimeSeries('BTC').then(data => {
    const dates = data.map(entry => entry.date);
    const closePrices = data.map(entry => entry.close);

    // IT: Traccia l'andamento del prezzo di chiusura nel tempo.
    // EN: Plot the closing price trend over time.
    const trace = {
        x: dates,
        y: closePrices,
        type: 'scatter',
        mode: 'lines+markers',
        marker: { color: 'blue' },
        name: 'Prezzo di Chiusura' // IT: Closing Price // EN: Closing Price
    };

    const layout = {
        title: 'Andamento Prezzo di Chiusura BTC', // IT: BTC Closing Price Trend
        xaxis: { title: 'Data' },
        yaxis: { title: 'Prezzo di Chiusura (USD)' },
        showlegend: true,
        legend: { x: 0, y: 1.1, orientation: 'h' }
    };

    // IT: Mostra il grafico iniziale del prezzo di chiusura.
    // EN: Display the initial closing price chart.
    Plotly.newPlot('cryptoChart', [trace], layout);

    // IT: Addestra e prevede con il modello LSTM.
    // EN: Train and predict with the LSTM model.
    trainAndPredictLSTM(data);
});
