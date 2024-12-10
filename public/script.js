async function fetchResults(symbol, interval) {
    const response = await fetch(`/api/results?symbol=${symbol}&interval=${interval}`);
    const results = await response.json();
    return results;
}

function plotResults(results) {
    // Grafico storico
    const historicalTrace = {
        x: results.historical.map(d => new Date(d.date)),
        y: results.historical.map(d => d.close),
        type: 'scatter',
        mode: 'lines+markers',
        marker: { color: 'blue' },
        name: 'Prezzo di Chiusura Storico'
    };

    const layoutHistorical = {
        title: 'Andamento Prezzo di Chiusura Storico',
        xaxis: { title: 'Data' },
        yaxis: { title: 'Prezzo (USD)' }
    };
    Plotly.newPlot('cryptoChart', [historicalTrace], layoutHistorical);

    // Grafico Errore di Addestramento
    const lossTrace = {
        x: results.lossHistory.map((_, i) => i+1),
        y: results.lossHistory,
        type: 'scatter',
        mode: 'lines+markers',
        marker: { color: 'green' },
        name: 'Errore di Addestramento'
    };

    const lossLayout = {
        title: 'Andamento Errore di Addestramento',
        xaxis: { title: 'Epoca' },
        yaxis: { title: 'Loss' }
    };
    Plotly.newPlot('lossChart', [lossTrace], lossLayout);

    // Grafico Test + Predizioni
    const testTrace = {
        x: results.testDates.map(d => new Date(d)),
        y: results.testActual,
        type: 'scatter',
        mode: 'lines+markers',
        marker: { color: 'blue' },
        name: 'Dati di Testing'
    };

    const predictionTrace = {
        x: results.testDates.map(d => new Date(d)),
        y: results.testPredictions,
        type: 'scatter',
        mode: 'markers',
        marker: { color: 'red', size: 8 },
        name: 'Predizioni'
    };

    const futureTrace = {
        x: [new Date(results.futureDate)],
        y: [results.futurePrediction],
        type: 'scatter',
        mode: 'markers',
        marker: { color: 'orange', size: 10 },
        name: 'Predizione Futura'
    };

    const testLayout = {
        title: 'Predizioni sui Dati di Testing e Future',
        xaxis: { title: 'Data' },
        yaxis: { title: 'Prezzo (USD)' },
        showlegend: true,
        legend: { x: 0, y: 1.1, orientation: 'h' }
    };
    Plotly.newPlot('testChart', [testTrace, predictionTrace, futureTrace], testLayout);
}

document.getElementById('updateBtn').addEventListener('click', async () => {
    const symbol = document.getElementById('symbol').value;
    const interval = document.getElementById('interval').value;
    const results = await fetchResults(symbol, interval);
    plotResults(results);
});

// Carica dati di default all'avvio (es: DOGE, 5m)
window.addEventListener('DOMContentLoaded', async () => {
    const symbol = document.getElementById('symbol').value;
    const interval = document.getElementById('interval').value;
    const results = await fetchResults(symbol, interval);
    plotResults(results);
});
