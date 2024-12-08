# README

## Italiano

Questo progetto dimostra come eseguire la previsione del prezzo di criptovalute (ad esempio BTC) utilizzando un modello LSTM in TensorFlow.js. Il flusso è il seguente:

- Recupero dei dati storici delle criptovalute dall’API Binance.
- Calcolo di indicatori tecnici base (SMA, RSI).
- Normalizzazione dei dati e addestramento di un modello LSTM per prevedere i futuri prezzi di chiusura.
- Visualizzazione dei risultati (dati storici, errore di training, previsioni su dati di test e una previsione futura) tramite Plotly.js.
- Utilizzo di Tailwind CSS per lo stile della pagina.

### Prerequisiti

- Connessione a internet (il progetto usa CDN per TensorFlow.js, Plotly.js, Tailwind CSS e `technicalindicators`).

### Come eseguire

1. Clona o scarica il repository.
2. Apri il file `index.html` in un browser web.
3. Attendi il caricamento dei dati e l'addestramento del modello (qualche secondo).
4. Visualizza i grafici:
   - Andamento storico dei prezzi di chiusura.
   - Errore di addestramento (loss).
   - Predizioni su dati di test e una predizione futura.

### Struttura dei file

- `index.html`: Contiene la struttura della pagina web e i riferimenti ai CDN.
- `script.js`: Contiene la logica per l’ottenimento dei dati, il pre-processing, il training del modello, le previsioni e la generazione dei grafici.

### Note

- Questo esempio è solo a scopo didattico, non fornisce consigli finanziari.
- L'accuratezza delle previsioni dipende dalla qualità/quantità dei dati, dal tuning del modello e dalla complessità dei mercati.

---

## English

This project demonstrates how to perform cryptocurrency price prediction (e.g. BTC) using an LSTM model in TensorFlow.js. The workflow is as follows:

- Fetch historical cryptocurrency data from the Binance API.
- Compute basic technical indicators (SMA, RSI).
- Normalize the data and train an LSTM model to predict future closing prices.
- Visualize results (historical data, training loss, test predictions, and one future prediction) using Plotly.js.
- Use Tailwind CSS for styling the page.

### Prerequisites

- Internet connection (the project uses CDN for TensorFlow.js, Plotly.js, Tailwind CSS, and `technicalindicators`).

### How to Run

1. Clone or download the repository.
2. Open `index.html` in a web browser.
3. Wait a few seconds for data loading and model training.
4. View the charts:
   - Historical closing price.
   - Training loss.
   - Test predictions and one future prediction.

### File Structure

- `index.html`: Contains the webpage structure and CDN references.
- `script.js`: Contains the logic for data fetching, preprocessing, model training, predictions, and chart rendering.

### Notes

- This example is for educational purposes only and does not provide financial advice.
- Prediction accuracy depends on data quality/quantity, model tuning, and market complexity.
