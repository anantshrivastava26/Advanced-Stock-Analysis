import React, { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  CartesianGrid,
  ReferenceDot,
} from "recharts";
import { format, parseISO } from "date-fns";

const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000";
const WINDOW_SIZE = 60; // must match backend

function formatDate(d) {
  try {
    return format(parseISO(d), "MMM dd");
  } catch {
    return d;
  }
}

export default function App() {
  const [ticker, setTicker] = useState("AAPL");
  const [data, setData] = useState([]);
  const [predicted, setPredicted] = useState(null);
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [trainStatus, setTrainStatus] = useState(null);
  const [error, setError] = useState(null);

  // === LOAD HISTORICAL & SENTIMENT DATA ===
  async function loadRealData() {
    setLoading(true);
    setError(null);
    setPredicted(null);

    try {
      const start = "2024-04-01";
      const histRes = await fetch(
        `${API_BASE}/historical?ticker=${encodeURIComponent(ticker)}&start=${encodeURIComponent(start)}`
      );
      const histJson = await histRes.json();
      if (!histRes.ok) throw new Error(histJson.detail || "Failed to fetch historical data");

      let merged = histJson.map((row) => ({
        date: row.Date,
        close: Number(row.Close),
      }));

      try {
        const sentRes = await fetch(
          `${API_BASE}/sentiment?ticker=${encodeURIComponent(ticker)}&start=${encodeURIComponent(start)}`
        );
        if (sentRes.ok) {
          const sentJson = await sentRes.json();
          const sentMap = Object.fromEntries(sentJson.map((s) => [s.date, s.sentiment]));
          merged = merged.map((d) => ({
            ...d,
            sentiment: sentMap[d.date] ?? 0.0,
          }));
        } else {
          merged = merged.map((d) => ({ ...d, sentiment: 0.0 }));
        }
      } catch {
        merged = merged.map((d) => ({ ...d, sentiment: 0.0 }));
      }

      if (merged.length === 0) throw new Error("No historical data found for ticker");

      setData(merged);
    } catch (err) {
      console.error("Error loading data:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  // === TRIGGER MODEL TRAINING ===
  async function triggerTrain() {
    setError(null);
    setTrainStatus(null);
    setTraining(true);

    try {
      const payload = { ticker, start: "2024-04-01", use_news: false, epochs: 20 };
      const resp = await fetch(`${API_BASE}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const json = await resp.json();
      if (!resp.ok) throw new Error(json.detail || "Failed to start training");
      setTrainStatus("started");

      // Poll /health until model_exists=true
      const maxTries = 60; // up to ~4 minutes if training is slow
      let tries = 0;
      while (tries < maxTries) {
        await new Promise((r) => setTimeout(r, 4000));
        tries += 1;
        try {
          const h = await fetch(`${API_BASE}/health`);
          const hjson = await h.json();
          if (hjson.model_exists) {
            setTrainStatus("completed");
            setTraining(false);
            await loadRealData();
            return;
          }
        } catch (e) {
          console.warn("Health check error while polling:", e);
        }
      }
      setTrainStatus("timeout");
      setTraining(false);
      setError("Training did not finish in expected time. Check server logs.");
    } catch (err) {
      console.error("Trigger train error:", err);
      setError(err.message || String(err));
      setTraining(false);
    }
  }

  // === MAKE PREDICTION ===
  async function predictNext() {
    setError(null);
    setPredicted(null);

    if (data.length < WINDOW_SIZE) {
      setError(`Need at least ${WINDOW_SIZE} days of data to predict.`);
      return;
    }

    setLoading(true);
    try {
      const prices = data.map((d) => d.close);
      const sents = data.map((d) => d.sentiment ?? 0.0);

      const resp = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prices, sents }),
      });

      const json = await resp.json();
      if (!resp.ok) throw new Error(json.detail || "Prediction failed");

      const lastDate = data[data.length - 1].date;
      const next = new Date(lastDate);
      next.setDate(next.getDate() + 1);
      setPredicted({ date: next.toISOString().slice(0, 10), price: json.predicted_price });
    } catch (err) {
      console.error("Predict error:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const chartData = predicted
    ? [...data, { date: predicted.date, close: predicted.price, predicted: true }]
    : data;

  return (
    <div style={styles.page}>
      <div style={styles.header}>
        <h2>Stock LSTM Dashboard</h2>
        <div style={styles.controls}>
          <label>Ticker:</label>
          <input
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            style={styles.input}
          />
          <button onClick={loadRealData} disabled={loading || training} style={styles.button}>
            {loading ? "Loading..." : "Load Data"}
          </button>
          <button onClick={predictNext} disabled={loading || training} style={styles.primary}>
            {loading ? "Predicting..." : "Predict Next"}
          </button>
          <button onClick={triggerTrain} disabled={training} style={styles.trainButton}>
            {training ? "Training..." : "Train Model"}
          </button>
        </div>
      </div>

      {error && <div style={styles.error}>{error}</div>}

      {trainStatus === "started" && (
        <div style={styles.info}>Training started... polling for completion ⏳</div>
      )}
      {trainStatus === "completed" && (
        <div style={styles.success}>✅ Model training completed successfully!</div>
      )}
      {trainStatus === "timeout" && (
        <div style={styles.error}>
          Training did not complete within expected time. Check backend logs.
        </div>
      )}

      <div style={styles.card}>
        <h4>Price Chart</h4>
        {data.length === 0 ? (
          <p>No data loaded yet.</p>
        ) : (
          <div style={{ width: "100%", height: 420 }}>
            <ResponsiveContainer>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tickFormatter={formatDate} minTickGap={20} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="close"
                  stroke="#1f77b4"
                  strokeWidth={2}
                  dot={false}
                  name="Close"
                />
                {predicted && (
                  <ReferenceDot
                    x={predicted.date}
                    y={predicted.price}
                    r={6}
                    label={{ position: "top", value: `Pred: ${predicted.price.toFixed(2)}` }}
                    stroke="red"
                    fill="red"
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {predicted && (
        <div style={styles.card}>
          <h4>Prediction</h4>
          <p>
            <b>Predicted Close for {predicted.date}: </b>
            {predicted.price.toFixed(2)}
          </p>
        </div>
      )}
    </div>
  );
}

const styles = {
  page: { fontFamily: "Inter, sans-serif", padding: 24, maxWidth: 1200, margin: "0 auto" },
  header: { display: "flex", justifyContent: "space-between", alignItems: "center" },
  controls: { display: "flex", gap: 8, alignItems: "center" },
  card: {
    background: "#fff",
    padding: 16,
    borderRadius: 8,
    boxShadow: "0 1px 6px rgba(0,0,0,0.08)",
    marginTop: 20,
  },
  input: { padding: 8, borderRadius: 6, border: "1px solid #ccc", width: 100 },
  button: {
    padding: "8px 12px",
    borderRadius: 6,
    border: "1px solid #ddd",
    background: "#f6f6f6",
    cursor: "pointer",
  },
  primary: {
    padding: "8px 12px",
    borderRadius: 6,
    border: "none",
    background: "#1f77b4",
    color: "#fff",
    cursor: "pointer",
  },
  trainButton: {
    padding: "8px 12px",
    borderRadius: 6,
    border: "none",
    background: "#16a34a",
    color: "#fff",
    cursor: "pointer",
  },
  error: {
    background: "#ffecec",
    color: "#9a1f1f",
    padding: 8,
    borderRadius: 6,
    marginTop: 12,
  },
  info: {
    background: "#fff8dc",
    color: "#665c00",
    padding: 8,
    borderRadius: 6,
    marginTop: 8,
  },
  success: {
    background: "#ecfdf5",
    color: "#065f46",
    padding: 8,
    borderRadius: 6,
    marginTop: 8,
  },
};
