import React, { useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid
} from "recharts";

export default function App() {
  const [ticker, setTicker] = useState("AAPL");
  const [model, setModel] = useState("random_forest");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const models = [
    { value: "linear", label: "Linear Regression" },
    { value: "random_forest", label: "Random Forest" },
    { value: "gb", label: "Gradient Boosting" },
    { value: "xgboost", label: "XGBoost" },
    { value: "lstm", label: "LSTM" }
  ];

  async function runPrediction(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    setResult(null);
    try {
      const resp = await fetch("http://localhost:8000/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker, model })
      });
      if (!resp.ok) {
        const j = await resp.json();
        throw new Error(j.detail || `Server error ${resp.status}`);
      }
      const data = await resp.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const chartData = (result && result.dates)
    ? result.dates.map((d, i) => ({ date: d, actual: result.actual[i], predicted: result.predicted[i] }))
    : [];

  return (
    <div className="app">
      <div className="card">
        <h2>StockSense — Predict & Visualize</h2>
        <form className="controls" onSubmit={runPrediction}>
          <input className="input" value={ticker} onChange={e => setTicker(e.target.value.toUpperCase())} />
          <select className="select" value={model} onChange={e => setModel(e.target.value)}>
            {models.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
          </select>
          <button className="button" type="submit" disabled={loading}>{loading ? "Running..." : "Run"}</button>
        </form>

        {error && <div style={{ color: "red" }}>{error}</div>}

        <div style={{ display: "flex", gap: 12 }}>
          <div style={{ flex: 2 }} className="card">
            <h3>Predictions</h3>
            <div style={{ height: 360 }}>
              {chartData.length ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="actual" stroke="#2563eb" dot={false} />
                    <Line type="monotone" dataKey="predicted" stroke="#f97316" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ padding: 20, color: "#6b7280" }}>Run prediction to show chart</div>
              )}
            </div>
          </div>

          <div style={{ flex: 1 }} className="card">
            <h3>Next Day Prediction</h3>
            <div style={{ marginTop: 12 }}>
              <div className="metric">
                {result ? Number(result.next_day_prediction).toFixed(2) : "—"}
              </div>
              <div style={{ color: "#6b7280", marginTop: 8 }}>Model: {result?.model_name ?? "—"}</div>

              <div style={{ marginTop: 20 }}>
                <h4>Metrics</h4>
                <div>MAE: {result?.metrics?.mae ? result.metrics.mae.toFixed(3) : "—"}</div>
                <div>Avg Sentiment: {result?.metrics?.avg_sentiment ? result.metrics.avg_sentiment.toFixed(3) : "—"}</div>
              </div>
            </div>
          </div>
        </div>

        <div style={{ marginTop: 12 }} className="card">
          <h3>Latest Headlines</h3>
          <div>
            {result?.headlines?.length ? result.headlines.map((h, i) => (
              <div key={i} className="headline">
                <div>{h}</div>
                <div style={{ color: "#6b7280", fontSize: 12 }}>Sentiment: {result.sentiment?.[i]?.toFixed?.(3) ?? "-"}</div>
              </div>
            )) : <div style={{ color: "#6b7280" }}>No headlines yet</div>}
          </div>
        </div>

      </div>
    </div>
  );
}
