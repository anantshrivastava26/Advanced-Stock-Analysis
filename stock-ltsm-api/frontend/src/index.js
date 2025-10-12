// src/index.js (or src/main.jsx)
import React from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.jsx";   // <-- explicitly point to App.jsx

const root = createRoot(document.getElementById("root"));
root.render(<App />);
