import React from "react";
import { createRoot } from "react-dom/client";
import { ResultGrid } from "./ResultGrid";
import "./style.css";

createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <ResultGrid />
  </React.StrictMode>,
);
