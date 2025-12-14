import React, { useEffect, useMemo, useState } from "react";
import { Streamlit, withStreamlitConnection, Theme } from "streamlit-component-lib";

type ResultItem = {
  path: string;
  url?: string;
  score?: number;
  caption?: string;
  metadata?: Record<string, unknown>;
};

type GridArgs = {
  results?: ResultItem[];
  columns?: number;
  thumbHeight?: number;
  showScore?: boolean;
  initialSelection?: string[];
};

type GridProps = {
  args?: GridArgs;
  disabled?: boolean;
  theme?: Theme;
};

const formatScore = (value?: number) => (value === undefined ? "" : value.toFixed(3));

const ResultGridComponent = ({ args }: GridProps) => {
  const { results = [], columns = 3, thumbHeight = 220, showScore = true, initialSelection = [] } = args || {};
  const [selection, setSelection] = useState<Set<string>>(new Set(initialSelection));

  useEffect(() => {
    Streamlit.setComponentReady();
  }, []);

  useEffect(() => {
    Streamlit.setFrameHeight();
  });

  const toggle = (path: string) => {
    setSelection((prev) => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      Streamlit.setComponentValue(Array.from(next));
      return next;
    });
  };

  const items = useMemo(
    () =>
      (results as ResultItem[]).map((item) => {
        const meta = item.metadata || {};
        const when =
          typeof meta["datetime_iso"] === "string"
            ? meta["datetime_iso"]
            : typeof meta["year"] === "number"
              ? String(meta["year"])
              : "";
        const where =
          typeof meta["latitude"] === "number" && typeof meta["longitude"] === "number"
            ? `${meta["latitude"]?.toFixed(3)}, ${meta["longitude"]?.toFixed(3)}`
            : "";
        return {
          ...item,
          selected: selection.has(item.path),
          when,
          where,
        };
      }),
    [results, selection],
  );

  return (
    <div
      className="grid"
      style={{
        gridTemplateColumns: `repeat(${Math.max(columns, 1)}, minmax(0, 1fr))`,
      }}
    >
      {items.map((item) => (
        <button
          key={item.path}
          className={`card ${item.selected ? "selected" : ""}`}
          onClick={() => toggle(item.path)}
          type="button"
        >
          <div className="thumb" style={{ height: `${thumbHeight}px` }}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={item.url || item.path} alt={item.caption || item.path} loading="lazy" />
            {item.selected && <div className="pill pill-selected">Selected</div>}
            {showScore && item.score !== undefined && (
              <div className="pill pill-score">Score {formatScore(item.score)}</div>
            )}
            {(item.caption || item.when || item.where) && (
              <div className="overlay">
                <div className="overlay-text">
                  {item.caption || (item.path || "").split(/[/\\]/).pop() || "Photo"}
                </div>
                <div className="overlay-meta">
                  {showScore && item.score !== undefined && <span>Score {formatScore(item.score)}</span>}
                  {item.when && <span>{item.when}</span>}
                  {item.where && <span>{item.where}</span>}
                </div>
              </div>
            )}
          </div>
        </button>
      ))}
      {items.length === 0 && <div className="empty">No results</div>}
    </div>
  );
};

export const ResultGrid = withStreamlitConnection(ResultGridComponent);
