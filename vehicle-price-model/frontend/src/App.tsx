import { useState, useEffect, FormEvent } from "react";
import "./App.css";

// ─── Types ───────────────────────────────────────────────────────────────────
type VehicleCategory = "cars" | "suv";

interface CarFormData {
  brand:        string;
  model:        string;
  variant:      string;
  model_year:   number;
  mileage_km:   number;
  fuel_type:    string;
  transmission: string;
  description:  string;
}

interface SUVFormData {
  brand:        string;
  model:        string;
  variant:      string;
  model_year:   number;
  mileage_km:   number;
  fuel_type:    string;
  transmission: string;
  engine_cc:    number;
  description:  string;
}

interface PredictionResult {
  predicted_price:      number;
  model_used:           string;
  vehicle_age:          number;
  mileage_per_year:     number;
  used_mileage_km:      number;
  is_mileage_estimated: boolean;
  confidence:           string;
  nlp_score:            number | null;
  nlp_signals:          string[] | null;
  nlp_verdict:          string | null;
}

interface Metadata {
  brands: string[];
  models: Record<string, string[]>;
}

type AppState = "idle" | "loading" | "success" | "error";

// ─── Constants ───────────────────────────────────────────────────────────────
const FUEL_TYPES    = ["Petrol", "Hybrid", "Diesel", "Electric"];
const TRANSMISSIONS = ["Automatic", "Manual"];
const CURRENT_YEAR  = new Date().getFullYear();

const DEFAULT_CAR_FORM: CarFormData = {
  brand:        "",
  model:        "",
  variant:      "Standard",
  model_year:   CURRENT_YEAR - 5,
  mileage_km:   50000,
  fuel_type:    "Petrol",
  transmission: "Automatic",
  description:  "",
};

const DEFAULT_SUV_FORM: SUVFormData = {
  brand:        "",
  model:        "",
  variant:      "Standard",
  model_year:   CURRENT_YEAR - 5,
  mileage_km:   60000,
  fuel_type:    "Petrol",
  transmission: "Automatic",
  engine_cc:    1500,
  description:  "",
};

// ─── Helpers ──────────────────────────────────────────────────────────────────
function formatLKR(amount: number): string {
  return new Intl.NumberFormat("en-LK", {
    style:                 "currency",
    currency:              "LKR",
    maximumFractionDigits: 0,
  }).format(amount);
}

function getVehicleAge(year: number): number {
  return CURRENT_YEAR - year;
}

// ─── Component ────────────────────────────────────────────────────────────────
export default function App() {
  const [category, setCategory] = useState<VehicleCategory>("cars");

  const [carForm,  setCarForm]  = useState<CarFormData>(DEFAULT_CAR_FORM);
  const [suvForm,  setSuvForm]  = useState<SUVFormData>(DEFAULT_SUV_FORM);

  const [carMeta, setCarMeta] = useState<Metadata>({ brands: [], models: {} });
  const [suvMeta, setSuvMeta] = useState<Metadata>({ brands: [], models: {} });
  const [metaError, setMetaError] = useState<string>("");

  const [state,    setState]    = useState<AppState>("idle");
  const [result,   setResult]   = useState<PredictionResult | null>(null);
  const [errorMsg, setErrorMsg] = useState<string>("");

  // ─── Fetch metadata on mount ──────────────────────────────────────────────
  useEffect(() => {
    const BASE = "http://localhost:8000";
    Promise.all([
      fetch(`${BASE}/metadata/cars`).then((r) => r.json()),
      fetch(`${BASE}/metadata/suv`).then((r) => r.json()),
    ])
      .then(([cars, suv]: [Metadata, Metadata]) => {
        setCarMeta(cars);
        setSuvMeta(suv);
      })
      .catch(() =>
        setMetaError(
          "Could not load brand/model lists from the server. Make sure it's running at localhost:8000."
        )
      );
  }, []);

  // ─── Derived ─────────────────────────────────────────────────────────────
  const isCarMode = category === "cars";
  const meta      = isCarMode ? carMeta : suvMeta;
  const form      = isCarMode ? carForm : suvForm;

  const availableModels: string[] =
    form.brand && meta.models[form.brand] ? meta.models[form.brand] : [];

  const vehicleAge = getVehicleAge(form.model_year);

  // ─── Handlers ─────────────────────────────────────────────────────────────
  function handleCategorySwitch(next: VehicleCategory) {
    if (next === category) return;
    setCategory(next);
    setState("idle");
    setResult(null);
    setErrorMsg("");
  }

  function handleCarChange(
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>
  ) {
    const { name, value } = e.target;
    setCarForm((prev) => ({
      ...prev,
      [name]:
        name === "model_year" || name === "mileage_km"
          ? Number(value)
          : value,
      ...(name === "brand" ? { model: "" } : {}),
    }));
  }

  function handleSuvChange(
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>
  ) {
    const { name, value } = e.target;
    setSuvForm((prev) => ({
      ...prev,
      [name]:
        name === "model_year" || name === "mileage_km" || name === "engine_cc"
          ? Number(value)
          : value,
      ...(name === "brand" ? { model: "" } : {}),
    }));
  }

  const handleChange = isCarMode ? handleCarChange : handleSuvChange;

  function setFuelType(ft: string) {
    if (isCarMode) setCarForm((p) => ({ ...p, fuel_type: ft }));
    else           setSuvForm((p) => ({ ...p, fuel_type: ft }));
  }

  function setTransmission(tr: string) {
    if (isCarMode) setCarForm((p) => ({ ...p, transmission: tr }));
    else           setSuvForm((p) => ({ ...p, transmission: tr }));
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setState("loading");
    setResult(null);
    setErrorMsg("");

    const endpoint = isCarMode
      ? "http://localhost:8000/api/predict"
      : "http://localhost:8000/api/predict/suv";

    const payload = {
      ...form,
      description: form.description.trim() || undefined,
    };

    try {
      const res = await fetch(endpoint, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(payload),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(
          (data as { detail?: string })?.detail ?? `Server error: ${res.status}`
        );
      }

      const data: PredictionResult = await res.json();
      setResult(data);
      setState("success");
    } catch (err: unknown) {
      const msg =
        err instanceof TypeError && err.message.includes("fetch")
          ? "Could not reach the prediction server. Make sure it's running at localhost:8000."
          : err instanceof Error
            ? err.message
            : "An unknown error occurred.";
      setErrorMsg(msg);
      setState("error");
    }
  }

  // ─── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="app-shell">
      {/* Header */}
      <header className="app-header">
        <div className="header-inner">
          <div className="logo-mark">VP</div>
          <div>
            <h1 className="app-title">Vehicle Price Predictor</h1>
            <p className="app-subtitle">Sri Lanka Market · ML-Powered Estimate</p>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="app-main">
        {/* Form Panel */}
        <section className="form-panel">
          <div className="panel-heading">
            <span className="panel-step">01</span>
            <h2 className="panel-title">Vehicle Details</h2>
          </div>

          {/* Category Toggle */}
          <div className="category-toggle">
            <button
              type="button"
              id="category-cars"
              className={`category-pill${category === "cars" ? " category-pill--active" : ""}`}
              onClick={() => handleCategorySwitch("cars")}
            >
              <span className="category-icon">🚗</span> Cars
            </button>
            <button
              type="button"
              id="category-suv"
              className={`category-pill${category === "suv" ? " category-pill--active" : ""}`}
              onClick={() => handleCategorySwitch("suv")}
            >
              <span className="category-icon">🚙</span> SUVs
            </button>
          </div>

          {metaError && (
            <p className="meta-error">⚠ {metaError}</p>
          )}

          <form onSubmit={handleSubmit} id="predict-form" className="predict-form">
            {/* Brand + Model */}
            <div className="form-row">
              <div className="field-group">
                <label htmlFor="brand" className="field-label">Brand</label>
                {meta.brands.length > 0 ? (
                  <select
                    id="brand"
                    name="brand"
                    value={form.brand}
                    onChange={handleChange}
                    required
                    className="field-select"
                  >
                    <option value="">Select brand</option>
                    {meta.brands.map((b) => (
                      <option key={b} value={b}>{b}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    id="brand"
                    name="brand"
                    type="text"
                    value={form.brand}
                    onChange={handleChange}
                    required
                    placeholder="e.g. Toyota"
                    className="field-input"
                  />
                )}
              </div>

              <div className="field-group">
                <label htmlFor="model" className="field-label">Model</label>
                {availableModels.length > 0 ? (
                  <select
                    id="model"
                    name="model"
                    value={form.model}
                    onChange={handleChange}
                    required
                    className="field-select"
                  >
                    <option value="">Select model</option>
                    {availableModels.map((m) => (
                      <option key={m} value={m}>{m}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    id="model"
                    name="model"
                    type="text"
                    value={form.model}
                    onChange={handleChange}
                    required
                    placeholder="e.g. Raize"
                    className="field-input"
                  />
                )}
              </div>
            </div>

            {/* Variant + Model Year */}
            <div className="form-row">
              <div className="field-group">
                <label htmlFor="variant" className="field-label">Variant</label>
                <input
                  id="variant"
                  name="variant"
                  type="text"
                  value={form.variant}
                  onChange={handleChange}
                  placeholder="e.g. GLS / Standard"
                  className="field-input"
                />
              </div>

              <div className="field-group">
                <label htmlFor="model_year" className="field-label">
                  Model Year
                  <span className="field-label-badge">
                    {vehicleAge === 0 ? "New" : `${vehicleAge}yr old`}
                  </span>
                </label>
                <input
                  id="model_year"
                  name="model_year"
                  type="number"
                  min={1990}
                  max={CURRENT_YEAR}
                  value={form.model_year}
                  onChange={handleChange}
                  required
                  className="field-input"
                />
              </div>
            </div>

            {/* Mileage */}
            <div className="form-row">
              <div className="field-group field-group--full">
                <label htmlFor="mileage_km" className="field-label">
                  Mileage (km)
                  <span className="field-label-badge">
                    {form.mileage_km.toLocaleString()} km
                  </span>
                </label>
                <div className="mileage-wrapper">
                  <input
                    id="mileage_km"
                    name="mileage_km"
                    type="range"
                    min={0}
                    max={500000}
                    step={1000}
                    value={form.mileage_km}
                    onChange={handleChange}
                    className="field-range"
                  />
                  <input
                    type="number"
                    name="mileage_km"
                    min={0}
                    max={500000}
                    value={form.mileage_km}
                    onChange={handleChange}
                    className="field-input mileage-input"
                  />
                </div>
              </div>
            </div>

            {/* SUV-only: Engine CC */}
            {!isCarMode && (
              <div className="form-row">
                <div className="field-group field-group--full">
                  <label htmlFor="engine_cc" className="field-label">
                    Engine Capacity (CC)
                    <span className="field-label-badge suv-badge">SUV Required</span>
                  </label>
                  <input
                    id="engine_cc"
                    name="engine_cc"
                    type="number"
                    min={650}
                    max={8000}
                    step={1}
                    value={(form as SUVFormData).engine_cc}
                    onChange={handleChange}
                    required
                    placeholder="e.g. 1490"
                    className="field-input"
                  />
                </div>
              </div>
            )}

            {/* Fuel + Transmission */}
            <div className="form-row">
              <div className="field-group">
                <label className="field-label">Fuel Type</label>
                <div className="pill-group">
                  {FUEL_TYPES.map((ft) => (
                    <button
                      key={ft}
                      type="button"
                      className={`pill${form.fuel_type === ft ? " pill--active" : ""}`}
                      onClick={() => setFuelType(ft)}
                    >
                      {ft}
                    </button>
                  ))}
                </div>
              </div>

              <div className="field-group">
                <label className="field-label">Transmission</label>
                <div className="pill-group">
                  {TRANSMISSIONS.map((tr) => (
                    <button
                      key={tr}
                      type="button"
                      className={`pill${form.transmission === tr ? " pill--active" : ""}`}
                      onClick={() => setTransmission(tr)}
                    >
                      {tr}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Optional: Listing Description for NLP */}
            <div className="form-row">
              <div className="field-group field-group--full">
                <label htmlFor="description" className="field-label">
                  Listing Description
                  <span className="field-label-badge optional-badge">Optional · NLP Score</span>
                </label>
                <textarea
                  id="description"
                  name="description"
                  value={form.description}
                  onChange={handleChange}
                  placeholder="e.g. One owner, accident free, full option, service records…"
                  className="field-input field-textarea"
                  rows={2}
                />
              </div>
            </div>

            {/* Submit */}
            <button
              type="submit"
              className={`submit-btn${state === "loading" ? " submit-btn--loading" : ""}`}
              disabled={state === "loading"}
            >
              {state === "loading" ? (
                <>
                  <span className="spinner" />
                  Predicting…
                </>
              ) : (
                <>
                  <span className="btn-icon">⚡</span>
                  Predict {isCarMode ? "Car" : "SUV"} Price
                </>
              )}
            </button>
          </form>
        </section>

        {/* Result Panel */}
        <section className="result-panel">
          <div className="panel-heading">
            <span className="panel-step">02</span>
            <h2 className="panel-title">Estimated Price</h2>
          </div>

          {state === "idle" && (
            <div className="result-placeholder">
              <div className="placeholder-icon">
                {isCarMode ? "🚗" : "🚙"}
              </div>
              <p className="placeholder-text">
                Fill in vehicle details and click<br />
                <strong>Predict {isCarMode ? "Car" : "SUV"} Price</strong> to get an estimate.
              </p>
            </div>
          )}

          {state === "loading" && (
            <div className="result-placeholder">
              <div className="loading-ring">
                <div /><div /><div /><div />
              </div>
              <p className="placeholder-text">Running model…</p>
            </div>
          )}

          {state === "error" && (
            <div className="result-error">
              <span className="error-icon">⚠</span>
              <p className="error-title">Prediction Failed</p>
              <p className="error-msg">{errorMsg}</p>
              <button className="retry-btn" onClick={() => setState("idle")}>
                Try Again
              </button>
            </div>
          )}

          {state === "success" && result && (
            <div className="result-card">
              {/* Price */}
              <p className="result-label">Predicted Market Price</p>
              <p className="result-price">{formatLKR(result.predicted_price)}</p>

              {/* Confidence + NLP Verdict */}
              <div className="result-badges-row">
                <span className={`confidence-badge confidence-badge--${result.confidence.toLowerCase().split(" ")[0]}`}>
                  {result.confidence === "High"   && "◉ High Confidence"}
                  {result.confidence === "Medium" && "◎ Medium Confidence"}
                  {result.confidence === "Low"    && "○ Low Confidence"}
                  {result.confidence === "Unknown" && "? Unknown"}
                </span>
                {result.nlp_verdict && (
                  <span className="nlp-verdict-badge">{result.nlp_verdict}</span>
                )}
              </div>

              {/* Meta badges */}
              <div className="result-meta">
                <span className="meta-badge">
                  📅 {vehicleAge === 0 ? "Brand New" : `${vehicleAge} Year${vehicleAge !== 1 ? "s" : ""} Old`}
                </span>
                <span className="meta-badge">⛽ {form.fuel_type}</span>
                <span className="meta-badge">⚙ {form.transmission}</span>
                <span className="meta-badge">🛣 {result.used_mileage_km.toLocaleString()} km
                  {result.is_mileage_estimated && " (est.)"}
                </span>
                {!isCarMode && (
                  <span className="meta-badge">🔧 {(form as SUVFormData).engine_cc} cc</span>
                )}
              </div>

              {/* NLP Signal Flags */}
              {result.nlp_signals && result.nlp_signals.length > 0 && (
                <div className="nlp-flags">
                  <p className="nlp-flags-label">Listing Signals</p>
                  <div className="nlp-flags-row">
                    {result.nlp_signals.map((sig) => (
                      <span
                        key={sig}
                        className={`nlp-flag${sig.startsWith("⚠") ? " nlp-flag--negative" : ""}`}
                      >
                        {sig}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Vehicle Name */}
              <div className="result-vehicle-name">
                {form.brand} {form.model}
                {form.variant && form.variant !== "Standard" && (
                  <span className="vehicle-variant"> · {form.variant}</span>
                )}
                <span className="vehicle-year"> ({form.model_year})</span>
              </div>

              <p className="result-disclaimer">
                * Estimate based on ML model trained on Sri Lanka {isCarMode ? "car" : "SUV"} market data.
                Actual resale prices may vary.
                {result.is_mileage_estimated && " Mileage was estimated from vehicle age."}
              </p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
