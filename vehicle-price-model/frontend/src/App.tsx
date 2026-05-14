import React, { useState, useMemo } from 'react';
import {
  Car,
  Settings,
  ChevronRight,
  Trophy,
  CheckCircle2,
  Activity,
  BarChart3,
  Gauge,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { VEHICLE_MODELS, VARIANTS, ML_MODELS, type ModelMetrics } from './data/carData';

/* ─── Constants ──────────────────────────────────────────────────────── */
const CHART_COLORS = ['#10b981', '#3b82f6', '#6366f1'];
const TOOLTIP_STYLE = {
  contentStyle: {
    backgroundColor: '#1e293b',
    border: '1px solid #334155',
    borderRadius: '8px',
    fontSize: '13px',
    color: '#e2e8f0',
  },
  itemStyle: { color: '#e2e8f0' },
};

/* ─── App ────────────────────────────────────────────────────────────── */
const App: React.FC = () => {
  const [model, setModel] = useState(VEHICLE_MODELS[0]);
  const [variant, setVariant] = useState('141');
  const [year, setYear] = useState(VARIANTS['141'].years[0]);
  const [transmission, setTransmission] = useState(VARIANTS['141'].transmissions[0]);
  const [fuelType, setFuelType] = useState(VARIANTS['141'].fuelTypes[0]);
  const [isPredicted, setIsPredicted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [livePredictions, setLivePredictions] = useState<any[]>([]);
  const [sampleSize, setSampleSize] = useState<number | null>(null);
  const [yearRange, setYearRange] = useState<string | null>(null);

  const variantInfo = VARIANTS[variant];

  /* ── Handlers ─────────────────────────────────────────────────────── */
  const handleModelChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const m = e.target.value;
    setModel(m);
    // Set default variant per vehicle
    const first = m === 'Toyota Aqua' ? 'Aqua'
                : m === 'Suzuki Alto' ? 'G1_Manual_2000-2012'
                : '141';
    setVariant(first);
    setYear(VARIANTS[first].years[0]);
    setTransmission(VARIANTS[first].transmissions[0]);
    setFuelType(VARIANTS[first].fuelTypes[0]);
    setIsPredicted(false);
  };

  const handleVariantChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const v = e.target.value;
    setVariant(v);
    setYear(VARIANTS[v].years[0]);
    setTransmission(VARIANTS[v].transmissions[0]);
    setFuelType(VARIANTS[v].fuelTypes[0]);
    setIsPredicted(false);
  };

  const bestModel = useMemo(() => {
    if (livePredictions.length === 0) return ML_MODELS[0];
    return [...livePredictions].sort((a, b) => {
      if (a.mae !== b.mae) return a.mae - b.mae;
      if (a.rmse !== b.rmse) return a.rmse - b.rmse;
      return b.r2 - a.r2;
    })[0];
  }, [livePredictions]);

  const handleAnalyze = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, model_year: year, variant, transmission, fuel_type: fuelType }),
      });
      const data = await res.json();
      if (data.status === 'success') {
        setLivePredictions(data.predictions);
        setSampleSize(data.sample_size);
        setYearRange(data.year_range);
        setIsPredicted(true);
      } else {
        alert('Error: ' + data.message);
      }
    } catch {
      alert('Failed to connect to backend. Make sure app.py is running on port 5000.');
    } finally {
      setLoading(false);
    }
  };

  const recommendedPrice = bestModel?.predictedPrice || 0;

  const formatLKR = (val: number) =>
    new Intl.NumberFormat('en-LK', { style: 'currency', currency: 'LKR', maximumFractionDigits: 0 }).format(val);

  const formatNumber = (val: number) => new Intl.NumberFormat('en-US').format(Math.round(val));

  const getConfidenceLabel = (r2: number) => {
    if (r2 >= 0.9) return { label: 'Excellent', color: 'text-emerald-400' };
    if (r2 >= 0.8) return { label: 'Very Good', color: 'text-emerald-400' };
    if (r2 >= 0.7) return { label: 'Good', color: 'text-blue-400' };
    if (r2 >= 0.5) return { label: 'Moderate', color: 'text-amber-400' };
    return { label: 'Low', color: 'text-slate-400' };
  };

  /* ── Chart data ───────────────────────────────────────────────────── */
  const chartData = livePredictions.length > 0 ? livePredictions : ML_MODELS;

  /* ── Render ────────────────────────────────────────────────────────── */
  return (
    <div className="min-h-screen py-10 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">

      {/* ─── Header ──────────────────────────────────────────────────── */}
      <header className="text-center mb-12">
        <div className="inline-flex items-center gap-3 mb-4">
          <Car className="w-8 h-8 text-emerald-500" />
          <h1 className="text-3xl font-bold text-white">AI Vehicle Price Predictor</h1>
        </div>
        <p className="text-slate-400 text-sm max-w-xl mx-auto">
          Machine learning ensemble analysis for fair market value estimation
        </p>
      </header>

      {/* ─── Main Grid ───────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">

        {/* ── Form Panel ────────────────────────────────────────────── */}
        <div className="lg:col-span-4">
          <div className="card p-6 space-y-5">
            <div className="flex items-center gap-2 pb-4 border-b border-slate-800">
              <Settings className="w-4 h-4 text-emerald-500" />
              <h2 className="text-base font-semibold text-white">Vehicle Configuration</h2>
            </div>

            <div className="space-y-4">
              <Field label="Vehicle Model">
                <select className="custom-dropdown" value={model} onChange={handleModelChange}>
                  {VEHICLE_MODELS.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </Field>

              <Field label="Variant / Generation">
                <select className="custom-dropdown" value={variant} onChange={handleVariantChange}>
                  {model === 'Toyota Corolla' ? (
                    <>
                      <option value="121">Corolla 121</option>
                      <option value="141">Corolla 141</option>
                      <option value="AE110">Corolla AE110</option>
                      <option value="DX/KE72">Corolla DX/KE72</option>
                    </>
                  ) : model === 'Toyota Aqua' ? (
                    <option value="Aqua">Aqua (Standard)</option>
                  ) : (
                    <>
                      <option value="G1_Manual_2000-2012">Manual 800cc (2000-2012)</option>
                      <option value="G2_Manual_2013-2015">Manual 800cc (2013-2015)</option>
                      <option value="G3_Manual_2016-2019">Manual 800cc (2016-2019)</option>
                      <option value="G4_Auto_lt700_2000-2015">Auto 660cc (2000-2015)</option>
                    </>
                  )}
                </select>
              </Field>

              <Field label="Year of Manufacture">
                <select className="custom-dropdown" value={year} onChange={e => setYear(Number(e.target.value))}>
                  {variantInfo.years.map(y => <option key={y} value={y}>{y}</option>)}
                </select>
              </Field>

              <Field label="Transmission">
                <select className="custom-dropdown" value={transmission} onChange={e => setTransmission(e.target.value)}>
                  {variantInfo.transmissions.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
              </Field>

              <Field label="Fuel Type">
                <select className="custom-dropdown" value={fuelType} onChange={e => setFuelType(e.target.value)}>
                  {variantInfo.fuelTypes.map(f => <option key={f} value={f}>{f}</option>)}
                </select>
              </Field>
            </div>

            <button
              onClick={handleAnalyze}
              disabled={loading}
              className="w-full bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-500 text-white font-semibold py-3 rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              {loading ? (
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <>
                  Analyze Fair Price
                  <ChevronRight className="w-4 h-4" />
                </>
              )}
            </button>
          </div>
        </div>

        {/* ── Results Panel ─────────────────────────────────────────── */}
        <div className="lg:col-span-8 space-y-6">
          {isPredicted ? (
            <>
              {/* Primary Result */}
              <div className="card p-6">
                <div className="flex items-start justify-between mb-6">
                  <div>
                    <p className="text-xs text-slate-400 uppercase tracking-wide font-medium mb-1">Recommended Fair Price</p>
                    <div className="text-4xl font-bold text-white">{formatLKR(recommendedPrice)}</div>
                  </div>
                  <span className="inline-flex items-center gap-1.5 bg-emerald-500/10 text-emerald-400 px-3 py-1 rounded-md text-xs font-semibold border border-emerald-500/20">
                    <Trophy className="w-3.5 h-3.5" />
                    Best Model
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 pt-5 border-t border-slate-800">
                  {/* Model Info */}
                  <div className="space-y-3">
                    <div className="flex items-center gap-2 text-slate-400">
                      <Activity className="w-4 h-4" />
                      <span className="text-xs font-medium uppercase tracking-wide">ML Engine</span>
                    </div>
                    <p className="text-lg font-semibold text-white">{bestModel.name}</p>
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-slate-400">R² Confidence:</span>
                      <span className={`text-sm font-semibold ${getConfidenceLabel(bestModel.r2).color}`}>
                        {(bestModel.r2 * 100).toFixed(1)}%
                      </span>
                    </div>
                    {yearRange && (
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-slate-500">Analysis Range:</span>
                        <span className="text-xs bg-slate-800 text-slate-300 px-2 py-0.5 rounded font-mono">{yearRange}</span>
                      </div>
                    )}
                  </div>

                  {/* Metric Boxes */}
                  <div className="grid grid-cols-2 gap-3">
                    <MetricBox label="MAE" value={formatNumber(bestModel.mae)} sub="LKR" />
                    <MetricBox label="R²" value={bestModel.r2.toFixed(4)} />
                  </div>
                </div>
              </div>

              {/* Model Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                {livePredictions.map(p => {
                  const isBest = p.name === bestModel.name;
                  return (
                    <div key={p.name} className={`card p-4 ${isBest ? 'border-emerald-500/40' : ''}`}>
                      <div className="flex justify-between items-center mb-3">
                        <span className="text-xs text-slate-400 font-semibold uppercase tracking-wide">{p.name.split(' ')[0]}</span>
                        {isBest && <CheckCircle2 className="w-4 h-4 text-emerald-500" />}
                      </div>
                      <p className="text-xl font-bold text-white mb-2">{formatLKR(p.predictedPrice)}</p>
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-400">R² Score</span>
                        <span className={`font-semibold ${getConfidenceLabel(p.r2).color}`}>{p.r2.toFixed(4)}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          ) : (
            <div className="card h-[400px] flex flex-col items-center justify-center text-center p-8 border-dashed border-slate-700">
              <Gauge className="w-12 h-12 text-slate-700 mb-4" />
              <h3 className="text-lg font-semibold text-slate-400">Ready for Analysis</h3>
              <p className="text-slate-500 text-sm max-w-sm mt-2">
                Configure the vehicle details and click &quot;Analyze&quot; to run the ML ensemble.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* ─── Charts Section ──────────────────────────────────────────── */}
      <section className="mt-14 pt-10 border-t border-slate-800 space-y-8">
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-emerald-500 uppercase tracking-wide text-xs font-semibold mb-1">
              <BarChart3 className="w-4 h-4" />
              Performance Metrics
            </div>
            <h2 className="text-2xl font-bold text-white">ML Model Performance Comparison</h2>
          </div>
          <div className="flex gap-4 text-xs text-slate-400">
            <Legend color="bg-emerald-500" label="Random Forest" />
            <Legend color="bg-blue-500" label="XGBoost" />
            <Legend color="bg-indigo-500" label="Gradient Boosting" />
          </div>
        </div>

        {/* Chart Row */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <ChartCard title="MAE Comparison" subtitle="Lower is Better" description="Mean Absolute Error — average LKR deviation from actual prices.">
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#1e293b" />
              <XAxis dataKey="name" hide />
              <YAxis stroke="#94a3b8" fontSize={11} tickFormatter={(v: number) => `${(v / 1000).toFixed(0)}k`} />
              <Tooltip {...TOOLTIP_STYLE} formatter={(v: number) => [formatNumber(v) + ' LKR', 'MAE']} />
              <Bar dataKey="mae" radius={[4, 4, 0, 0]}>
                {chartData.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % 3]} />)}
              </Bar>
            </BarChart>
          </ChartCard>

          <ChartCard title="R² Score Comparison" subtitle="Higher is Better" description="Proportion of variance explained by the model (Max: 1.0).">
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#1e293b" />
              <XAxis dataKey="name" hide />
              <YAxis stroke="#94a3b8" fontSize={11} domain={[0, 1]} />
              <Tooltip {...TOOLTIP_STYLE} formatter={(v: number) => [v.toFixed(4), 'R²']} />
              <Bar dataKey="r2" radius={[4, 4, 0, 0]}>
                {chartData.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % 3]} />)}
              </Bar>
            </BarChart>
          </ChartCard>
        </div>
      </section>

      {/* ─── Footer ──────────────────────────────────────────────────── */}
      <footer className="text-center py-10 mt-12 border-t border-slate-800 text-slate-500 text-xs">
        &copy; 2026 AI Vehicle Price Intelligence Platform &middot; Built with ML Ensemble Analysis
      </footer>
    </div>
  );
};

/* ─── Reusable Sub-Components ────────────────────────────────────────── */

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-xs font-medium text-slate-400 mb-1.5">{label}</label>
      {children}
    </div>
  );
}

function MetricBox({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-slate-800/60 rounded-lg p-3 text-center border border-slate-700/50">
      <p className="text-[11px] text-slate-400 uppercase tracking-wide font-medium mb-1">{label}</p>
      <p className="text-sm font-bold text-white">{value}</p>
      {sub && <p className="text-[10px] text-slate-500 mt-0.5">{sub}</p>}
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className={`w-2 h-2 rounded-full ${color}`} />
      {label}
    </span>
  );
}

function ChartCard({
  title,
  subtitle,
  description,
  children,
}: {
  title: string;
  subtitle: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <div className="card p-5 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-white">{title}</h3>
        <span className="text-[11px] text-slate-500">{subtitle}</span>
      </div>
      <div className="h-60">
        <ResponsiveContainer width="100%" height="100%">
          {children as any}
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-slate-500 leading-relaxed">{description}</p>
    </div>
  );
}

export default App;
