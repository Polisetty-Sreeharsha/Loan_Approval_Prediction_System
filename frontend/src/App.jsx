import { useState } from "react";
import {
  GraduationCap,
  ShieldCheck,
  ChartNoAxesCombined,
  ArrowRight,
  RotateCcw,
  CheckCircle2,
  XCircle,
  LockKeyhole,
} from "lucide-react";
import "./App.css";

const initialForm = {
  gender: "Male",
  married: "No",
  dependents: "0",
  education: "Graduate",
  self_employed: "No",
  applicantincome: "50000",
  coapplicantincome: "0",
  loanamount: "4000000",
  loan_amount_term: "240",
  credit_history: "1",
  property_area: "Urban",
};

const modelOptions = [
  { value: "current", label: "Current Model" },
  { value: "candidate", label: "Candidate Model" },
];

function Field({ label, name, value, onChange, options, hint }) {
  return (
    <label className="field">
      <span>{label}</span>
      {options ? (
        <select name={name} value={value} onChange={onChange}>
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      ) : (
        <input
          name={name}
          value={value}
          onChange={onChange}
          required
        />
      )}
      {hint && <small className="field-hint">{hint}</small>}
    </label>
  );
}

function NumberField({
  label,
  name,
  value,
  onChange,
  min = "0",
  step = "any",
  hint,
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        type="number"
        name={name}
        value={value}
        onChange={onChange}
        min={min}
        step={step}
        required
      />
      {hint && <small className="field-hint">{hint}</small>}
    </label>
  );
}

export default function App() {
  const [form, setForm] = useState(initialForm);
  const [selectedModel, setSelectedModel] = useState("current");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function handleChange(event) {
    const { name, value } = event.target;
    setForm((previous) => ({ ...previous, [name]: value }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    const payload = {
      ...form,
      applicantincome: Number(form.applicantincome),
      coapplicantincome: Number(form.coapplicantincome),
      loanamount: Number(form.loanamount),
      loan_amount_term: Number(form.loan_amount_term),
      credit_history: Number(form.credit_history),
    };

    const endpoint =
      selectedModel === "candidate"
        ? "http://127.0.0.1:8000/predict-candidate"
        : "http://127.0.0.1:8000/predict";

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail
            ? typeof data.detail === "string"
              ? data.detail
              : JSON.stringify(data.detail)
            : "Prediction failed. Please check your inputs."
        );
      }

      setResult({
        ...data,
        model:
          selectedModel === "candidate"
            ? data.model || "Logistic Regression Candidate"
            : "Current Saved Model",
      });
    } catch (err) {
      setError(
        err.message.includes("fetch")
          ? "Cannot connect to the backend. Make sure the FastAPI server is running at http://127.0.0.1:8000."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }

  function resetApplication() {
    setForm(initialForm);
    setSelectedModel("current");
    setResult(null);
    setError("");
  }

  const isApproved = result?.prediction === "Approved";
  const probability = result
    ? Math.max(
        0,
        Math.min(100, Number(result.approval_probability) * 100)
      )
    : 0;

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" onClick={(e) => e.preventDefault()}>
          <span className="brand-icon">
            <GraduationCap size={23} />
          </span>
          <span>LoanWise</span>
        </a>

        <div className="topbar-label">
          <ShieldCheck size={17} />
          <span>Loan Prediction System</span>
        </div>
      </header>

      <main className="main-content">
        <section className="hero">
          <div className="eyebrow">SMARTER FINANCIAL INSIGHTS</div>
          <h1>
            Your loan journey,
            <br />
            <span>made clearer.</span>
          </h1>
          <p>
            Enter your application details to receive a machine-learning
            prediction. Compare two models using the same information.
          </p>

          <div className="feature-row">
            <span>
              <ShieldCheck size={17} /> Secure prediction
            </span>
            <span>
              <ChartNoAxesCombined size={17} /> Model comparison
            </span>
            <span>
              <GraduationCap size={17} /> AI-powered insights
            </span>
          </div>
        </section>

        <div className="content-grid">
          <form className="form-card" onSubmit={handleSubmit}>
            <div className="card-heading">
              <div>
                <h2>Loan application</h2>
                <p>Provide the information requested below.</p>
              </div>
              <span className="step-label">01 / APPLICATION</span>
            </div>

            <div className="section-label">01 / SELECT MODEL</div>

            <div className="form-grid">
              <Field
                label="Prediction Model"
                name="selectedModel"
                value={selectedModel}
                onChange={(event) => {
                  setSelectedModel(event.target.value);
                  setResult(null);
                  setError("");
                }}
                options={modelOptions}
              />
            </div>

            <div className="section-label section-spaced">
              02 / PERSONAL INFORMATION
            </div>

            <div className="form-grid">
              <Field
                label="Gender"
                name="gender"
                value={form.gender}
                onChange={handleChange}
                options={[
                  { value: "Male", label: "Male" },
                  { value: "Female", label: "Female" },
                ]}
              />

              <Field
                label="Marital Status"
                name="married"
                value={form.married}
                onChange={handleChange}
                options={[
                  { value: "Yes", label: "Married" },
                  { value: "No", label: "Not married" },
                ]}
              />

              <Field
                label="Dependents"
                name="dependents"
                value={form.dependents}
                onChange={handleChange}
                options={[
                  { value: "0", label: "0" },
                  { value: "1", label: "1" },
                  { value: "2", label: "2" },
                  { value: "3+", label: "3+" },
                ]}
              />

              <Field
                label="Education"
                name="education"
                value={form.education}
                onChange={handleChange}
                options={[
                  { value: "Graduate", label: "Graduate" },
                  { value: "Not Graduate", label: "Not Graduate" },
                ]}
              />

              <Field
                label="Self Employed"
                name="self_employed"
                value={form.self_employed}
                onChange={handleChange}
                options={[
                  { value: "Yes", label: "Yes" },
                  { value: "No", label: "No" },
                ]}
              />

              <Field
                label="Property Area"
                name="property_area"
                value={form.property_area}
                onChange={handleChange}
                options={[
                  { value: "Urban", label: "Urban" },
                  { value: "Semiurban", label: "Semiurban" },
                  { value: "Rural", label: "Rural" },
                ]}
              />
            </div>

            <div className="section-label section-spaced">
              03 / FINANCIAL INFORMATION
            </div>

            <div className="form-grid">
              <NumberField
                label="Applicant Income (₹ / month)"
                name="applicantincome"
                value={form.applicantincome}
                onChange={handleChange}
              />

              <NumberField
                label="Co-applicant Income (₹ / month)"
                name="coapplicantincome"
                value={form.coapplicantincome}
                onChange={handleChange}
              />

              <NumberField
                label="Loan Amount (₹)"
                name="loanamount"
                value={form.loanamount}
                onChange={handleChange}
                min="0.01"
                hint="Enter the total requested loan amount."
              />

              <NumberField
                label="Loan Term (months)"
                name="loan_amount_term"
                value={form.loan_amount_term}
                onChange={handleChange}
                min="1"
                step="1"
              />

              <Field
                label="Credit History"
                name="credit_history"
                value={form.credit_history}
                onChange={handleChange}
                options={[
                  { value: "1", label: "Good / available (1)" },
                  { value: "0", label: "Poor / unavailable (0)" },
                ]}
              />
            </div>

            <button className="submit-button" type="submit" disabled={loading}>
              <span>
                {loading ? "Analyzing application..." : "Predict loan outcome"}
              </span>
              {loading ? (
                <span className="loader" />
              ) : (
                <ArrowRight size={19} />
              )}
            </button>

            <p className="form-note">
              Use accurate information. This tool provides an educational
              prediction and does not guarantee loan approval.
            </p>
          </form>

          <aside className="result-card">
            <div className="result-icon">
              <ChartNoAxesCombined size={24} />
            </div>

            <span className="result-overline">PREDICTION SUMMARY</span>
            <h2>Your results</h2>

            {!result && !loading && (
              <div className="empty-result">
                <div className="empty-symbol">↗</div>
                <h3>Ready when you are</h3>
                <p>
                  Complete the application and select Predict loan outcome to
                  see the model's prediction here.
                </p>
              </div>
            )}

            {loading && (
              <div className="empty-result">
                <div className="loader" />
                <h3>Analyzing your details</h3>
                <p>The selected machine-learning model is processing your application.</p>
              </div>
            )}

            {result && (
              <div className="prediction-result">
                <span className="result-caption">PREDICTED OUTCOME</span>

                <div
                  className={`status-pill ${
                    isApproved ? "approved" : "rejected"
                  }`}
                >
                  {isApproved ? (
                    <CheckCircle2 size={20} />
                  ) : (
                    <XCircle size={20} />
                  )}
                  {result.prediction}
                </div>

                <div className="probability-heading">
                  <span>Approval probability</span>
                  <strong>{probability.toFixed(1)}%</strong>
                </div>

                <div
                  className="progress-track"
                  role="progressbar"
                  aria-label="Approval probability"
                  aria-valuenow={probability}
                  aria-valuemin={0}
                  aria-valuemax={100}
                >
                  <div
                    className="progress-fill"
                    style={{ width: `${probability}%` }}
                  />
                </div>

                <div className="section-label section-spaced">
                  MODEL USED
                </div>
                <p>{result.model}</p>

                <p className="disclaimer">
                  This probability is the model's estimated probability for
                  approval. Actual lending decisions may depend on additional
                  checks and lender-specific policies.
                </p>

                <button
                  type="button"
                  className="reset-button"
                  onClick={resetApplication}
                >
                  <RotateCcw size={15} /> Reset application
                </button>
              </div>
            )}

            {error && <div className="error-box">{error}</div>}

            <div className="privacy-note">
              <LockKeyhole size={19} />
              <p>
                <strong>Use responsibly</strong>
                <br />
                Enter accurate information. This tool provides an educational
                prediction and does not guarantee loan approval.
              </p>
            </div>
          </aside>
        </div>
      </main>

      <footer>
        LoanWise · Machine Learning Loan Prediction · Academic Project
      </footer>
    </div>
  );
}