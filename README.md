# ⚡ AI SQL Data Analyst Agent (CSV → SQL → Insights)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![Groq LPU](https://img.shields.io/badge/Groq-High--Speed%20LPU-F55036.svg)](https://groq.com/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-003B57.svg)](https://www.sqlite.org/)
[![Plotly](https://img.shields.io/badge/Visuals-Plotly%20Interactive-3F4F75.svg)](https://plotly.com/)
[![Tests](https://img.shields.io/badge/Tests-56%20Passed%20(100%25)-brightgreen.svg)]()

An AI Data Analyst Agent that converts natural language questions into safe, executable SQLite queries, interactive Plotly visualizations, and grounded analytical findings.

Featuring **Defense-in-Depth C-Engine Security**, **AST-based SQL Validation**, and **Ultra-fast Groq LPU Inference**.

---

## 🎯 The Problem & The Solution

| The Challenge | Our Solution |
|---|---|
| **SQL Barrier:** Business stakeholders and non-technical users need access to relational data but lack SQL expertise. | **Conversational Text-to-SQL:** Converts plain English into syntactically valid SQLite queries using schema-bound LLM prompts. |
| **SQL Injection & Hallucinations:** Naive LLM pipelines are susceptible to destructive commands (`DROP`, `DELETE`), stacked queries, and hallucinated columns. | **Hardware & Engine-Level Guardrails:** Multi-tier validation combining `sqlglot` AST parsing, token blacklisting, and native SQLite C-engine read-only authorizers (`conn.set_authorizer`). |
| **Dirty Data:** Real-world CSV uploads frequently suffer from inconsistent casing, duplicate rows, whitespace, and corrupt numeric symbols. | **Dedicated Data Processing:** Automatic encoding detection, robust header sanitization, type coercion, and non-destructive data handling. |
| **Meaningless Visualizations:** Traditional tools often generate misleading or fabricated charts. | **Intelligent Visualizer:** Heuristic chart recommendation and 12+ interactive chart types (Bar, Horizontal Bar, Line, Pie/Donut, Area, Scatter, Histogram, Box Plot, Treemap, Funnel, Heatmap) anchored strictly to query results. |

---

## 🏛️ System Architecture

```text
User Natural Language Question + CSV Dataset
                    │
                    ▼
          Streamlit Frontend Dashboard (app.py)
                    │
        ┌───────────┴────────────────────────┐
        ▼                                    ▼
modules/data_loader.py            modules/data_cleaner.py
(Encoding detection &             (Health Score, Whitespace,
 Header Sanitization)              Deduplication, Imputation)
        │                                    │
        └───────────┬────────────────────────┘
                    ▼
          modules/database.py
(SQLite Dynamic Schema Introspection & PRAGMA DDL)
                    │
                    ▼
        modules/sql_generator.py
(Schema-Bound Prompting with Sample Value Categorization)
                    │
                    ▼
         modules/llm_service.py
(Official Groq SDK + Ultra-fast LPU Inference)
                    │
                    ▼
        modules/sql_validator.py
(AST Parse via sqlglot, Single Statement Check, DDL/DML Block)
                    │
                    ▼
        modules/query_executor.py
(SQLite Engine Authorizer, 10s Timeout Watchdog, Row Truncation)
                    │
        ┌───────────┴────────────────────────┐
        ▼                                    ▼
modules/visualizer.py             modules/insights.py
(Plotly Intelligent Chart Suite)   (Deterministic Math + Grounded Narrative)
```

---

## 🔒 Defense-in-Depth Security Model

1. **AST Validation (`sqlglot`):** Queries are parsed into abstract syntax trees to verify the root operation is an un-nested `SELECT` or `UNION`.
2. **Single-Statement Enforcement:** Stacked queries (e.g. `SELECT 1; DROP TABLE data_table;`) are rejected before reaching the database.
3. **SQLite C-Engine Authorizer (`conn.set_authorizer`):** Python's native SQLite authorizer callback intercepts every execution step at the virtual machine layer, denying all modifying actions (`SQLITE_INSERT`, `SQLITE_UPDATE`, `SQLITE_DELETE`, `SQLITE_DROP_TABLE`, `SQLITE_ALTER_TABLE`, `SQLITE_PRAGMA`, `SQLITE_ATTACH`).
4. **Execution Watchdog & Row Limiting:** Infinite loops and runaway Cartesian joins are aborted via progress handlers (`set_progress_handler`), and output tables are safely bounded to prevent browser memory exhaustion.

---

## 🚀 Key Features

### 1. Dynamic Dataset Support
- Supports arbitrary CSV files with varied column types and names.
- Automatic header sanitization (handles whitespace, special symbols, and duplicates).
- Includes two built-in sample datasets: `sales_data.csv` (Retail Sales) and `customer_churn.csv` (Telecom Churn).

### 2. Schema-Aware AI SQL Generation
- Uses the latest **Groq LPU Models** (such as `openai/gpt-oss-120b` and `llama-3.3-70b-versatile`).
- Enriches LLM system prompts with table DDL and distinct categorical sample values to ensure exact spelling adherence.
- Generates dynamic suggested questions tailored to the active dataset's columns.

### 3. Intelligent Interactive Visualizations & 12+ Chart Types
- Auto-detects the optimal chart type based on DataFrame geometry:
  - **Single Scalar Value:** Metric KPI card.
  - **Categorical + Numeric:** Bar Chart or Donut Chart (for $\le 6$ slices).
  - **Date + Numeric:** Interactive Line Chart with time-series zoom.
  - **Two Numeric Columns:** Scatter Plot with trendline.
  - **Multi-column Raw Records:** Formatted sortable data table.
- **Interactive User Controls & Advanced Chart Library:**
  - **Vertical Bar Chart** & **Horizontal Bar Chart** (ranking analysis)
  - **Interactive Line Chart** & **Area Chart** (trend analysis)
  - **Donut / Pie Chart** & **Treemap** (part-to-whole share analysis)
  - **Scatter Plot** (correlation analysis)
  - **Histogram** (distribution frequency with box margin)
  - **Box & Whisker Plot** (IQR, quartiles, median, and outlier detection)
  - **Funnel Chart** (stage and drop-off flow)
  - **2D Density Heatmap** (density intersections)
  - **Data Table View** (sortable and downloadable)
- All charts feature theme-adaptive transparent canvases for seamless dark and light mode rendering.

### 4. Grounded Insights & Performance Metrics
- Sleek, compact execution stats (execution latency, row count, timestamp).
- Grounded AI executive summary synthesizing findings strictly from returned records.
- Query history timeline with execution latency tracking, row metrics, and one-click re-run capability.

---

## 📦 Project Structure

```text
ai-sql-data-analyst/
│
├── app.py                     # Professional Streamlit multi-tab application
├── requirements.txt           # Verified, compatible production dependencies
├── README.md                  # Comprehensive project documentation & guide
├── .env.example               # Template for API credentials and configs
├── .gitignore                 # Excludes .env, *.db, and cache files
│
├── modules/                   # Modular, decoupled architecture
│   ├── __init__.py
│   ├── data_loader.py         # CSV loading, encoding resolution, header sanitization
│   ├── data_cleaner.py        # Data-quality audit and interactive cleaning operations
│   ├── database.py            # SQLite connection lifecycle & schema DDL generation
│   ├── llm_service.py         # Official Groq SDK client with error handling & fallbacks
│   ├── sql_generator.py       # Schema-aware prompt engineering & dynamic question suggestions
│   ├── sql_validator.py       # Markdown stripping, sqlglot AST verification & security rules
│   ├── query_executor.py      # SQLite authorizer, timeout watchdog, and row limits
│   ├── visualizer.py          # Plotly intelligent chart selection & custom renderers
│   └── insights.py            # Grounded statistical calculations & AI narrative
│
├── tests/                     # Comprehensive automated test suite (56 tests)
│   ├── __init__.py
│   ├── test_data_loader.py    # Scenarios 1, 2, 6, 7
│   ├── test_data_cleaner.py   # Scenarios 3, 4, 5
│   ├── test_sql_validator.py  # Scenarios 11, 12, 13
│   ├── test_query_executor.py # Scenarios 14, 19, timeouts & authorizers
│   ├── test_llm_service.py    # Scenarios 15, 16, 20
│   ├── test_sql_generator.py  # Scenarios 8, 9, 10
│   ├── test_visualizer.py     # Scenarios 17, 18, advanced chart types
│   ├── test_insights.py       # Grounded insights & math assertions
│   └── test_scenarios_comprehensive.py # Full 20-scenario certification
│
└── sample_data/
    ├── sales_data.csv         # Retail sales sample dataset
    └── customer_churn.csv     # Telecom churn arbitrary schema validation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ai-sql-data-analyst.git
cd ai-sql-data-analyst
```

### 2. Create and Activate a Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Note: `sqlite3` is part of Python's standard library and is not installed through pip.)*

### 4. Configure Environment Variables
Copy the template to `.env`:
```powershell
cp .env.example .env
```
Edit `.env` and add your Groq API key:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
SQL_QUERY_TIMEOUT_SECONDS=10
SQL_MAX_RESULT_ROWS=5000
```
*(You can also input your Groq API key directly in the Streamlit application sidebar.)*

### 5. Launch the Application
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 🧪 Automated Testing (56 Tests Passed)

All 20 test scenarios specified in the project requirements are covered using mocked Groq API calls, enabling 100% offline, deterministic testing without API credits:

```bash
pytest tests -v
```

### Test Coverage Highlights:
- **Scenario 1–7 (Data Ingestion & Quality):** Valid CSVs, empty CSVs, missing values, duplicate records, invalid numeric values (currency symbols, strings), arbitrary column names, special characters.
- **Scenario 8–10 (SQL Generation):** Schema-bound queries, unanswerable questions, empty prompts, dynamic suggestions.
- **Scenario 11–13 (SQL Validation & Defense):** Markdown code fence stripping, syntax errors, modifying query rejection (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `PRAGMA`), stacked injection blocking.
- **Scenario 14–16 (Execution & AI Resilience):** Empty query results, Groq API errors (429, 500), decommissioned model handling, missing API keys.
- **Scenario 17–19 (Visuals & Performance):** Suitable chart generation (Bar, Line, Donut, Scatter), non-visualizable queries (table fallback), large result set truncation, and query timeouts.

---

## 🎤 How to Present This Project in Technical Interviews

> *"My project is an AI SQL Data Analyst Agent that allows non-technical stakeholders to analyze CSV datasets using natural-language questions instead of writing SQL queries manually.*
> 
> *The application ingests arbitrary CSVs using Pandas and dynamically creates an SQLite relational database. When a user asks a question, a Groq-hosted language model (`llama-3.3-70b-versatile`) generates an SQLite-compatible query based on the database schema and sample categorical values.*
> 
> *What sets this project apart from simple LLM wrappers is its defense-in-depth security model: it validates queries via AST parsing with `sqlglot` to prevent stacked injections, and enforces engine-level read-only access using Python's native SQLite authorizer (`conn.set_authorizer`). Additionally, it includes an interactive Data Studio for data-quality audits and controlled cleaning, intelligent Plotly visualization selection, and grounded insights that prevent hallucinated statistics."*

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
