# ⚗️ ChemForge

**ChemForge** is a modular Python meta-framework for constructing custom chemical property prediction pipelines, on-the-fly surrogate models, and database-driven estimation workflows.

Rather than locking you into a single fixed model architecture, **ChemForge** provides a construction-focused workflow where you can freely mix, match, and stack custom chemical encoders, comparibility metrics, structural mutators, thermodynamic database adapters, and surrogate model backends.

---

## 🌟 Key Features

* **🔨 Construction-Focused Architecture:** Modular design decoupled into clear responsibilities: `encoders`, `metrics`, `mutators`, `adapters`, `models`, `evaluators`, and `data_processing`.
* **📡 Dynamic Context Harvesting:** Automatically explore chemical space around uncharacterized target molecules using mutations to harvest nearest neighbors with complete empirical database records.
* **🧠 Plug-and-Play Surrogate Backends:** Seamlessly switch between ML based or traditional regression ensembles while preserving evaluation logic.
* **🔌 Heterogeneous Database Adapters:** Query thermodynamic and empirical databases on the fly.

---

## 🤖 AI Disclosure & Disclaimer

Parts of the code and documentation in this repository were generated or assisted by **Google Gemini**.

---

## 📦 Package Architecture

```text
chemforge/
├── data/            # User defined storage space
├── data_processing/ # Data manipulation for running predictions
├── evaluators/      # Context-building & surrogate training evaluators
├── models/          # Model families (Gaussian Processes)
├── encoders/        # Molecular representations (Morgan bits, physical descriptors)
├── metrics/         # Vector distance & chemical similarity functions
├── mutators/        # Combinatorial & procedural molecule transformation engines
├── adapters/        # External database connectors & caching wrappers
├── MAIN.py
└── README.md
