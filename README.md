# Marketing Mix Modeling (MMM) &amp; Brand Strategy Impact Simulator
### Quantifying the Business Impact of Positioning, Focus, and Category Leadership Using Econometrics

## 🧠 Project Overview
This project builds a full-stack Marketing Mix Modeling (MMM) platform to quantify how brand positioning and strategic focus decisions impact long-term revenue and ROI.


Inspired by principles from The 22 Immutable Laws of Marketing, this system tests strategic marketing hypotheses using econometric modeling, adstock transformations, and counterfactual simulations.
The goal is to bridge marketing strategy and data science — turning qualitative brand principles into measurable financial impact.

## 🎯 Business Questions
- Does being a category leader improve long-term ROI?
- Do focused campaigns outperform fragmented messaging strategies?
- How does brand perception affect revenue over time?
- What is the lagged impact of marketing spend?
- How do short-term performance campaigns compare to long-term brand investments?

## 🏗️ System Architecture
Data Generation / Ingestion
        ↓
Data Cleaning & Feature Engineering
        ↓
Adstock & Saturation Transformations
        ↓
Regression / Bayesian MMM
        ↓
Channel Contribution Decomposition
        ↓
Counterfactual Simulation Engine
        ↓
Executive Dashboard

## 📦 Tech Stack
- Python
- pandas / numpy
- statsmodels / scikit-learn
- PyMC (for Bayesian MMM)
- matplotlib / seaborn
- Streamlit (dashboard)
- SQL (optional data storage)
- Docker (optional deployment)

## 📊 Dataset
The project uses either:
- Synthetic multi-year marketing data (generated programmatically), OR
- Public marketing dataset (if available)

Example Variables
| Category            | Variables                             |
| ------------------- | ------------------------------------- |
| Revenue             | Weekly sales                          |
| Media Spend         | TV, Search, Social, Display, Radio    |
| Brand Metrics       | Awareness, Consideration              |
| Strategy Indicators | Focus score, Category leadership flag |
| Controls            | Seasonality, Holidays, Pricing        |

## 🧮 Methodology
1️⃣ Data Preparation
- Handle missing values
- Normalize media spend
- Create lag variables
- Encode strategic indicators

2️⃣ Adstock Transformation
- Captures lagged marketing impact over time:
- Adstock_t = Spend_t + λ * Adstock_(t-1)

3️⃣ Saturation Modeling
- Applies diminishing returns curve:
- Saturation(x) = α * x / (β + x)

4️⃣ Econometric Modeling
Baseline Model:
Revenue ~ TV_adstock + Search_adstock + Social_adstock 
         + Brand_awareness_lag 
         + Focus_score 
         + Category_leader 
         + Controls

Outputs:
- Channel ROI
- Contribution decomposition
- Long-term elasticity
- Brand vs performance impact split

5️⃣ Counterfactual Simulation Engine
Simulate strategic decisions:
- Increase focus score by 30%
- Remove line extension
- Reposition as challenger brand
- Shift 20% budget to brand channels

Estimate:
- Incremental revenue
- ROI change
- Long-term brand lift

## 📈 Dashboard Features
Built with Streamlit:
- Channel contribution breakdown
- Short-term vs long-term impact visualization
- ROI curves
- Strategy simulation sliders
- Executive summary export

## 🔬 Strategic Marketing Laws Tested
This project operationalizes key principles:
- Leadership vs challenger positioning
- Focus vs line extension
- Long-term vs short-term marketing impact
- Category creation strategy
- Brand perception vs product features

## 📊 Sample Insights (Example)
- Focused campaigns generate 18% higher long-term ROI
- Category leaders experience stronger adstock retention
- Over-extension reduces elasticity by 12%
-  Brand investment drives 40% of long-term incremental revenue

## 🚀 How to Run
git clone https://github.com/yourusername/mmm-brand-strategy-simulator.git
cd mmm-brand-strategy-simulator
pip install -r requirements.txt
python src/run_model.py
streamlit run dashboard/app.py

## 📁 Project Structure
<img width="538" height="752" alt="image" src="https://github.com/user-attachments/assets/203d09c4-ddc0-48ac-8f59-b487d1e107d6" />

## 🌍 Why This Project Matters
Most marketing discussions remain qualitative.

This project proves:
- Positioning is measurable.
- Focus is quantifiable.
- Strategy has elasticity.

It demonstrates the ability to combine:
- Business strategy
- Econometrics
- Marketing analytics
- Executive storytelling

— exactly the skill set required in marketing effectiveness consulting.
