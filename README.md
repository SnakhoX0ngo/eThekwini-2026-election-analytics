# eThekwini 2026 Local Government Election Analytics

## Project Overview

This project analyses historical South African Local Government Election data for the eThekwini Metropolitan Municipality and develops model-based estimates for the 2026 Local Government Election.

The analysis uses historical election data from 2011, 2016 and 2021 and focuses on:

- Historical party performance
- Proportional Representation (PR) vote shares
- Ward-level election performance
- Three selected wards
- 2026 metro-level vote-share estimates
- 2026 ward-level model estimates
- Voter turnout analysis
- Machine-learning model evaluation
- Model uncertainty and limitations

**Important:** The 2026 election estimates in this project are model outputs. They are not actual election results and should not be interpreted as official election results.

---

## Research Questions

The project addresses the following questions:

1. Which relevant party has the largest projected aggregate vote share in eThekwini in 2026?
2. What are the projected leading parties in three selected wards?
3. What are the projected vote shares for the relevant parties?
4. What level of voter turnout can reasonably be estimated from the available historical data?
5. How well do the forecasting and classification models perform on historical validation data?
6. What limitations and uncertainties affect the 2026 estimates?

---

## Study Area

**Municipality:** eThekwini Metropolitan Municipality  
**Municipality Code:** ETH  
**Province:** KwaZulu-Natal, South Africa

The 2026 ward structure is considered when assessing geographic comparability. Historical ward boundaries may not be identical across election years, so voting-district continuity is considered when selecting wards for detailed analysis.

---

# Data Sources

The project uses publicly available and traceable sources.

## 1. Electoral Commission of South Africa (IEC)

Official IEC election results portal:

https://results.elections.org.za/

The IEC results system provides election results at different geographic levels, including municipality, ward and voting district.

### Historical eThekwini datasets

#### 2011 Local Government Election

Official IEC eThekwini detailed results:

https://results.elections.org.za/home/LGEPublicReports/197/Detailed%20Results/KN/ETH.pdf

#### 2016 Local Government Election

Official IEC eThekwini detailed results:

https://results.elections.org.za/home/LGEPublicReports/402/Detailed%20Results/KN/ETH.pdf

#### 2021 Local Government Election

Official IEC eThekwini detailed results:

https://results.elections.org.za/home/LGEPublicReports/1091/Detailed%20Results/KN/ETH.pdf

---

## 2. Municipal Demarcation Board (MDB)

Official MDB website:

https://www.demarcation.org.za/

The Municipal Demarcation Board provides information concerning municipal and ward demarcation.

### 2026 ward finalisation

MDB 2026 ward finalisation information:

https://www.demarcation.org.za/wp-content/uploads/2026/05/MDB-finalises-4488-wards-for-2026-Local-Government-Elections.pdf

This source is used when considering changes to the 2026 ward structure and geographic comparability.

---

## 3. Statistics South Africa

Official Statistics South Africa website:

https://www.statssa.gov.za/

Census 2022 Municipal Factsheet:

https://census.statssa.gov.za/assets/documents/2022/Census_2022_Municipal_factsheet-Web.pdf

Statistics South Africa provides demographic and population information that can be used to provide contextual information about the municipality.

---

## 4. eThekwini Municipality

Official eThekwini Municipality website:

https://results.elections.org.za/home/Downloads/ME-Results

Municipal planning documents are used where relevant for municipal and ward-structure context.

---

# Dataset Description

The repository contains the cleaned historical election datasets used in the analysis:

```text
Datasets/
├── 2011_ETH.csv
├── 2016_ETH.csv
└── 2021_ETH.csv
