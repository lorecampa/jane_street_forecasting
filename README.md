# Jane Street Forecasting

<p align="center">
  <a href="http://recsys.deib.polimi.it">
    <img src="https://i.imgur.com/mPb3Qbd.gif" width="180" />
  </a>
</p>

We participated in the challenge as [...], a team of 3 MSc students from Politecnico di Milano:

* **[Andrea Alari](https://github.com/andrea-alari)**
* **[Lorenzo Campana](https://github.com/lorecampa)**
* **[Carlo Sgaravatti](https://github.com/CarloSgaravatti)**

## Setup Instructions

### 1. Create a Virtual Environment

To manage dependencies, create a virtual environment and install the required packages:

```bash
# Create the virtual environment
python3.10 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download and Organize the Dataset

Download the dataset and ensure the following directory structure is maintained:

```bash
.
├── README.md
├── features.csv
├── lags
│   └── date_id=0
├── responders.csv
├── sample_submission.csv
├── test
│   └── date_id=0
└── train
    ├── partition_id=0
    ├── partition_id=1
    ├── partition_id=2
    ├── partition_id=3
    ├── partition_id=4
    ├── partition_id=5
    ├── partition_id=6
    ├── partition_id=7
    ├── partition_id=8
    └── partition_id=9
```

Create a .env file at the root of the project and set your PYTHONPATH:

```bash
PYTHONPATH="path/to/your/project"
```
