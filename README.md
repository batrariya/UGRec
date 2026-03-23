# UGRec

This project implements a fairness-aware recommendation system using **LightGCN** on the **MovieLens 1M dataset**.

## Installation

Clone the repository:

```bash
git clone  https://github.com/batrariya/FairRS.git
cd FairRS
```

### Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
```

Activate the environment:

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

### Install Dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

---

## Dataset

This project uses the **MovieLens 1M dataset**.

Download the dataset from:
https://grouplens.org/datasets/movielens/1m/

After downloading, extract it into the following directory:

```
data/ml-1m/
```

Expected folder structure:

```
data/
 └── ml-1m/
      ├── movies.dat
      ├── ratings.dat
      ├── users.dat
```

---

## Running the Project

Run the training script using:

```bash
python main.py
```

---

## Requirements

The main dependencies include:

- torch
- numpy
- scipy
- pandas
- scikit-learn
