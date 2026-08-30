# Track 1 — Health Diagnosis Assistant

An MVP multiclass classifier that predicts one of nine diagnosis categories from patient demographics, symptoms, exposures, vital signs, and laboratory values.

## Run it

```bash
python -m pip install -r requirements.txt
python train.py --data /path/to/track1_participant_dataset.csv
streamlit run app.py
```

## Approach

- Removes `patient_id` to avoid learning an identifier rather than clinical patterns.
- Uses categorical encoding and median/mode imputation inside a single pipeline.
- Uses a stratified 80/20 validation split to preserve the class mix.
- Evaluates accuracy and macro F1, which gives minority diagnoses equal weight.

## Validation result

On this participant dataset, the MVP scored **95.60% accuracy** and **0.9288 macro-F1** on a stratified 20% holdout set.

The web interface shows the most likely diagnosis and the full probability ranking. It is a hackathon demonstration and not a replacement for clinical diagnosis.
