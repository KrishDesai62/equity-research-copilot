"""
Track B: manipulation/anomaly classifier. NOT in MVP scope — build in
Phase 2 after Track A is validated. Same architecture pattern as
train_distress.py (logistic baseline -> XGBoost -> NN comparison in
Phase 3), applied to the Beneish-derived feature table.

Reminder from README fix #3: if you're training on the
aaer_enforcement_flag as primary target, consider restatement_flag as an
auxiliary feature rather than merging both into one label.
"""

# TODO: implement train_logistic_baseline / train_xgboost /
# run_walk_forward_evaluation mirroring train_distress.py once Phase 2
# starts. Left empty intentionally — don't build this during the MVP.
