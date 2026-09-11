"""
Phase 4 (post-MVP). Surfaces Track A (distress probability) and Track B
(manipulation flag) model outputs as structured risk callouts — this is
a TOOL CALL to your trained models, not a free-text LLM guess. This is
the piece that differentiates the agent layer from a plain RAG chatbot
(see README discussion on uniqueness) — don't let it degrade into the
LLM eyeballing the filing and guessing a risk level instead of calling
the actual classifiers.
"""
# TODO: build in Phase 4. Load the trained Track A/B models
# (from ml/train_distress.py, ml/train_manipulation.py) and expose a
# function like get_risk_profile(ticker, as_of_date) -> {distress_prob,
# manipulation_flag, insider_sentiment} that the synthesis agent calls.
