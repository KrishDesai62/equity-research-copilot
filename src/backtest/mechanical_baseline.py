"""Fixed, untuned risk/news comparator for the same dated agent inputs."""
import numpy as np


def positions(probabilities, sentiments, *, max_risk=.1):
    risk, news = np.asarray(probabilities, float), np.asarray(sentiments, float)
    if risk.shape != news.shape or risk.ndim != 1 or not np.isfinite(risk).all() or not np.isfinite(news).all():
        raise ValueError('Mechanical comparator requires matching finite risk/news inputs')
    if ((risk < 0) | (risk > 1)).any() or ((news < -1) | (news > 1)).any() or not 0 < max_risk < 1:
        raise ValueError('Invalid risk or sentiment ranges')
    return ((risk < max_risk) & (news > 0)).astype(float)
