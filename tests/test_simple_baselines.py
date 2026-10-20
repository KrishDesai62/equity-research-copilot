import numpy as np
import pandas as pd
from src.ml.simple_baselines import altman_proxy,fit_logistic,FINANCIAL,TARGET


def test_book_equity_formula_and_missing_inputs():
    row={'Assets':100.,'Liabilities':50.,'wc_to_ta':.1,'re_to_ta':.2,'ebit_to_ta':.03}
    z=altman_proxy(pd.DataFrame([row,{**row,'Liabilities':0},{**row,'re_to_ta':np.nan}]))
    assert abs(z[0]-2.5596)<1e-12
    assert np.isnan(z[1:]).all()


def test_prediction_does_not_refit_preprocessing_on_evaluation_values():
    frame=pd.DataFrame(1.,index=np.arange(12),columns=FINANCIAL)
    frame['cik']=[str(i) for i in range(12)];frame[TARGET]=np.arange(12)%2;frame['design_weight']=1.
    frame.loc[0,'wc_to_ta']=np.nan
    model,features=fit_logistic(frame,'logistic_financial')
    before=model.named_steps['imputer'].statistics_.copy()
    evaluation=frame[features].copy();evaluation['wc_to_ta']=1e20
    model.predict_proba(evaluation)
    np.testing.assert_array_equal(before,model.named_steps['imputer'].statistics_)
