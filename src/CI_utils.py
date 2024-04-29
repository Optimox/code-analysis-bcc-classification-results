from math import sqrt
from scipy.special import ndtri
from sklearn.metrics import  classification_report
from statsmodels.stats.contingency_tables import mcnemar
import numpy as np


def get_sens_spec(df, pred_col="lcoct_phase_answer"):
    """
    Returns sensibility, specificity, accuracy and f1-score
    """
    report =  classification_report(y_true=df.is_diagnostic_bcc,
                                    y_pred=df[pred_col].values,
                                    output_dict=True
                         )
    return {"sensitivity": report["True"]["recall"], #["recall"], precision
            "specificity": report["False"]["recall"], #["recall"], precision
            "accuracy": report["accuracy"],
            "f1-score" : report['macro avg']["f1-score"]
           }
    
# Taken from here : https://gist.github.com/maidens/29939b3383a5e57935491303cf0d8e0b
def _proportion_confidence_interval(r, n, z):
    """Compute confidence interval for a proportion.
    
    Follows notation described on pages 46--47 of [1]. 
    
    References
    ----------
    [1] R. G. Newcombe and D. G. Altman, Proportions and their differences, in Statisics
    with Confidence: Confidence intervals and statisctical guidelines, 2nd Ed., D. G. Altman, 
    D. Machin, T. N. Bryant and M. J. Gardner (Eds.), pp. 45-57, BMJ Books, 2000. 
    
    r : ratio
    n : num examples
    z : = -ndtri((1.0-alpha)/2)
    """
    
    A = 2*r + z**2
    B = z*sqrt(z**2 + 4*r*(1 - r/n))
    C = 2*(n + z**2)
    return ((A-B)/C, (A+B)/C)

def sensitivity_and_specificity_with_confidence_intervals(TP, FP, FN, TN, alpha=0.95):
    """Compute confidence intervals for sensitivity and specificity using Wilson's method. 
    
    This method does not rely on a normal approximation and results in accurate 
    confidence intervals even for small sample sizes.
    
    Parameters
    ----------
    TP : int
        Number of true positives
    FP : int 
        Number of false positives
    FN : int
        Number of false negatives
    TN : int
        Number of true negatives
    alpha : float, optional
        Desired confidence. Defaults to 0.95, which yields a 95% confidence interval. 
    
    Returns
    -------
    sensitivity_point_estimate : float
        Numerical estimate of the test sensitivity
    specificity_point_estimate : float
        Numerical estimate of the test specificity
    sensitivity_confidence_interval : Tuple (float, float)
        Lower and upper bounds on the alpha confidence interval for sensitivity
    specificity_confidence_interval
        Lower and upper bounds on the alpha confidence interval for specificity 
        
    References
    ----------
    [1] R. G. Newcombe and D. G. Altman, Proportions and their differences, in Statisics
    with Confidence: Confidence intervals and statisctical guidelines, 2nd Ed., D. G. Altman, 
    D. Machin, T. N. Bryant and M. J. Gardner (Eds.), pp. 45-57, BMJ Books, 2000. 
    [2] E. B. Wilson, Probable inference, the law of succession, and statistical inference,
    J Am Stat Assoc 22:209-12, 1927. 
    """
    
    # 
    z = -ndtri((1.0-alpha)/2)
    
    # Compute sensitivity using method described in [1]
    sensitivity_point_estimate = TP/(TP + FN)
    sensitivity_confidence_interval = _proportion_confidence_interval(TP, TP + FN, z)
    
    # Compute specificity using method described in [1]
    specificity_point_estimate = TN/(TN + FP)
    specificity_confidence_interval = _proportion_confidence_interval(TN, TN + FP, z)
    
    return sensitivity_point_estimate, specificity_point_estimate, sensitivity_confidence_interval, specificity_confidence_interval
