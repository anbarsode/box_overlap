"""Check log-fit equations and total-cost accounting against known analytic curves."""
from pathlib import Path
import os
import sys
import tempfile
for setting in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[setting]='1'
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from validation.analysis_plots import overlap_fits,plot_costs


def test_fits():
    x=np.geomspace(1e-9,1e3,100)
    y=100*x**.6
    fit=overlap_fits(np.r_[x,0.,1.],np.r_[y,1.,0.])
    assert fit['fit_pairs']==100
    np.testing.assert_allclose(fit['fits'][0]['coefficients'],[.6,2.],atol=1e-13)
    z=np.log10(x)
    fit=overlap_fits(x,10**(.08*z*z+.6*z+2))
    np.testing.assert_allclose(fit['fits'][1]['coefficients'],[.08,.6,2.],atol=1e-13)
    n=np.geomspace(100,10000,15)
    coefficients = {
        'box_generation_s': [0., 1., -1.],
        'preparation_s': [0., 1., -4.],
        'index_build_s': [.02, 1., -4.],
        'find_NvN_s': [0., 2., -8.],
        'overlap_NvN_s': [0., 2., -7.],
        'find_1vN_s': [0., 1., -6.],
        'overlap_1vN_s': [.01, 1., -5.],
    }
    data = {'N': n, 'NvN_pairs': .04*n*(n-1)/2}
    for key, values in coefficients.items():
        data[key] = 10**np.polyval(values, np.log10(n))
    with tempfile.TemporaryDirectory() as directory:
        result = plot_costs(data, Path(directory)/'scaling.jpg')
    for panel in result['panels']:
        assert panel['fit_points'] == 15
        for component in panel['components']:
            expected_coefficients = coefficients[component['key']]
            np.testing.assert_allclose(component['coefficients'], expected_coefficients, atol=1e-12)
        for count, prediction in panel['predicted_seconds'].items():
            expected = sum(10**np.polyval(coefficients[c['key']], np.log10(int(count)))
                           for c in panel['components'])
            np.testing.assert_allclose(prediction, expected, rtol=1e-12)
    comparison = result['panels'][0]['comparison']
    np.testing.assert_allclose(comparison['extrapolated_rejection_fraction'], .96)
    np.testing.assert_allclose(comparison['PO2.0_seconds'], 600*n+n*(n-1)/2)
    expected = result['panels'][0]['measured_total_seconds'] + 600*n + .04*n*(n-1)/2
    np.testing.assert_allclose(comparison['hybrid_seconds'], expected)
    pairs = 1e8*(1e8-1)/2
    total = result['panels'][0]['predicted_seconds']['100000000']
    np.testing.assert_allclose(comparison['hybrid_predicted_seconds']['100000000'],
                               total+600*1e8+.04*pairs)
    # The quadratic integration stage starts below generation but dominates at large N.
    assert data['overlap_NvN_s'][-1] < data['box_generation_s'][-1]
    assert result['panels'][0]['predicted_seconds']['100000000'] > 1e9
    print('Fit checks passed: power law, log-quadratic, zero exclusion, total stages and extension')


if __name__=='__main__':
    test_fits()
