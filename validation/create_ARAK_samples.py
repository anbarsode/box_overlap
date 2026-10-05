import numpy as np
rng = np.random.default_rng()
from RandomDistributions import RandomDistribution
import matplotlib.pyplot as plt
import corner
from tqdm import tqdm

for i in tqdm(range(10000)):
    try:
        intrinsic = dict(n_components=1, bounds=[[0.,3.],[-2.,0.],[-1.,1.]], dimension=3,
            reference_center=[rng.uniform(0,3), 2*rng.uniform(0,1)**0.5-2, rng.uniform(-1,1)], mode_scale=[0.3,1.5,3.],
            alignment=rng.uniform(0,1), K=2, epsilon=1., p=4, n_basis=5, coefficient_scale=.3,
            boundary_treatment='truncate', truncation_pilot_samples=int(1e3), truncation_normalization_samples=int(1e4))

        skymap = dict(n_components=4, bounds=[[0.,2*np.pi],[-1.,1.]], dimension=2,
            reference_center=[rng.uniform(0,2*np.pi),rng.uniform(-1,1)], mode_scale=[5,1], center_scale=[2,1],
            weight_decay=1., scale_decay=1., alignment=rng.uniform(0,1,4),
            K=4, epsilon=2.5, p=4, n_basis=5, coefficient_scale=1.8,
            boundary_treatment='truncate', truncation_pilot_samples=int(1e3), truncation_normalization_samples=int(1e4))

        p = RandomDistribution().product([intrinsic, skymap])
        s = p.rvs(int(1e4))
        np.save(f'ARAK_samples/event_{i}.npy', s)

        if i < 100:
            corner.corner(s, labels=[r"$\log_{10}\mathcal{M}$", r"$\log_{10}q$", r"$\chi_\mathrm{eff}$", r"$\alpha$", r"$\sin \delta$"], show_titles=True)
            plt.savefig(f'ARAK_samples/event_{i}.png')
            plt.close()

    except KeyboardInterrupt:
        raise

    except:
        print('Failed to create distribution, retrying...')
        i = i - 1
        continue