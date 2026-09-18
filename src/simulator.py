"""
lenstronomy-based simulator for galaxy-galaxy strong lensing images.

Build the ImageModel ONCE (simulator.__init__), then call simulate(theta)
repeatedly -- only kwargs_lens / kwargs_source change per draw.
"""

import numpy as np

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Util import util, image_util
import lenstronomy.Util.simulation_util as sim_util

from priors import theta_to_kwargs, FIXED_SOURCE


class LensSimulator:
    def __init__(
        self,
        num_pix=64,
        delta_pix=0.05,
        psf_fwhm=0.1,
        exposure_time=100.0,
        background_rms=0.05,
        target_snr_range=(10, 30),
        include_kappa=True,
        seed=None,
    ):
        self.num_pix = num_pix
        self.delta_pix = delta_pix
        self.exposure_time = exposure_time
        self.background_rms = background_rms
        self.target_snr_range = target_snr_range
        self.include_kappa = include_kappa
        self.rng = np.random.default_rng(seed)

        kwargs_data = sim_util.data_configure_simple(
            num_pix, delta_pix, exposure_time, background_rms
        )
        from lenstronomy.Data.imaging_data import ImageData
        self.data_class = ImageData(**kwargs_data)

        kwargs_psf = {
            "psf_type": "GAUSSIAN",
            "fwhm": psf_fwhm,
            "pixel_size": delta_pix,
            "truncation": 5,
        }
        self.psf_class = PSF(**kwargs_psf)

        self.lens_model_class = LensModel(
            lens_model_list=["SIE", "SHEAR", "CONVERGENCE"]
        )
        self.source_model_class = LightModel(light_model_list=["SERSIC_ELLIPSE"])

        kwargs_numerics = {"supersampling_factor": 2, "supersampling_convolution": True}

        self.image_model = ImageModel(
            data_class=self.data_class,
            psf_class=self.psf_class,
            lens_model_class=self.lens_model_class,
            source_model_class=self.source_model_class,
            lens_light_model_class=None,
            point_source_class=None,
            kwargs_numerics=kwargs_numerics,
        )

    def _clean_image(self, kwargs_lens, kwargs_source):
        return self.image_model.image(
            kwargs_lens, kwargs_source, kwargs_lens_light=None, kwargs_ps=None
        )

    def _peak_snr(self, image):
        peak_signal = image.max() * self.exposure_time
        noise = np.sqrt(peak_signal + (self.background_rms * self.exposure_time) ** 2)
        return peak_signal / noise if noise > 0 else 0.0

    def _rescale_amplitude(self, kwargs_lens, kwargs_source, target, max_iter=15):
        lo, hi = self.target_snr_range

        amp = kwargs_source[0]["amp"]
        image, snr = self._clean_image(kwargs_lens, kwargs_source), 0.0
        for _ in range(max_iter):
            image = self._clean_image(kwargs_lens, kwargs_source)
            snr = self._peak_snr(image)
            if snr <= 0:
                amp *= 3.0
            else:
                amp *= target / snr
                if abs(snr - target) < 0.05 * target:
                    break
            kwargs_source[0]["amp"] = amp
        image = self._clean_image(kwargs_lens, kwargs_source)
        snr = self._peak_snr(image)
        return image, snr

    def _add_noise(self, image):
        poisson = image_util.add_poisson(image, exp_time=self.exposure_time)
        bkg = image_util.add_background(image, sigma_bkd=self.background_rms)
        return image + poisson + bkg

    def simulate(self, theta, add_noise=True, return_snr=False):
        kwargs_lens, kwargs_source = theta_to_kwargs(theta, include_kappa=self.include_kappa)
        kwargs_source[0]["amp"] = FIXED_SOURCE["amp"]

        target = self.rng.uniform(*self.target_snr_range)
        clean_image, snr = self._rescale_amplitude(kwargs_lens, kwargs_source, target)

        image = self._add_noise(clean_image) if add_noise else clean_image
        image = image.astype(np.float32)

        if return_snr:
            return image, snr
        return image

    def simulate_batch(self, thetas):
        return np.stack([self.simulate(t) for t in thetas], axis=0)
