from __future__ import division
from matplotlib import use
use('agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.special import erf, erfc, erfinv

__all__ = ['ManyBackgroundCollection',
           'lnprob', 'lnprior', 'lnlike']
class ManyBackgroundCollection(object):
    """
    A set of events with an associated ranking statistic.
    Events are divided into foreground and several different
    classes of background events.
    """
    def __init__(self, glitch_dict, xmin=3.5):
        """
        Initialize the sample collector

        Parameters:
        -----------
        glitch_dict: `dict`
            dictionary with glitch class as key (i.e. 'Scratchy')
            and corresponding snr time-series as value

        xmin: `float`, optional, default: 3.5
            Minimum threshold SNR

        Returns
        -------
        `ManyBackgroundCollection`: with the following
        attrs, `xmin`, `glitch_dict`

        Notes
        -----"""

        self.xmin = xmin
        self.glitch_dict = glitch_dict


    def draw_samples(self, foreground_count, gaussian_background_count,
                     **kwargs):
        """
        Draw a full set of foreground, background, and Gravity Spy
        events

        Parameters:
        -----------
        foreground_count : `int`
            known count of foreground events

        gaussian_background_count : `int`
            known count of background events

        glitch_classes : `list`, optional, default: `self.glitch_dict.keys()`
            if you would like
            to only populate samples from some of the gravityspy
            categories provide a list like `['Scratchy', 'Blip']`

        Returns
        -------
        self : `ManyBackgroundCollection` now has an attr
            `samples` that contains keys of 'Foreground', 'Gaussian'
            and list of glitch_classes. In addition, the attrs
            `foreground_count`, `gaussian_background_count`,
            `unlabeled_samples` and `num_samples` are set.

        Notes
        -----
        """

        glitch_classes = kwargs.pop('glitch_classes', self.glitch_dict.keys())
        glitch_counts = kwargs.pop('glitch_counts', [10]*len(self.glitch_dict.keys()))
        self.samples = {}

        # Draw foreground samples
        self.foreground_count = foreground_count
        self.samples['Foreground'] = self.xmin * (1 -
                                         np.random.uniform(size=foreground_count))**(-1/3)



        # Draw gaussian background samples
        self.gaussian_background_count = gaussian_background_count
        self.samples['Gaussian'] = np.sqrt(2) * erfinv(1 -
                                       (1 - np.random.uniform(
                                       size=gaussian_background_count))*
                                       erfc(self.xmin / np.sqrt(2)))

        # Define each glitch class to have SNRs defined in the glitch_dict
        for glitch_class, glitch_count in zip(glitch_classes, glitch_counts):
            self.samples[glitch_class] = np.random.choice(
                np.array(self.glitch_dict[glitch_class]), size=int(glitch_count))

        # Create array of all samples, regardless of label
        self.unlabeled_samples = np.array([])
        for key in self.samples.keys():
            self.unlabeled_samples = np.append(self.unlabeled_samples,
                np.array(self.samples[key]))

        self.num_samples = len(self.unlabeled_samples)


    def plot_hist(self):
        """
        Make a histogram of all drawn samples.
        """
        num_classes = len(self.samples.keys())
        num_bins = int(np.floor(np.sqrt(self.num_samples)))
        colors = plt.cm.viridis(np.linspace(0, 1, num_classes))

        # FIXME: need a robust and uniform way to define bins
        bins = np.linspace(self.xmin, max(self.unlabeled_samples), num_bins)

        plot = plt.figure(figsize=(20,10))
        ax = plot.gca()

        for idx, icategory in enumerate(self.samples.keys()):
            ax.hist(self.samples[icategory], label=icategory,
                     color=colors[idx], bins=bins, cumulative=-1,
                     histtype='step')

        plot.legend(loc='upper right')
        ax.set_yscale('log', nonposy='clip')
        ax.set_xlim(self.xmin, max(self.unlabeled_samples) + 1)
        ax.set_ylim(1, None)
        ax.set_xlabel('SNR')
        ax.set_ylabel('Number of Events  with SNR > Corresponding SNR')
        ax.set_title('%i Samples with Minimum SNR of %.2f' % (int(self.num_samples), self.xmin))
        return plot


    def lnlike(self, counts, glitch_classes=[]):
        """
        Log Likelihood

        Parameters:
        -----------
        counts: array
            each entry is a count for each source type in the following order:
            [foreground_counts, gaussian_counts, all_other_glitch_counts]
        """
        if np.all(counts >= 0):
            # Foreground likelihood
            fg_likelihood = getattr(self,  'Foreground' + '_evaluted')* \
                 counts[0]

            # Gaussian noise likelihood
            gauss_likelihood = getattr(self,  'Gaussian' + '_evaluted') * counts[1]

            # Likelihood for all other glitch sources of interest
            glitch_likelihood = 0
            for idx, iglitchtype in enumerate(glitch_classes):
                # Evaluate likelihood
                glitch_likelihood += counts[idx+2] * getattr(self, iglitchtype + '_evaluted')

            return np.sum(np.log(fg_likelihood + gauss_likelihood + \
                glitch_likelihood))
        else:
            return -np.inf


    def lnprior(self, counts):
        """
        Log Prior

        Parameters:
        -----------
        counts: array
            each entry is a count for each source type in the following order:
            [foreground_counts, gaussian_counts, all_other_glitch_counts]

        N.B.: technically, the exp^(-Sum(counts)) term is part of the likelihood in FGMC
        """
        if np.all(counts >= 0):
            return -np.sum(counts) - 0.5*np.log(np.prod(counts))
        else:
            return -np.inf


    def lnprob(self, counts, glitch_classes=[]):
        """
        Combine log likelihood and log prior

        Parameters:
        -----------
        counts: array
            each entry is a count for each source type in the following order:
            [foreground_counts, gaussian_counts, all_other_glitch_counts]
         """

        prior = self.lnprior(counts)
        posterior = self.lnlike(counts, glitch_classes)
        if not np.isfinite(prior):
            return -np.inf
        return prior + posterior



def lnlike(theta, samples, xmin):
    """
    Log Likelihood

    Parameters:
    -----------
    theta: iterable of two floats - (rate_f, rate_b)
        first entry corresponds to the rate of foreground events;
        second entry is the rate of background events;

    samples: array
        all SNR values in the given distribution

    xmin: float
        minimum threshold SNR
    """
    rate_f, rate_b = theta
    if rate_f > 0 and rate_b > 0:
        return np.sum(np.log(3 * xmin**3 * rate_f * samples**(-4) \
            + rate_b * (np.sqrt(np.pi/2) * erfc(
            xmin / np.sqrt(2)))**(-1) * np.exp(-samples**2 / 2)))
    else:
        return -np.inf


def lnprior(theta):
    """
    Log Prior

    Parameters:
    -----------
    theta: iterable of two floats - (rate_f, rate_b)
        first entry corresponds to the rate of foreground events;
        second entry is the rate of background events;


    N.B.: technically, the exp^(-Rf - Rb) term is part of the likelihood in FGMC
    """

    rate_f, rate_b = theta
    if rate_f > 0 and rate_b > 0:
        return -rate_f - rate_b - 0.5*np.log(rate_f * rate_b)
    else:
        return -np.inf


def lnprob(theta, samples, xmin):
    """
    Combine log likelihood and log prior

    Parameters:
    -----------
    theta: iterable of two floats - (rate_f, rate_b)
        first entry corresponds to the rate of foreground events;
        second entry is the rate of background events;

    samples: array
        all SNR values in the given distribution

    xmin: float
        minimum threshold SNR
    """

    prior = lnprior(theta)
    posterior = lnlike(theta, samples, xmin)
    if not np.isfinite(prior):
        return -np.inf

    return prior + posterior


def compute_pastro(collection, lambda_post_samples, glitch_classes, glitch_kdes,
                   xmin=3.5, num_draw_from_lamda=1000, category='Foreground',
                   random_state=1986):
    """
    Calculate the probability that a given SNR sample
    comes from an astrophysical distribution.

    Parameters:
    -----------
    snr: float
        value of signal to noise ratio of interest

    post_samples: array of floats
        array of posterior samples of dimension
        (num_samples, num_source_classes) where
        num_samples is the number of posterior
        samples and num_source_classes is the number
        of classes considered (i.e. foreground, gaussian, etc.)

    source_classes: array of strings
        Each entry corresponds to the name of a type of
        source class: Foreground, Background, Blip, Scratchy, etc.

    glitch_kdes: dictionary
        Evaluated KDE for each GravitySpy glitch class.
        Keys are strings of GravitySpy class names.

    xmin: float
        Minimum threshold SNR

    num_iters: int
        number of samples drawn for Monte Carlo integration

    Returns:
    --------
    float or array
        P(astro) for each SNR value provided.
        This only returns one P(astro) if only one SNR is provided.

    """
    # Draw a sample from the posterior (some set of counts)
    counts = lambda_post_samples.sample(n=num_draw_from_lamda,
        random_state=random_state)

    # Compute the "likelihood ratio" for the drawn posterior sample
    setattr(collection, '{0}_likelihood'.format('Foreground'),
            np.multiply.outer(counts['Foreground'],
                              3 * xmin**3 * collection.unlabeled_samples**(-4)
                              ))

    setattr(collection, '{0}_likelihood'.format('Gaussian'),
            np.multiply.outer(counts['Gaussian'],
            (np.sqrt(np.pi/2) * erfc(
                xmin / np.sqrt(2)))**(-1) * np.exp(-collection.unlabeled_samples**2 / 2)
                                        ))

    # All GSpy glitches have an SNR greater than 7.5
    for idx, iglitchtype in enumerate(glitch_classes):
        setattr(collection, '{0}_likelihood'.format(iglitchtype),
            np.multiply.outer(counts[iglitchtype],
            np.exp(
            glitch_kdes[iglitchtype].score_samples(collection.unlabeled_samples.reshape(-1,1)))
            ))

    # Add to other samples
    denominator = 0
    for ikey in collection.samples.keys():
        denominator += getattr(collection, '{0}_likelihood'.format(ikey))

    likelihood_ratio = getattr(collection, '{0}_likelihood'.format(category))/\
        denominator

    # Report sum divided by N
    return likelihood_ratio.sum(axis=0) / num_draw_from_lamda
