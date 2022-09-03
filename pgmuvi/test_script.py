import numpy as np
import torch
import gpytorch
import pandas as pd
from gps import SpectralMixtureGPModel as SMG
from gps import SpectralMixtureKISSGPModel as SMKG
from gps import TwoDSpectralMixtureGPModel as TMG
import matplotlib.pyplot as plt
from trainers import train
from gpytorch.constraints import Interval
from gpytorch.priors import LogNormalPrior, NormalPrior, UniformPrior
import pyro
from pyro.infer.mcmc import NUTS, MCMC, HMC
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist

from astropy import units

def parse_results(gp, results):
    loss = results['loss']
    print("Final loss: ",loss[-1])
    
    pass

def plot_results(gp, results, zscale = None):
    for key, value in results.items():
        #print(key, value, torch.Tensor(value))
        #loss = results['loss']
        #print("Final loss: ",loss[-1])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        try:
            ax.plot(value, "-")
        except ValueError:
            pass
        ax.set_ylabel(key)
        ax.set_xlabel("Iteration")
        if "means" in key and zscale is not None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(zscale/torch.Tensor(value),"-")
            ax.set_ylabel(key)
            ax.set_xlabel("Iteration")
    #plt.show()
    #pass

def plot_psd(gp, results):
    #if isinstance(results['covar_module.raw_mixture_weights'][-1], float): #only one mixture component
    #    n_dim, n_mix = 1, 1
    #elif isinstance(results['covar_module.raw_mixture_weights'][-1], ): #only one mixture component
    #    pass
    n_mix = results['covar_module.raw_mixture_means'][-1].size()[0]
    n_dim = result['covar_module.raw_mixture_means'][-1].size()[1]
    pass

def plot_data():
    pass

# if __name__=="__main__":
def run_pgmuvi(LCfile = 'AlfOriAAVSO_Vband.csv', timecolumn = 'JD', \
               magcolumn = 'Magnitude', synthetic_data=False):
    """
    Arguments:
    ----------
    LCfile -- full path to file containing light curve. It should contain
                two columns, one containing the time coordinate and
                another containing the magnitude (or flux) variable
    timecolumn -- name of column in LCfile containing the time coordinate
    magcolumn -- name of column in LCfile containing the magnitude variable
    """

    if synthetic_data:


        """ Let's generate some synthetic data from a perturbed sine curve 
            but on the same time sampling as the real data"""

        P = np.random.uniform(30, 300)#137. #Days!
        print("True period: ",P," days")
        n_data = 400
        jd_min = 2450000
        n_periods = np.random.uniform(3,100)
        jd_max = jd_min + P*(n_periods)
        print("Simulating for ",n_periods," periods")
        train_jd = torch.Tensor(np.random.uniform(jd_min, jd_max, size=n_data))
        train_mag = torch.sin(train_jd*(2*np.pi/P))
        train_mag = train_mag + 0.1*torch.randn_like(train_mag)
        train_mag_err = 0.1*train_mag

        period_guess = P*(np.random.uniform()+0.5)#147 #this number is in the same units as our original input.
    else:
        #testdata = pd.read_csv("~/projects/betelgeuseScuba2/AlfOriAAVSO_Vband.csv")#[-700:]
        testdata = pd.read_csv(LCfile)
        #train_jd = torch.Tensor(testdata['JD'].to_numpy())
        train_jd = torch.Tensor(testdata[timecolumn].to_numpy())
        train_mag = torch.Tensor(testdata[magcolumn].to_numpy())
        print(len(train_jd))
        #train_mag = torch.Tensor(testdata['Magnitude'].to_numpy())
        train_mag_err = 0.05*train_mag

        period_guess = 400.

    #For homoscedastic data, or when the scatter dominates instead of uncertainty, you can use this likelihood, which will attempt to learn a single number of the noise
    #likelihood = gpytorch.likelihoods.GaussianLikelihood()
    #If you want to constrain the amount of noise, you can pass a constraint to the likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=Interval(0.01, 0.15))
    
    #We can also try:
    #likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(train_mag_err, learn_additional_noise = True)
    #when we have measured uncertainty. While this wouldn't make a lot of sense for this set of synthetic data where the uncertainty is always the same
    #it can be a powerful addition if you have measured uncertainty. Setting learn_additional_noise = True will tell it to also try to infer and
    #additional set of Gaussian noise on the diagonal of the covariance matrix, as the simple GaussianLikelihood does. If you're very confident about the
    #measured uncertainty, set it to False instead, but if you think there may be extra physics (e.g. stochastic variation/jitter) that isn't captured
    #well by this model, or that the uncertainties may be underestimated, it is good to turn it on.

    #We're going to put a constraint on the noise in the likelihood, because we want to use the information we have about the data
    #likelihood.register_constraint("noise_constraint", Interval(0.08, 0.2), "noise")

    n_mix = 1

    #train_jd = train_jd/365.25
    #z-scoring (i.e. transforming the data so that it fills the [0,1)
    #interval and does not have values outside that interval)the data is
    #important. In our case, that is particularly true of the x-values
    #(dates here).
    #This also has knock-on effects on the interpretation of the frequencies
    #in the power spectrum (as they will be inferred for the [0,1) interval!)
    date_range = torch.max(train_jd) - torch.min(train_jd)
    train_jd = train_jd - torch.min(train_jd)
    train_jd = train_jd/torch.max(train_jd)

    #model = SMKG(train_jd, train_mag, likelihood, num_mixtures = n_mix, grid_size=4000)
    model = SMG(train_jd, train_mag, likelihood, num_mixtures = n_mix)

    train_method=False #"NUTS"

    #model.covar_module.base_kernel.raw_mixture_means.item = torch.Tensor([[[np.log10(1/100)]]])

    
    #Now we should z-scale it
    period_guess = period_guess/date_range

    hypers = {
        'likelihood.noise_covar.noise': torch.tensor(0.2),
        'covar_module.base_kernel.raw_mixture_means': (1/period_guess.clone().detach()),
        'covar_module.base_kernel.raw_mixture_weights': torch.tensor(0.0),
        'covar_module.base_kernel.raw_mixture_scales': torch.tensor(0.),
        'mean_module.constant': torch.tensor(0.),
    }

    hypers = {
        'likelihood.noise_covar.noise': torch.tensor(0.2),
        'covar_module.raw_mixture_means': (1/period_guess.clone().detach()), #torch.tensor(-2.),
        'covar_module.raw_mixture_weights': torch.tensor(0.),
        'covar_module.raw_mixture_scales': torch.tensor(0.),
        'mean_module.constant': torch.tensor(0.),
    }

    hypers = {
        'likelihood.noise_covar.noise': torch.tensor(0.1),
        'covar_module.mixture_means': (1/period_guess.clone().detach()), #torch.tensor(-2.),
        'covar_module.mixture_weights': torch.tensor(1.),
        'covar_module.mixture_scales': torch.tensor(1.),
        'mean_module.constant': torch.tensor(0.),
    }
    model.initialize(**hypers)

    print("Initial z-scaled period: ", 1./(model.sci_kernel.mixture_means.item())) #for a single component
    print("Initial period: ", date_range * 1./(model.sci_kernel.mixture_means.item())) #for a single component
    print("Inital width: ", (model.sci_kernel.mixture_scales.item()/model.sci_kernel.mixture_means.item()) * (date_range/(model.sci_kernel.mixture_means.item())))
    print("Initial weight: ", model.sci_kernel.mixture_weights.item())
    print("Additional noise: ", model.likelihood.noise_covar.noise.item())

    #print(model)
    #print(model.covar_module)
    #help(model)
    #exit()

    #for name, param in model.named_parameters():
    #    print(name, param)
    #[print(p) for p in model.parameters()]


    #print(model.covar_module.raw_mixture_means,#.item(),
    #      1/torch.exp(model.covar_module.raw_mixture_means),#.item(),
    #      2*np.pi/torch.exp(model.covar_module.raw_mixture_means)#.item())
    #      )

    #exit()
    if train_method=="NUTS":
        cuda = False #Change this to True if you have access to a GPU
        if cuda:
            likelihood.cuda()
            model.cuda()

        num_samples = 100
        warmup_steps= 100
        model.mean_module.register_prior("mean_prior", UniformPrior(-1, 1), "constant")
        model.covar_module.register_prior("means_prior", LogNormalPrior(1/P, 0.1), "mixture_means")
        model.covar_module.register_prior("weights_prior", UniformPrior(0,1), "mixture_weights")
        model.covar_module.register_prior("scales_prior", UniformPrior(0,1/P), "mixture_scales")
        #model.covar_module.base_kernel.register_prior("lengthscale_prior", UniformPrior(0.01, 0.5), "lengthscale")
        #model.covar_module.register_prior("outputscale_prior", UniformPrior(1, 2), "outputscale")
        likelihood.register_prior("noise_prior", UniformPrior(0.01, 0.15), "noise")
        
        def pyro_model(x, y):
            with gpytorch.settings.fast_computations(False, False, False):
                sampled_model = model.pyro_sample_from_prior()
                #help(sampled_model)
                output = sampled_model.likelihood(sampled_model(x))
                pyro.sample("obs", output, obs=y)
            return y

        jit = False
        nuts_kernel = NUTS(pyro_model, adapt_step_size=True, jit_compile=jit)
        mcmc_run = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, disable_progbar=False)
        if cuda:
            mcmc_run.run(train_jd.cuda(), train_mag.cuda())
        else:
            mcmc_run.run(train_jd, train_mag)

        print(mcmc_run.summary(prob=0.5))

        model.pyro_load_from_samples(mcmc_run.get_samples())

        exit()

    elif train_method=="VI":
        cuda = False #Change this to True if you have access to a GPU
        if cuda:
            likelihood.cuda()
            model.cuda()

        num_samples = 100
        warmup_steps= 100
        model.mean_module.register_prior("mean_prior", UniformPrior(-1, 1), "constant")
        model.covar_module.register_prior("means_prior", LogNormalPrior(1/P, 0.1), "mixture_means")
        model.covar_module.register_prior("weigthts_prior", UniformPrior(0,1), "mixture_weights")
        model.covar_module.register_prior("scales_prior", UniformPrior(0,1/P), "mixture_scales")
        #model.covar_module.base_kernel.register_prior("lengthscale_prior", UniformPrior(0.01, 0.5), "lengthscale")
        #model.covar_module.register_prior("outputscale_prior", UniformPrior(1, 2), "outputscale")
        likelihood.register_prior("noise_prior", UniformPrior(0.01, 0.5), "noise")
        
        def pyro_model(x, y):
            with gpytorch.settings.fast_computations(False, False, False):
                sampled_model = model.pyro_sample_from_prior()
                #help(sampled_model)
                output = sampled_model.likelihood(sampled_model(x))
                pyro.sample("obs", output, obs=y)
            return y

        def pyro_guide(x,y):
            pass
    else:
        model.train()
        likelihood.train()
    

        #help(likelihood)

        training_iter = 300#0

        print(model.parameters())
        for p in model.parameters():
            print(p)

        for param_name, param in model.named_parameters():
            print(f'Parameter name: {param_name:42} value = {param.data}')

        with gpytorch.settings.max_cg_iterations(10000):
            results = train(model, likelihood, train_jd, train_mag, maxiter = training_iter, miniter = 50, stop = 0.00001, lr=0.1, optim="AdamW", stopavg=30)

    
    for key, value in results.items():
        print(key)
        print(value)
        try:
            print(value[0].size())
        except:
            pass
    plot_results(results)
    
    ## Use the adam optimizer
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    ##help(optimizer)
    
    ## "Loss" for GPs - the marginal log likelihood
    #mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    ##help(mll)
    ##exit()
    
    #for i in range(training_iter):
    #    optimizer.zero_grad()
    #    output = model(train_jd)
    #    loss = -mll(output, train_mag)
    #    loss.backward()
    #    #print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    #    optimizer.step()


    ##Add saving the model/optimizer stuff

    ##print(model)

    ##for name, param in model.named_parameters():
    ##    print(name, param)

    ##print(model.covar_module.raw_mixture_means,#.item(),
    ##      1/torch.exp(model.covar_module.raw_mixture_means),#.item(),
    ##      2*np.pi/torch.exp(model.covar_module.raw_mixture_means)#.item())
    ##      )
    ##print(#model.covar_module.mixture_means,
    ##      1./model.covar_module.mixture_means,#.item(),
    ##      #2*np.pi/model.covar_module.mixture_means,
    ##      model.covar_module.mixture_weights)
    ##      #1/torch.exp(model.covar_module.raw_mixture_means),#.item(),
    ##      #2*np.pi/torch.exp(model.covar_module.raw_mixture_means)#.item())

    if len(model.sci_kernel.mixture_means) == 1:
        #For a single component mixture
        print("Estimated mean magnitude: ", model.mean_module.constant.item())
        #print("Estimated frequency: ", model.covar_module.mixture_means.item()) #for a single component
        #print("Estimated width: ", model.covar_module.mixture_scales.item())
        #print("Fractional width: ", model.covar_module.mixture_scales.item()/model.covar_module.mixture_means.item())
        print("Estimated period: ", date_range * 1./(model.sci_kernel.mixture_means.item())) #for a single component
        print("Estimated width: ", model.sci_kernel.mixture_scales.item()/model.sci_kernel.mixture_means.item() * (1./(model.sci_kernel.mixture_means.item()*date_range)))
        print("Estimated weight: ", model.sci_kernel.mixture_weights.item())
        print("Additional noise: ", model.likelihood.noise_covar.noise.item())
        #print("Estimated width: ", 1./model.covar_module.mixture_scales.item())
        
    plot_results(model, results, zscale = date_range)
    
        
    #exit()
    # 1000 test points across the range of the data
    test_x = torch.linspace(train_jd.min(), train_jd.max(), 10000)

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    
    # The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
    # See https://arxiv.org/abs/1803.06058
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Make predictions
        observed_pred = likelihood(model(test_x))
        
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))
        
        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(train_jd.numpy(), train_mag.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([3, -3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        plt.show()


if __name__=="__main__":
    run_pgmuvi(LCfile="~/projects/betelgeuseScuba2/AlfOriAAVSO_Vband.csv")
