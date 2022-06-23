import numpy as np
import torch
import gpytorch
import pandas as pd
from gps import SpectralMixtureGPModel as SMG
from gps import TwoDSpectralMixtureGPModel as TMG
import matplotlib.pyplot as plt
from trainers import train
from astropy import units

def parse_results(gp, results):
    loss = results['loss']
    print("Final loss: ",loss[-1])
    
    pass

def plot_results(gp, results):
    for key, value in results.items():
        #loss = results['loss']
        #print("Final loss: ",loss[-1])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        try:
            ax.plot(value, "-")
        except ValueError:
            
        ax.set_ylabel(key)
        ax.set_xlabel("Iteration")
        plt.show()
    pass

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
              magcolumn = 'Magnitude'):
    """
    Arguments:
    ----------
    LCfile -- full path to file containing light curve. It should contain
                two columns, one containing the time coordinate and
                another containing the magnitude (or flux) variable
    timecolumn -- name of column in LCfile containing the time coordinate
    magcolumn -- name of column in LCfile containing the magnitude variable
    """
    # testdata = pd.read_csv("~/projects/betelgeuseScuba2/AlfOriAAVSO_Vband.csv")
    testdata = pd.read_csv(LCfile)

    print(testdata)

    # train_jd = torch.Tensor(testdata['JD'].to_numpy())
    # train_mag = torch.Tensor(testdata['Magnitude'].to_numpy())
    train_jd = torch.Tensor(testdata[timecolumn].to_numpy())
    train_mag = torch.Tensor(testdata[magcolumn].to_numpy())


    # """ Let's generate some synthetic data from a perturbed sine curve 
    #     but on the same time sampling as the real data"""

    # P = 137. #Days!
    # if isinstance(period, units.Quantity):
    #     P = period.to('day').value
    # else:
    #     P = period
    # 
    # train_mag = torch.sin(train_jd*(2*np.pi/P))
    # train_mag = train_mag + 0.1*torch.randn_like(train_mag)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    #We can also try:
    #likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(train_mag_err, learn_additional_noise = True)
    #when we have measured uncertainty. While this wouldn't make a lot of sense for this set of synthetic data where the uncertainty is always the same
    #it can be a powerful addition if you have measured uncertainty. Setting learn_additional_noise = True will tell it to also try to infer and
    #additional set of Gaussian noise on the diagonal of the covariance matrix, as the simple GaussianLikelihood does. If you're very confident about the
    #measured uncertainty, set it to False instead, but if you think there may be extra physics (e.g. stochastic variation/jitter) that isn't captured
    #well by this model, or that the uncertainties may be underestimated, it is good to turn it on.

    model = SMG(train_jd, train_mag, likelihood, num_mixtures = 2)
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
    model.train()
    likelihood.train()

    #help(likelihood)

    training_iter = 100

    print(model.parameters())
    for p in model.parameters():
        print(p)

    for param_name, param in model.named_parameters():
        print(f'Parameter name: {param_name:42} value = {param.data}')


    results = train(model, likelihood, train_jd, train_mag, maxiter = training_iter, miniter = 20, stop = 0.01, optim="Adam")
    #print(results)
    for key, value in results.items():
        print(key)
        print(value)
        try:
            print(value[0].size())
        except:
            pass
    plot_results(results)
    exit()

    
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    #help(optimizer)
    
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    #help(mll)
    #exit()
    
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_jd)
        loss = -mll(output, train_mag)
        loss.backward()
        #print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
        optimizer.step()


    #Add saving the model/optimizer stuff

    #print(model)

    #for name, param in model.named_parameters():
    #    print(name, param)

    #print(model.covar_module.raw_mixture_means,#.item(),
    #      1/torch.exp(model.covar_module.raw_mixture_means),#.item(),
    #      2*np.pi/torch.exp(model.covar_module.raw_mixture_means)#.item())
    #      )
    #print(#model.covar_module.mixture_means,
    #      1./model.covar_module.mixture_means,#.item(),
    #      #2*np.pi/model.covar_module.mixture_means,
    #      model.covar_module.mixture_weights)
    #      #1/torch.exp(model.covar_module.raw_mixture_means),#.item(),
    #      #2*np.pi/torch.exp(model.covar_module.raw_mixture_means)#.item())

    if len(model.covar_module.mixture_means) == 1:
        #For a single component mixture
        print("Estimated mean magnitude: ", model.mean_module.constant.item())
        print("Estimated frequency: ", model.covar_module.mixture_means.item()) #for a single component
        print("Estimated width: ", model.covar_module.mixture_scales.item())
        print("Fractional width: ", model.covar_module.mixture_scales.item()/model.covar_module.mixture_means.item())
        print("Estimated period: ", 1./model.covar_module.mixture_means.item()) #for a single component
        print("Estimated width: ", model.covar_module.mixture_scales.item()/model.covar_module.mixture_means.item() * 1./model.covar_module.mixture_means.item())
        #print("Estimated width: ", 1./model.covar_module.mixture_scales.item())
        
        
    
        
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
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        plt.show()
