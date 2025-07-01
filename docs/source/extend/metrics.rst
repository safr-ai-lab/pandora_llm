Metrics
=======

We provide a wide array of both `matplotlib` and `plotly` visualization tools in `pandora_llm.utils.plot_utils`, including histograms and ROC plots.

But you may also wish to include a new metric or evlauation plot that allows you to measure different dimensions of privacy leakage. To do so:

**1. Add the function to compute the metric in pandora_llm.utils.plot_utils**

.. code:: python
    
    def compute_metric(ground_truth, predictions, **kwargs):
        
        metric = metric_fn(ground_truth,predictions)
        
        return metric

**2. It is good practice to include the option of computing the 95% confidence interval.**

This is a template that will enable you to compute the bootstrapped confidence interval for potentially multiple statistics at the same time.

.. code:: python

    def compute_metric_with_ci(ground_truth, predictions, **kwargs):

        metric = metric_fn(ground_truth,predictions)

        def metric_bootstrap(data,axis):
            ground_truth = data[0,0,:].T
            predictions = data[1,0,:].T
            metric = metric_fn(ground_truth, predictions)
            other_derived_values = ... 
            return np.array([[metric]+[other_derived_values]]).T
        
        data = torch.cat((ground_truth[:,None],predictions[:,None]),dim=1)
        from scipy.stats import bootstrap
        bootstrap_result = bootstrap((data,), metric_bootstrap, confidence_level=0.95, n_resamples=num_bootstraps, batch=1, method='percentile',axis=0)
        metric_se = bootstrap_result.standard_error[0]
        other_derived_values_se = bootstrap_result.standard_error[1:]

        return metric, metric_se

**3. Plotting can be done with the plotting library of your choice. Here is an example in ``matplotlib`` for ROC.**

.. code:: python
    
    def plot_metric(ground_truth, predictions, **kwargs):
        
        ...
    
        plt.figure(figsize=(7,7),dpi=300)
        plt.plot([0, 1], [0, 1], linestyle="--", c="k")
        if not log_scale:
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:0.4f}',c=color)
            plt.xlim([0,1] if lims is None else lims)
            plt.ylim([0,1] if lims is None else lims)
        else:
            plt.loglog(fpr, tpr, label=f'AUC = {roc_auc:0.4f}',c=color)
            plt.xlim([10**(-int(np.log10(n_points))),1] if lims is None else lims)
            plt.ylim([10**(-int(np.log10(n_points))),1] if lims is None else lims)
        if ci:
            plt.fill_between(fpr_range,bootstrap_result.confidence_interval.low[1:],bootstrap_result.confidence_interval.high[1:],alpha=0.1,color=color)
        plt.title(plot_title)
        plt.legend(loc="lower right")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.minorticks_on()
        plt.grid(which="major",alpha=0.2)
        plt.grid(which="minor",alpha=0.1)
    
        ...