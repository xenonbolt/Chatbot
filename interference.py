from sources import model_ensemble

GPU_frac = 0.06

models = {
    {'model':'15k', 'gpu': 0, 'memory_fraction': GPU_frac},
    {'model':'33k', 'gpu': 0, 'memory_fraction': GPU_frac},
    {'model':'52k', 'gpu': 0, 'memory_fraction': GPU_frac},
    {'model':'54k', 'gpu': 0, 'memory_fraction': GPU_frac},
    {'model':'55k', 'gpu': 0, 'memory_fraction': GPU_frac},
    {'model':'75k', 'gpu': 0, 'memory_fraction': GPU_frac},
    {'model':'76k', 'gpu': 0, 'memory_fraction': GPU_frac},
    {'model':'80k', 'gpu': 0, 'memory_fraction': GPU_frac},
    {'model':'83k', 'gpu': 0, 'memory_fraction': GPU_frac},
    {'model':'84k', 'gpu': 0, 'memory_fraction': GPU_frac},
    {'model':'87k', 'gpu': 0, 'memory_fraction': GPU_frac},
    # {'model':'99k', 'gpu': 0, 'memory_fraction': GPU_frac},

    {'model':'140k', 'gpu': 1, 'memory_fraction': GPU_frac},
    {'model':'141k', 'gpu': 1, 'memory_fraction': GPU_frac},
    {'model':'160k', 'gpu': 1, 'memory_fraction': GPU_frac},
    {'model':'166k', 'gpu': 1, 'memory_fraction': GPU_frac},
    # {'model':'170k', 'gpu': 1, 'memory_fraction': GPU_frac},
    {'model':'199k', 'gpu': 1, 'memory_fraction': GPU_frac},
    {'model':'200k', 'gpu': 1, 'memory_fraction': GPU_frac},
    {'model':'214k', 'gpu': 1, 'memory_fraction': GPU_frac},
    {'model':'215k', 'gpu': 1, 'memory_fraction': GPU_frac},
    {'model':'217k', 'gpu': 1, 'memory_fraction': GPU_frac},
    {'model':'228k', 'gpu': 1, 'memory_fraction': GPU_frac},
    {'model':'259k', 'gpu': 1, 'memory_fraction': GPU_frac}, 
}

if __name__ == '__main__':
    ensemble_interference = model_ensemble.init(models)

    # responses = ensemble_interference('hello world')
    # print(responses)

    model_ensemble.live_ensemble_interference()