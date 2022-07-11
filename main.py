def execute_pipeline_salad_to_generate_samples_eps(alphas=[[.1] *4, [1]*4, [5]*4, [10]*4], losses=['CE', 'KL', 'Rao', 'g'], epsilons=[0.03125, 0.0625, 0.125, 0.25, 0.3125, 0.5]):
    from tabulate import tabulate

    accuracies = [['LOSS', 'EPS', 'ALPHA', 'ACC_CLASS', 'ACC_DET']]
    for loss in losses:
        for eps in epsilons:
            for alpha in alphas:
                acc_c, acc_d = execute_pipeline_salad(loss, eps, alpha)
                accuracies.append([loss, eps, alpha, acc_c, acc_d])
    print(tabulate(accuracies))
    return tabulate(accuracies)

def execute_pipeline_salad(loss=None, eps=None, alpha=None):
    from pipelines.pipeline_salad import main_pipeline_wb
    from config_reader_utls import attrDict, config_reader_utls

    config_file_path = 'config/config_salad_wb.yaml'
    args_ = attrDict.AttrDict.from_nested_dicts(config_reader_utls.read_file(file_path=config_file_path))
    acc_c, acc_d = main_pipeline_wb(args=args_, loss=loss, eps=eps, alpha=alpha)
    return acc_c, acc_d

def execute_pipeline_nss():
    from pipelines.pipeline_nss import main_pipeline_wb
    from config_reader_utls import attrDict, config_reader_utls

    config_file_path = 'config/config_nss_wb.yaml'
    args_ = attrDict.AttrDict.from_nested_dicts(config_reader_utls.read_file(file_path=config_file_path))
    main_pipeline_wb(args=args_)

def execute_pipeline_plots(alpha=None):
    from pipelines.pipeline_plots import main_pipeline_plots
    from config_reader_utls import attrDict, config_reader_utls

    config_file_path = 'config/config_salad_wb.yaml'
    args_ = attrDict.AttrDict.from_nested_dicts(config_reader_utls.read_file(file_path=config_file_path))
    main_pipeline_plots(args=args_, alphs=alpha)

def execute_pipeline_hamper():
    from pipelines.pipeline_hamper import main_pipeline_wb
    from config_reader_utls import attrDict, config_reader_utls

    config_file_path = 'config/config_hamper_wb.yaml'
    args_ = attrDict.AttrDict.from_nested_dicts(config_reader_utls.read_file(file_path=config_file_path))
    main_pipeline_wb(args=args_)

if __name__ == '__main__':
    # alphas = [.1, 1, 5, 10]
    # execute_pipeline_plots(alpha=alphas)
    #execute_pipeline_hamper()
    #execute_pipeline_salad()
    #execute_pipeline_salad_to_generate_samples_eps()
    execute_pipeline_nss()