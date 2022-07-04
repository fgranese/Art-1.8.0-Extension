import matplotlib.pyplot as plt
import random

def execute_pipeline_salad(loss=None, eps=None, alpha=None):
    from pipelines.pipeline_salad import main_pipeline_wb
    from config_reader_utls import attrDict, config_reader_utls

    config_file_path = 'config/config_salad_wb.yaml'
    args_ = attrDict.AttrDict.from_nested_dicts(config_reader_utls.read_file(file_path=config_file_path))
    acc_c, acc_d = main_pipeline_wb(args=args_, loss=loss, eps=eps, alpha=alpha)
    return acc_c, acc_d

def main_pipeline_plots(args, alphs=None, lss=None):
    alphas = alphs if alphs is not None else args.ADV_CREATION.alpha
    get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF), range(n)))
    colors = get_colors(len(alphas))
    markers = ['^', '+', 'x', 'o']
    losses_alpha_dict = [[],[],[],[]]

    for i in range(len(alphas)):
        c = colors[i]
        alpha = alphas[i]
        acc_c, acc_d = execute_pipeline_salad(alpha=alpha, loss='CE')
        plt.scatter(acc_c, acc_d, s=200, c=c, marker=markers[0])
        losses_alpha_dict[0].append((acc_c, acc_d))

        acc_c, acc_d = execute_pipeline_salad(alpha=alpha, loss='KL')
        plt.scatter(acc_c, acc_d, s=200, c=c, marker=markers[1])
        losses_alpha_dict[1].append((acc_c, acc_d))

        acc_c, acc_d = execute_pipeline_salad(alpha=alpha, loss='Rao')
        plt.scatter(acc_c, acc_d, s=200, c=c, marker=markers[2])
        losses_alpha_dict[2].append((acc_c, acc_d))

        acc_c, acc_d = execute_pipeline_salad(alpha=alpha, loss='g')
        plt.scatter(acc_c, acc_d, s=200, c=c, marker=markers[3])
        losses_alpha_dict[3].append((acc_c, acc_d))

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]

    for j in range(len(losses_alpha_dict)):

        plt.plot([acc_c[0] for acc_c in losses_alpha_dict[j]], [acc_c[1] for acc_c in losses_alpha_dict[j]], c='lightgray')

    handles = [f("s", colors[i]) for i in range(len(colors))]
    handles += [f(markers[i], "k") for i in range(len(markers))]

    labels = alphas + ['CE', 'KL', 'FR', 'Gini']
    plt.legend(handles, labels)

    plt.xlabel('Accuracy classifier')
    plt.ylabel('Accuracy detector')
    plt.title('ATTACK {} - {} - {}'.format(args.ADV_CREATION.strategy, args.ADV_CREATION.norm, args.ADV_CREATION.epsilon))
    plt.savefig('prova.png')


