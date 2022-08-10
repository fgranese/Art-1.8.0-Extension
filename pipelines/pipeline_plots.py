import matplotlib.pyplot as plt
import random


def execute_pipeline_salad(loss=None, eps=None, alpha=None):
    from pipelines.pipeline_salad import main_pipeline_wb
    from config_reader_utls import attrDict, config_reader_utls

    config_file_path = 'config/config_salad_wb.yaml'
    args_ = attrDict.AttrDict.from_nested_dicts(config_reader_utls.read_file(file_path=config_file_path))
    acc_c, acc_d = main_pipeline_wb(args=args_, loss=loss, eps=eps, alpha=alpha)
    return acc_c, acc_d


def main_pipeline_plots(args, alphs=None):
    alphas = alphs if alphs is not None else args.ADV_CREATION.alpha
    losses_train = ['CE', 'KL', 'Rao', 'g']
    losses_test = ['CE', 'KL', 'Rao', 'g']

    random.seed(10)
    get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF), range(n)))
    dict_markers = {'CE': '^', 'KL': '+', 'Rao': 'x', 'g': 'o'}  # losses train
    markers = [dict_markers[loss] for loss in losses_train]

    colors = get_colors(len(losses_test))

    # for i in range(len(losses)):
    #     c = colors[i]
    alpha = alphas[0]
    # acc_c, acc_d = execute_pipeline_salad(alpha=alpha, loss='CE')
    # for j in range(len(losses)):
    #     plt.scatter(acc_c, acc_d[j], s=200, c=colors[j], marker=markers[0])
    #     losses_alpha_dict[0].append((acc_c, acc_d[j]))

    for loss_train in losses_train:
        losses_alpha_dict = []

        print(alpha)
        acc_c, acc_ds = execute_pipeline_salad(alpha=alpha, loss=loss_train)
        for j in range(len(losses_test)):
            plt.scatter(acc_c, acc_ds[j], s=200, c=colors[j], marker=dict_markers[loss_train])

        # for j in range(len(losses_alpha_dict)):
        plt.plot([acc_c] * len(acc_ds), [acc_d for acc_d in acc_ds], c='lightgray')


    # acc_c, acc_d = execute_pipeline_salad(alpha=alpha, loss='Rao')
    # for j in range(len(losses_test)):
    #     plt.scatter(acc_c, acc_d[j], s=200, c=colors[j], marker=markers[2])
    #     losses_alpha_dict[2].append((acc_c, acc_d[j]))
    #
    # acc_c, acc_d = execute_pipeline_salad(alpha=alpha, loss='g')
    # for j in range(len(losses_test)):
    #     plt.scatter(acc_c, acc_d[j], s=200, c=colors[j], marker=markers[3])
    #     losses_alpha_dict[3].append((acc_c, acc_d[j]))

        # acc_c, acc_d = execute_pipeline_salad(alpha=alpha, loss='KL')
        # for j in range(len(losses)):
        #     plt.scatter(acc_c, acc_d[j], s=200, c=c, marker=markers[1])
        #     losses_alpha_dict[1].append((acc_c, acc_d[j]))
        #
        # acc_c, acc_d = execute_pipeline_salad(alpha=alpha, loss='Rao')
        # for j in range(len(losses)):
        #     plt.scatter(acc_c, acc_d[j], s=200, c=c, marker=markers[2])
        #     losses_alpha_dict[2].append((acc_c, acc_d[j]))
        #
        # acc_c, acc_d = execute_pipeline_salad(alpha=alpha, loss='g')
        # for j in range(len(losses)):
        #     plt.scatter(acc_c, acc_d[j], s=200, c=c, marker=markers[3])
        #     losses_alpha_dict[3].append((acc_c, acc_d[j]))

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]

    plt.axhline(y=0.5, color='green', linestyle='dashed')
    plt.axvline(x=0.5, color='green', linestyle='dashed')

    handles = [f("s", colors[i]) for i in range(len(colors))]
    handles += [f(markers[i], "k") for i in range(len(markers))]

    # labels = alphas + ['CE', 'KL', 'FR', 'Gini']
    labels = ['Gini test' if loss == 'g' else loss + ' test' for loss in losses_test] +\
             ['Gini train' if loss == 'g' else loss + ' train' for loss in losses_train]
             #['CE train', 'KL train', 'FR train', 'Gini train'] + ['CE test', 'KL test', 'FR test', 'Gini test']
    plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xlabel('Accuracy classifier')
    plt.ylabel('Accuracy detector')
    plt.title('ATTACK {} - {} - {} - {}'.format(args.ADV_CREATION.strategy, args.ADV_CREATION.norm, args.ADV_CREATION.epsilon, alpha))
    plt.tight_layout()
    plt.savefig('plots/{}_{}_{}_all_2.png'.format(args.ADV_CREATION.strategy, args.ADV_CREATION.norm, args.ADV_CREATION.epsilon))
