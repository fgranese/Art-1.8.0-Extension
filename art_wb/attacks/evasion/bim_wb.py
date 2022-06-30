from art_wb.attacks.evasion.pgd_wb import ProjectedGradientDescent_WB

class BasicIterativeMethod_WB(ProjectedGradientDescent_WB):
    def __init__(self, **kwargs):
        kwargs['num_random_init'] = 0
        super().__init__(**kwargs)