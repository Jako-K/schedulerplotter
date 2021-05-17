import seaborn; seaborn.set_style("darkgrid")
from ._scheduler_parent import Scheduler
import ipywidgets as widgets
import torch

class WidgetLambdaLR(Scheduler):
    def __init__(self, refresh_func):
        """
        DESCRIPTION COPIED FROM PYTORCH: https://pytorch.org/docs/stable/optim.html

        Sets the learning rate of each parameter group to the initial lr times a given function. When last_epoch=-1, sets initial lr as lr.
        torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1, verbose=False)
        """
        super().__init__(name="LambdaLR", refresh_func=refresh_func)
        self.scheduler_class = torch.optim.lr_scheduler.LambdaLR

    def _get_lambda(self, value):
        """ This is not an elegant solution, but it gets the jub done i.e. printable lambda functions"""

        class PrintableLambda:
            def __init__(self, value):
                self.value = value

            def __call__(self, epoch):
                return self.value ** epoch

            def __str__(self):
                return f"lambda epoch: {self.value} ** epoch"

        return PrintableLambda(value)

    # Overrides
    def _set_kwargs(self, param_name, value):
        self.kwargs[param_name] = self._get_lambda(value)

    # Overrides
    def _init(self):
        slider = widgets.FloatSlider(value=0.75, max=1,
                                     step=0.01,
                                     description=r"$lr_{k}=\lambda^{epoch}$",
                                     **self.widget_extra
                                     )
        self.kwargs = {"lr_lambda": self._get_lambda(slider.value)}
        self._get_widget_name[slider] = "lr_lambda"


class WidgetMultiplicativeLR(Scheduler):
    def __init__(self, refresh_func):
        """
        DESCRIPTION COPIED FROM PYTORCH: https://pytorch.org/docs/stable/optim.html

        torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda, last_epoch=-1, verbose=False)

        Multiply the learning rate of each parameter group by the factor given in the specified function.
        When last_epoch=-1, sets initial lr as lr.
        """
        super().__init__(name="MultiplicativeLR", refresh_func=refresh_func)
        self.scheduler_class = torch.optim.lr_scheduler.MultiplicativeLR

    def _get_lambda(self, value):
        """ This is not an elegant solution, but it gets the jub done i.e. printable lambda funtions"""

        class PrintableLambda:
            def __init__(self, value):
                self.value = value

            def __call__(self, epoch):
                return self.value

            def __str__(self):
                return f"lambda epoch: {self.value}"

        return PrintableLambda(value)

    # Overrides
    def _set_kwargs(self, param_name, value):
        """This is only here to allow custom parameter updates e.g. lambda epoch: value**epoch in LambdaLR"""
        self.kwargs[param_name] = self._get_lambda(value)

    def _init(self):
        slider = widgets.FloatSlider(value=0.75, max=1,
                                     step=0.01,
                                     description=r"$lr_{k+1}=\lambda \cdot lr_{k}$",
                                     **self.widget_extra
                                     )

        self.kwargs = {"lr_lambda": self._get_lambda(slider.value)}
        self._get_widget_name[slider] = "lr_lambda"


class WidgetStepLR(Scheduler):
    def __init__(self, refresh_func):
        """
        DESCRIPTION COPIED FROM PYTORCH: https://pytorch.org/docs/stable/optim.html

        torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False)

        Decays the learning rate of each parameter group by gamma every step_size epochs.
        Notice that such decay can happen simultaneously with other changes to the learning rate from outside this scheduler.
        When last_epoch=-1, sets initial lr as lr.
        """
        super().__init__(name="StepLR", refresh_func=refresh_func)
        self.scheduler_class = torch.optim.lr_scheduler.StepLR

    def _init(self):
        gamma_slider = widgets.FloatSlider(value=0.1, max=1, step=0.01,
                                           description="gamma",
                                           **self.widget_extra
                                           )

        step_size_slider = widgets.IntSlider(value=1, max=self.epochs, step=1,
                                             description="Step size",
                                             **self.widget_extra
                                             )

        self.kwargs = {"gamma": gamma_slider.value, "step_size": step_size_slider.value}
        self._get_widget_name[gamma_slider] = "gamma"
        self._get_widget_name[step_size_slider] = "step_size"


class WidgetCosineAnnealingLR(Scheduler):
    def __init__(self, refresh_func):
        """
        DESCRIPTION COPIED FROM PYTORCH: https://pytorch.org/docs/stable/optim.html

        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False)

        Set the learning rate of each parameter group using a cosine annealing schedule,
        where eta_{max} is set to the initial lr and T_{cur} is the number of epochs since the last restart in SGDR
        """

        super().__init__(name="CosineAnnealingLR", refresh_func=refresh_func)
        self.scheduler_class = torch.optim.lr_scheduler.CosineAnnealingLR

    def _init(self):
        T_max_slider = widgets.IntSlider(value=1, max=20, step=1,
                                         description="T max",
                                         **self.widget_extra
                                         )

        eta_min_slider = widgets.FloatLogSlider(value=self.lr * 10,
                                                min=-10, max=1, step=0.001,
                                                description="Eta min",
                                                readout_format='.2e',
                                                **self.widget_extra
                                                )

        self.kwargs = {"eta_min": eta_min_slider.value, "T_max": T_max_slider.value}

        self._get_widget_name[T_max_slider] = "T_max"
        self._get_widget_name[eta_min_slider] = "eta_min"


__all__ = ["WidgetLambdaLR", "WidgetMultiplicativeLR", "WidgetStepLR", "WidgetCosineAnnealingLR"]