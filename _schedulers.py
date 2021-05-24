import seaborn; seaborn.set_style("darkgrid")
from ._scheduler_parent import Scheduler
import ipywidgets as widgets
import torch


# TODO: Find a solution for ReduceLROnPlateau

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

        step_size_slider = widgets.IntSlider(value=1, max=self.steps, step=1,
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


class WidgetCosineAnnealingWarmRestarts(Scheduler):
    def __init__(self, refresh_func):
        """
        DESCRIPTION COPIED FROM PYTORCH: https://pytorch.org/docs/stable/optim.html

        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False)

        Set the learning rate of each parameter group using a cosine annealing schedule,
        where eta_{max} is set to the initial lr and T_{cur} is the number of epochs since the last restart in SGDR
        """

        super().__init__(name="CosineAnnealingWarmRestarts", refresh_func=refresh_func)
        self.scheduler_class = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts

    def _init(self):
        T_0_slider = widgets.IntSlider(value=5, min=1, max=100, step=1,
                                       description="T_0",
                                       **self.widget_extra
                                       )

        eta_min_slider = widgets.FloatLogSlider(value=self.lr * 10,
                                                min=-10, max=1, step=0.001,
                                                description="Eta min",
                                                readout_format='.2e',
                                                **self.widget_extra
                                                )

        T_mult_slider = widgets.IntSlider(value=1, min=0, step=1,
                                       description="T_mult",
                                       **self.widget_extra
                                       )

        self.kwargs = {"eta_min": eta_min_slider.value, 
                       "T_0": T_0_slider.value, 
                       "T_mult": T_mult_slider.value
                       }

        self._get_widget_name[T_0_slider] = "T_0"
        self._get_widget_name[eta_min_slider] = "eta_min"
        self._get_widget_name[T_mult_slider] = "T_mult"



class WidgetMultiStepLR(Scheduler):
    def __init__(self, refresh_func):
        """
        DESCRIPTION COPIED FROM PYTORCH: https://pytorch.org/docs/stable/optim.html

        torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False)

        Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones. 
        Notice that such decay can happen simultaneously with other changes to the learning rate from outside this scheduler. 
        When last_epoch=-1, sets initial lr as lr.
        
        """

        super().__init__(name="MultiStepLR", refresh_func=refresh_func)
        self.scheduler_class = torch.optim.lr_scheduler.MultiStepLR

    def _init(self):
        gamma_slider = widgets.FloatSlider(value=0.1, min=0, max=1, step=0.001,
                                           description="gamma", **self.widget_extra
                                           )
        milestones_text = widgets.Textarea(value="5,10", description = "Milestones (sep ,)",
                                           layout={'width': f'{self.layout_width-81}px', "height": "29px"},
                                           style={'description_width': f'{self.description_width}px'}
                                           )

        submit_button = widgets.Button(description="Submit milestones")
        submit_button.on_click(self._handle_milestones_submit)
        milestones_combined = widgets.HBox( children = [milestones_text, submit_button] )

        self.kwargs = {"gamma": gamma_slider.value, "milestones": milestones_text.value}

        self._get_widget_name[gamma_slider] = "gamma"
        self._get_widget_name[milestones_combined] = "milestones"

    def _handle_milestones_submit(self, e):
        # This is a hack, no two ways about it. But it seemed stupid to change the
        # general structure for this scheduler alone.
        milestones_text = list(self._get_widget_name.keys())[-1].children[0].value
        self.kwargs["milestones"] = milestones_text.replace(" ", "")
        self.kwargs["milestones"] = [int(number) for number in self.kwargs["milestones"].split(",")]
        self.refresh()


class WidgetExponentialLR(Scheduler):
    def __init__(self, refresh_func):
        """
        DESCRIPTION COPIED FROM PYTORCH: https://pytorch.org/docs/stable/optim.html

        torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1, verbose=False)

        Decays the learning rate of each parameter group by gamma every epoch. When last_epoch=-1, sets initial lr as lr.
        """

        super().__init__(name="ExponentialLR", refresh_func=refresh_func)
        self.scheduler_class = torch.optim.lr_scheduler.ExponentialLR

    def _init(self):
        gamma_slider = widgets.FloatSlider(value=0.1, min=0, max=1, step=0.001,
                                           description="gamma", **self.widget_extra
                                           )

        self.kwargs = {"gamma": gamma_slider.value}
        self._get_widget_name[gamma_slider] = "gamma"


class WidgetCyclicLR(Scheduler):
    def __init__(self, refresh_func):
        """
        DESCRIPTION COPIED FROM PYTORCH: https://pytorch.org/docs/stable/optim.html

        torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None,
                                          mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle',
                                          cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1,
                                          verbose=False)

        Sets the learning rate of each parameter group according to cyclical learning rate policy (CLR).
        The policy cycles the learning rate between two boundaries with a constant frequency,
        as detailed in the paper Cyclical Learning Rates for Training Neural Networks.
        The distance between the two boundaries can be scaled on a per-iteration or per-cycle basis.

        Cyclical learning rate policy changes the learning rate after every batch.
        ´step´ should be called after a batch has been used for training.
        """

        super().__init__(name="CyclicLR", refresh_func=refresh_func)
        self.scheduler_class = torch.optim.lr_scheduler.CyclicLR


    def _init(self):
        base_lr_slider = widgets.FloatLogSlider(value=self.lr, min=-10, max=1, step=0.001, description="base_lr", readout_format='.2e', **self.widget_extra)
        max_lr_slider = widgets.FloatLogSlider(value=self.lr*10, min=-10, max=1, step=0.001, description="max_lr", readout_format='.2e', **self.widget_extra)
        step_size_up_slider = widgets.IntSlider(value=1, min=1, max=2500, step=1, description="step_size_up", **self.widget_extra )
        step_size_down_slider = widgets.IntSlider(value=1, min=1, max=2500, step=1, description="step_size_down", **self.widget_extra )
        mode_dropdown = widgets.Dropdown(description="Mode",value="triangular",options=["triangular", "triangular2", "exp_range"],layout={'width': f'{self.layout_width-80}px'},style = self.widget_extra["style"])
        gamma_slider = widgets.FloatSlider(value=0.95, min=0, max=1, step=0.001, description="gamma", **self.widget_extra)
        cycle_momentum_slider = widgets.Dropdown(description="cycle_momentum", value=True, options=[True, False], layout={'width': f'{self.layout_width-80}px'}, style = self.widget_extra["style"])
        base_momentum_slider = widgets.FloatSlider(value=0.8, min=0, max=2, step=0.001, description="base_momentum", **self.widget_extra)
        max_momentum_slider = widgets.FloatSlider(value=0.9, min=0, max=2, step=0.001, description="max_momentum", **self.widget_extra)

        sliders = [base_lr_slider, max_lr_slider, step_size_up_slider, step_size_down_slider, gamma_slider,
                   mode_dropdown,cycle_momentum_slider, base_momentum_slider, max_momentum_slider]
        names = ["base_lr", "max_lr", "step_size_up", "step_size_down", "gamma", "mode", "cycle_momentum",
                 "base_momentum", "max_momentum"]

        self.kwargs = {}
        for name, slider in {names[i]:sliders[i] for i in range(len(sliders))}.items():
            self.kwargs[name] = slider.value
            self._get_widget_name[slider] = name


class WidgetOneCycleLR(Scheduler):
    def __init__(self, refresh_func):
        """
        DESCRIPTION COPIED FROM PYTORCH: https://pytorch.org/docs/stable/optim.html

        torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, total_steps=None, epochs=None, steps_per_epoch=None,
                                            pct_start=0.3, anneal_strategy='cos', cycle_momentum=True,
                                            base_momentum=0.85, max_momentum=0.95,
                                            div_factor=25.0, final_div_factor=10000.0,
                                            three_phase=False, last_epoch=-1, verbose=False)

        Sets the learning rate of each parameter group according to the 1cycle learning rate policy.
        The 1cycle policy anneals the learning rate from an initial learning rate to some maximum learning rate and
        then from that maximum learning rate to some minimum learning rate much lower than the initial learning rate.
        This policy was initially described in the paper Super-Convergence:
        Very Fast Training of Neural Networks Using Large Learning Rates.

        The 1cycle learning rate policy changes the learning rate after every batch.
        step should be called after a batch has been used for training.

        This scheduler is not chainable.
        Note also that the total number of steps in the cycle can be determined in one of two ways
        (listed in order of precedence):
        1.) A value for total_steps is explicitly provided.
        2.) A number of epochs (epochs) and a number of steps per epoch (steps_per_epoch) are provided.
            In this case, the number of total steps is inferred by total_steps = epochs * steps_per_epoch

        You must either provide a value for total_steps or provide a value for both epochs and steps_per_epoch.
        The default behaviour of this scheduler follows the fastai implementation of 1cycle, which claims that
        “unpublished work has shown even better results by using only two phases”.
        To mimic the behaviour of the original paper instead, set three_phase=True.
        """

        super().__init__(name="OneCycleLR", refresh_func=refresh_func)
        self.scheduler_class = torch.optim.lr_scheduler.OneCycleLR



    def _init(self):
        max_lr_slider = widgets.FloatLogSlider(value=self.lr*10, min=-10, max=1, step=0.001, description="max_lr", readout_format='.2e', **self.widget_extra)
        pct_start_slider = widgets.FloatSlider(value=0.3, min=0, max=1, step=0.001, description="pct_start", **self.widget_extra)
        anneal_strategy_dropdown = widgets.Dropdown(description="anneal_strategy ", value="cos",options=["cos", "linear"], layout={'width': f'{self.layout_width-80}px'},style = self.widget_extra["style"])
        cycle_momentum_dropdown = widgets.Dropdown(description="cycle_momentum", value=True, options=[True, False], layout={'width': f'{self.layout_width-80}px'}, style = self.widget_extra["style"])
        base_momentum_slider = widgets.FloatSlider(value=0.85, min=0, max=2, step=0.001, description="base_momentum", **self.widget_extra)
        max_momentum_slider = widgets.FloatSlider(value=0.95, min=0, max=2, step=0.001, description="max_momentum", **self.widget_extra)
        div_factor_slider = widgets.FloatSlider(value=25.0, min=0, max=1000, step=1, description="div_factor", **self.widget_extra)
        final_div_factor_slider = widgets.FloatSlider(value=10000, min=-10, max=20000, step=100, description="final_div_factor ", readout_format='.2e', **self.widget_extra)
        three_phase_dropdown = widgets.Dropdown(description="three_phase", value=False, options=[True, False], layout={'width': f'{self.layout_width-80}px'}, style = self.widget_extra["style"])


        sliders = [max_lr_slider, pct_start_slider, anneal_strategy_dropdown, cycle_momentum_dropdown,
                   base_momentum_slider, max_momentum_slider, div_factor_slider, final_div_factor_slider, three_phase_dropdown]
        names = ["max_lr", "pct_start", "anneal_strategy", "cycle_momentum", "base_momentum", "max_momentum",
                 "div_factor", "final_div_factor", "three_phase"]

        self.kwargs = {}
        for name, slider in {names[i]:sliders[i] for i in range(len(sliders))}.items():
            self.kwargs[name] = slider.value
            self._get_widget_name[slider] = name

        self.kwargs["total_steps"] = self.steps



__all__ = ["WidgetLambdaLR",
           "WidgetMultiplicativeLR", 
           "WidgetStepLR", 
           "WidgetCosineAnnealingLR", 
           "WidgetCosineAnnealingWarmRestarts",
           "WidgetMultiStepLR",
           "WidgetExponentialLR",
           "WidgetCyclicLR",
           "WidgetOneCycleLR"
           ]