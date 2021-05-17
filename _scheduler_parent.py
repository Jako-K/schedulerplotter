import matplotlib.pyplot as plt
import seaborn; seaborn.set_style("darkgrid")

import ipywidgets as widgets

import torch
import torch.nn as nn
import warnings

class Scheduler:
    def __init__(self, name, refresh_func):
        self.name = name
        self.refresh = refresh_func
        self.lr = 1e-4
        self.epochs = 10
        self.scheduler_class = None
        self.kwargs = None
        self._to_display = []

        self.description_width = 125
        self.layout_width = 500
        self.widget_extra = dict(continuous_update=False,
                                 layout={'width': f'{self.layout_width}px'},
                                 style={'description_width': f'{self.description_width}px', "color":"lightblue"}
                                 )

        self._get_widget_name = self._init_get_widget_name()
        self._init()

    def _init(self):
        raise NotImplementedError

    def _set_kwargs(self, param_name, value):
        """This is only here to allow custom parameter update implementation e.g. lambda epoch: value**epoch in LambdaLR"""
        self.kwargs[param_name] = value

    def _init_get_widget_name(self):
        slider = widgets.FloatLogSlider(value=self.lr,
                                        min=-10, max=1, step=0.001,
                                        description="Learning rate",
                                        readout_format='.2e',
                                        **self.widget_extra
                                        )

        slider2 = widgets.IntSlider(value=self.epochs,
                                    min=1, max=100, step=1,
                                    description="epochs",
                                    **self.widget_extra
                                    )
        return {slider: "learning_rate", slider2: "epochs"}

    def _get_optimizer(self):
        dummy_model = nn.Linear(1, 1)
        dummy_optimizer = torch.optim.SGD(dummy_model.parameters(), self.lr)
        return dummy_optimizer

    def _plot(self, semilogy):
        warnings.filterwarnings("ignore", category=UserWarning)

        scheduler = self.scheduler_class(self._get_optimizer(), **self.kwargs)
        learning_rates = []

        for epoch in range(1, self.epochs):
            learning_rates.append(scheduler.optimizer.param_groups[0]["lr"])
            scheduler.step()

        fig, ax = plt.subplots(figsize=(15, 5))
        if semilogy:
            ax.semilogy(learning_rates, '-o')
        else:
            ax.plot(learning_rates, '-o')
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        ax.set_title(f"{self.name} - Learning rate over {self.epochs} epochs")
        ax.set_ylabel("Learning rate")
        ax.set_xlabel("Epoch")

        warnings.filterwarnings("default", category=UserWarning)  # reset to default warnings

    def print_scheduler(self):
        print(f"optimizer = torch.optim.?(?.parameters(), lr={format(self.lr, '.3e')})")
        print(f"torch.optim.lr_scheduler.{self.name}(\n\toptimizer,")
        for k, v in self.kwargs.items():
            if type(v) == float:
                v = format(v, ".3e")
            print(f"\t{k} = {v},")
        print(")")

    def _signal(self, change):
        widget_changed = self._get_widget_name[change.owner]
        value_changed = change.new

        if widget_changed == "learning_rate":
            self.lr = value_changed
        elif widget_changed == "epochs":
            self.epochs = value_changed
        else:
            self._set_kwargs(widget_changed, value_changed)

        self.refresh()

    def update_display(self, semilogy):
        widgets_to_display = []
        for widget in list(self._get_widget_name.keys()):
            widgets_to_display.append(widget)
            widget.observe(self._signal, names="value")

        display(widgets.VBox(children=widgets_to_display))

        self._plot(semilogy)