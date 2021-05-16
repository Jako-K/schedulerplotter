from IPython.display import clear_output
import ipywidgets as widgets
from ._schedulers import *

class Plotter:
    def __init__(self):
        self.active_scheduler = None
        self.schedulers = {"LambdaLR": WidgetLambdaLR,
                           "MultiplicativeLR": WidgetMultiplicativeLR,
                           "StepLR": WidgetStepLR,
                           "CosineAnnealingLR": WidgetCosineAnnealingLR,
                           }
        self.terminate = False
        self.semilogy = False
        self._refresh()

    def _get_exit_bottom(self):
        def on_buttom_press(_):
            clear_output()
            self.active_scheduler.print_scheduler()

        buttom = widgets.Button(description='Get scheduler')
        buttom.on_click(on_buttom_press)
        return buttom

    def _get_scheduler_dropdown(self):
        def scheduler_change(change):
            self.active_scheduler = self.schedulers[change.new](self._refresh)
            self._refresh()

        init_value = None if self.active_scheduler is None else self.active_scheduler.name
        dropdown = widgets.Dropdown(description="Scheduler",
                                    value=init_value,
                                    options=list(self.schedulers.keys())
                                    )
        dropdown.observe(scheduler_change, names="value")
        return dropdown

    def _get_log_checkbox(self):
        def checkbox_change(change):
            self.semilogy = change.new
            self._refresh()

        check_box = widgets.Checkbox(
            value=self.semilogy,
            description='Logarithmic',
            disabled=False,
            indent=False
        )

        check_box.observe(checkbox_change, names="value")
        return check_box

    def _refresh(self):
        clear_output()
        exit_buttom = self._get_exit_bottom()
        scheduler_dropdown = self._get_scheduler_dropdown()
        log_check_box = self._get_log_checkbox()

        display(widgets.HBox(children=[scheduler_dropdown, log_check_box, exit_buttom]))
        print("." * 118)
        if self.active_scheduler:
            self.active_scheduler.update_display(self.semilogy)
