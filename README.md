# What's the point of this repository?
I find the whole process of picking and tuning learning rate schedulers in Pytorch a bit tedious. After thinking about a way of making it more enjoyable, I came to the conclusion that most of the annoyances associated with schedulers, could be alleviated with a very simple GUI that would allow fast prototyping. I feel like a python package which runs directly in Jupyter notebook would be most beneficial, since it would be cross-platform and lightweight almost by default. And let's face it, the vast majority of us use Jupyter notebook anyway, so why not :)

# Demo
Import the class `Plotter` and make an instance of it


```python
from schedulerplotter import Plotter
Plotter();
```

![png](README_IMG/scheduler_image.png)

## Get scheduler
Ones you're happy with the settings just click the `Get scheduler` button and use the printed values to construct the scheduler you have created.

    optimizer = torch.optim.?(?.parameters(), lr=1.000e-04)
    torch.optim.lr_scheduler.CosineAnnealingLR(
    	optimizer,
    	eta_min = 3.200e-03,
    	T_max = 7,
    )

