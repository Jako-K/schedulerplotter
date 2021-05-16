# Why build this repo?
I find the whole process of picking and tuning learning rate schedulers tedious and somewhat frustrating. I came to the conclusion that this was primarily due to the lack of interactive options. I feel like a python package which runs directly in jupyter notebook would be most beneficial, and since i couldn't find any proper solutions online I decided to make my own.  

# Demo
Import the class `PLotter` and make an instance of it


```python
from schedulerplotter import Plotter
Plotter();
```


    HBox(children=(Dropdown(description='Scheduler', index=3, options=('LambdaLR', 'MultiplicativeLR', 'StepLR', '…


    ......................................................................................................................
    


    VBox(children=(FloatLogSlider(value=0.0001, continuous_update=False, description='Learning rate', layout=Layou…



![png](output_2_3.png)


## Get scheduler
Ones you're happy with the settings just click the `Get scheduler` buttom and use the printed values to construct the scheduler you have created.


```python
Plotter();
```

    optimizer = torch.optim.?(?.parameters(), lr=1.000e-04)
    torch.optim.lr_scheduler.CosineAnnealingLR(
    	optimizer,
    	eta_min = 1.000e-03,
    	T_max = 7,
    )
    
