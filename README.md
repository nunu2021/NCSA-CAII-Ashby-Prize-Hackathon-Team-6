# NCSA-CAII-Ashby-Prize-Hackathon-Team-6
Challenge: The objective of this project is to create a machine learning model trained on accurate WRF-PartMC data that predicts climate-relevant aerosol properties from only the features that current GCMs can output.
[Problem Statement](https://ai.ncsa.illinois.edu/wp-content/uploads/2022/04/instructions_04052022-min-1.pdf)

## Analyze Data
### Normalize Data
To start solving this question, we first needed to normalize our input and output data points to reduce floating point error and to clean up major discrepancies between the norms of each input params.

- Method 1: Normalizing Using Mean and STD
    1. If value is zero, we replace it with a minimum non-zero value (so we can log).
    2. When calculating mean and standard deviation, we use the log(value) to ensure floating point precision .
    3. All variables except 'z' are converted to log space.
    4. Global mean is subtracted and normalized by standard deviation for each variable.
    5. Added cos(Time) as additional feature.
    6. Used for MLP and TabNet.
   
    
- Method 2: Normalizing for Each height
    1. Variables are converted to log space similar to method 1.
    2. Mean and standard deviation are calculated at each height instead of global.
    3. Dataset used for final TabNet model.

    
### Determine Strong correlation Inputs
Next, we visualized the correlation of the input variables with the output variables at a single timestamp, then over the course of the time range given.

- 1 Timestamp Correlation Plot:
![1 Timestamp Correlation](abs_correlation_t0_v2.png)
- Correlation Plot over time:
![Correlation Over Time](t_loop_all_z.gif)


## Develop Models
### MLP - Multilayer Perceptrons


### TabNet
![Tabnet1](tabnet.png)
![Tabnet2](tabnet2.png)
![Tabnet3](tabnet3.png)

### Gradient Boosting Tree

## Get Predictions
![MLE / MSE](mle.png)
(z2cnn001.gif)

Read Our Full presentation [HERE](https://docs.google.com/presentation/d/14Tt9RcEZN6glRenaNbsKrEFVbnRaBpxLx9ZVkvc7fX0/edit?usp=sharing)

# Team:
Sunny Tang, Kedar Phadke, Chu-Chun Chen, Labdhi Jain
