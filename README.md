# \[shashi buhle\]

**Group members:**
- Francesco Arous   (zxs354)
- Lukas Heidtmann   (prw583)
- Albert Sniec      (zsr832)  

**This repository contains the following files:**
1. Inaugural project.
2. Data project. 
3. Model project.
4. The take-home exam from 25.06-27.06


The **Inaugural project** basically just answers the problem sets given to us without leaving to much room for interpretation. Please note that we are making use of the HouseholdSpecialisationModel.py-file which contains some of our code. We will hand that in as well. 
In this project we examine how both males and females chose their time whether it be working from home or working in the market, and how this varies based on the relative wage ratio. 
Notably, we examine the effect of varying the household production parameters alpha and sigma and evaluating the impact they have on the ratio of hours worked at home by males and females and the wage ratio. We create plots to illustrate these effects in both a discrete and continous model setting, with varying females wages and a constant male wage variable.
We likewise run a regression analysis and complete an optimization problem to find the optimal values of alpha and sigma so our data fits the model.
Finally, we suggest an extension of the model in which we hold the household production parameter alpha constant and examine the effect this has on the vale of sigma, and whether this helps match the data.


The **Data project** is titled _"Impact of GDP per capita on employment in worldwide agriculture over the past 30 years"_ where we examine the impact of GDP per capita on the % of people employed in agriculture for all countries in the world. We then conduct further analysis by calculating the arable land per person employed in agriculture for the top ten largest countries in the world (by total arable land) and compare how this has evolved as the GDP per capita has increased/decreased over time providing interpretation as to why the plotted relationships do exist. All of our datatsets are read in through the World Bank API and adapted into our .ipynb-file. 


Our **Model project**, titled _"Options pricing according to the Black-Scholes(-Merton)-Model"_ deals with the pricing of financial options according to the mathematical model proposed by Fischer Black and Myron Samuel Scholes. The model was regarded as groundbreaking when it was first published in 1973 and is (in somewhat modified form) still widely used up to this day by options market participants. Today some of the model's assumptions have been relaxed and generalized in many directions, leading to a plethora of models that are currently used in derivative pricing and risk management.
In general, the model in its basic form arrives at a pricing formula (Black–Scholes formula) for financial derivatives of some non-dividend-paying underlying asset and thus gives a theoretical estimate of the price of European-style options. According to the model a financial option can be assigned a price given the risk of the underlying security and its expected return. 
We first specify the parameters of the model and show you how the authors arrived at the pricing formulas, bevor implementing them into python code. An interactive figure allows the reader to change all parameters of the model and see the resulting changes in a financial options valuation for different stock prices of a hypothetical underlying financial asset. We then import some real-world financial data for several well-known stocks and apply the model accordingly. As an extension of the model we added in dividend payments, which are not part of the standard model.


The **Take-home exam** features our solutions to the provided three problem sets. All the code is contained in the exams ipynb-file. 


Further requirements for python packages needed can be found in the individual ReadMe-files or in the ipynb-files of the projects themself. **Before you run our projects please clear all outputs, restart your kernel and follow the instructions for pip-installs and imports at the beginning of the file!**


We hope you find our solutions interesting and would really like to get some constructive feedback on what to improve in the furture!














