# Model analysis project from team shashi-buhle

Our project, titled **Options pricing according to the Black-Scholes(-Merton)-Model** deals with the pricing of financial options according to the mathematical model proposed by Fischer Black and Myron Samuel Scholes. Robert C. Merton had also helped to work on the problem but published a paper on his own, which is why the model is commonly only referred to as the Black-Scholes-model. The model was regarded as groundbreaking when it was first published in 1973 and is (in somewhat modified form) still widely used up to this day by options market participants. Today some of the model's assumptions have been relaxed and generalized in many directions, leading to a plethora of models that are currently used in derivative pricing and risk management.
   
In general, the model in its basic form arrives at a pricing formula (Black–Scholes formula) for financial derivatives of some non-dividend-paying underlying asset and thus gives a theoretical estimate of the price of European-style options. According to the model a financial option can be assigned a price given the risk of the underlying security and its expected return. 

We first specify the parameters of the model and show you how the authors arrived at the pricing formulas, bevor implementing them into python code. An interactive figure allows the reader to change all parameters of the model and see the resulting changes in a financial options valuation for different stock prices of a hypothetical underlying financial asset. We then import some real-world financial data for several well-known stocks and apply the model accordingly. As an extension of the model we added in dividend payments, which are not part of the standard model.  

The **results** of the project can be seen from running [TheBlack-ScholesModel.ipynb](TheBlack-ScholesModel.ipynb).

**Dependencies:** Apart from a standard Anaconda Python 3 installation, the project requires some further packages. We do list the pip-installers in the beginning of out notebook so it should be easy to quickly install them by removing the "#" infront of said code. 
