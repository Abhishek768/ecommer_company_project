import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn import metrics

# fetching data from ecommerce csv file
ecommerce_data = pd.read_csv('Ecommerce Customers')

# checking the head of ecommerce_data
print(ecommerce_data.head())

# Let's expolre data with the help of visualization 
snp.set_palette("GnBu_d")
snp.set_style('whitegrid')

snp.pairplot(data = ecommerce_data)

# As we can see that Length of Membership looks to be the most
# Correlated feature with Yearly Amount Spent
# so will create a linear regression plot
snp.lmplot(x = 'Length of Membership', y = 'Yearly Amount Spent',data = ecommerce_data)

# Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets. 
X = ecommerce_data[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y = ecommerce_data['Yearly Amount Spent']

# train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Now its time to train our model on our training data
lm = LinearRegression()
lm.fit(X_train, y_train)

# Now that we have fit our model, let's evaluate its performance by predicting off the test values!
prediction = lm.predict(X_test)

# will show the scatter plot predicted values versus true value
snp.scatterplot(y_test, prediction)
plt.xlabel('True Test Values')
plt.ylabel('Predicted Values')

# Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2)
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

plt.show()
