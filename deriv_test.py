
def function(x):
    return 4*(3*x+5)**2

epsilon = 1e-7
a=2.6666667
print(f"Derivative is {round((function(a+epsilon)-function(a))/epsilon, 3)}, estimate is {24*(3*a+5)}")
