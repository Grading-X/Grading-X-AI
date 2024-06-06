import numpy as np
import matplotlib.pyplot as plt

x_sc = [10,8,3,0, 10,8,5,5, 10,5,2,0, 10,8,0,0, 10,0,0,0, 10,8,0,0]
key_sc = [10,10,10,2, 10,3,3,10, 10,4,0,0, 10,3,0,0, 10,0,0,0, 10,10,1,0]
llm_sc = [10,10,3,0, 10,2,4,10, 10,3,0,0, 10,3,1,2, 10,3,0,0, 10,10,1,0]

# Keyword
plt.figure(figsize=(5, 3))
plt.scatter(x_sc, key_sc, color='blue', label='Data points')

m, b = np.polyfit(x_sc, key_sc, 1)
plt.plot(x_sc, np.array(x_sc) * m + b, color='red', label='Trend line')
plt.plot(x_sc, x_sc, color='green', linestyle='--', label='y = x')

plt.xlabel('Human Grading')
plt.ylabel('Keyword Grading')
plt.title('Science')
plt.legend()

plt.show()

# LLM
plt.figure(figsize=(5, 3))
plt.scatter(x_sc, llm_sc, color='blue', label='Data points')

m, b = np.polyfit(x_sc, llm_sc, 1)
plt.plot(x_sc, np.array(x_sc) * m + b, color='red', label='Trend line')
plt.plot(x_sc, x_sc, color='green', linestyle='--', label='y = x')

plt.xlabel('Human Grading')
plt.ylabel('LLM Grading')
plt.title('Science')
plt.legend()

plt.show()