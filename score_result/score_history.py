import numpy as np
import matplotlib.pyplot as plt

x_hist = [8,7,7,8, 8,7,9,9, 8,8,10,10, 7,8,8,10, 7,7,8,9, 7,7,8,9]
key_hist = [10,9,10,4, 10,10,10,10, 10,10,10,10, 10,10,10,10, 10,10,10,10, 10,10,10,10]
llm_hist = [10,5,10,4, 10,10,10,10, 10,10,10,10, 10,10,10,10, 10,10,10,10, 10,10,10,10]

# Keyword
plt.figure(figsize=(5, 3))
plt.scatter(x_hist, key_hist, color='blue', label='Data points')

m, b = np.polyfit(x_hist, key_hist, 1)
plt.plot(x_hist, np.array(x_hist) * m + b, color='red', label='Trend line')
plt.plot(x_hist, x_hist, color='green', linestyle='--', label='y = x')

plt.xlabel('Human Grading')
plt.ylabel('Keyword Grading')
plt.title('History')
plt.legend()

plt.show()

# LLM
plt.figure(figsize=(5, 3))
plt.scatter(x_hist, llm_hist, color='blue', label='Data points')

m, b = np.polyfit(x_hist, llm_hist, 1)
plt.plot(x_hist, np.array(x_hist) * m + b, color='red', label='Trend line')
plt.plot(x_hist, x_hist, color='green', linestyle='--', label='y = x')

plt.xlabel('Human Grading')
plt.ylabel('LLM Grading')
plt.title('History')
plt.legend()

plt.show()

