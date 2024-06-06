import numpy as np
import matplotlib.pyplot as plt

x_com = [10,6,6,0, 10,8,6,0, 10,6,8,0, 10,5,3,0, 10,5,7,0, 10,10,10,10]
key_com = [10,10,10,2, 10,10,10,0, 10,10,10,9, 10,0,0,0, 10,9,10,0, 10,10,10,7]
llm_com = [10,10,10,2, 10,10,10,0, 10,10,10,4, 10,0,0,0, 10,3,10,0, 10,10,10,8]

# Keyword
plt.figure(figsize=(5, 3))
plt.scatter(x_com, key_com, color='blue', label='Data points')

m, b = np.polyfit(x_com, key_com, 1)
plt.plot(x_com, np.array(x_com) * m + b, color='red', label='Trend line')
plt.plot(x_com, x_com, color='green', linestyle='--', label='y = x')

plt.xlabel('Human Grading')
plt.ylabel('Keyword Grading')
plt.title('Computer Science')
plt.legend()

plt.show()

# LLM
plt.figure(figsize=(5, 3))
plt.scatter(x_com, llm_com, color='blue', label='Data points')

m, b = np.polyfit(x_com, llm_com, 1)
plt.plot(x_com, np.array(x_com) * m + b, color='red', label='Trend line')
plt.plot(x_com, x_com, color='green', linestyle='--', label='y = x')

plt.xlabel('Human Grading')
plt.ylabel('LLM Grading')
plt.title('Computer Science')
plt.legend()

plt.show()