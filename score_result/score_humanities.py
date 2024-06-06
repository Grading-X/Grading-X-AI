import numpy as np
import matplotlib.pyplot as plt

x_hu = [10,10,5,0, 10,5,0,0, 10,7,10,0, 10,8,4,0, 0,8,10,0, 10,5,8,2]
key_hu = [10,10,10,7, 9,1,0,0, 10,10,10,0, 10,9,2,0, 5,8,10,2, 10,10,10,0]
llm_hu = [10,10,10,6, 3,3,0,0, 10,10,10,0, 10,4,3,0, 2,6,10,2, 10,10,10,2]

# Keyword
plt.figure(figsize=(5, 3))
plt.scatter(x_hu, key_hu, color='blue', label='Data points')

m, b = np.polyfit(x_hu, key_hu, 1)
plt.plot(x_hu, np.array(x_hu) * m + b, color='red', label='Trend line')
plt.plot(x_hu, x_hu, color='green', linestyle='--', label='y = x')

plt.xlabel('Human Grading')
plt.ylabel('Keyword Grading')
plt.title('Humanities')
plt.legend()

plt.show()

# LLM
plt.figure(figsize=(5, 3))
plt.scatter(x_hu, llm_hu, color='blue', label='Data points')

m, b = np.polyfit(x_hu, llm_hu, 1)
plt.plot(x_hu, np.array(x_hu) * m + b, color='red', label='Trend line')
plt.plot(x_hu, x_hu, color='green', linestyle='--', label='y = x')

plt.xlabel('Human Grading')
plt.ylabel('LLM Grading')
plt.title('Humanities')
plt.legend()

plt.show()