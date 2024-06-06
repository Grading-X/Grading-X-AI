import numpy as np
import matplotlib.pyplot as plt

x_com = [10,6,6,0, 10,8,6,0, 10,6,8,0, 10,5,3,0, 10,5,7,0, 10,10,10,10]
key_com = [10,10,10,2, 10,10,10,0, 10,10,10,9, 10,0,0,0, 10,9,10,0, 10,10,10,7]
llm_com = [10,10,10,2, 10,10,10,0, 10,10,10,4, 10,0,0,0, 10,3,10,0, 10,10,10,8]

x_sc = [10,8,3,0, 10,8,5,5, 10,5,2,0, 10,8,0,0, 10,0,0,0, 10,8,0,0]
key_sc = [10,10,10,2, 10,3,3,10, 10,4,0,0, 10,3,0,0, 10,0,0,0, 10,10,1,0]
llm_sc = [10,10,3,0, 10,2,4,10, 10,3,0,0, 10,3,1,2, 10,3,0,0, 10,10,1,0]

x_hu = [10,10,5,0, 10,5,0,0, 10,7,10,0, 10,8,4,0, 0,8,10,0, 10,5,8,2]
key_hu = [10,10,10,7, 9,1,0,0, 10,10,10,0, 10,9,2,0, 5,8,10,2, 10,10,10,0]
llm_hu = [10,10,10,6, 3,3,0,0, 10,10,10,0, 10,4,3,0, 2,6,10,2, 10,10,10,2]

x_hist = [8,7,7,8, 8,7,9,9, 8,8,10,10, 7,8,8,10, 7,7,8,9, 7,7,8,9]
key_hist = [10,9,10,4, 10,10,10,10, 10,10,10,10, 10,10,10,10, 10,10,10,10, 10,10,10,10]
llm_hist = [10,5,10,4, 10,10,10,10, 10,10,10,10, 10,10,10,10, 10,10,10,10, 10,10,10,10]

x_total = x_com + x_sc + x_hu + x_hist
key_total = key_com + key_sc + key_hu + key_hist
llm_total = llm_com + llm_sc + llm_hu + llm_hist

# Keyword
plt.figure(figsize=(5, 3))
plt.scatter(x_total, key_total, color='blue', label='Data points')

m, b = np.polyfit(x_total, key_total, 1)
plt.plot(x_total, np.array(x_total) * m + b, color='red', label='Trend line')
plt.plot(x_total, x_total, color='green', linestyle='--', label='y = x')

plt.xlabel('Human Grading')
plt.ylabel('Keyword Grading')
plt.title('Total')
plt.legend()

plt.show()

# LLM
plt.figure(figsize=(5, 3))
plt.scatter(x_total, llm_total, color='blue', label='Data points')

m, b = np.polyfit(x_total, llm_total, 1)
plt.plot(x_total, np.array(x_total) * m + b, color='red', label='Trend line')
plt.plot(x_total, x_total, color='green', linestyle='--', label='y = x')

plt.xlabel('Human Grading')
plt.ylabel('LLM Grading')
plt.title('Total')
plt.legend()

plt.show()
