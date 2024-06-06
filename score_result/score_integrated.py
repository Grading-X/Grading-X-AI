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

fig, axes = plt.subplots(2, 4, figsize=(15, 5))

# 1
axes[0, 0].scatter(x_com, key_com, color='blue', label='Data points')
m, b = np.polyfit(x_com, key_com, 1)
axes[0, 0].plot(x_com, np.array(x_com) * m + b, color='red', label='Trend line')
axes[0, 0].plot(x_com, x_com, color='green', linestyle='--', label='y = x')
axes[0, 0].set_ylabel('Keyword Grading')
axes[0, 0].set_title('Computer Science')

# 2
axes[1, 0].scatter(x_com, llm_com, color='blue', label='Data points')
m, b = np.polyfit(x_com, llm_com, 1)
axes[1, 0].plot(x_com, np.array(x_com) * m + b, color='red', label='Trend line')
axes[1, 0].plot(x_com, x_com, color='green', linestyle='--', label='y = x')
axes[1, 0].set_xlabel('Human Grading')
axes[1, 0].set_ylabel('LLM Grading')

# 3
axes[0, 1].scatter(x_sc, key_sc, color='blue', label='Data points')
m, b = np.polyfit(x_sc, key_sc, 1)
axes[0, 1].plot(x_sc, np.array(x_sc) * m + b, color='red', label='Trend line')
axes[0, 1].plot(x_sc, x_sc, color='green', linestyle='--', label='y = x')
axes[0, 1].set_title('Science')

# 4
axes[1, 1].scatter(x_sc, llm_sc, color='blue', label='Data points')
m, b = np.polyfit(x_sc, llm_sc, 1)
axes[1, 1].plot(x_sc, np.array(x_sc) * m + b, color='red', label='Trend line')
axes[1, 1].plot(x_sc, x_sc, color='green', linestyle='--', label='y = x')
axes[1, 1].set_xlabel('Human Grading')

# 5
axes[0, 2].scatter(x_hu, key_hu, color='blue', label='Data points')
m, b = np.polyfit(x_hu, key_hu, 1)
axes[0, 2].plot(x_hu, np.array(x_hu) * m + b, color='red', label='Trend line')
axes[0, 2].plot(x_hu, x_hu, color='green', linestyle='--', label='y = x')
axes[0, 2].set_title('Humanities')

# 6
axes[1, 2].scatter(x_hu, llm_hu, color='blue', label='Data points')
m, b = np.polyfit(x_hu, llm_hu, 1)
axes[1, 2].plot(x_hu, np.array(x_hu) * m + b, color='red', label='Trend line')
axes[1, 2].plot(x_hu, x_hu, color='green', linestyle='--', label='y = x')
axes[1, 2].set_xlabel('Human Grading')

# 7
axes[0, 3].scatter(x_hist, key_hist, color='blue', label='Data points')
m, b = np.polyfit(x_hist, key_hist, 1)
axes[0, 3].plot(x_hist, np.array(x_hist) * m + b, color='red', label='Trend line')
axes[0, 3].plot(x_hist, x_hist, color='green', linestyle='--', label='y = x')
axes[0, 3].set_title('History')
axes[0, 3].set_xlim(0, 10)
axes[0, 3].set_ylim(0, 10)

# 8
axes[1, 3].scatter(x_hist, llm_hist, color='blue', label='Data points')
m, b = np.polyfit(x_hist, llm_hist, 1)
axes[1, 3].plot(x_hist, np.array(x_hist) * m + b, color='red', label='Trend line')
axes[1, 3].plot(x_hist, x_hist, color='green', linestyle='--', label='y = x')
axes[1, 3].set_xlabel('Human Grading')
axes[1, 3].set_xlim(0, 10)
axes[1, 3].set_ylim(0, 10)

fig.subplots_adjust(hspace=0.3)
plt.show()