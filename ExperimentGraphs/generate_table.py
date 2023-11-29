import matplotlib.pyplot as plt

# Data
tasks = ['imdb, copa', 'imdb, amazon', 'sst2, mrpc', 'yelp review full, qqp', 'mrpc, qqp', 'wic, yelp review full']
prompt_tuning = [52.0, 60.56, 87.3, 90.2, 91.2, 45.32]
progressive_prompts = [49.0, 61.06, 86.3, 89.4, 90.0, 64.32]
selective_prompt_07 = [50.0, 62.76, 85.0, 90.4, 90.6, 64.8]
selective_prompt_085 = [48.0, 62.44, 86.03, 87.6, 89.9, 64.56]

# Bar width and padding
bar_width = 0.11
padding = 0.11
index = range(len(tasks))

# Set a larger figure size
fig, ax = plt.subplots(figsize=(16, 10))  # Adjust the size here

# Choose more vibrant and contrasting colors for the bars
colors = ['#1f78b4', '#e31a1c', '#33a02c', '#ff7f00']

# Plotting the bars with increased spacing
bars1 = ax.bar(index, prompt_tuning, width=bar_width, label='Prompt Tuning', color=colors[0])
bars2 = ax.bar([i + bar_width + padding for i in index], progressive_prompts, width=bar_width, label='Progressive Prompts', color=colors[1])
bars3 = ax.bar([i + 2 * (bar_width + padding) for i in index], selective_prompt_07, width=bar_width, label='Selective Progressive Prompts (0.7)', color=colors[2])
bars4 = ax.bar([i + 3 * (bar_width + padding) for i in index], selective_prompt_085, width=bar_width, label='Selective Progressive Prompts (0.85)', color=colors[3])

# Adding labels
ax.set_xlabel('Task')
ax.set_ylabel('Accuracy (%)')
ax.set_xticks([i + 1.5 * (bar_width + padding) for i in index])
ax.set_xticklabels(tasks)
ax.legend()

# Adding labels on top of the bars with a slightly larger font size
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # Dynamic offset based on bar height
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10.5)  # Set the font size

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)

# Show the plot
plt.show()