import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

cs = {
    'general': {
        'grad': 150.10,
        'undergrad': 79.67
    },
    'salary': {
        'grad': 128.87,
        'undergrad': 91.97
    }
}

ds = {
    'general': {
        'grad': 153.01,
        'undergrad': 77.98
    },
    'salary': {
        'grad': 147.96,
        'undergrad': 80.92
    }
}

it = {
    'general': {
        'grad': 153.26,
        'undergrad': 77.84
    },
    'salary': {
        'grad': 128.87,
        'undergrad': 91.97
    }
}

pm = {
    'general': {
        'grad': 152.44,
        'undergrad': 78.31
    },
    'salary': {
        'grad': 146.96,
        'undergrad': 81.50
    }
}

swe = {
    'general': {
        'grad': 155.23,
        'undergrad': 76.70
    },
    'salary': {
        'grad': 138.52,
        'undergrad': 86.38
    }
}

fields = ['CYS', 'DS', 'SWE', 'TPM', 'IT']
data_dicts = [cs, ds, swe, pm, it]

plt.figure(figsize=(8, 6))

# Set bar width and positions with custom spacing
bar_width = 0.11
spacing_factor = 0.5  # Controls gap between field groups
r1 = np.arange(len(fields)) * spacing_factor
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

# Color palette with light/dark pairs - reordered to match new sequence
colors = ['#6BAED6', '#FDD0A2', '#2171B5', '#D94801']

# Starting point for bars
start_point = 210

# Create bars with border - reordered sequence
for i, (field, data) in enumerate(zip(fields, data_dicts)):
    # Calculate heights (difference from start_point)
    grad_general = start_point - data['general']['grad']
    undergrad_general = start_point - data['general']['undergrad']
    grad_salary = start_point - data['salary']['grad']
    undergrad_salary = start_point - data['salary']['undergrad']
    
    plt.bar(r1[i], grad_general, bottom=start_point - grad_general, width=bar_width, 
            label='Grad General' if i == 0 else "", color=colors[0], edgecolor='black', linewidth=1.1)
    plt.bar(r2[i], undergrad_general, bottom=start_point - undergrad_general, width=bar_width, 
            label='Undergrad General' if i == 0 else "", color=colors[1], edgecolor='black', linewidth=1.1)
    plt.bar(r3[i], grad_salary, bottom=start_point - grad_salary, width=bar_width, 
            label='Grad Top Salary' if i == 0 else "", color=colors[2], edgecolor='black', linewidth=1.1)
    plt.bar(r4[i], undergrad_salary, bottom=start_point - undergrad_salary, width=bar_width, 
            label='Undergrad Top Salary' if i == 0 else "", color=colors[3], edgecolor='black', linewidth=1.1)

# Add border around plot
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['left'].set_linewidth(0.5)
plt.gca().spines['right'].set_linewidth(0.5)
plt.gca().spines['top'].set_linewidth(0.5)
plt.gca().spines['bottom'].set_linewidth(0.5)

# Update x-ticks to align with the center of each group of bars
group_centers = r1 + (1.5 * bar_width)  # Calculate center of each group
plt.xticks(group_centers, fields, fontsize=14)
plt.yticks(fontsize=14)

# Adjust the plot limits to show all bars with some padding
plt.xlim(min(r1) - bar_width, max(r4) + bar_width)
plt.ylim(210, 25)

# Add legend with slightly adjusted position
plt.legend(loc='upper right', frameon=True, edgecolor='black', fancybox=False, fontsize=15.5)

plt.xlabel('Field', fontsize=15.5)
plt.ylabel("Course Rank", fontsize=15.5)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show plot
plt.show()