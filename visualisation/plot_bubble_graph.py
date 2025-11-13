import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D

data = {
    'Model': [
        'convnextv2-huge-22k-384',
        'convnextv2-base-22k-384',
        'resnet-152',
        'resnet-50',
        'resnet-18',
        'convnextv2-tiny-22k-384'
    ],
    'Model_Size_Str': ['659M', '89M', '60M', '25.6M', '11.7M', '28M'],
    'Acc_V2_SelfSup': [99.3894993894993, 98.2905982905982, 98.2905982905982, 96.9474969474969, 92.4298, 99.1452991452991],
}

df = pd.DataFrame(data)

# Data preprocessing
# 1. Clean 'Model Size': remove 'M' and convert to float
df['Model_Size_M'] = df['Model_Size_Str'].str.replace('M', '').astype(float)

# 2. LOG TRANSFORM X-AXIS
df['Log_Model_Size'] = np.log10(df['Model_Size_M'])

# 3. Extract Model Family (for color/marker)
df['Model_Family'] = df['Model'].apply(lambda x: 'ConvNeXt' if 'convnext' in x.lower() else 'ResNet')

# 4. Simplify Model Name for labels (still useful for annotation)
def simplify_model_name(name):
    if 'convnext' in name.lower():
        return name.split('-')[1].capitalize()
    elif 'resnet' in name.lower():
        return name.split('-')[0].capitalize() + '-' + name.split('-')[1]
    return name

df['Acc_Proportion'] = df['Acc_V2_SelfSup'] / 100.0
df['Exp_Acc_V2_SelfSup_Scaled'] = np.exp(df['Acc_Proportion'])

df['Simplified_Model'] = df['Model'].apply(simplify_model_name)

# Ensure data is sorted by size for the trend line
df = df.sort_values('Model_Size_M')

# --- MODIFICATION: Apply exp() to Test Accuracy and set Y-scale to log ---
df['Exp_Acc_V2_SelfSup'] = np.exp(df['Acc_V2_SelfSup'])
Y_COLUMN = 'Exp_Acc_V2_SelfSup'
# 使用 \mathrm{} 解决 Mathtext 渲染问题
Y_LABEL = r'$\exp(\mathrm{Test\; Accuracy})$' 
# --------------------------------------------------------------------------

# Plotting code starts here

# Set plotting style
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(10, 6))

# Set background color and grid
plt.grid(True, which='major', linestyle='--', alpha=0.3)
plt.gca().set_facecolor((1, 1, 1, 0.9))

# Bubble size scaled directly by the parameter count (Model_Size_M)
size_scale = 1.0

# Define colors for Model Family
colors = {
    'ConvNeXt': (100/255, 149/255, 237/255, 0.8), # Cornflower blue
    'ResNet': (240/255, 128/255, 128/255, 0.8)    # Light coral
}

# PLOT TREND LINES (Dashed Light Gray)
for family in ['ConvNeXt', 'ResNet']:
    df_family = df[df['Model_Family'] == family].copy()
    
    # Sort by log size to ensure line connects points in order
    df_family = df_family.sort_values('Log_Model_Size')
    
    plt.plot(df_family['Log_Model_Size'], df_family[Y_COLUMN],
             linestyle='--', color='lightgray', linewidth=1.5, zorder=1) # zorder=1 ensures it's behind the bubbles

# PLOT SCATTER POINTS
for family in ['ConvNeXt', 'ResNet']:
    df_family = df[df['Model_Family'] == family].copy()

    # Scatter plot: X=Log_Model_Size (continuous), Y=Exp_Acc_V2_SelfSup, S=Model_Size_M
    plt.scatter(df_family['Log_Model_Size'], df_family[Y_COLUMN],
                s=df_family['Model_Size_M'] * size_scale, # Use original size for bubble area
                color=colors[family],
                edgecolors='white', linewidths=1.5, marker='o',
                label=f'_nolegend_', zorder=2) # zorder=2 ensures it's above the lines

# Add parameter annotation (below the point)
for i in range(len(df)):
    row = df.iloc[i]
    x_pos = row['Log_Model_Size'] # Log transformed X position
    y_pos = row[Y_COLUMN]
    params_m = row['Model_Size_M']
    
    # Annotation placement adjusted for log scale Y-axis: place it slightly below the point by using a multiplicative factor (0.9)
    plt.text(x_pos, y_pos * 0.9,
             f"{params_m:.1f}M",
             ha='center', fontsize=9, color='gray')


# Axis settings
# Mathtext fix: using \mathrm{} instead of \text{}
plt.xlabel(r'$\log_{10}(\mathrm{Model\; Parameters})$', fontsize=12) 
plt.ylabel(Y_LABEL, fontsize=12)

# plt.yscale('log')
# Set Y-limits with a small buffer on the log scale
y_min = df[Y_COLUMN].min() * 0.9 
y_max = df[Y_COLUMN].max() * 1.1
plt.ylim(y_min, y_max)
# ---------------------------------------------------------

# Set X-ticks to show actual Parameter count instead of raw log values
log_ticks = np.array([10, 20, 50, 100, 200, 500])
plt.xticks(np.log10(log_ticks), [f'{t:.0f}M' for t in log_ticks])
plt.xlim(np.log10(df['Model_Size_M'].min() * 0.8), np.log10(df['Model_Size_M'].max() * 1.1)) # Auto scale X-limit based on log values

# Mathtext fix: using \mathrm{} instead of \text{}
plt.title(r'Test Accuracy vs. Model Parameters', fontsize=14, weight='bold', pad=10)
# ----------------------------------------------------------------------------

# Custom Legend to show Model Family (Color)
legend_elements = [
    # Family Color Legend
    Line2D([0], [0], marker='o', color='w', label='ConvNeXt', markerfacecolor=colors['ConvNeXt'][:3], markersize=10, markeredgecolor='white'),
    Line2D([0], [0], marker='o', color='w', label='ResNet', markerfacecolor=colors['ResNet'][:3], markersize=10, markeredgecolor='white'),
]

# Plot the custom legend
plt.legend(handles=legend_elements, title='Legend', frameon=False, loc='lower right')

plt.tight_layout()
plt.savefig('accuracy_vs_model_size.pdf')