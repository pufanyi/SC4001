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
        'convnextv2-tiny-22k-384',
        'ViT-B/16',
        'ViT-L/16',
        'convnextv2-atto-1k-224',
        'convnextv2-tiny-1k-224'
    ],
    'Model_Size_Str': ['659M', '89M', '60M', '25.6M', '11.7M', '28M', '86M', '307M', '3.7M', '9.1M'],
    'Acc_V2_SelfSup': [99.3894993894993, 99.2673992673992, 98.2905982905982, 96.9474969474969, 92.4298, 99.1452991452991, 99.2673992673992, 99.2673992673992, 89.98778998779, 98.046398046398],
}

df = pd.DataFrame(data)

# 1. Clean 'Model Size': remove 'M' and convert to float
df['Model_Size_M'] = df['Model_Size_Str'].str.replace('M', '').astype(float)

# 2. LOG TRANSFORM X-AXIS
df['Log_Model_Size'] = np.log10(df['Model_Size_M'])

# 3. Extract Model Family (for color/marker)
df['Model_Family'] = df['Model'].apply(lambda x: 'ConvNeXt' if 'convnext' in x.lower() else ('ViT' if 'vit' in x.lower() else 'ResNet'))

def simplify_model_name(name):
    if 'convnext' in name.lower():
        return name.split('-')[1].capitalize()
    elif 'resnet' in name.lower():
        return name.split('-')[0].capitalize() + '-' + name.split('-')[1]
    elif 'vit' in name.lower():
        return name
    return name

df['Acc_Proportion'] = df['Acc_V2_SelfSup'] / 100.0
df['Exp_Acc_V2_SelfSup_Scaled'] = np.exp(df['Acc_Proportion'])
df['Simplified_Model'] = df['Model'].apply(simplify_model_name)

df = df.sort_values('Model_Size_M')

df['Exp_Acc_V2_SelfSup'] = np.exp(df['Acc_V2_SelfSup'])
Y_COLUMN = 'Exp_Acc_V2_SelfSup'
Y_LABEL = r'$\exp(\mathrm{Test\; Accuracy})$' 
# --------------------------------------------------------------------------

sns.set_theme(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(10, 6))

# Set background color and grid
plt.grid(True, which='major', linestyle='--', alpha=0.3)
plt.gca().set_facecolor((1, 1, 1, 0.9))

size_scale = 1.0

colors = {
    'ConvNeXt': (100/255, 149/255, 237/255, 0.8), 
    'ResNet': (240/255, 128/255, 128/255, 0.8),   
    'ViT': (0/255, 128/255, 128/255, 0.8) 
}

for family in ['ConvNeXt', 'ResNet', 'ViT']: 
    df_family = df[df['Model_Family'] == family].copy()
    
    df_family = df_family.sort_values('Log_Model_Size')
    
    if len(df_family) >= 2:
        plt.plot(df_family['Log_Model_Size'], df_family[Y_COLUMN],
                 linestyle='--', color='lightgray', linewidth=1.5, zorder=1) 

for family in ['ConvNeXt', 'ResNet', 'ViT']: 
    df_family = df[df['Model_Family'] == family].copy()

    # Scatter plot: X=Log_Model_Size (continuous), Y=Exp_Acc_V2_SelfSup, S=Model_Size_M
    plt.scatter(df_family['Log_Model_Size'], df_family[Y_COLUMN],
                s=df_family['Model_Size_M'] * size_scale, 
                color=colors[family],
                edgecolors='white', linewidths=1.5, marker='o',
                label=f'_nolegend_', zorder=2) 

offset_mapping = {
    'ViT-B/16': 1.01, 
}

for i in range(len(df)):
    row = df.iloc[i]
    x_pos = row['Log_Model_Size'] 
    y_pos = row[Y_COLUMN]
    params_m = row['Model_Size_M']
    model_name = row['Model']
    
    y_factor = 0.9 
    ha_align = 'center'
    
    if model_name in offset_mapping:
        y_factor = offset_mapping[model_name] 
        ha_align = 'left' 
        
    plt.text(x_pos, y_pos * y_factor,
             f"{params_m:.1f}M",
             ha=ha_align, fontsize=9, color='gray')


# Axis settings
plt.xlabel(r'$\log_{10}(\mathrm{Model\; Parameters})$', fontsize=12) 
plt.ylabel(Y_LABEL, fontsize=12)

# Set Y-limits with a small buffer on the log scale
y_min = df[Y_COLUMN].min() * 0.9 
y_max = df[Y_COLUMN].max() * 1.1
plt.ylim(y_min, y_max)
# ---------------------------------------------------------

# Set X-ticks to show actual Parameter count instead of raw log values
log_ticks = np.array([10, 20, 50, 100, 200, 500, 800])
plt.xticks(np.log10(log_ticks), [f'{t:.0f}M' for t in log_ticks])
plt.xlim(np.log10(df['Model_Size_M'].min() * 0.8), np.log10(df['Model_Size_M'].max() * 1.1))

plt.title(r'Test Accuracy vs. Model Parameters', fontsize=14, weight='bold', pad=10)
# ----------------------------------------------------------------------------

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='ConvNeXt', markerfacecolor=colors['ConvNeXt'][:3], markersize=10, markeredgecolor='white'),
    Line2D([0], [0], marker='o', color='w', label='ResNet', markerfacecolor=colors['ResNet'][:3], markersize=10, markeredgecolor='white'),
    Line2D([0], [0], marker='o', color='w', label='ViT', markerfacecolor=colors['ViT'][:3], markersize=10, markeredgecolor='white'),
]

plt.legend(handles=legend_elements, frameon=False, loc='lower right')

plt.tight_layout()
plt.show()
# plt.savefig('accuracy_vs_model_size.pdf')