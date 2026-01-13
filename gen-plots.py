"""
script to generate the plots for the precision-recall comparison across different block sizes (K={5, 7, 9})
TODO: The current implementation requires manually entering these values; instead, better create a function that calculates these values 
from the predicted maps and ground-truth mappings;
"""

import os
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif"
})


## helper function to generate 2x2 plots
def gen_plots_2x2(datasets, titles, micro: bool):
    K = [5, 7, 10]
    chapters = ['dig', 'inf', 'resp']
    colors = {
        'dig': 'tab:blue',
        'inf': 'tab:orange',
        'resp': 'tab:green'
    }

    out = './out/plots'
    os.makedirs(out, exist_ok=True)

    filename = f'{out}/micro-avg-plot.png' if micro else f'{out}/macro-avg-plot.png'

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, data, title in zip(axes, datasets, titles):
        for chapter in chapters:
            precision = [data[str(k)][chapter][0] for k in K]
            recall = [data[str(k)][chapter][1] for k in K]

            ax.plot(
                K, precision,
                color=colors[chapter],
                linestyle='-',
                marker='o',
                label=f'{chapter} Precision'
            )

            ax.plot(
                K, recall,
                color=colors[chapter],
                linestyle='--',
                marker='o',
                label=f'{chapter} Recall'
            )

        ax.set_xlabel(
            'K',
            fontsize=14,
            fontstyle='italic'
        )
        ax.set_ylabel(
            f'Score ({"micro" if micro else "macro"}-averaged)',
            fontsize=14,
            fontstyle='italic'
        )
        ax.set_xticks(K)
        ax.set_ylim(0.4, 1)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.text(
            0.5, -0.25, title,                # x centered, y below axis
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=14,
            family='serif'
        )


    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(filename, dpi=300)


if __name__ == '__main__':
    ## Micro Averaged Precision and Recall Values
    d9cm210cm = {
        '5': {'dig': [0.71, 0.85],'inf': [0.63, 0.81],'resp': [0.65, 0.75]},
        '7': {'dig': [0.66, 0.86],'inf': [0.59, 0.84],'resp': [0.59, 0.77]},
        '10': {'dig': [0.60, 0.87],'inf': [0.52, 0.85],'resp': [0.59, 0.76]}
        }
    d10cm9cm = {
        '5': {'dig': [0.64 , 0.8],'inf': [0.7 , 0.83],'resp': [0.6 , 0.68]},
        '7': {'dig': [0.59 , 0.81],'inf': [0.63 , 0.85],'resp': [0.55 , 0.69]},
        '10': {'dig': [0.5 , 0.8],'inf': [0.57 , 0.85],'resp': [0.49 , 0.7]}
        }
    d10am11 = {
        '5': {'dig': [0.47 , 0.64],'inf': [0.55 , 0.74],'resp': [0.58 , 0.76]},
        '7': {'dig': [0.45 , 0.66],'inf': [0.49 , 0.75],'resp': [0.61 , 0.8]},
        '10': {'dig': [0.41 , 0.66],'inf': [0.44 , 0.77],'resp': [0.53 , 0.82]}
        }
    d1110am = {
        '5': {'dig': [0.54 , 0.53],'inf': [0.59 , 0.76],'resp': [0.63 , 0.77]},
        '7': {'dig': [0.49 , 0.54],'inf': [0.56 , 0.79],'resp': [0.62 , 0.82]},
        '10': {'dig': [0.43 , 0.57],'inf': [0.47 , 0.82],'resp': [0.53 , 0.79]}
        }

    ## Macro Averaging
    d9cm210cm_macro = {
        '5': {'dig': [0.82 , 0.9],'inf': [0.75 , 0.85],'resp': [0.78 , 0.83]},
        '7': {'dig': [0.82 , 0.91],'inf': [0.74 , 0.87],'resp': [0.74 , 0.83]},
        '10': {'dig': [0.79 , 0.92],'inf': [0.71 , 0.89],'resp': [0.77 , 0.84]}
        }

    d10cm9cm_macro = {
        '5': {'dig': [0.74 , 0.83],'inf': [0.78 , 0.86],'resp': [0.67 , 0.74]},
        '7': {'dig': [0.72 , 0.84],'inf': [0.76 , 0.88],'resp': [0.64 , 0.74]},
        '10': {'dig': [0.67 , 0.84],'inf': [0.72 , 0.88 ],'resp': [0.72 , 0.88 ]}
        }

    d10am11_macro = {
        '5': {'dig': [0.64 , 0.73],'inf': [0.67 , 0.78],'resp': [0.7 , 0.81]},
        '7': {'dig': [0.62 , 0.74],'inf': [0.64 , 0.79],'resp': [0.73 , 0.85]},
        '10': {'dig': [0.58 , 0.74],'inf': [0.62 , 0.81],'resp': [0.68 , 0.87]}
        }

    d1110am_macro = {
        '5': {'dig': [0.61 , 0.66],'inf': [0.71 , 0.8],'resp': [0.71 , 0.77]},
        '7': {'dig': [0.59 , 0.66],'inf': [0.69 , 0.82],'resp': [0.72 , 0.82]},
        '10': {'dig': [0.58 , 0.69],'inf': [0.65 , 0.84],'resp': [0.66 , 0.79]}
        }
    
    ## titles
    titles = ['ICD-9-CM → ICD-10-CM', 'ICD-10-CM → ICD-9-CM', 'ICD-10-AM → ICD-11', 'ICD-11 → ICD-10-AM']
    ## generate micro-averaged plots
    data_micro = [d9cm210cm, d10cm9cm, d10am11, d1110am]
    gen_plots_2x2(data_micro, titles, True)

    ## generate macro-averaged plots
    data_macro = [d9cm210cm_macro, d10cm9cm_macro, d10am11_macro, d1110am_macro]
    gen_plots_2x2(data_macro, titles, False)


