import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visual_by_tsne(data, labels, name):
    tsne = TSNE(n_components=2)
    embedded_data = tsne.fit_transform(data)


    colors = plt.cm.rainbow(np.linspace(0, 1, 20))
    
    plt.figure(figsize=(10, 8))

    for task_id in range(20):

        task_indices = np.where(labels == task_id)
        task_data = embedded_data[task_indices]

        plt.scatter(task_data[:, 0], task_data[:, 1], label=f'Task {task_id}', color=colors[task_id])

    # plt.legend()
    # plt.title("t-SNE Visualization with Different Colors for Tasks")
    plt.savefig(name + '.png', dpi=400, quality=95)
    # plt.show()

