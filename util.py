#util.py
import matplotlib.pyplot as plt
import seaborn as sns

# draw confusion matrix
def draw_confusion_matrix(confusion_matrix, class_names):
    fig = plt.figure(figsize=(6, 6))
    heatmap = sns.heatmap(
        confusion_matrix, xticklabels=class_names, yticklabels=class_names,
        annot=True, fmt="d", cbar=True, square=True, cmap='YlGnBu')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    return fig