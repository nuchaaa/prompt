from emotion.analysis import (
    analyze_dataset,
    print_summary,
    plot_confusion_matrix,
    plot_per_emotion_accuracy
)

dataset_path = "../dataset"  # folder with angry/happy/sad/... subfolders

df = analyze_dataset(dataset_path, output_csv="results.csv")
overall_acc, per_emotion_acc = print_summary(df)

plot_confusion_matrix(df, save_path="../confusion_matrix.png")
plot_per_emotion_accuracy(per_emotion_acc, save_path="per_emotion_accuracy.png")