def mean_average_f1(average_precision: float, average_recall: float) -> float:
    return 2 * (average_precision * average_recall) / (average_precision + average_recall + 1e-10)
