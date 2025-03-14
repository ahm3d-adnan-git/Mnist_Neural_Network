import sys
import re

def parse_log_file(log_file):
    """Parses the log file to extract predictions and labels."""
    predictions = []
    labels = []
    pattern = re.compile(r"Prediction=(\d+)\.\s+Label=(\d+)")

    try:
        with open(log_file, 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    predictions.append(int(match.group(1)))
                    labels.append(int(match.group(2)))
    except FileNotFoundError:
        print(f"Error: File '{log_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error while reading the file: {e}")
        sys.exit(1)

    if not predictions or not labels:
        print("Warning: No predictions found in the log file.")
        sys.exit(1)

    return predictions, labels

def calculate_accuracy(predictions, labels):
    """Calculates the accuracy of the predictions."""
    correct_predictions = sum(p == l for p, l in zip(predictions, labels))
    total = len(predictions)
    return (correct_predictions / total) * 100 if total > 0 else 0.0

def main():
    """Main function to parse the log file and compute accuracy."""
    if len(sys.argv) != 2:
        print("Usage: python evaluate_accuracy_from_log.py <log_predictions.txt>")
        sys.exit(1)

    log_file = sys.argv[1]
    predictions, labels = parse_log_file(log_file)
    accuracy = calculate_accuracy(predictions, labels)

    print(f"Total Predictions: {len(predictions)}")
    print(f"Correct Predictions: {sum(p == l for p, l in zip(predictions, labels))}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
