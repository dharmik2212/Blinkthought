import json
import sys

def validate_file(filepath):
    print(f"Validating {filepath}...")
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            item = json.loads(line)
            assert "state" in item, f"Missing state at line {line_num}"
            assert "questions" in item, f"Missing questions at line {line_num}"
            for q in item["questions"]:
                assert "id" in q, f"Missing question id at line {line_num}"
                assert "type" in q, f"Missing question type at line {line_num}"
                assert "options" in q, f"Missing options at line {line_num}"
                assert "target" in q, f"Missing target at line {line_num}"
                
                # Check target distribution sums to ~1.0
                target_sum = sum(q["target"].values())
                assert abs(target_sum - 1.0) < 1e-5, f"Target sum {target_sum} != 1.0 for question {q['id']} at line {line_num}"
            count += 1
    print(f"{filepath}: {count} examples validated successfully.")

if __name__ == "__main__":
    for f in ["train.jsonl", "calibration.jsonl", "test.jsonl"]:
        validate_file(f)
    print("All datasets validated successfully!")
