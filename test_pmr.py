import jsonlines


def read_jsonl(fname) -> list:
    data = []
    with open(fname, 'r+', encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            data.append(item)
    return data


print(len(read_jsonl('./run_scripts/pmr/checkpoints_large/20_2e-5/test_text_predict.jsonl')))