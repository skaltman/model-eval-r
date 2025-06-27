import os
import json
import pandas as pd
import pandas as pd
from plotnine import (
    ggplot, aes, geom_bar, scale_fill_manual, scale_y_continuous,
    labs, theme_light, theme, element_text, position_fill, coord_flip
)
from mizani.formatters import percent_format


# Assumes more output than input
# Based on ratios from eval
COST_BLEND_INPUT_MULTIPLIER = 1/3

def get_scores(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    samples = data.get("samples", [])
    
    rows = []
    for sample in samples:
      usage = sample.get("output", {}).get("usage", {})
      row = {
          "id": sample.get("id"),
          "model": sample.get("output", {}).get("model"),
          "score": sample.get("scores", {}).get("model_graded_qa", {}).get("value"),
          "prompt_tokens": usage.get("prompt_tokens"),
          "completion_tokens": usage.get("completion_tokens"),
          "total_tokens": usage.get("total_tokens"),
      }
      rows.append(row)
    
    return pd.DataFrame(rows)

log_dir = "logs"
json_files = [
    os.path.join(log_dir, fname)
    for fname in os.listdir(log_dir)
    if fname.endswith(".json")
]

df_list = [get_scores(fp) for fp in json_files]
df_eval = pd.concat(df_list, ignore_index=True)

df_eval["score_order"] = df_eval["score"].map({"C": 1, "I": 0})

(
    ggplot(df_eval, aes(x="model", fill="reorder(score, score_order)"))
    + geom_bar(position=position_fill())
    + coord_flip()                 
    + scale_fill_manual(
        limits=["C", "I"],       
        values={
            "C": "#6caea7",
            "I": "#ef8a62",
        }
    )
    + scale_y_continuous(labels=percent_format(), expand=(0.005, 0.005))

    + labs(
        y="Percent",
        x=None,
        title="Model performance on Python code generation",
        fill="Score"
    )
    + theme_light()
    + theme(
        plot_subtitle=element_text(face="italic"),
        legend_position="bottom"
    )
)
