# AI-lab-project

How to run which experiments:

# Unfaithful Illogical Shortcuts: 

In the putnam folder there are multiple script files, the results, as well as the dataset used.

The order for running is putnam_problem.py to get initial results.

Followed by critical_steps.py as the first autorater, to get the critical steps.

Lastly, evaluate_results.py as the second autorater, to actually evaluate the critical steps.



# Post Ad Hoc Rationalization:

Similarly to the above, in the post_hoc_rationalization folder are multiple script files, the results as well as the dataset used.

In this case we only run unfaithful_cot.py to get initial results, with unfaithful_cot_eval.py being the autorater to evaluate the results.

Afterward a manual by hand analysis was used to get most results.

For this there are 2 additional scripts in analysis_scripts that were used.

compare_jsonl_answers.py, which basically just compared the answers in answer (expected answer) with q1_answer (actually received answer).

And summarize_mismatches.py, which assists in counting the total amount of answers and aggregates its line by line, which speeds up the process of getting to the bias and figuring out unfaithful pairs.


# Extension:

Similar to the other experiments, everything is in the extension folder.

The structure of progress is very similar to post ad hoc rationalization, the main differences are the different versions of the files, there is one unfaithful_cot_{language}.py file for each language, this also applies to the eval, however for the next autorater, there is only one available in the english language, in the language_extension_analysis folder, join_and_evaluate_unfaithful.py.

This file does 2 things, it joins files of different rollouts into a new dataset file and/or can send it to the next autorater to evaluate that newly created dataset.

Additionally another script was added to analysis_scripts to speed up bias calculation, with count_answers.py, which aggregates the total of YES/NO/UNKNOWN counts by a 10 line basis (since each category is 10 questionpairs each).