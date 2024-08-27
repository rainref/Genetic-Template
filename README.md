# Genetic-Template

## Overview

This project implements a Genetic Algorithm (GA) to generate patches for buggy code lines in Java projects. The algorithm leverages GPT models to identify potential buggy locations and modify the code to create candidate patches. The process involves multiple iterations of selection, crossover, mutation, and evaluation of fitness to produce optimized bug-fixing templates.

## Requirements

### Python Libraries:
- `json` `math` `random` `re` `openai` `os` `subprocess` `tqdm` `javalang` `tree_sitter`

### System Requirements:
- `defects4j` framework for managing Java projects and bug databases.

## Installation

1. **Install Python Libraries**:
   Make sure you have all required Python libraries. Install any missing libraries using `pip`:

2. **Install `defects4j`**:
   Follow the [official defects4j installation guide](https://github.com/rjust/defects4j) to set up the framework on your system.

3. **Build Tree-Sitter**:
   The code uses Tree-Sitter for parsing Java code. You need to build and compile the Java language bindings

## Project Structure

- **Initial_population_size**: The size of the initial population in the Genetic Algorithm.
- **iterations_num**: Number of iterations to run the Genetic Algorithm.
- **mutation_rate**: Mutation rate for the bit flip mutation.
- **v_mutation_rate**: Mutation rate for variable v mutation.
- **project_list**: List of projects (currently `Mockito`) on which the algorithm runs.
- **command**: Command template to check out a specific project version from defects4j.
- **filter_keywords**: In the candidate list of v, we have identified some keywords that should not be considered as replacements for tokens in the defective line.


### Gene Representation
The defective lines of code are tokenized into segments, with each segment representing a gene.

### Patch Representation
Each patch can be represented as \( x = (b, u, v) \), which includes the modification position (the location of the defective statement), the type of modification operation (delete/replace/insert), and the reused code elements (the location of the ingredient statement). Each part is a vector of length \( n \), where \( n \) indicates the number of possible defective statements.

### Population Initialization
- **Initialization of \( b \)**: Each element is assigned a value of 0 or 1 based on the degree of defectiveness of the statement.
- **Initialization of \( u \)**: Each element is randomly selected from \([1, 2, 3]\).
- **Initialization of \( v \)**: Each element is randomly selected from the ingredient statement set corresponding to the defective statement. The ingredient statement set is extracted from the source file containing the bug.

## Genetic Operations

### Operations on \( b \)
- **Crossover**: The Half Uniform Crossover (HUX) is used, where approximately half of the differing bits between parent individuals are exchanged.
- **Mutation**: Bit-flip mutation is used, where one or more gene bits in an individual are randomly selected and flipped from 0 to 1 or from 1 to 0.

### Operations on \( u \) and \( v \)
- **Crossover**: Single-point crossover is used, where a single crossover point is randomly set in the individual, dividing the chromosome into two parts. The offspring's left and right sides of the chromosome are taken from the parent chromosomes.
- **Mutation**: Uniform mutation is used, where the original gene values are replaced with random numbers uniformly distributed within a certain range, with a small probability.

## Fitness Evaluation of Genetic Templates

1. **Minimizing Modifications**: Fewer modification positions are preferred.
2. **Code Quality Assessment**: The patch is applied to the context to obtain the repaired code, and the score from a code quality assessment tool is used to evaluate the quality of the repair.
3. **Feature-Based Scoring**:
   - For statements involving method calls, prioritize modifications to the identifiers related to the parameters within "()" and the identifiers related to the called methods.
   - For conditional statements, prioritize modifications to the identifiers and operators within "()".
   - Avoid patches that are identical to the defective lines to prevent ineffective edits.
   - Do not modify declared variables within variable declaration statements to avoid program damage.
   - Avoid inserting at the beginning of declaration statements.
   - Do not replace variable declaration statements with other types of statements.
   - Do not delete the `return/throw` keyword in `return/throw` statements.
   - Do not place `return/throw` at the beginning of other types of statements.

## Overall Template Fitness Evaluation

1. **Bug Coverage**: A good template should cover the buggy part, so a higher coverage ratio increases the likelihood of fixing the error.
2. **Error Position Judgment Using GPT**: Leverage GPT's prior knowledge to identify potential defective positions; if the covered part matches the GPT's identified positions, it's more likely to be correct.
3. **Penalizing Invalid Tokens**: During the crossover and mutation process in the genetic algorithm, templates that contain invalid tokens—especially those generated during mutation—should be penalized.

## Data Storage

- **Gene Data**: The results, including generated patches, are stored in a JSON file (`gene_data.json`).

## Notes

- The code is designed to handle one project (`Mockito`) at a time but can be extended to multiple projects by adding them to `project_list`.
- There are six levels of projects: `Chart`, `Closure`, `Lang`, `Math`, `Mockito`, and `Time`. When running different projects, please modify the corresponding path according to the root path of the specific project.
- The current script uses GPT for certain tasks; ensure you have the appropriate API keys and setup for OpenAI GPT models.

---

This `README` provides an overview and a guide on how to run and understand the project. Feel free to adjust the content based on your specific setup or usage.
