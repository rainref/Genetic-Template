import json
import math
import random
import re
import openai
import os
import subprocess
from tqdm import tqdm
from javalang.tokenizer import tokenize
from tree_sitter import Language, Parser


Initial_population_size = 10
iterations_num = 10
mutation_rate = 0.1
v_mutation_rate = 0.4

project_list = ['Mockito']

command = "defects4j checkout -p {} -v {}b -w ./{}{}"

fold_path = '/root/zh/repair/defects4j/framework/projects/{}/modified_classes'
path = '/root/zh/repair/defects4j/framework/projects/{}/patches/{}.src.patch'
pos_path = '/root/zh/repair/defects4j/framework/projects/{}/modified_classes/{}.src'

prompt_fitness = '''
{}
This code is the defect code line. Please judge the three locations where he is most likely to parody the problem and return me a location linked list like [[start1,end1],[start2,end2],[start3,end3]]
Just output the list of results, not superfluous explanations.
'''
prompt_bug = '''
This code is the defect code line. 
'''
prompt_bug2 = '''
This is the token list of defect lines. Please give the defect probability of each token in the token list according to the defect lines, and return the result to me in the form of a list. 
For example, [p1, p2, p2, p3...pn].Pn is probability value.
Just output the list of results, not superfluous explanations.
'''
filter_keywords = ['public', 'private', 'protected', 'native', 'abstract', 'interface', 'class', 'goto', 'enum',
                   'extends', 'default', 'strictfp', 'implements', 'package', 'void', 'import', 'throws', 'return',
                   'try', 'catch']
pattern = r"\[\[\w+,\w+\],\[\w+,\w+\],\[\w+,\w+\]\]"
pattern2 = r'\[([\d.,\s]+)\]'

JAVA_LANGUAGE = Language("build/my-languages.so", "java")
java_parser = Parser()
java_parser.set_language(JAVA_LANGUAGE)

error_text = '(ERROR) @error'

method_invocation_text = '''
(method_invocation) @method_invocation
'''
if_text = '''
(if_statement) @if_statement
'''
local_vardecl_text = '''
(local_variable_declaration) @local_variable_declaration
'''

def count_files_in_folder(folder_path):
    try:
        items = os.listdir(folder_path)
        file_count = sum(os.path.isfile(os.path.join(folder_path, item)) for item in items)
        return file_count
    except OSError:
        return -1


def ask_gpt(messages):
    try:
        openai.organization = ''
        openai.base_url = ""
        openai.api_key = ""
        completion = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            timeout=30)
        return completion.choices[0].message.content
    except Exception as e:
        print(e)
        return ask_gpt(messages)

def roulette_wheel_selection(probabilities):
    cumulative_probabilities = []
    cumulative_sum = 0
    for p in probabilities:
        cumulative_sum += p
        cumulative_probabilities.append(cumulative_sum)
    r1 = random.uniform(-50, sum(probabilities))
    r2 = random.uniform(-50, sum(probabilities))
    index1 = next(i for i, cp in enumerate(cumulative_probabilities) if r1 <= cp)
    index2 = next(i for i, cp in enumerate(cumulative_probabilities) if r2 <= cp)
    return index1, index2


def generate_random_value(probability_1):
    random_value = random.uniform(0, 1)
    if random_value < probability_1:
        return 1
    else:
        return 0


def population_init(prob_list, v_dict, token_list):
    total_sum = sum(prob_list)
    if total_sum == 0:
        total_sum = 1
    numbers = [1, 2, 3]
    b_list, u_list, v_list = [], [], []
    for i in range(Initial_population_size):
        b, u, v = [], [], []
        for prob in prob_list:
            b.append(generate_random_value(prob / total_sum))
            u.append(random.choice(numbers))
        if all(x == 0 for x in b):
            b[random.randint(0, len(b) - 1)] = 1
        b_list.append(b)
        u_list.append(u)
        for token in token_list:
            try:
                v.append(v_dict[type(token)][random.randint(0, len(v_dict[type(token)]) - 1)])
            except Exception as e:
                print(e)
                v.append(token)
        v_list.append(v)
    return b_list, u_list, v_list


def HUX(parent1, parent2):
    offspring = []
    for gene1, gene2 in zip(parent1, parent2):
        if random.uniform(0, 1) < 0.5:
            offspring.append(gene1)
        else:
            offspring.append(gene2)
    return offspring


def bit_flip_mutation(individual):
    mutated_individual = []
    for gene in individual:
        if random.uniform(0, 1) < mutation_rate:
            mutated_gene = 1 - gene  # Flip the bit
        else:
            mutated_gene = gene
        mutated_individual.append(mutated_gene)
    return mutated_individual


def single_point_crossover(parent1, parent2):
    try:
        crossover_point = random.randint(0, len(parent1) - 1)
    except Exception as e:
        crossover_point=0
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    # offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    return offspring1


def u_mutation(individual):
    mutated_individual = []
    for gene in individual:
        if random.uniform(0, 1) < mutation_rate:
            mutated_gene = random.choice([1, 2, 3])  # Generate a new gene uniformly between 0 and 1
        else:
            mutated_gene = gene
        mutated_individual.append(mutated_gene)
    return mutated_individual


def v_mutation(individual, token_list, v_dict):
    mutated_individual = []
    for i in range(len(individual)):
        if random.uniform(0, 1) < v_mutation_rate:
            try:
                mutated_gene = v_dict[type(token_list[i])][random.randint(0, len(
                v_dict[type(token_list[i])]) - 1)]  # Generate a new gene uniformly between 0 and 1
            except Exception as e:
                print(e)
                mutated_gene =token_list[i]
        else:
            mutated_gene = individual[i]
        mutated_individual.append(mutated_gene)
    return mutated_individual


def get_template(b, u, v, token_list, buggy_line):
    one_indices = [index for ident, value in enumerate(b) if value == 1]
    if len(one_indices) > 1:
        selected_indices = random.sample(one_indices, k=random.randint(1, 2))
    else:
        selected_indices = one_indices
    for i in selected_indices:
        try:
            buggy_line = buggy_line.replace(token_list[i].value, '<SPAN>')
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            buggy_line = buggy_line.replace(token_list[-1].value, '<SPAN>')
    for i in range(len(b) - 1):
        buggy_line = re.sub(r'<SPAN>\s*<SPAN>', '<SPAN>', buggy_line)
    fix_buggy_line = buggy_line

    for bi, ui, vi, ti in zip(b, u, v, token_list):
        if bi == 1:
            if ui == 1:
                fix_buggy_line = fix_buggy_line.replace(str(ti.value), '')
            elif ui == 2:
                fix_buggy_line = fix_buggy_line.replace(str(ti.value), str(vi))
            else:
                fix_buggy_line = fix_buggy_line.replace(str(ti.value), str(vi) + ' ' + str(ti.value))

    return fix_buggy_line


def mask_pos(buggy_line_cl, mask_line):
    begin_index = -1
    for i in range(min(len(buggy_line_cl), len(mask_line))):
        if buggy_line_cl[i] != mask_line[i]:
            begin_index = i
            break
    end_index = -1
    for i in range(-1, -min(len(buggy_line_cl), len(mask_line)) - 1, -1):
        if buggy_line_cl[i] != mask_line[i]:
            end_index = len(buggy_line_cl) + i
            break
    return [begin_index, end_index]


def fitness_score(template, buggy_line):
    buggy_clean = buggy_line.replace(' ', '')
    all = len(buggy_clean)

    unmask = len(template.replace(' ', '').replace('<SPAN>', ''))
    if all == 0:
        score1 = 1
    else:
        score1 = 1 - unmask / all
    buggy_line = buggy_line.strip()
    resp = ask_gpt([{"role": "user", "content": prompt_fitness.format(buggy_line)}])
    pattern = r'\[\[\d+,\s*\d+\](?:,\s*\[\d+,\s*\d+\])*\]'
    match = re.findall(pattern, resp)
    while not match:
        resp = ask_gpt([{"role": "user", "content": prompt_fitness.format(buggy_line)}])
        match = re.findall(pattern, resp)
    print('match[0]', match[0])
    try:
        pos_list = json.loads(match[0])
    except Exception as e:
        return fitness_score(template,buggy_line)

    mask = mask_pos(buggy_line, template)
    x = 0
    for pos in pos_list:
        intersection_start = max(pos[0], mask[0])
        intersection_end = min(pos[1], mask[1])
        if pos[1] != pos[0]:
            x += max(0, intersection_end - intersection_start + 1) / (pos[1] - pos[0])
        else:
            x += 0
    if x == 0:
        score2 = 0
    else:
        score2 = math.exp(1 - x)

    return score1 + score2


def allmask_template(b, token_list, buggy_line):
    for i, t in zip(b, token_list):
        if i == 1:
            buggy_line = buggy_line.replace(t.value, '<SPAN>')
    merged_text = ''
    for i in range(len(b) - 1):
        merged_text = re.sub(r'<SPAN>\s*<SPAN>', '<SPAN>', buggy_line)

    return merged_text


# token_list is a node with a type
def fitness(b, u, v, token_list, buggy_line):
    one_indices = [index for index, value in enumerate(b) if value == 1]
    # Modifying the position is appropriate within a limit of 1-2 tokens.
    score1 = math.exp(2 - len(one_indices))

    fix_buggy_line = buggy_line
    # Modify the line according to the BUV.
    for bi, ui, vi, ti in zip(b, u, v, token_list):
        if bi == 1:
            if ui == 1:
                fix_buggy_line = fix_buggy_line.replace(str(ti.value), '')
            elif ui == 2:
                fix_buggy_line = fix_buggy_line.replace(str(ti.value), str(vi))
            else:
                fix_buggy_line = fix_buggy_line.replace(str(ti.value), str(vi) + ' ' + str(ti.value))

    tree = java_parser.parse(bytes(fix_buggy_line, "utf8", ))
    root_node = tree.root_node
    error_query = JAVA_LANGUAGE.query(error_text)
    error_capture = error_query.captures(root_node)
    # Restore the line to ensure the syntax is correct.
    if len(error_capture) == 0:
        score2 = 1
    else:
        score2 = -10

    # Obtain the corresponding template.
    template = allmask_template(b, token_list, buggy_line)
    patternkuo = r'\((.*?)\)'
    # For statements with method calls, there is a preference for modifying the identifiers related to parameters within the parentheses and the identifiers related to the method being called.
    method_query = JAVA_LANGUAGE.query(method_invocation_text)
    method_capture = method_query.captures(root_node)
    if len(method_capture) != 0:
        substrings = re.findall(patternkuo, template)
        if '<SPAN>' in ''.join(substrings):
            score3 = 1
        else:
            score3 = 0
    else:
        score3 = 0
    # For conditional statements, there is a tendency to modify the identifiers and operators within the parentheses.
    if_query = JAVA_LANGUAGE.query(if_text)
    if_capture = if_query.captures(root_node)
    if len(if_capture) != 0:
        substrings = re.findall(patternkuo, template)
        if '<SPAN>' in ''.join(substrings):
            score4 = 1
        else:
            score4 = 0
    else:
        score4 = 0

    # Remove patches that are consistent with defective lines to avoid ineffective edits
    # Do not modify the declared variables in variable declaration statements to prevent program disruption
    # Do not perform insertion operations at the beginning of declaration statements
    local_query = JAVA_LANGUAGE.query(local_vardecl_text)
    local_capture = local_query.captures(root_node)
    if len(local_capture) != 0:
        for bi, ti in zip(b, token_list):

            if 'Identifier' in str(type(ti)) and bi == 1:
                score5 = -10
            else:
                score5 = 1

        if b[0] == 1 and u[0] == 3:
            score6 = -10
        else:
            score6 = 1
    else:
        score5 = 0
        score6 = 0

    # Do not delete the return or throw keywords in return or throw statements.
    if 'return' in buggy_line:
        if b[0] == 1 and u[0] == 1:
            score7 = -10
        else:
            score7 = 1
    else:
        score7 = 0

    if 'throw' in buggy_line:
        if b[0] == 1 and u[0] == 1:
            score8 = -10
        else:
            score8 = 1
    else:
        score8 = 0
    return score1 + score2 + score3 + score4 + score5 + score6 + score7 + score8


def extract_defect_lines(diff_content):
    defect_lines = []
    current_defect = []
    lines = diff_content
    for line in lines:
        if line.startswith("-") and line[1] != '-':
            # If a line starts with +, it's part of a defect block.
            current_defect.append(line[1:].strip())
        else:
            if current_defect:
                # If we reach a non-defect line and have accumulated defect lines, save them as a block.
                defect_lines.append("\n".join(current_defect))
                current_defect = []

    # If the last lines were a defect block, save them too.
    if current_defect:
        defect_lines.append("\n".join(current_defect))
    return defect_lines

with open('./data/gene_data.json', 'r') as file:
    all_data = json.load(file)
for pro in project_list:
    folder = fold_path.format(pro)
    num = count_files_in_folder(folder)
    for i in tqdm(range(int(num))):
        id = i + 1
        cmd = command.format(pro, id, pro, id)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("The command executed successfully:")
        else:
            print("The command failed to execute:")
            continue
        position = pos_path.format(pro, id)
        with open(position, 'r', encoding='utf-8', errors="ignore") as f:
            pos_data = f.readline()

        path_i = path.format(pro, id)
        with open(path_i, 'r', encoding='utf-8', errors="ignore") as f:
            data = f.readlines()
        # Obtain the lists of fixed lines and buggy lines.
        fix_line = []
        buggy_line = extract_defect_lines(data)

        if not buggy_line:
            continue
        # Obtain the file containing the buggy lines
        bugfile_path = './' + pro + str(id) + '/src/' + pos_data.replace('.', '/').replace('\n', '') + '.java'
        try:
            with open(bugfile_path, 'r', encoding='utf-8', errors="ignore") as f:
                javadata = f.readlines()
        except Exception as e:
            bugfile_path = './' + pro + str(id) + '/src/java/' + pos_data.replace('.', '/').replace('\n', '') + '.java'
            with open(bugfile_path, 'r', encoding='utf-8', errors="ignore") as f:
                javadata = f.readlines()

        v_list = list(tokenize(''.join(javadata)))
        v_dict = {}
        for v in v_list:
            if v_dict.get(type(v)) is not None and v.value not in filter_keywords and v.value not in v_dict[type(v)]:
                v_dict[type(v)].append(v.value)
            else:
                v_dict[type(v)] = []
                v_dict[type(v)].append(v.value)

        for index, buggy in enumerate(buggy_line):
            key_name = pro + '-' + str(id) + '-place-' + str(index) + '.java'
            if key_name in all_data:
                bs = all_data.get(key_name).get('bs_templates')
            else:
                continue
            if 'ours' in all_data[key_name]:
                continue

            token_list = list(tokenize(buggy))
            if not token_list:
                continue
            t_list = []
            for token in token_list:
                t_list.append(token.value)
            bpos = json.dumps(t_list)

            prompt = buggy + prompt_bug + bpos + prompt_bug2
            print('prompt', prompt)
            response = ask_gpt([{"role": "user", "content": prompt}])
            matches = re.findall(pattern2, response)
            print('response', response)
            while not matches:
                response = ask_gpt([{"role": "user", "content": prompt}])
                matches = re.findall(pattern2, response)
                print('response', response)

            print('Match results', matches)
            prob_list = matches[0].split(',')
            prob_list = [float(num_str) for num_str in prob_list]
            # Population initialization
            b_list, u_list, v_list = population_init(prob_list, v_dict, token_list)

            print('b:', b_list)
            print('u:', u_list)
            print('v:', v_list)
            print('Defective code line：', buggy)

            fitness_list = []
            for i in range(len(b_list)):
                fitness_list.append(fitness(b_list[i], u_list[i], v_list[i], token_list, buggy))

            # selected_parents = random.sample(range(0, len(b_list)-1), 2)
            for i in range(iterations_num):
                index1, index2 = roulette_wheel_selection(fitness_list)
                offspringb = HUX(b_list[index1], b_list[index2])
                offspringb = bit_flip_mutation(offspringb)

                offspringu = single_point_crossover(u_list[index1], u_list[index2])
                offspringu = u_mutation(offspringu)

                offspringv = single_point_crossover(v_list[index1], v_list[index2])
                offspringv = v_mutation(offspringv, token_list, v_dict)

                print('b', offspringb)
                print('u', offspringu)
                print('v', offspringv)
                b_list.append(offspringb)
                u_list.append(offspringu)
                v_list.append(offspringv)
                fitness_list.append(fitness(offspringb, offspringu, offspringv, token_list, buggy))

            index1, index2 = roulette_wheel_selection(fitness_list)
            print('Selected template1', get_template(b_list[index1], u_list[index1], v_list[index1], token_list, buggy))
            print('Selected template2', get_template(b_list[index2], u_list[index2], v_list[index2], token_list, buggy))
            gene_template = []
            for b, u, v in zip(b_list, u_list, v_list):
                template = get_template(b, u, v, token_list, buggy)
                gene_template.append((fitness_score(template, buggy), template))

            bs_template = []

            for template in bs:
                bs_template.append((fitness_score(template, buggy), template))

            ours = gene_template + bs_template
            ours.sort(reverse=True, key=lambda x: x[0])
            ours_list = []
            for our in ours:
                ours_list.append(our[1])
            all_data[key_name]['ours'] = ours_list

with open('./data/gene_data.json', 'w') as file:
    json.dump(all_data, file, indent=4)

